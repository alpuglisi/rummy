"""Model factories, stage-2 rank/suit classifier, inference wrapper and export helpers.

This is section (C) of the card-vision pipeline.  It defines *what* is trained and
*how* it is run, but no training loop (that is ``vision/train.py``).

Detector flavours (all ultralytics YOLO11)
------------------------------------------
* ``build_detector(cfg)``   52-class detector (or 56-class ``all`` space) of the **corner index**
  of a card (the rank+suit glyphs in the top-left / bottom-right corner) - the primary model,
  robust to heavily overlapping cards in fanned hands and spread piles.
* ``build_localizer(cfg)``  single-class ("CARD") localizer = stage 1 of the two-stage recogniser.
* ``build_obb(cfg)``        oriented-bounding-box variant (``yolo11<scale>-obb``).

``cfg.pretrained`` selects ``<arch>.pt`` (COCO weights, downloaded by ultralytics on first use)
versus ``<arch>.yaml`` (random init, fully offline); ``cfg.weights`` overrides both with an
explicit checkpoint.  Models built from a yaml get their head rebuilt for the target class
count and canonical class names right away (ultralytics would otherwise do it at train time).

Stage-2 classifier
------------------
``CardClassifier`` is a torchvision backbone (``mobilenet_v3_small`` by default) with two
heads, ``rank`` (13 logits) and ``suit`` (4 logits).  Inputs are square RGB corner crops of
``cfg.crop_size`` pixels normalised with ImageNet statistics (see ``preprocess_crops``).
``predict`` returns engine card ids ``suit * 13 + rank`` exactly like ``vision.cards``.

Inference wrapper
-----------------
``CardRecognizer`` turns a BGR image into ``Detection`` records (engine ``card_id``, canonical
``name``, ``conf``, pixel ``xyxy`` and the card's 500-rummy ``points``) either directly from a
52-class detector (``single_stage``) or from localizer boxes + classifier (``two_stage``).
``CardRecognizer.to_engine_ids`` yields the ids sorted left-to-right for the C++/PPO engine.

Export
------
``export_detector`` wraps ``YOLO.export`` (ONNX always works; ``tflite``/LiteRT needs the
TensorFlow / LiteRT stack: missing modules are reported with install instructions instead of
crashing) and ``export_classifier`` writes an ONNX (opset 17, dynamic batch) checked against
onnxruntime, plus a ``<name>.json`` describing the input contract for the Android app.

Usage::

    from vision.config import ModelConfig
    from vision import model as M

    det = M.build_detector(ModelConfig(task="detector", arch="yolo11n", pretrained=False))
    print(M.model_summary(det))

    clf = M.CardClassifier(ModelConfig(task="classifier", pretrained=False))
    out = clf(x)                 # {"rank": (N, 13), "suit": (N, 4)}
    ids = clf.predict(x)         # (N,) engine ids in 0..51
    loss = clf.loss(out, rank_targets, suit_targets)

    rec = M.CardRecognizer("two_stage", detector=localizer, classifier=clf)
    dets = rec.recognize(image_bgr)
    engine_ids = M.CardRecognizer.to_engine_ids(dets)

CLI::

    python -m vision.model --task detector --arch yolo11n --no-pretrained --summary
    python -m vision.model --export-classifier-demo --out runs/vision/demo
    python -m vision.model --task detector --weights runs/vision/det/weights/best.pt \\
        --export-detector --formats onnx,tflite --out runs/vision/det/export
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import logging
import os
import re
import shutil
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision import cards  # noqa: E402
from vision.config import ModelConfig, Paths  # noqa: E402

if TYPE_CHECKING:  # ultralytics is imported lazily (slow import, optional at inference time)
    from ultralytics import YOLO

logger = logging.getLogger("vision.model")

PathLike = Union[str, os.PathLike]

# --------------------------------------------------------------------------- constants
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
NUM_RANKS: int = len(cards.RANKS)   # 13
NUM_SUITS: int = len(cards.SUITS)   # 4
#: padding (fraction of the box side, each side) around a corner box before cutting a classifier
#: crop.  Must match ``generate_synthetic.CROP_PAD`` so inference crops look like training crops.
CROP_PAD: float = 0.15
#: cv2 interpolation for the final square resize of a classifier crop.  ``generate_synthetic.cut_crop``
#: always uses ``INTER_AREA`` (also when enlarging), so inference must too or the crops differ.
RESIZE_INTERPOLATION: int = cv2.INTER_AREA
#: pipeline task -> ultralytics task
YOLO_TASKS: Dict[str, str] = {"detector": "detect", "localizer": "detect", "obb": "obb"}
RECOGNIZER_MODES: Tuple[str, ...] = ("single_stage", "two_stage")
CLASSIFIER_CHECKPOINT_FORMAT = "rummy-card-classifier-v1"

#: backbone -> (torchvision constructor, weights enum, feature channels, hidden width of the neck)
_BACKBONES: Dict[str, Tuple[str, str, int, int]] = {
    "mobilenet_v3_small": ("mobilenet_v3_small", "MobileNet_V3_Small_Weights", 576, 1024),
    "mobilenet_v3_large": ("mobilenet_v3_large", "MobileNet_V3_Large_Weights", 960, 1280),
    "resnet18": ("resnet18", "ResNet18_Weights", 512, 512),
}
BACKBONES: Tuple[str, ...] = tuple(_BACKBONES)

#: export format -> alternative sets of importable modules, any one of which makes the format
#: exportable.  Checked *before* calling ultralytics so its auto-``pip install`` never fires.
_EXPORT_DEPS: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    "onnx": (("onnx",),),
    "torchscript": ((),),
    # ultralytics >= 8.4 exports tflite ("litert") through litert-torch; older versions go via onnx2tf
    "tflite": (("litert_torch", "ai_edge_litert"), ("tensorflow", "onnx2tf")),
    "litert": (("litert_torch", "ai_edge_litert"), ("tensorflow", "onnx2tf")),
    "saved_model": (("tensorflow", "onnx2tf"),),
    "pb": (("tensorflow", "onnx2tf"),),
    "edgetpu": (("tensorflow", "onnx2tf"),),
    "tfjs": (("tensorflow", "tensorflowjs"),),
    "openvino": (("openvino",),),
    "coreml": (("coremltools",),),
    "ncnn": (("ncnn",),),
    "paddle": (("paddle", "x2paddle"),),
    "engine": (("tensorrt",),),
    "mnn": (("MNN",),),
}
#: import name -> pip distribution name (for the install hint)
_PIP_NAMES: Dict[str, str] = {
    "litert_torch": "litert-torch", "ai_edge_litert": "ai-edge-litert", "tensorflow": "tensorflow",
    "onnx2tf": "onnx2tf", "tensorflowjs": "tensorflowjs", "coremltools": "coremltools", "tensorrt": "tensorrt",
    "MNN": "MNN", "x2paddle": "x2paddle", "paddle": "paddlepaddle",
}


# =========================================================================== YOLO factories
def class_names(cfg: ModelConfig) -> List[str]:
    """Canonical class names the model of ``cfg`` is trained on (localizer -> ``["CARD"]``)."""
    space = "card" if cfg.task == "localizer" else (cfg.class_space or "cards52")
    return cards.class_names_for_space(space)


def num_classes(cfg: ModelConfig) -> int:
    """Number of output classes for ``cfg`` (52 cards, 56 with the extra pile classes, 1 for the localizer)."""
    return len(class_names(cfg))


def yolo_source(cfg: ModelConfig, variant: str = "") -> str:
    """Model source string handed to ``ultralytics.YOLO``.

    ``cfg.weights`` wins; otherwise ``<arch><variant>.pt`` when ``cfg.pretrained`` (ultralytics
    downloads COCO weights from the github release assets) or ``<arch><variant>.yaml`` (random
    init, offline).  ``variant`` is e.g. ``"-obb"``; an arch that already carries it is left alone.
    """
    if cfg.weights:
        return str(cfg.weights)
    stem = re.sub(r"\.(pt|yaml|yml)$", "", str(cfg.arch).strip(), flags=re.IGNORECASE)
    if variant and not stem.endswith(variant):
        stem = f"{stem}{variant}"
    return f"{stem}.pt" if cfg.pretrained else f"{stem}.yaml"


def _is_yaml_source(source: str) -> bool:
    return str(source).lower().endswith((".yaml", ".yml"))


def is_yolo(obj: Any) -> bool:
    """True for an ``ultralytics.YOLO`` (or duck-typed) model: has ``.task``, ``.predict`` and an inner ``.model``."""
    return hasattr(obj, "task") and hasattr(obj, "predict") and isinstance(getattr(obj, "model", None), nn.Module)


def set_num_classes(model: "YOLO", names: Sequence[str]) -> "YOLO":
    """Give a yaml-built YOLO a freshly initialised head with ``len(names)`` outputs and canonical names.

    Ultralytics builds yaml models with the yaml's default ``nc`` (80) and only rebuilds the head
    once ``train(data=...)`` runs.  Rebuilding here makes ``model.names`` / ``model.model.nc`` /
    the head's ``nc`` truthful immediately (summaries, exports of untrained models, tests).
    Weights-loaded models are left untouched (the trainer transfers their weights into a new head).
    """
    inner = model.model
    nc = len(names)
    yaml_cfg = getattr(inner, "yaml", None)
    if isinstance(yaml_cfg, dict) and yaml_cfg.get("nc") != nc:
        cfg = dict(yaml_cfg)
        cfg["nc"] = nc
        logger.debug("rebuilding %s head: nc %s -> %d", type(inner).__name__, yaml_cfg.get("nc"), nc)
        rebuilt = type(inner)(cfg, cfg.get("channels", 3), nc, verbose=False)
        rebuilt.args = getattr(inner, "args", None)
        rebuilt.task = getattr(inner, "task", model.task)
        rebuilt.train(inner.training)
        model.model = rebuilt
    model.model.names = {i: str(n) for i, n in enumerate(names)}
    model.model.nc = nc
    return model


def _build_yolo(source: str, task: str, names: Sequence[str]) -> "YOLO":
    from ultralytics import YOLO  # slow import, keep lazy

    logger.info("building %s model from %s (%d classes)", task, source, len(names))
    model = YOLO(source, task=task)
    if _is_yaml_source(source):
        set_num_classes(model, names)
    else:
        have = len(model.names) if getattr(model, "names", None) else 0
        if have != len(names):
            logger.info("%s carries %d classes; the head will be rebuilt for %d classes when training starts",
                        source, have, len(names))
    return model


def build_detector(cfg: ModelConfig) -> "YOLO":
    """52-class (``cards52``) or 56-class (``all``) corner-index detector: ``yolo11<scale>[.pt|.yaml]``."""
    return _build_yolo(yolo_source(cfg), YOLO_TASKS["detector"], cards.class_names_for_space(cfg.class_space or "cards52"))


def build_localizer(cfg: ModelConfig) -> "YOLO":
    """Single-class card localizer (stage 1 of the two-stage recogniser); ``class_space`` is forced to ``card``."""
    return _build_yolo(yolo_source(cfg), YOLO_TASKS["localizer"], cards.class_names_for_space("card"))


def build_obb(cfg: ModelConfig) -> "YOLO":
    """Oriented-box detector ``yolo11<scale>-obb[.pt|.yaml]`` in ``cfg.class_space`` (default ``cards52``)."""
    return _build_yolo(yolo_source(cfg, "-obb"), YOLO_TASKS["obb"], cards.class_names_for_space(cfg.class_space or "cards52"))


def build_classifier(cfg: ModelConfig) -> "CardClassifier":
    """Stage-2 rank/suit classifier; ``cfg.weights`` (a ``CardClassifier.save`` checkpoint) overrides arch/pretrained."""
    if cfg.weights:
        logger.info("loading classifier checkpoint %s", cfg.weights)
        return CardClassifier.load(cfg.weights)
    return CardClassifier(cfg)


def build_model(cfg: ModelConfig) -> Union["YOLO", "CardClassifier"]:
    """Dispatch on ``cfg.task``: detector | localizer | obb | classifier."""
    builders = {"detector": build_detector, "localizer": build_localizer, "obb": build_obb, "classifier": build_classifier}
    if cfg.task not in builders:
        raise ValueError(f"unknown task {cfg.task!r}; expected one of {tuple(builders)}")
    return builders[cfg.task](cfg)


# =========================================================================== stage-2 classifier
def _make_backbone(name: str, pretrained: bool) -> Tuple[nn.Module, int, int]:
    """Return ``(feature_extractor, feature_channels, neck_width)`` for a supported torchvision backbone.

    ImageNet weights are requested only when ``pretrained``; any failure (offline sandbox, blocked
    ``download.pytorch.org``, checksum error) is logged and the network starts from random init.
    """
    if name not in _BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; expected one of {BACKBONES}")
    from torchvision import models as tvm  # lazy: torchvision import is not free

    ctor_name, weights_name, feat_dim, hidden = _BACKBONES[name]
    ctor = getattr(tvm, ctor_name)
    net: Optional[nn.Module] = None
    if pretrained:
        try:
            weights = getattr(tvm, weights_name).DEFAULT
            net = ctor(weights=weights)
            logger.info("loaded ImageNet weights for %s", name)
        except Exception as exc:  # URLError / HTTPError / RuntimeError(hash) / AttributeError ...
            first = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            logger.warning("ImageNet weights for %s unavailable (%s: %.200s); starting from random init. "
                           "Download them on a connected machine into %s or use pretrained=False.",
                           name, type(exc).__name__, first, torch.hub.get_dir())
            net = None
    if net is None:
        net = ctor(weights=None)
    if name.startswith("mobilenet"):
        features: nn.Module = net.features
    else:  # resnet: everything before the global pool / fc
        features = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4)
    return features, feat_dim, hidden


class CardClassifier(nn.Module):
    """Rank (13) + suit (4) two-head classifier over square corner crops.

    * input: ``(N, 3, crop_size, crop_size)`` float tensor, RGB, ImageNet-normalised
      (build it with :func:`preprocess_crops`);
    * ``forward`` -> ``{"rank": (N, 13) logits, "suit": (N, 4) logits}``;
    * ``predict`` -> ``(N,)`` engine ids ``suit * 13 + rank`` (``cards.CARD_CLASSES`` order);
    * ``loss`` -> cross-entropy(rank) + cross-entropy(suit).

    The graph is static (no data-dependent python control flow) so ``export_classifier`` can
    trace it to ONNX.  ``save``/``load`` persist ``state_dict`` + the construction config.
    """

    def __init__(self, cfg: Optional[ModelConfig] = None, *, backbone: Optional[str] = None,
                 crop_size: Optional[int] = None, dropout: Optional[float] = None,
                 pretrained: Optional[bool] = None) -> None:
        super().__init__()
        cfg = cfg if cfg is not None else ModelConfig(task="classifier", pretrained=False)
        self.backbone_name: str = backbone or cfg.backbone
        self.crop_size: int = int(crop_size or cfg.crop_size)
        self.dropout: float = float(cfg.dropout if dropout is None else dropout)
        use_pretrained = bool(cfg.pretrained if pretrained is None else pretrained)
        self.features, feat_dim, hidden = _make_backbone(self.backbone_name, use_pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.neck = nn.Sequential(nn.Linear(feat_dim, hidden), nn.Hardswish(), nn.Dropout(self.dropout))
        self.rank_head = nn.Linear(hidden, NUM_RANKS)
        self.suit_head = nn.Linear(hidden, NUM_SUITS)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    # ---- config / persistence
    @property
    def config(self) -> Dict[str, Any]:
        return {"backbone": self.backbone_name, "crop_size": self.crop_size, "dropout": self.dropout,
                "num_ranks": NUM_RANKS, "num_suits": NUM_SUITS}

    def save(self, path: PathLike, extra: Optional[Dict[str, Any]] = None) -> Path:
        """Write ``{"format", "config", "state_dict", "class_names", "extra"}`` to ``path`` (``.pt``)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": CLASSIFIER_CHECKPOINT_FORMAT, "config": self.config, "state_dict": self.state_dict(),
                   "class_names": list(cards.CARD_CLASSES), "ranks": list(cards.RANKS), "suits": list(cards.SUITS),
                   "extra": dict(extra or {})}
        torch.save(payload, p)
        return p

    @classmethod
    def load(cls, path: PathLike, map_location: Union[str, torch.device] = "cpu") -> "CardClassifier":
        """Rebuild a classifier saved with :meth:`save` (or a bare ``state_dict`` with the default config)."""
        try:  # our payload is plain tensors / dicts / lists, so the safe loader suffices
            ckpt = torch.load(Path(path), map_location=map_location, weights_only=True)
        except Exception:  # older torch or a checkpoint with extra python objects in "extra"
            ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            c = ckpt.get("config", {})
            model = cls(backbone=c.get("backbone"), crop_size=c.get("crop_size"), dropout=c.get("dropout"), pretrained=False)
            model.load_state_dict(ckpt["state_dict"])
        else:  # plain state_dict
            model = cls(pretrained=False)
            model.load_state_dict(ckpt)
        model.eval()
        return model

    # ---- inference
    def normalize(self, x01: torch.Tensor) -> torch.Tensor:
        """ImageNet-normalise an RGB tensor in [0, 1] (used when the normalisation is embedded in the ONNX)."""
        return (x01 - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.pool(self.features(x)).flatten(1)
        h = self.neck(feats)
        return {"rank": self.rank_head(h), "suit": self.suit_head(h)}

    @staticmethod
    def ids_from_logits(rank_logits: torch.Tensor, suit_logits: torch.Tensor) -> torch.Tensor:
        return suit_logits.argmax(dim=1) * NUM_RANKS + rank_logits.argmax(dim=1)

    @staticmethod
    def split_card_ids(ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Engine ids -> ``(rank_idx, suit_idx)`` targets for :meth:`loss`."""
        ids = ids.long()
        return ids % NUM_RANKS, ids // NUM_RANKS

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Engine card ids ``(N,)`` in ``0..51`` for a batch of preprocessed crops."""
        out = self.forward(x)
        return self.ids_from_logits(out["rank"], out["suit"])

    @torch.no_grad()
    def predict_with_conf(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(ids, conf)`` where ``conf = max softmax(rank) * max softmax(suit)`` in (0, 1]."""
        out = self.forward(x)
        p_rank = F.softmax(out["rank"], dim=1).max(dim=1).values
        p_suit = F.softmax(out["suit"], dim=1).max(dim=1).values
        return self.ids_from_logits(out["rank"], out["suit"]), p_rank * p_suit

    # ---- training
    @staticmethod
    def loss_components(out: Dict[str, torch.Tensor], rank_t: torch.Tensor, suit_t: torch.Tensor,
                        label_smoothing: float = 0.0) -> Dict[str, torch.Tensor]:
        l_rank = F.cross_entropy(out["rank"], rank_t.long(), label_smoothing=label_smoothing)
        l_suit = F.cross_entropy(out["suit"], suit_t.long(), label_smoothing=label_smoothing)
        return {"rank": l_rank, "suit": l_suit, "total": l_rank + l_suit}

    def loss(self, out: Dict[str, torch.Tensor], rank_t: torch.Tensor, suit_t: torch.Tensor,
             label_smoothing: float = 0.0) -> torch.Tensor:
        """Sum of the rank and suit cross-entropies (targets are index tensors ``(N,)``)."""
        return self.loss_components(out, rank_t, suit_t, label_smoothing)["total"]


class _ClassifierExportWrapper(nn.Module):
    """Tuple-output (rank, suit) view of ``CardClassifier`` for ONNX tracing; optionally embeds the normalisation."""

    def __init__(self, model: CardClassifier, embed_preprocess: bool) -> None:
        super().__init__()
        self.model = model
        self.embed_preprocess = bool(embed_preprocess)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.embed_preprocess:  # static python flag, not data-dependent -> trace-safe
            x = self.model.normalize(x)
        out = self.model(x)
        return out["rank"], out["suit"]


# =========================================================================== crops / preprocessing
def square_crop(image: np.ndarray, xyxy: Sequence[float], out_size: int, pad: float = CROP_PAD) -> np.ndarray:
    """Square ``out_size`` crop centred on a box, side ``max(w, h) * (1 + 2 * pad)``, reflect-padded at borders.

    Mirrors ``generate_synthetic.cut_crop`` (same framing *and* ``INTER_AREA`` resize, also when enlarging)
    so the classifier sees pixel-identical crops at inference and in training.
    """
    h, w = image.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in xyxy)
    side = max(x1 - x0, y1 - y0, 1.0) * (1.0 + 2.0 * pad)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    X0, Y0 = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
    X1, Y1 = int(round(cx + side / 2.0)), int(round(cy + side / 2.0))
    if X1 <= X0:
        X1 = X0 + 1
    if Y1 <= Y0:
        Y1 = Y0 + 1
    pad_l, pad_t = max(0, -X0), max(0, -Y0)
    pad_r, pad_b = max(0, X1 - w), max(0, Y1 - h)
    if pad_l or pad_t or pad_r or pad_b:
        image = cv2.copyMakeBorder(image, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101)
        X0, X1, Y0, Y1 = X0 + pad_l, X1 + pad_l, Y0 + pad_t, Y1 + pad_t
    crop = image[Y0:Y1, X0:X1]
    if crop.size == 0:
        crop = np.zeros((4, 4, 3), np.uint8)
    return cv2.resize(crop, (out_size, out_size), interpolation=RESIZE_INTERPOLATION)


def preprocess_crops(crops: Sequence[np.ndarray], crop_size: int, *, bgr: bool = True,
                     device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    """uint8 HxWx3 crops (BGR by default, as OpenCV loads them) -> ``(N, 3, S, S)`` float tensor, RGB, ImageNet-normalised."""
    n = len(crops)
    if n == 0:
        return torch.zeros((0, 3, crop_size, crop_size), dtype=torch.float32, device=device)
    arr = np.empty((n, crop_size, crop_size, 3), dtype=np.float32)
    for i, c in enumerate(crops):
        c = np.asarray(c)
        if c.ndim == 2:
            c = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
        c = c[:, :, :3]
        if c.shape[0] != crop_size or c.shape[1] != crop_size:
            c = cv2.resize(c, (crop_size, crop_size), interpolation=RESIZE_INTERPOLATION)
        if bgr:
            c = c[:, :, ::-1]
        arr[i] = c
    arr /= 255.0
    arr -= np.asarray(IMAGENET_MEAN, dtype=np.float32)
    arr /= np.asarray(IMAGENET_STD, dtype=np.float32)
    t = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 3, 1, 2)))
    return t.to(device) if device is not None else t


# =========================================================================== recogniser
@dataclass
class Detection:
    """One recognised object.  ``card_id`` is the engine id (0..51); ids 52..55 are the ``EXTRA_CLASSES``
    (joker / piles / card back) a detector trained in the ``all`` space may report - they carry 0 points
    and are excluded from :meth:`CardRecognizer.to_engine_ids`."""
    card_id: int
    name: str
    conf: float
    xyxy: Tuple[float, float, float, float]
    points: int
    quad: Optional[List[Tuple[float, float]]] = None   # OBB polygon (4 pixel points) when available

    @property
    def is_card(self) -> bool:
        return 0 <= self.card_id < cards.NUM_CARD_CLASSES

    @property
    def cx(self) -> float:
        return (self.xyxy[0] + self.xyxy[2]) / 2.0

    @property
    def cy(self) -> float:
        return (self.xyxy[1] + self.xyxy[3]) / 2.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RawBox(NamedTuple):
    """A detector output before card semantics are attached."""
    xyxy: Tuple[float, float, float, float]
    conf: float
    cls: int
    name: str
    quad: Optional[List[Tuple[float, float]]] = None


def _to_numpy(t: Any) -> np.ndarray:
    if hasattr(t, "detach"):
        t = t.detach()
    if hasattr(t, "cpu"):
        t = t.cpu()
    if hasattr(t, "numpy"):
        t = t.numpy()
    return np.asarray(t)


def parse_yolo_results(results: Iterable[Any]) -> List[RawBox]:
    """Flatten ultralytics ``Results`` (detect ``.boxes`` or OBB ``.obb``) into :class:`RawBox` records.

    Works on duck-typed objects too (numpy arrays instead of tensors) so tests can mock a detector.
    """
    out: List[RawBox] = []
    for r in results:
        boxes = getattr(r, "boxes", None)
        obb = getattr(r, "obb", None)
        src = boxes if boxes is not None else obb
        if src is None:
            continue
        conf = _to_numpy(src.conf).reshape(-1).astype(np.float64)
        if conf.size == 0:
            continue
        xyxy = _to_numpy(src.xyxy).reshape(-1, 4).astype(np.float64)
        cls = _to_numpy(src.cls).reshape(-1).astype(int)
        quads = None
        if boxes is None and hasattr(src, "xyxyxyxy"):
            quads = _to_numpy(src.xyxyxyxy).reshape(-1, 4, 2).astype(np.float64)
        names = getattr(r, "names", None) or {}
        for i in range(conf.size):
            c = int(cls[i])
            if isinstance(names, dict):
                name = str(names.get(c, c))
            else:
                name = str(names[c]) if c < len(names) else str(c)
            quad = [(float(px), float(py)) for px, py in quads[i]] if quads is not None else None
            out.append(RawBox(tuple(float(v) for v in xyxy[i]), float(conf[i]), c, name, quad))
    return out


def card_id_from_class_name(name: str) -> Optional[int]:
    """Detector class name -> id in the ``all`` space (``"10C"`` -> 9, ``"JOKER"`` -> 52); ``None`` if unknown.

    Numeric names (a detector trained without names) are taken as ids directly.
    """
    n = str(name).strip()
    if n.upper() in cards.CLASS_TO_ID:
        return cards.CLASS_TO_ID[n.upper()]
    if n.isdigit():
        i = int(n)
        return i if 0 <= i < len(cards.ALL_CLASSES) else None
    canon = cards.normalize_class_name(n)
    return cards.CLASS_TO_ID.get(canon) if canon else None


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def dedupe_detections(dets: Sequence[Detection], iou_thr: float = 0.6, same_card: bool = True) -> List[Detection]:
    """Greedy class-agnostic NMS (keep the highest confidence among boxes with IoU > ``iou_thr``) and, when
    ``same_card``, keep only the most confident instance of each card id (a card shows two corner indices;
    500 rummy is played with a single deck so a card cannot legitimately appear twice)."""
    kept: List[Detection] = []
    seen: set = set()
    for d in sorted(dets, key=lambda d: d.conf, reverse=True):
        if same_card and d.is_card and d.card_id in seen:
            continue
        if any(iou_xyxy(d.xyxy, k.xyxy) > iou_thr for k in kept):
            continue
        kept.append(d)
        if d.is_card:
            seen.add(d.card_id)
    return kept


class CardRecognizer:
    """Image -> :class:`Detection` list for the app / engine bridge.

    ``mode="single_stage"``: the detector's classes are card names (52 or 56 classes).
    ``mode="two_stage"``: the detector only localises (any class space); every box is cut into a
    square crop (:func:`square_crop`) and classified by a :class:`CardClassifier`;
    ``conf = detector_conf * classifier_conf``.  Boxes the localizer labels as an extra class
    (joker, piles) are passed through unclassified.

    ``detector`` is an ``ultralytics.YOLO`` (detect or OBB) or any object with
    ``predict(image, **kw) -> results`` understood by :func:`parse_yolo_results`.
    """

    def __init__(self, mode: str = "single_stage", detector: Any = None, classifier: Optional[CardClassifier] = None, *,
                 conf: float = 0.25, iou: float = 0.5, imgsz: int = 640, device: str = "cpu", max_det: int = 100,
                 crop_size: Optional[int] = None, crop_pad: float = CROP_PAD, min_cls_conf: float = 0.0,
                 batch_size: int = 64, dedupe: bool = True, dedupe_iou: float = 0.6) -> None:
        if mode not in RECOGNIZER_MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {RECOGNIZER_MODES}")
        if detector is None:
            raise ValueError("a detector (or localizer) model is required")
        if mode == "two_stage" and classifier is None:
            raise ValueError("two_stage mode needs a CardClassifier")
        self.mode = mode
        self.detector = detector
        self.classifier = classifier
        self.conf, self.iou, self.imgsz, self.device, self.max_det = float(conf), float(iou), int(imgsz), str(device), int(max_det)
        self.crop_pad, self.min_cls_conf, self.batch_size = float(crop_pad), float(min_cls_conf), int(batch_size)
        self.dedupe, self.dedupe_iou = bool(dedupe), float(dedupe_iou)
        self.crop_size = int(crop_size or (classifier.crop_size if classifier is not None else ModelConfig().crop_size))
        self._cls_device = torch.device("cpu")
        if classifier is not None:
            classifier.eval()
            params = list(classifier.parameters())
            if params:
                self._cls_device = params[0].device

    @classmethod
    def from_paths(cls, mode: str, detector_weights: PathLike, classifier_weights: Optional[PathLike] = None,
                   **kwargs: Any) -> "CardRecognizer":
        """Load ``best.pt`` files: an ultralytics checkpoint and (two-stage) a ``CardClassifier.save`` checkpoint."""
        from ultralytics import YOLO

        detector = YOLO(str(detector_weights))
        classifier = CardClassifier.load(classifier_weights) if classifier_weights else None
        return cls(mode, detector, classifier, **kwargs)

    # ---- stages
    def detect_raw(self, image_bgr: np.ndarray) -> List[RawBox]:
        results = self.detector.predict(image_bgr, conf=self.conf, iou=self.iou, imgsz=self.imgsz, device=self.device,
                                        max_det=self.max_det, verbose=False)
        return parse_yolo_results(results)

    def _single_stage(self, raw: Sequence[RawBox]) -> List[Detection]:
        dets: List[Detection] = []
        for rb in raw:
            cid = card_id_from_class_name(rb.name)
            if cid is None:
                logger.debug("ignoring detection with unknown class %r", rb.name)
                continue
            name = cards.ALL_CLASSES[cid]
            pts = cards.points_500_rummy(name) if cid < cards.NUM_CARD_CLASSES else 0
            dets.append(Detection(cid, name, rb.conf, rb.xyxy, pts, rb.quad))
        return dets

    def _two_stage(self, image_bgr: np.ndarray, raw: Sequence[RawBox]) -> List[Detection]:
        assert self.classifier is not None
        dets: List[Detection] = []
        to_classify: List[RawBox] = []
        for rb in raw:
            cid = card_id_from_class_name(rb.name)
            if cid is not None and cid >= cards.NUM_CARD_CLASSES:  # joker / pile / back: keep the localizer's label
                dets.append(Detection(cid, cards.ALL_CLASSES[cid], rb.conf, rb.xyxy, 0, rb.quad))
            else:
                to_classify.append(rb)
        for start in range(0, len(to_classify), self.batch_size):
            chunk = to_classify[start:start + self.batch_size]
            crops = [square_crop(image_bgr, rb.xyxy, self.crop_size, self.crop_pad) for rb in chunk]
            x = preprocess_crops(crops, self.crop_size, device=self._cls_device)
            ids, cconf = self.classifier.predict_with_conf(x)
            for rb, cid, cc in zip(chunk, ids.tolist(), cconf.tolist()):
                if cc < self.min_cls_conf:
                    continue
                name = cards.card_name(int(cid))
                dets.append(Detection(int(cid), name, rb.conf * float(cc), rb.xyxy, cards.points_500_rummy(name), rb.quad))
        return dets

    def recognize(self, image_bgr: np.ndarray) -> List[Detection]:
        """Run the pipeline on a HxWx3 uint8 BGR image; detections come back sorted left-to-right."""
        raw = self.detect_raw(image_bgr)
        dets = self._single_stage(raw) if self.mode == "single_stage" else self._two_stage(image_bgr, raw)
        if self.dedupe:
            dets = dedupe_detections(dets, self.dedupe_iou, same_card=True)
        dets.sort(key=lambda d: (d.cx, d.cy))
        return dets

    @staticmethod
    def to_engine_ids(dets: Iterable[Detection]) -> List[int]:
        """Engine card ids (0..51 only) ordered by box centre x, left to right."""
        return [d.card_id for d in sorted(dets, key=lambda d: (d.cx, d.cy)) if d.is_card]


# =========================================================================== export
def missing_export_modules(fmt: str) -> List[str]:
    """Python modules that must be importable before ultralytics can export ``fmt`` (empty = ready or unknown format)."""
    stacks = _EXPORT_DEPS.get(fmt.lower())
    if stacks is None:
        return []
    best: Optional[List[str]] = None
    for stack in stacks:
        missing = []
        for mod in stack:
            try:
                present = importlib.util.find_spec(mod) is not None
            except (ImportError, ValueError):
                present = False
            if not present:
                missing.append(mod)
        if not missing:
            return []
        if best is None or len(missing) < len(best):
            best = missing
    return best or []


def _install_hint(missing: Sequence[str]) -> str:
    pkgs = " ".join(_PIP_NAMES.get(m, m) for m in missing)
    return f'install the export extras with  pip install "ultralytics[export]"  (or: pip install {pkgs})'


@contextlib.contextmanager
def _working_directory(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _move_into(produced: PathLike, out_dir: Path) -> Path:
    src = Path(produced)
    dst = out_dir / src.name
    if src.resolve() == dst.resolve():
        return dst
    if dst.is_dir() and not dst.is_symlink():
        shutil.rmtree(dst)
    elif dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.move(str(src), str(dst))
    return dst


def export_detector(model: "YOLO", out_dir: PathLike, formats: Sequence[str] = ("onnx", "tflite"), imgsz: int = 640,
                    half: bool = False, int8: bool = False, nms: bool = False, *, data: Optional[str] = None,
                    simplify: bool = False, opset: Optional[int] = None, dynamic: bool = False, batch: int = 1,
                    device: str = "cpu", auto_install: bool = False, **kwargs: Any) -> Dict[str, Optional[Path]]:
    """Export a YOLO model with ``model.export`` into ``out_dir``; returns ``{format: path or None}``.

    Formats whose python dependencies are missing (e.g. ``tflite`` without the TensorFlow / LiteRT
    stack) are skipped with a logged install hint unless ``auto_install`` lets ultralytics pip-install
    them.  Any exception raised by an export is logged and the next format is attempted, so a
    training run never dies in its export step.  ``int8`` needs calibration ``data`` (a data.yaml).
    Note: the process working directory is switched to ``out_dir`` for the duration of each export
    (ultralytics writes yaml-built models relative to the CWD); artefacts written elsewhere
    (next to a ``.pt``) are moved into ``out_dir``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Optional[Path]] = {}
    for fmt in formats:
        f = str(fmt).strip().lower()
        if not f:
            continue
        missing = [] if auto_install else missing_export_modules(f)
        if missing:
            logger.warning("skipping %s export: missing python module(s) %s; %s", f, missing, _install_hint(missing))
            results[fmt] = None
            continue
        kw: Dict[str, Any] = dict(format=f, imgsz=imgsz, half=half, int8=int8, nms=nms, simplify=simplify,
                                  dynamic=dynamic, batch=batch, device=device, verbose=False)
        if opset is not None:
            kw["opset"] = opset
        if data is not None:
            kw["data"] = data
        kw.update(kwargs)
        logger.info("exporting %s -> %s (imgsz=%s half=%s int8=%s nms=%s)", f, out_dir, imgsz, half, int8, nms)
        try:
            with _working_directory(out_dir):  # yaml-built models export relative to the CWD
                produced = Path(model.export(**kw)).resolve()  # resolve while still inside out_dir
            results[fmt] = _move_into(produced, out_dir)
            logger.info("%s export written to %s", f, results[fmt])
        except Exception as exc:  # ImportError / RuntimeError / SyntaxError (bad arg) ...
            first = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            hint = ""
            if isinstance(exc, (ImportError, ModuleNotFoundError)) or "No module named" in str(exc):
                hint = f"; {_install_hint([getattr(exc, 'name', None) or 'the missing package'])}"
            logger.error("%s export failed: %s: %.300s%s", f, type(exc).__name__, first, hint)
            results[fmt] = None
    return results


def _torch_onnx_export(module: nn.Module, dummy: torch.Tensor, path: Path, opset: int) -> None:
    kw: Dict[str, Any] = dict(input_names=["images"], output_names=["rank", "suit"], opset_version=opset,
                              dynamic_axes={"images": {0: "batch"}, "rank": {0: "batch"}, "suit": {0: "batch"}},
                              do_constant_folding=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the TorchScript exporter is deprecated but the dynamo one needs onnxscript
        try:
            torch.onnx.export(module, (dummy,), str(path), dynamo=False, **kw)
        except TypeError:  # torch < 2.5 has no ``dynamo`` keyword
            torch.onnx.export(module, (dummy,), str(path), **kw)


def check_onnx_classifier(onnx_path: PathLike, reference: nn.Module, crop_size: int, batch: int = 2,
                          seed: int = 0) -> float:
    """Run the ONNX with onnxruntime on random input and return the max |onnx - torch| over both heads."""
    import onnxruntime as ort

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, 3, crop_size, crop_size, generator=g)
    # snapshot every module's flag (not just the root's): ``export_classifier`` passes an eval-mode wrapper
    # around a possibly training-mode classifier, and ``reference.train(root_flag)`` would clobber the inner one
    modes = [(m, m.training) for m in reference.modules()]
    reference.eval()
    try:
        with torch.no_grad():
            ref = reference(x)
    finally:
        for m, training in modes:  # a mid-training export must not leave the model in eval mode
            m.training = training
    if isinstance(ref, dict):
        ref = (ref["rank"], ref["suit"])
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})
    return max(float(np.abs(o - r.numpy()).max()) for o, r in zip(outs, ref))


def export_classifier(model: CardClassifier, out_dir: PathLike, crop_size: Optional[int] = None, *,
                      name: str = "card_classifier", opset: int = 17, check: bool = True, atol: float = 1e-3,
                      embed_preprocess: bool = False) -> Path:
    """Export ``model`` to ``<out_dir>/<name>.onnx`` (opset ``opset``, dynamic batch, outputs ``rank``/``suit``).

    A ``<name>.json`` next to it documents the input contract.  With ``embed_preprocess`` the graph
    takes RGB in [0, 1] and normalises internally (handy for the app); otherwise the input is already
    ImageNet-normalised like the torch model.  ``check`` runs onnxruntime and raises ``RuntimeError``
    if the outputs differ from torch by more than ``atol``.
    """
    import onnx

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_size = int(crop_size or model.crop_size)
    path = out_dir / f"{name}.onnx"
    was_training = model.training
    model.eval()
    wrapper = _ClassifierExportWrapper(model, embed_preprocess).eval()
    dummy = torch.zeros(1, 3, crop_size, crop_size)
    try:
        _torch_onnx_export(wrapper, dummy, path, opset)
    finally:
        model.train(was_training)
    onnx.checker.check_model(onnx.load(str(path)))
    meta = {
        "format": "rummy-card-classifier-onnx-v1", "opset": opset, "backbone": model.backbone_name,
        "input": {"name": "images", "layout": "NCHW", "dtype": "float32", "shape": [-1, 3, crop_size, crop_size],
                  "color": "RGB", "range": "0-1 (normalised inside the graph)" if embed_preprocess else "imagenet-normalised",
                  "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD), "crop_pad": CROP_PAD,
                  "resize_interpolation": "area"},
        "outputs": {"rank": list(cards.RANKS), "suit": list(cards.SUITS)},
        "card_id": "suit_index * 13 + rank_index (matches the C++ engine)",
        "class_names": list(cards.CARD_CLASSES),
    }
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2))
    logger.info("classifier ONNX written to %s (%.1f KB)", path, path.stat().st_size / 1024)
    if check:
        diff = check_onnx_classifier(path, wrapper, crop_size)
        logger.info("onnxruntime check: max |onnx - torch| = %.2e", diff)
        if not diff <= atol:
            raise RuntimeError(f"ONNX output differs from torch by {diff:.3e} (> {atol})")
    return path


# =========================================================================== summary / CLI
def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


def model_summary(model: Any, imgsz: Optional[int] = None) -> str:
    """One-paragraph description of a YOLO model or a :class:`CardClassifier` (or any ``nn.Module``)."""
    if is_yolo(model):
        inner = model.model
        names = getattr(model, "names", None) or getattr(inner, "names", None) or {}
        head = inner.model[-1] if hasattr(inner, "model") and len(inner.model) else None
        size = int(imgsz or (getattr(model, "overrides", {}) or {}).get("imgsz") or 640)
        flops = None
        try:
            from ultralytics.utils.torch_utils import get_flops

            flops = get_flops(inner, size)
        except Exception as exc:  # thop missing or unsupported layer
            logger.debug("GFLOPs unavailable: %s", exc)
        names_list = [str(names[k]) for k in sorted(names)] if isinstance(names, dict) else [str(n) for n in names]
        preview = ", ".join(names_list[:6]) + (", ..." if len(names_list) > 6 else "")
        lines = [
            f"YOLO task={model.task} source={getattr(model, 'ckpt_path', None) or getattr(model, 'cfg', None) or '?'}",
            f"  type={type(inner).__name__} head={type(head).__name__ if head is not None else '?'}"
            f" head.nc={getattr(head, 'nc', '?')} classes={len(names_list)} [{preview}]",
            f"  params={count_parameters(inner) / 1e6:.2f}M"
            + (f" GFLOPs@{size}={flops:.2f}" if flops else "") + f" stride={getattr(inner, 'stride', None)}",
        ]
        return "\n".join(lines)
    if isinstance(model, CardClassifier):
        return (f"CardClassifier backbone={model.backbone_name} crop_size={model.crop_size} dropout={model.dropout}"
                f" heads=rank({NUM_RANKS})+suit({NUM_SUITS}) params={count_parameters(model) / 1e6:.2f}M"
                f" input=RGB imagenet-normalised NCHW")
    if isinstance(model, nn.Module):
        return f"{type(model).__name__} params={count_parameters(model) / 1e6:.2f}M"
    return f"{type(model).__name__}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m vision.model",
        description="Inspect / export the card-vision models (detector, localizer, OBB, rank-suit classifier).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  python -m vision.model --task detector --arch yolo11n --no-pretrained --summary\n"
               "  python -m vision.model --export-classifier-demo --out runs/vision/demo\n"
               "  python -m vision.model --task detector --weights best.pt --export-detector --formats onnx,tflite --out export/",
    )
    d = ModelConfig()
    p.add_argument("--task", choices=("detector", "localizer", "obb", "classifier"), default=d.task)
    p.add_argument("--arch", default=d.arch, help="ultralytics arch/scale, e.g. yolo11n, yolo11s (default %(default)s)")
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=d.pretrained,
                   help="start from COCO / ImageNet weights (downloads them); --no-pretrained builds from yaml offline")
    p.add_argument("--weights", default=None, help="explicit checkpoint (.pt) overriding --arch/--pretrained")
    p.add_argument("--class-space", choices=cards.CLASS_SPACES, default=d.class_space)
    p.add_argument("--backbone", choices=BACKBONES, default=d.backbone, help="classifier backbone")
    p.add_argument("--crop-size", type=int, default=d.crop_size, help="classifier input size (default %(default)s)")
    p.add_argument("--imgsz", type=int, default=d.imgsz, help="detector image size for summaries/exports")
    p.add_argument("--summary", action="store_true", help="print model_summary() (default action)")
    p.add_argument("--export-classifier-demo", action="store_true",
                   help="build a CardClassifier and export it to ONNX under --out (checks it with onnxruntime)")
    p.add_argument("--export-detector", action="store_true", help="export the detector model with ultralytics")
    p.add_argument("--formats", default="onnx,tflite", help="comma-separated export formats (default %(default)s)")
    p.add_argument("--out", type=Path, default=None, help="output dir for exports (default runs/vision/model_exports)")
    p.add_argument("--half", action="store_true")
    p.add_argument("--int8", action="store_true")
    p.add_argument("--nms", action="store_true", help="bake NMS into the detector export")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    torch.manual_seed(args.seed)
    out_dir = args.out or (Paths().runs / "model_exports")
    cfg = ModelConfig(task=args.task, arch=args.arch, pretrained=bool(args.pretrained), imgsz=args.imgsz,
                      class_space=args.class_space, backbone=args.backbone, crop_size=args.crop_size, weights=args.weights)
    try:
        if args.export_classifier_demo:
            clf = CardClassifier(cfg)
            print(model_summary(clf))
            path = export_classifier(clf, out_dir, cfg.crop_size)
            print(f"classifier ONNX: {path}\nmetadata:        {path.with_suffix('.json')}")
            return 0
        model = build_model(cfg)
        if args.summary or not args.export_detector:
            print(model_summary(model, imgsz=args.imgsz))
        if args.export_detector:
            if not is_yolo(model):
                logger.error("--export-detector needs a detector/localizer/obb task (use --export-classifier-demo for the classifier)")
                return 1
            formats = [f for f in args.formats.split(",") if f.strip()]
            done = export_detector(model, out_dir, formats, imgsz=args.imgsz, half=args.half, int8=args.int8, nms=args.nms)
            for fmt, path in done.items():
                print(f"{fmt:12s} {path if path else 'SKIPPED/FAILED (see log)'}")
            return 0 if any(done.values()) else 2
        return 0
    except Exception as exc:
        logger.error("%s: %s", type(exc).__name__, exc, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
