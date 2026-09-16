"""Train the card-vision models (section D of the pipeline): dataset merging, training, evaluation, export.

Sub-commands
------------
``detector``    52-class (``cards52``) or 56-class (``all``) corner-index detector, ultralytics YOLO11.
``localizer``   single-class "CARD" detector (stage 1 of the two-stage recogniser): every accepted dataset's
                labels are rewritten to class 0 into a ``localizer`` view.
``obb``         oriented-box detector trained on the ``views/obb`` labels of each dataset.
``classifier``  stage-2 rank/suit classifier (``model.CardClassifier``) trained on ``crops/labels.csv``.
``all``         synthetic check (a small synthetic set is generated when the named one is missing),
                detector, classifier, one ``summary.json`` and a final artefact table.

Dataset resolution and merging
------------------------------
``--datasets`` takes a comma-separated list of ``synthetic:<name>`` (``Paths.synthetic/<name>``), registry
keys (``Paths.datasets/<key>``, see ``config.DATASETS``) or plain directory paths.  Each root must carry the
``manifest.json`` written by ``download_datasets`` / ``generate_synthetic``.  A dataset is refused, with an
error naming every offender, when its ``box_semantics`` differs from ``--box-semantics`` (a corner-labelled
dataset may serve ``card`` semantics through its ``views/card``) or its ``class_space`` differs from the
task's space.  The localizer accepts every card-like space (``cards52``, ``all``, ``card``); ``obb`` skips
datasets without an OBB view with a warning.  The run's ``data.yaml`` lists the absolute image directories
of every accepted dataset (ultralytics accepts lists for ``train``/``val``), so nothing is copied.

Ultralytics resolves symlinks before it derives label paths (``.../images/x.jpg`` -> ``.../labels/x.txt``),
which defeats the ``views/<name>/images -> ../../images`` directory symlink of the canonical layout: a view
trained through it would silently use the primary labels.  ``train.py`` therefore *materialises* every view
it trains on under ``<run>/views/<dataset>_<tag>_<view>/`` (``tag`` = short hash of the dataset root, so two
datasets sharing a basename never collide) as real directories of per-file symlinks (images) and per-file
symlinks or rewritten label files (labels).  A view is rebuilt whenever the source's image count or label
files (name, size, mtime) change.

Smoke mode
----------
``--smoke`` = 1 epoch, imgsz 320, batch 8, at most 5 % of the training images, ``workers 0``, CPU, no
pretrained download (a yaml model is built when ``<arch>.pt`` is not already present) and no export unless
``--export`` is given; the classifier trains 1 epoch on at most 512 crops.

Usage::

    python -m vision.train detector --datasets synthetic:rummy_v1,augstartups,andy8744 --epochs 50 --name detector_v1
    python -m vision.train localizer --datasets synthetic:rummy_v1 --box-semantics card --name localizer_v1
    python -m vision.train obb --datasets synthetic:rummy_v1,jackfurby --name obb_v1
    python -m vision.train classifier --crops synthetic:rummy_v1 --cls-epochs 15 --name classifier_v1
    python -m vision.train all --smoke --name smoke_v1
    python vision/train.py detector --smoke --model yolo11n.yaml --datasets synthetic:rummy_v1

Every run writes ``<runs-root>/<name>/summary.json`` (metrics, artefacts, configs) next to the ultralytics
outputs (``weights/best.pt``, ``weights/last.pt``, ``results.csv``, ``export/``) or the classifier outputs
(``weights/best.pt``, ``weights/last.pt``, ``results.csv``, ``export/card_classifier.onnx``).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision import cards as C  # noqa: E402
from vision import labels as L  # noqa: E402
from vision.config import DATASETS, ModelConfig, Paths, SynthConfig, TrainConfig  # noqa: E402

log = logging.getLogger("vision.train")

PathLike = Union[str, os.PathLike]

# --------------------------------------------------------------------------- constants
TASKS: Tuple[str, ...] = ("detector", "localizer", "obb", "classifier", "all")
YOLO_TASKS: Tuple[str, ...] = ("detector", "localizer", "obb")
#: class spaces whose boxes cover (parts of) single cards, all usable by the single-class localizer
CARD_LIKE_SPACES: Tuple[str, ...] = ("cards52", "all", "card")
DETECTOR_SPACES: Tuple[str, ...] = ("cards52", "all")
SYNTHETIC_PREFIX = "synthetic:"
#: ``--smoke`` overrides (spec: epochs 1, imgsz 320, batch 8, fraction <= 0.05, workers 0, cpu)
SMOKE: Dict[str, Any] = {"epochs": 1, "imgsz": 320, "batch": 8, "fraction": 0.05, "workers": 0, "device": "cpu",
                         "cls_epochs": 1, "max_crops": 512, "max_val_crops": 256, "synth_num": 24, "synth_img_size": 320}
DEFAULT_EXPORT: Tuple[str, ...] = ("onnx", "tflite")
CROPS_CSV = "labels.csv"
VIEW_STAMP = "view.json"
SUMMARY_NAME = "summary.json"


class DatasetError(ValueError):
    """A requested dataset is missing or incompatible with the training task."""


# --------------------------------------------------------------------------- options
@dataclass
class TrainOptions(TrainConfig):
    """``config.TrainConfig`` plus the knobs only ``train.py`` needs (additive, nothing renamed)."""
    class_space: str = "cards52"            # detector / obb label space (localizer is always "card")
    crops_from: List[str] = field(default_factory=list)   # corner-semantics datasets to cut classifier crops from
    max_crops: Optional[int] = None         # cap on training crops (smoke)
    max_val_crops: Optional[int] = None
    val_ratio: float = 0.1                  # classifier hold-out when a crop set has no val split
    label_smoothing: float = 0.05
    weight_decay: float = 0.01
    cache_crops: bool = True                # keep crops in RAM when they fit (see CACHE_LIMIT_BYTES)
    synth_num: int = 500                    # images generated by ``all`` when the synthetic set is missing
    synth_img_size: int = 640
    plots: bool = True
    verbose: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}


def apply_smoke(opts: TrainOptions) -> TrainOptions:
    """Force the smoke settings onto ``opts`` (in place) and return it."""
    opts.smoke = True
    opts.epochs = SMOKE["epochs"]
    opts.imgsz = min(int(opts.imgsz), SMOKE["imgsz"])
    opts.batch = min(int(opts.batch), SMOKE["batch"])
    opts.fraction = min(float(opts.fraction), SMOKE["fraction"])
    opts.workers = SMOKE["workers"]
    opts.device = SMOKE["device"]
    opts.cls_epochs = SMOKE["cls_epochs"]
    opts.max_crops = SMOKE["max_crops"] if opts.max_crops is None else min(opts.max_crops, SMOKE["max_crops"])
    opts.max_val_crops = SMOKE["max_val_crops"] if opts.max_val_crops is None else min(opts.max_val_crops, SMOKE["max_val_crops"])
    opts.synth_num = min(int(opts.synth_num), SMOKE["synth_num"])
    opts.synth_img_size = min(int(opts.synth_img_size), SMOKE["synth_img_size"])
    opts.plots = False
    return opts


def resolve_device(device: Optional[str]) -> str:
    """``auto`` -> ``"0"`` when CUDA is available else ``"cpu"``; anything else is passed through."""
    d = (device or "auto").strip().lower()
    if d != "auto":
        return d
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def torch_device(device: Optional[str]) -> Any:
    """``torch.device`` for the classifier from a ``--device`` value: ``cpu``, a CUDA index (``0`` -> ``cuda:0``)
    or an explicit torch device string.  A comma list (``0,1``, valid for ultralytics) uses its first index."""
    torch = _torch()
    d = resolve_device(device)
    first = d.split(",")[0].strip()
    if "," in d:
        log.info("classifier trains on a single GPU: using %s of --device %s", first, d)
    if first == "cpu":
        return torch.device("cpu")
    return torch.device(f"cuda:{first}" if first.isdigit() else first)


def split_csv(value: Optional[Union[str, Sequence[str]]]) -> List[str]:
    """``"a, b,,c"`` -> ``["a", "b", "c"]`` (lists are flattened the same way)."""
    if value is None:
        return []
    items = [value] if isinstance(value, str) else list(value)
    out: List[str] = []
    for item in items:
        out.extend(s.strip() for s in str(item).split(",") if s.strip())
    return out


# --------------------------------------------------------------------------- model selection
def local_weights_available(source: str) -> bool:
    """True when ``source`` (``yolo11n.pt``) exists in the CWD or ultralytics' weights dir (no download needed)."""
    p = Path(source)
    if p.exists():
        return True
    try:
        from ultralytics.utils import SETTINGS
        wd = SETTINGS.get("weights_dir")
        return bool(wd) and (Path(wd) / p.name).exists()
    except Exception:  # noqa: BLE001
        return False


def make_model_config(task: str, model: str, pretrained: bool = True, *, smoke: bool = False, imgsz: int = 640,
                      class_space: str = "cards52", backbone: Optional[str] = None, crop_size: int = 96,
                      dropout: float = 0.2) -> ModelConfig:
    """Turn ``--model`` into a ``ModelConfig``.

    ``yolo11n`` -> arch (COCO weights when ``pretrained``); ``yolo11n.yaml`` -> random init; ``path/to/best.pt``
    (existing file) -> explicit weights; ``yolo11s.pt`` (absent) -> arch with weights to download.  For the
    classifier ``model`` may name a backbone.  In smoke mode a pretrained arch whose ``.pt`` is not already
    on disk falls back to the yaml so nothing is downloaded.
    """
    m = (model or "").strip()
    arch, weights, pre = ModelConfig().arch, None, bool(pretrained)
    if task == "classifier":
        from vision import model as M
        bb = backbone or (m if m in M.BACKBONES else ModelConfig().backbone)
        if m and m not in M.BACKBONES and not backbone:
            log.debug("--model %r is not a classifier backbone; using %s", m, bb)
        return ModelConfig(task="classifier", pretrained=pre, imgsz=imgsz, class_space=class_space, backbone=bb,
                           crop_size=int(crop_size), dropout=float(dropout))
    if m.lower().endswith((".yaml", ".yml")):
        arch, pre = Path(m).stem, False
    elif m.lower().endswith(".pt"):
        if Path(m).exists():
            weights = str(Path(m))
        else:
            arch, pre = Path(m).stem, True
    elif m:
        arch = m
    cfg = ModelConfig(task=task, arch=arch, pretrained=pre, imgsz=imgsz, class_space=class_space, weights=weights)
    if smoke and pre and not weights:
        from vision import model as M
        source = M.yolo_source(cfg, "-obb" if task == "obb" else "")
        if not local_weights_available(source):
            log.info("smoke: %s is not present locally; building from yaml instead of downloading", source)
            cfg.pretrained = False
    return cfg


# --------------------------------------------------------------------------- dataset resolution
def dataset_key(spec: str) -> str:
    """File-system friendly key for a dataset spec (``synthetic:rummy_v1`` -> ``synthetic_rummy_v1``)."""
    s = spec.strip()
    if s.startswith(SYNTHETIC_PREFIX):
        return "synthetic_" + s[len(SYNTHETIC_PREFIX):].strip().strip("/").replace("/", "_")
    if "/" in s or s.startswith("."):
        return Path(s).resolve().name
    return s


def resolve_dataset(spec: str, paths: Paths) -> Path:
    """``synthetic:<name>`` -> ``Paths.synthetic/<name>``; a registry key -> ``Paths.datasets/<key>``;
    an existing directory path is used as is."""
    s = spec.strip()
    if not s:
        raise DatasetError("empty dataset spec")
    if s.startswith(SYNTHETIC_PREFIX):
        name = s[len(SYNTHETIC_PREFIX):].strip()
        if not name:
            raise DatasetError(f"bad dataset spec {spec!r}: expected synthetic:<name>")
        return paths.synthetic / name
    p = Path(s)
    if ("/" in s or s.startswith(".")) and p.is_dir():
        return p.resolve()
    if s not in DATASETS:
        log.warning("%r is not a registry key (%s); looking for it under %s anyway", s, ", ".join(DATASETS), paths.datasets)
    return paths.datasets / s


@dataclass
class DatasetInfo:
    """A resolved dataset root plus the manifest facts ``train.py`` decides on."""
    spec: str
    root: Path
    manifest: Dict[str, Any]

    @property
    def key(self) -> str:
        return dataset_key(self.spec)

    @property
    def class_space(self) -> str:
        return str(self.manifest.get("class_space") or "")

    @property
    def box_semantics(self) -> str:
        return str(self.manifest.get("box_semantics") or "")

    @property
    def names(self) -> List[str]:
        names = self.manifest.get("names")
        if names:
            return [str(n) for n in names]
        try:
            return C.class_names_for_space(self.class_space)
        except ValueError:
            return []

    def view_root(self, name: str) -> Optional[Path]:
        """Root of view ``name`` (``views/<name>`` with a ``labels`` dir) or ``None``."""
        rel = (self.manifest.get("views") or {}).get(name) or f"views/{name}"
        root = self.root / rel
        return root if (root / "labels").is_dir() else None

    def labels_root(self, view: Optional[str] = None) -> Optional[Path]:
        """``labels`` dir of the primary label set or of view ``view`` (``None`` when the view is missing)."""
        if view is None:
            return self.root / "labels"
        vr = self.view_root(view)
        return None if vr is None else vr / "labels"

    def splits(self) -> List[str]:
        return [s for s in L.SPLITS if L.list_images(self.root / "images" / s)]

    def image_dir(self, split: str) -> Path:
        return self.root / "images" / split

    def describe(self) -> str:
        return f"{self.spec} [{self.root}] box_semantics={self.box_semantics or '?'} class_space={self.class_space or '?'}"


def load_dataset_info(spec: str, paths: Paths) -> DatasetInfo:
    """Resolve ``spec`` and read its manifest (falling back to the ``data.yaml`` extras with a warning)."""
    root = resolve_dataset(spec, paths)
    if not root.is_dir():
        hint = ("generate it with  python -m vision.generate_synthetic --out " + str(root)
                if spec.startswith(SYNTHETIC_PREFIX) else "download it with  python -m vision.download_datasets --datasets " + spec)
        raise DatasetError(f"dataset {spec!r} not found at {root}; {hint}")
    manifest = L.read_manifest(root)
    if not manifest:
        data_yaml = root / "data.yaml"
        if data_yaml.exists():
            d = L.read_data_yaml(data_yaml)
            if d.get("box_semantics") and d.get("class_space"):
                log.warning("%s has no manifest.json; using box_semantics/class_space from its data.yaml", root)
                manifest = {"key": root.name, "box_semantics": d["box_semantics"], "class_space": d["class_space"],
                            "names": L.yaml_names_list(d), "views": {}}
    if not manifest:
        raise DatasetError(f"dataset {spec!r} at {root} has no manifest.json (run download_datasets / generate_synthetic first)")
    info = DatasetInfo(spec=spec, root=root, manifest=manifest)
    if not info.splits():
        raise DatasetError(f"dataset {spec!r} at {root} has no images under images/{{train,val}}")
    return info


# --------------------------------------------------------------------------- source selection
@dataclass
class TrainSource:
    """One dataset as used by a run: which label set (primary or a view) and an optional class remap."""
    info: DatasetInfo
    view: Optional[str] = None                     # None = primary labels, else the source view name
    semantics: str = "corner"
    remap: Optional[Dict[int, Optional[int]]] = None   # applied while materialising (localizer)
    out_view: Optional[str] = None                 # name of the materialised view (None = train on the primary layout)

    @property
    def labels_root(self) -> Path:
        root = self.info.labels_root(self.view)
        if root is None:
            raise DatasetError(f"view {self.view!r} missing in {self.info.root}")
        return root

    @property
    def needs_materialising(self) -> bool:
        return self.view is not None or self.remap is not None

    def describe(self) -> str:
        src = f"views/{self.view}" if self.view else "labels"
        extra = " (classes -> 0)" if self.remap else ""
        return f"{self.info.spec}: {src}{extra}"


def task_class_space(task: str, class_space: str = "cards52") -> str:
    return "card" if task == "localizer" else (class_space or "cards52")


def task_names(task: str, class_space: str = "cards52") -> List[str]:
    return C.class_names_for_space(task_class_space(task, class_space))


def _semantics_source(info: DatasetInfo, box_semantics: str) -> Optional[Tuple[Optional[str], str]]:
    """``(view, semantics)`` giving ``box_semantics`` for ``info`` or ``None`` (primary labels preferred)."""
    if info.box_semantics == box_semantics:
        return None, box_semantics
    if box_semantics == "card" and info.box_semantics == "corner" and info.view_root("card") is not None:
        return "card", "card"
    return None


def select_sources(task: str, infos: Sequence[DatasetInfo], box_semantics: str = "corner",
                   class_space: str = "cards52") -> List[TrainSource]:
    """Decide how each dataset takes part in a ``task`` run, or raise ``DatasetError`` naming every offender."""
    if task not in YOLO_TASKS:
        raise ValueError(f"select_sources: task must be one of {YOLO_TASKS}, got {task!r}")
    if box_semantics not in ("corner", "card"):
        raise DatasetError(f"--box-semantics must be corner or card, got {box_semantics!r}")
    sources: List[TrainSource] = []
    offenders: List[str] = []
    for info in infos:
        if task == "obb":
            if info.class_space != class_space:
                offenders.append(f"{info.describe()}: class_space {info.class_space!r} != {class_space!r}")
                continue
            if info.view_root("obb") is None:
                log.warning("skipping %s for obb: no views/obb labels", info.spec)
                continue
            sources.append(TrainSource(info, view="obb", semantics="obb", out_view="obb"))
            continue
        chosen = _semantics_source(info, box_semantics)
        if chosen is None:
            offenders.append(f"{info.describe()}: box_semantics {info.box_semantics!r} != {box_semantics!r}"
                             + (" and no views/card" if box_semantics == "card" and info.box_semantics == "corner" else ""))
            continue
        view, sem = chosen
        if task == "detector":
            if info.class_space != class_space:
                offenders.append(f"{info.describe()}: class_space {info.class_space!r} != {class_space!r}")
                continue
            sources.append(TrainSource(info, view=view, semantics=sem, out_view=view))
        else:  # localizer: any card-like space, every kept class -> 0
            if info.class_space not in CARD_LIKE_SPACES:
                offenders.append(f"{info.describe()}: class_space {info.class_space!r} is not card-like {CARD_LIKE_SPACES}")
                continue
            remap = {i: C.canonical_id(n, "card") for i, n in enumerate(info.names)}
            sources.append(TrainSource(info, view=view, semantics=sem, remap=remap, out_view="localizer"))
    if offenders:
        raise DatasetError(f"refusing {len(offenders)} dataset(s) for task {task!r} (box_semantics={box_semantics!r}, "
                           f"class_space={task_class_space(task, class_space)!r}):\n  - " + "\n  - ".join(offenders))
    if not sources:
        raise DatasetError(f"no usable dataset for task {task!r} among {[i.spec for i in infos]}")
    return sources


# --------------------------------------------------------------------------- views / merging
def _link_file(src: Path, dst: Path) -> None:
    """Relative symlink ``dst -> src`` (copy fallback); existing links are left alone."""
    if dst.exists() or dst.is_symlink():
        return
    L.link_or_copy(src, dst)


def label_signature(labels_dir: Path) -> str:
    """Cheap content signature of a label directory: sha1 over ``name:size:mtime_ns`` of every ``*.txt``.
    Rewritten (remapped) view labels are copies, not symlinks, so a view must be rebuilt when the source labels
    change even if the image count does not."""
    h = hashlib.sha1()
    if labels_dir.is_dir():
        for p in sorted(labels_dir.glob("*.txt")):
            try:
                st = p.stat()
            except OSError:
                continue
            h.update(f"{p.name}:{st.st_size}:{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def materialize_view(source: TrainSource, out_root: Path, splits: Sequence[str] = ("train", "val")) -> Dict[str, Path]:
    """Create ``out_root/{images,labels}/<split>`` as real directories of per-file symlinks (rewritten label
    files when ``source.remap`` is set) and return ``{split: image_dir}``.  Idempotent through a stamp of the
    dataset root, label root, remap and, per split, the image count and ``label_signature``."""
    info = source.info
    labels_root = source.labels_root
    stamp = {"dataset": str(info.root), "labels": str(labels_root), "remap": None if source.remap is None else
             {str(k): v for k, v in source.remap.items()}, "splits": {}}
    for split in splits:
        n = len(L.list_images(info.image_dir(split)))
        if n:
            stamp["splits"][split] = {"n": n, "labels": label_signature(labels_root / split)}
    stamp_path = out_root / VIEW_STAMP
    if stamp_path.exists():
        try:
            if json.loads(stamp_path.read_text()) == stamp:
                log.debug("view %s is up to date", out_root)
                return {s: out_root / "images" / s for s in stamp["splits"]}
        except (OSError, ValueError):
            pass
    dropped = 0
    for split in stamp["splits"]:
        img_out, lbl_out = out_root / "images" / split, out_root / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        stale = list(img_out.iterdir()) + list(lbl_out.iterdir()) + [lbl_out.with_suffix(".cache")]
        for old in stale:   # entries (and the ultralytics label cache) of an older stamp
            if old.is_symlink() or old.is_file():
                old.unlink()
        for img in L.list_images(info.image_dir(split)):
            _link_file(img, img_out / img.name)
            lbl = labels_root / split / f"{img.stem}.txt"
            if not lbl.exists():
                continue   # image without labels = background sample
            if source.remap is None:
                _link_file(lbl, lbl_out / lbl.name)
            else:
                kept, drop = L.remap_boxes(L.read_boxes(lbl), source.remap)
                dropped += drop
                L.write_boxes(lbl_out / lbl.name, kept)
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    stamp_path.write_text(json.dumps(stamp, indent=2))
    log.info("materialised view %s from %s (%s)%s", out_root.name, labels_root,
             ", ".join(f"{s}={v['n']}" for s, v in stamp["splits"].items()), f", {dropped} boxes dropped by the remap" if dropped else "")
    return {s: out_root / "images" / s for s in stamp["splits"]}


def view_dir(run_dir: Path, source: TrainSource) -> Path:
    """``<run_dir>/views/<key>_<tag>_<view>`` of a materialised view; ``tag`` hashes the resolved dataset root so
    two datasets sharing a basename (``a/cards`` and ``b/cards`` -> key ``cards``) never share a directory."""
    tag = hashlib.sha1(str(Path(source.info.root).resolve()).encode()).hexdigest()[:8]
    return Path(run_dir) / "views" / f"{source.info.key}_{tag}_{source.out_view or source.view}"


def prepare_sources(sources: Sequence[TrainSource], run_dir: Path) -> List[Dict[str, Any]]:
    """Materialise what needs it and return ``[{"source", "dirs": {split: image_dir}}]``."""
    out: List[Dict[str, Any]] = []
    for src in sources:
        if src.needs_materialising:
            dirs = materialize_view(src, view_dir(run_dir, src))
        else:
            dirs = {s: src.info.image_dir(s) for s in src.info.splits()}
        out.append({"source": src, "dirs": dirs})
    return out


def write_merged_yaml(path: PathLike, prepared: Sequence[Dict[str, Any]], names: Sequence[str],
                      extra: Optional[Dict[str, Any]] = None) -> Path:
    """Write the run's ``data.yaml`` with list-style absolute ``train``/``val`` image dirs (no copying)."""
    train = [str(Path(p["dirs"]["train"]).resolve()) for p in prepared if "train" in p["dirs"]]
    val = [str(Path(p["dirs"]["val"]).resolve()) for p in prepared if "val" in p["dirs"]]
    if not train:
        raise DatasetError("no training images: none of the datasets has an images/train split")
    if not val:
        raise DatasetError("no validation images: none of the datasets has an images/val split")
    return L.write_data_yaml(path, names, train, val, extra=extra)


def count_images(prepared: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for p in prepared:
        for split, d in p["dirs"].items():
            counts[split] = counts.get(split, 0) + len(L.list_images(d))
    return counts


def build_data_yaml(task: str, opts: TrainOptions, paths: Paths, run_dir: Path) -> Tuple[Path, List[Dict[str, Any]]]:
    """Resolve ``opts.datasets`` for ``task``, materialise views and write ``<run_dir>/data.yaml``."""
    infos = [load_dataset_info(s, paths) for s in opts.datasets]
    sources = select_sources(task, infos, opts.box_semantics, opts.class_space)
    for s in sources:
        log.info("dataset %s", s.describe())
    prepared = prepare_sources(sources, run_dir)
    names = task_names(task, opts.class_space)
    sem = "obb" if task == "obb" else opts.box_semantics
    yaml_path = write_merged_yaml(run_dir / "data.yaml", prepared, names,
                                  extra={"box_semantics": sem, "class_space": task_class_space(task, opts.class_space),
                                         "datasets": [s.describe() for s in sources]})
    counts = count_images(prepared)
    log.info("merged data.yaml -> %s (%s, %d classes)", yaml_path, ", ".join(f"{k}={v}" for k, v in counts.items()), len(names))
    return yaml_path, prepared


# --------------------------------------------------------------------------- summaries
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_summary(run_dir: Path, summary: Dict[str, Any]) -> Path:
    p = Path(run_dir) / SUMMARY_NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2, default=str))
    return p


def metrics_dict(res: Any) -> Dict[str, Optional[float]]:
    """``{"map50", "map50_95", "precision", "recall", "fitness"}`` from a ``DetMetrics``/``OBBMetrics`` (or a dict)."""
    if res is None:
        return {}
    if isinstance(res, dict):
        return {"map50": res.get("metrics/mAP50(B)"), "map50_95": res.get("metrics/mAP50-95(B)"),
                "precision": res.get("metrics/precision(B)"), "recall": res.get("metrics/recall(B)"), "fitness": res.get("fitness")}
    box = getattr(res, "box", None)
    out: Dict[str, Optional[float]] = {}
    for key, attr in (("map50", "map50"), ("map50_95", "map"), ("precision", "mp"), ("recall", "mr")):
        try:
            out[key] = float(getattr(box, attr))
        except Exception:  # noqa: BLE001
            out[key] = None
    try:
        out["fitness"] = float(res.fitness)
    except Exception:  # noqa: BLE001
        out["fitness"] = None
    return out


def artefact_rows(summary: Dict[str, Any], prefix: str = "") -> List[Tuple[str, str]]:
    """Flatten ``summary["artefacts"]`` (and nested step summaries) into ``(label, path)`` rows."""
    rows: List[Tuple[str, str]] = []
    for k, v in (summary.get("artefacts") or {}).items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                rows.append((f"{prefix}{k}/{k2}", str(v2) if v2 else "-"))
        else:
            rows.append((f"{prefix}{k}", str(v) if v else "-"))
    for step, sub in (summary.get("steps") or {}).items():
        if isinstance(sub, dict):
            rows.extend(artefact_rows(sub, prefix=f"{step}/"))
    return rows


def format_table(rows: Sequence[Tuple[str, str]]) -> str:
    if not rows:
        return "(no artefacts)"
    w = max(len(r[0]) for r in rows)
    lines = [f"{'artefact':<{w}}  exists  path", f"{'-' * w}  ------  ----"]
    for label, path in rows:
        exists = "yes" if path != "-" and Path(path).exists() else "no"
        lines.append(f"{label:<{w}}  {exists:<6}  {path}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- YOLO training
def train_yolo(task: str, opts: TrainOptions, mcfg: ModelConfig, paths: Paths, run_dir: Path) -> Dict[str, Any]:
    """Train / validate / export a detector, localizer or OBB model; returns (and writes) the run summary."""
    if task not in YOLO_TASKS:
        raise ValueError(f"train_yolo: task must be one of {YOLO_TASKS}, got {task!r}")
    from vision import model as M

    t0 = time.time()
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(opts.device)
    yaml_path, prepared = build_data_yaml(task, opts, paths, run_dir)
    mcfg.task = task
    if task == "localizer":
        mcfg.class_space = "card"
    model = M.build_model(mcfg)
    log.info("model: %s", M.model_summary(model, imgsz=opts.imgsz).splitlines()[0])
    train_kw: Dict[str, Any] = dict(
        data=str(yaml_path), epochs=int(opts.epochs), imgsz=int(opts.imgsz), batch=int(opts.batch), device=device,
        workers=int(opts.workers), seed=int(opts.seed), fraction=float(opts.fraction), project=str(run_dir.parent),
        name=run_dir.name, patience=int(opts.patience), lr0=float(opts.lr0), exist_ok=True, plots=bool(opts.plots),
        verbose=bool(opts.verbose),
    )
    log.info("training %s: %s", task, {k: v for k, v in train_kw.items() if k not in ("data", "project")})
    results = model.train(**train_kw)
    trainer = getattr(model, "trainer", None)
    save_dir = Path(getattr(trainer, "save_dir", None) or getattr(results, "save_dir", None) or run_dir)
    best = Path(getattr(trainer, "best", save_dir / "weights" / "best.pt"))
    last = Path(getattr(trainer, "last", save_dir / "weights" / "last.pt"))
    train_metrics = metrics_dict(results)
    log.info("training done in %.1fs: mAP50=%s mAP50-95=%s (best %s)", time.time() - t0,
             _fmt(train_metrics.get("map50")), _fmt(train_metrics.get("map50_95")), best)

    val_metrics: Dict[str, Optional[float]] = {}
    errors: List[str] = []
    try:
        val_res = model.val(data=str(yaml_path), imgsz=int(opts.imgsz), batch=int(opts.batch), device=device,
                            workers=int(opts.workers), plots=bool(opts.plots), project=str(save_dir), name="val",
                            exist_ok=True, verbose=bool(opts.verbose), split="val")
        val_metrics = metrics_dict(val_res)
        log.info("validation: mAP50=%s mAP50-95=%s precision=%s recall=%s", _fmt(val_metrics.get("map50")),
                 _fmt(val_metrics.get("map50_95")), _fmt(val_metrics.get("precision")), _fmt(val_metrics.get("recall")))
    except Exception as exc:  # noqa: BLE001 - the checkpoints exist; report and carry on
        log.error("validation failed: %s: %s", type(exc).__name__, exc, exc_info=opts.verbose)
        errors.append(f"val: {type(exc).__name__}: {exc}")

    exports: Dict[str, Optional[str]] = {}
    if opts.export:
        done = M.export_detector(model, save_dir / "export", formats=list(opts.export), imgsz=int(opts.imgsz), device="cpu")
        exports = {k: (str(v) if v else None) for k, v in done.items()}
    else:
        log.info("export skipped (no formats requested)")

    summary = {
        "task": task, "name": run_dir.name, "run_dir": str(save_dir), "created": now_iso(), "smoke": bool(opts.smoke),
        "elapsed_s": round(time.time() - t0, 1), "device": device,
        "config": {"train": opts.to_dict(), "model": asdict(mcfg)},
        "datasets": [{"spec": p["source"].info.spec, "root": str(p["source"].info.root), "labels": p["source"].describe(),
                      "box_semantics": p["source"].semantics, "class_space": p["source"].info.class_space,
                      "images": {s: len(L.list_images(d)) for s, d in p["dirs"].items()}} for p in prepared],
        "images": count_images(prepared), "class_names": task_names(task, opts.class_space), "data_yaml": str(yaml_path),
        "metrics": {"train_final": train_metrics, "val": val_metrics},
        "artefacts": {"best": str(best) if best.exists() else None, "last": str(last) if last.exists() else None,
                      "results_csv": str(save_dir / "results.csv") if (save_dir / "results.csv").exists() else None,
                      "data_yaml": str(yaml_path), "exports": exports},
        "errors": errors,
    }
    write_summary(save_dir, summary)
    if save_dir != run_dir:
        write_summary(run_dir, summary)
    log.info("summary written to %s", save_dir / SUMMARY_NAME)
    return summary


def _fmt(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.4f}"


# --------------------------------------------------------------------------- classifier: crops
@dataclass
class CropRecord:
    path: Path
    card_id: int
    split: str = "train"
    source: str = ""


def read_crops_csv(root: Path, source: str = "") -> List[CropRecord]:
    """Rows of ``<root>/labels.csv`` (``path,card_id,rank_idx,suit_idx``; ``path`` relative to the dataset root
    or to ``root``).  The split is the first directory after ``crops/`` (``crops/val/AS/x.jpg`` -> ``val``)."""
    csv_path = Path(root) / CROPS_CSV
    if not csv_path.exists():
        raise DatasetError(f"no {CROPS_CSV} under {root}")
    out: List[CropRecord] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rel = Path(row["path"])
            cid = int(row["card_id"])
            if not 0 <= cid < C.NUM_CARD_CLASSES:
                continue
            cand = [root / rel, root.parent / rel, root / rel.name]
            path = next((p for p in cand if p.exists()), cand[0])
            parts = rel.parts
            split = "train"
            for i, part in enumerate(parts[:-1]):
                if part in L.SPLITS:
                    split = part
                    break
                if part == "crops" and i + 1 < len(parts) - 1 and parts[i + 1] in L.SPLITS:
                    split = parts[i + 1]
                    break
            out.append(CropRecord(path=path, card_id=cid, split=split, source=source))
    return out


def crops_from_dataset(root: Path, out_dir: Path, crop_size: int, splits: Sequence[str] = ("train", "val"),
                       pad: Optional[float] = None) -> Path:
    """Cut square classifier crops from a corner-semantics dataset's primary labels into ``out_dir``
    (``<split>/<CANONICAL>/<stem>_<k>.jpg`` + ``labels.csv``); skipped when ``labels.csv`` already exists."""
    from vision.generate_synthetic import CROP_PAD, cut_crop

    out_dir = Path(out_dir)
    csv_path = out_dir / CROPS_CSV
    if csv_path.exists():
        log.info("crops already cut under %s", out_dir)
        return out_dir
    rows: List[Tuple[str, int, int, int]] = []
    n_img = 0
    for split in splits:
        for img_path, lbl_path in L.iter_split(root, split):
            boxes = [b for b in L.read_boxes(lbl_path) if 0 <= b.cls < C.NUM_CARD_CLASSES]
            if not boxes:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            n_img += 1
            for k, box in enumerate(boxes):
                name = C.CARD_CLASSES[box.cls]
                rel = Path(split) / name / f"{img_path.stem}_{k}.jpg"
                (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_dir / rel), cut_crop(img, box, int(crop_size), pad=CROP_PAD if pad is None else pad),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                rows.append((rel.as_posix(), box.cls, box.cls % 13, box.cls // 13))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "card_id", "rank_idx", "suit_idx"])
        w.writerows(rows)
    log.info("cut %d crops from %d labelled images of %s -> %s", len(rows), n_img, root, out_dir)
    return out_dir


def collect_crop_records(opts: TrainOptions, paths: Paths, run_dir: Path, crop_size: int) -> List[CropRecord]:
    """Crops from every ``--crops`` dataset (``<root>/crops/labels.csv``) plus crops cut from ``--crops-from``."""
    records: List[CropRecord] = []
    for spec in opts.crops:
        root = resolve_dataset(spec, paths)
        crops_root = root / "crops"
        if not (crops_root / CROPS_CSV).exists():
            raise DatasetError(f"{spec!r}: no crops/{CROPS_CSV} under {root} (generate with write_crops or use --crops-from)")
        recs = read_crops_csv(crops_root, source=spec)
        log.info("crops %s: %d records (%s)", spec, len(recs), _split_counts(recs))
        records.extend(recs)
    for spec in opts.crops_from:
        info = load_dataset_info(spec, paths)
        if info.box_semantics != "corner":
            raise DatasetError(f"--crops-from {spec!r}: box_semantics is {info.box_semantics!r}, crops need corner boxes")
        if info.class_space not in ("cards52", "all"):
            raise DatasetError(f"--crops-from {spec!r}: class_space {info.class_space!r} has no card ids")
        target = info.root / "crops"
        try:
            target.mkdir(parents=True, exist_ok=True)
            probe = target / ".write_test"
            probe.touch()
            probe.unlink()
        except OSError:
            target = run_dir / "crops" / info.key
        out = crops_from_dataset(info.root, target, crop_size)
        recs = read_crops_csv(out, source=spec)
        log.info("crops-from %s: %d records (%s)", spec, len(recs), _split_counts(recs))
        records.extend(recs)
    if not records:
        raise DatasetError("no classifier crops found (use --crops synthetic:<name> or --crops-from <dataset>)")
    return records


def _split_counts(recs: Sequence[CropRecord]) -> str:
    counts: Dict[str, int] = {}
    for r in recs:
        counts[r.split] = counts.get(r.split, 0) + 1
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


def split_records(records: Sequence[CropRecord], val_ratio: float, seed: int, max_train: Optional[int] = None,
                  max_val: Optional[int] = None) -> Tuple[List[CropRecord], List[CropRecord]]:
    """Train/val lists.  Records already marked ``val`` are validation; sources without any ``val`` crops get a
    deterministic hold-out of ``val_ratio``.  Optional caps take a seeded random subset (smoke)."""
    rng = np.random.default_rng(int(seed))
    by_source: Dict[str, List[CropRecord]] = {}
    for r in records:
        by_source.setdefault(r.source, []).append(r)
    train: List[CropRecord] = []
    val: List[CropRecord] = []
    for src, recs in by_source.items():
        v = [r for r in recs if r.split == "val"]
        t = [r for r in recs if r.split != "val"]
        if not v and val_ratio > 0 and len(t) >= 2:
            perm = rng.permutation(len(t))
            n_val = max(1, int(round(len(t) * val_ratio)))
            v = [t[i] for i in sorted(perm[:n_val])]
            t = [t[i] for i in sorted(perm[n_val:])]
            log.info("crops %s: no val split, holding out %d of %d", src or "?", len(v), len(v) + len(t))
        train.extend(t)
        val.extend(v)
    if max_train is not None and len(train) > max_train:
        train = [train[i] for i in sorted(rng.choice(len(train), size=int(max_train), replace=False))]
    if max_val is not None and len(val) > max_val:
        val = [val[i] for i in sorted(rng.choice(len(val), size=int(max_val), replace=False))]
    return train, val


# --------------------------------------------------------------------------- classifier: augmentation
@dataclass
class AugConfig:
    """Crop augmentation strengths.  No horizontal / vertical flips: rank glyphs are chiral (6/9, J/L...)."""
    rotate_deg: float = 15.0
    scale: Tuple[float, float] = (0.85, 1.15)
    translate: float = 0.08          # fraction of the side
    brightness: float = 0.25         # +- fraction of 255
    contrast: float = 0.25
    saturation: float = 0.35
    hue_shift: float = 6.0           # OpenCV hue units (0..180)
    channel_gain: float = 0.08
    gamma: Tuple[float, float] = (0.7, 1.4)
    gray_p: float = 0.05
    blur_p: float = 0.2
    noise_p: float = 0.25
    jpeg_p: float = 0.15
    cutout_p: float = 0.3
    cutout_size: Tuple[float, float] = (0.1, 0.35)


def augment_crop(rng: np.random.Generator, img: np.ndarray, cfg: Optional[AugConfig] = None) -> np.ndarray:
    """Augment one uint8 BGR crop with numpy / OpenCV only (rotation, scale, shift, colour, blur, noise, cutout)."""
    cfg = cfg or AugConfig()
    h, w = img.shape[:2]
    out = img
    # geometry: rotation about the centre with scale and a small shift, reflect padding, NO flips
    ang = float(rng.uniform(-cfg.rotate_deg, cfg.rotate_deg))
    sc = float(rng.uniform(*cfg.scale))
    tx, ty = (float(v) for v in rng.uniform(-cfg.translate, cfg.translate, size=2) * (w, h))
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, sc)
    M[:, 2] += (tx, ty)
    out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    # colour
    f = out.astype(np.float32)
    if rng.random() < cfg.gray_p:
        g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
        f = np.repeat(g[:, :, None], 3, axis=2)
    else:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-cfg.hue_shift, cfg.hue_shift)) % 180.0
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(1 - cfg.saturation, 1 + cfg.saturation), 0, 255)
        f = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    c = float(rng.uniform(1 - cfg.contrast, 1 + cfg.contrast))
    b = float(rng.uniform(-cfg.brightness, cfg.brightness)) * 255.0
    mean = float(f.mean())
    f = (f - mean) * c + mean + b
    f *= rng.uniform(1 - cfg.channel_gain, 1 + cfg.channel_gain, size=3).astype(np.float32)
    f = np.clip(f, 0, 255)
    gamma = float(rng.uniform(*cfg.gamma))
    if abs(gamma - 1.0) > 1e-3:
        f = 255.0 * np.power(f / 255.0, gamma)
    # blur / noise / jpeg
    if rng.random() < cfg.blur_p:
        if rng.random() < 0.5:
            f = cv2.GaussianBlur(f, (0, 0), float(rng.uniform(0.3, 1.2)))
        else:
            k = int(rng.integers(3, 8))
            kernel = np.zeros((k, k), np.float32)
            kernel[k // 2, :] = 1.0 / k
            kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), float(rng.uniform(0, 180)), 1.0), (k, k))
            s = kernel.sum()
            f = cv2.filter2D(f, -1, kernel / s if s > 0 else kernel)
    if rng.random() < cfg.noise_p:
        f = f + rng.normal(0.0, float(rng.uniform(2.0, 10.0)), size=f.shape).astype(np.float32)
    out = np.clip(f, 0, 255).astype(np.uint8)
    if rng.random() < cfg.jpeg_p:
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 90))])
        if ok:
            dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if dec is not None:
                out = dec
    # cutout: 1-2 rectangles filled with a flat colour (simulates fingers / other cards over the index)
    if rng.random() < cfg.cutout_p:
        out = out.copy()
        for _ in range(int(rng.integers(1, 3))):
            cw = int(max(2, rng.uniform(*cfg.cutout_size) * w))
            ch = int(max(2, rng.uniform(*cfg.cutout_size) * h))
            x0 = int(rng.integers(0, max(1, w - cw + 1)))
            y0 = int(rng.integers(0, max(1, h - ch + 1)))
            colour = rng.integers(0, 256, size=3) if rng.random() < 0.5 else np.full(3, int(rng.integers(60, 200)))
            out[y0:y0 + ch, x0:x0 + cw] = colour.astype(np.uint8)
    return np.ascontiguousarray(out)


# --------------------------------------------------------------------------- classifier: dataset / loop
CACHE_LIMIT_BYTES = 2 * 1024 ** 3


def _torch():
    import torch
    return torch


class CropDataset:
    """``torch.utils.data.Dataset`` of ``(uint8 BGR crop, card_id)``; augmentation is seeded per (epoch, index)."""

    def __init__(self, records: Sequence[CropRecord], crop_size: int, augment: bool = False, seed: int = 0,
                 aug: Optional[AugConfig] = None, cache: bool = True) -> None:
        self.records = list(records)
        self.crop_size = int(crop_size)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.aug = aug or AugConfig()
        self.epoch = 0
        self._cache: Optional[List[Optional[np.ndarray]]] = None
        est = len(self.records) * self.crop_size * self.crop_size * 3
        if cache and est <= CACHE_LIMIT_BYTES:
            self._cache = [None] * len(self.records)
        elif cache:
            log.info("crop cache disabled: %d crops would need %.1f GB", len(self.records), est / 1024 ** 3)

    def __len__(self) -> int:
        return len(self.records)

    def _read(self, i: int) -> np.ndarray:
        if self._cache is not None and self._cache[i] is not None:
            return self._cache[i]
        img = cv2.imread(str(self.records[i].path), cv2.IMREAD_COLOR)
        if img is None:
            log.warning("unreadable crop %s; using a blank", self.records[i].path)
            img = np.full((self.crop_size, self.crop_size, 3), 128, np.uint8)
        if img.shape[0] != self.crop_size or img.shape[1] != self.crop_size:
            interp = cv2.INTER_AREA if img.shape[0] > self.crop_size else cv2.INTER_LINEAR
            img = cv2.resize(img, (self.crop_size, self.crop_size), interpolation=interp)
        if self._cache is not None:
            self._cache[i] = img
        return img

    def __getitem__(self, i: int) -> Tuple[np.ndarray, int]:
        img = self._read(i)
        if self.augment:
            rng = np.random.default_rng([self.seed, self.epoch, i])
            img = augment_crop(rng, img, self.aug)
        return img, int(self.records[i].card_id)

    def class_histogram(self) -> Dict[int, int]:
        h: Dict[int, int] = {}
        for r in self.records:
            h[r.card_id] = h.get(r.card_id, 0) + 1
        return dict(sorted(h.items()))


def collate_crops(batch: Sequence[Tuple[np.ndarray, int]]):
    """Batch of ``(crop, id)`` -> ``(tensor (N,3,S,S) via model.preprocess_crops, LongTensor ids)`` so training
    and inference share exactly one normalisation path (BGR -> RGB, /255, ImageNet mean/std)."""
    from vision import model as M
    torch = _torch()
    crops = [b[0] for b in batch]
    size = crops[0].shape[0] if crops else 1
    x = M.preprocess_crops(crops, size)
    y = torch.tensor([int(b[1]) for b in batch], dtype=torch.long)
    return x, y


def make_loader(ds: CropDataset, batch: int, shuffle: bool, workers: int, seed: int):
    torch = _torch()
    gen = torch.Generator().manual_seed(int(seed))
    return torch.utils.data.DataLoader(ds, batch_size=max(1, int(batch)), shuffle=shuffle, num_workers=max(0, int(workers)),
                                       collate_fn=collate_crops, generator=gen, drop_last=False,
                                       persistent_workers=False)


def evaluate_classifier(model, loader, device, label_smoothing: float = 0.0) -> Dict[str, float]:
    """Mean loss and rank / suit / joint card accuracy over ``loader``."""
    torch = _torch()
    from vision.model import CardClassifier
    model.eval()
    n = 0
    loss_sum = 0.0
    rank_ok = suit_ok = card_ok = 0
    with torch.no_grad():
        for x, ids in loader:
            x, ids = x.to(device), ids.to(device)
            rank_t, suit_t = CardClassifier.split_card_ids(ids)
            out = model(x)
            loss_sum += float(model.loss(out, rank_t, suit_t, label_smoothing)) * len(ids)
            pr, ps = out["rank"].argmax(1), out["suit"].argmax(1)
            rank_ok += int((pr == rank_t).sum())
            suit_ok += int((ps == suit_t).sum())
            card_ok += int(((pr == rank_t) & (ps == suit_t)).sum())
            n += len(ids)
    if n == 0:
        return {"loss": float("nan"), "rank_acc": 0.0, "suit_acc": 0.0, "card_acc": 0.0, "n": 0}
    return {"loss": loss_sum / n, "rank_acc": rank_ok / n, "suit_acc": suit_ok / n, "card_acc": card_ok / n, "n": n}


def cosine_with_warmup(step: int, total: int, warmup: int) -> float:
    """LR multiplier: linear warm-up to 1 over ``warmup`` steps, then cosine decay to ~0 at ``total``."""
    if total <= 0:
        return 1.0
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train_classifier(opts: TrainOptions, mcfg: ModelConfig, paths: Paths, run_dir: Path) -> Dict[str, Any]:
    """Train the rank/suit crop classifier; writes ``weights/best.pt``, ``weights/last.pt``, ``results.csv``,
    ``export/card_classifier.onnx`` and ``summary.json`` under ``run_dir``; returns the summary."""
    torch = _torch()
    from vision import model as M

    t0 = time.time()
    run_dir = Path(run_dir).resolve()
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    seed = int(opts.seed)
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    device = torch_device(opts.device)
    use_amp = device.type == "cuda"
    mcfg.task = "classifier"

    records = collect_crop_records(opts, paths, run_dir, mcfg.crop_size)
    train_recs, val_recs = split_records(records, opts.val_ratio, seed, opts.max_crops, opts.max_val_crops)
    if not train_recs:
        raise DatasetError("no training crops after splitting")
    train_ds = CropDataset(train_recs, mcfg.crop_size, augment=True, seed=seed, cache=opts.cache_crops)
    val_ds = CropDataset(val_recs, mcfg.crop_size, augment=False, seed=seed, cache=opts.cache_crops)
    log.info("classifier crops: train=%d val=%d classes_seen=%d crop_size=%d", len(train_ds), len(val_ds),
             len(train_ds.class_histogram()), mcfg.crop_size)
    train_loader = make_loader(train_ds, opts.cls_batch, True, opts.workers, seed)
    val_loader = make_loader(val_ds, opts.cls_batch, False, opts.workers, seed) if len(val_ds) else None

    model = M.CardClassifier(mcfg).to(device)
    log.info("model: %s", M.model_summary(model))
    optim = torch.optim.AdamW(model.parameters(), lr=float(opts.cls_lr), weight_decay=float(opts.weight_decay))
    epochs = max(1, int(opts.cls_epochs))
    steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * steps_per_epoch
    warmup = min(steps_per_epoch, max(1, int(0.05 * total_steps))) if epochs > 1 else 0
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: cosine_with_warmup(s, total_steps, warmup))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_path, last_path = run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"
    history: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {"card_acc": -1.0, "epoch": -1}
    step = 0
    with open(results_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "lr", "train_loss", "val_loss", "rank_acc", "suit_acc", "card_acc", "time_s"])
        for epoch in range(epochs):
            te = time.time()
            model.train()
            train_ds.epoch = epoch
            loss_sum, n_seen = 0.0, 0
            for x, ids in train_loader:
                x, ids = x.to(device, non_blocking=True), ids.to(device)
                rank_t, suit_t = M.CardClassifier.split_card_ids(ids)
                optim.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    out = model(x)
                    loss = model.loss(out, rank_t, suit_t, float(opts.label_smoothing))
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optim)
                scaler.update()
                sched.step()
                step += 1
                loss_sum += float(loss.detach()) * len(ids)
                n_seen += len(ids)
            train_loss = loss_sum / max(1, n_seen)
            metrics = evaluate_classifier(model, val_loader, device) if val_loader is not None else \
                {"loss": float("nan"), "rank_acc": 0.0, "suit_acc": 0.0, "card_acc": 0.0, "n": 0}
            lr = float(optim.param_groups[0]["lr"])
            row = {"epoch": epoch, "lr": lr, "train_loss": train_loss, "val_loss": metrics["loss"], "rank_acc": metrics["rank_acc"],
                   "suit_acc": metrics["suit_acc"], "card_acc": metrics["card_acc"], "time_s": round(time.time() - te, 1)}
            history.append(row)
            w.writerow([row[k] for k in ("epoch", "lr", "train_loss", "val_loss", "rank_acc", "suit_acc", "card_acc", "time_s")])
            f.flush()
            log.info("epoch %d/%d: train_loss=%.4f val_loss=%.4f rank=%.3f suit=%.3f card=%.3f (%.1fs)", epoch + 1, epochs,
                     train_loss, metrics["loss"], metrics["rank_acc"], metrics["suit_acc"], metrics["card_acc"], row["time_s"])
            extra = {"epoch": epoch, "metrics": metrics, "train_loss": train_loss, "train": opts.to_dict(), "model": asdict(mcfg)}
            model.save(last_path, extra=extra)
            score = metrics["card_acc"] if val_loader is not None else -train_loss
            if score > best["card_acc"] or best["epoch"] < 0:
                best = {"card_acc": score, "epoch": epoch, "metrics": metrics}
                model.save(best_path, extra=extra)

    errors: List[str] = []
    exports: Dict[str, Optional[str]] = {"onnx": None, "onnx_meta": None}
    try:
        best_model = M.CardClassifier.load(best_path)
        onnx_path = M.export_classifier(best_model, run_dir / "export", mcfg.crop_size)
        exports = {"onnx": str(onnx_path), "onnx_meta": str(onnx_path.with_suffix(".json"))}
    except Exception as exc:  # noqa: BLE001 - checkpoints exist; report and carry on
        log.error("classifier ONNX export failed: %s: %s", type(exc).__name__, exc, exc_info=opts.verbose)
        errors.append(f"export: {type(exc).__name__}: {exc}")

    summary = {
        "task": "classifier", "name": run_dir.name, "run_dir": str(run_dir), "created": now_iso(), "smoke": bool(opts.smoke),
        "elapsed_s": round(time.time() - t0, 1), "device": str(device),
        "config": {"train": opts.to_dict(), "model": asdict(mcfg)},
        "crops": {"sources": list(opts.crops) + [f"from:{s}" for s in opts.crops_from], "train": len(train_ds), "val": len(val_ds),
                  "classes_seen": len(train_ds.class_histogram())},
        "metrics": {"best": best.get("metrics", {}), "best_epoch": best["epoch"], "final": history[-1] if history else {},
                    "history": history},
        "artefacts": {"best": str(best_path) if best_path.exists() else None, "last": str(last_path) if last_path.exists() else None,
                      "results_csv": str(results_csv), "exports": exports},
        "errors": errors,
    }
    write_summary(run_dir, summary)
    log.info("classifier done in %.1fs: best card_acc=%.3f (epoch %d); summary %s", summary["elapsed_s"],
             max(0.0, best["card_acc"]), best["epoch"] + 1, run_dir / SUMMARY_NAME)
    return summary


# --------------------------------------------------------------------------- all
def ensure_synthetic(spec: str, paths: Paths, opts: TrainOptions, fetch: bool = True) -> Dict[str, Any]:
    """Generate ``Paths.synthetic/<name>`` with ``generate_synthetic.generate`` when it is missing."""
    root = resolve_dataset(spec, paths)
    if (root / "manifest.json").exists():
        return {"spec": spec, "root": str(root), "generated": False}
    from vision.generate_synthetic import generate

    cfg = SynthConfig(out_dir=root, cards_dir=paths.assets_cards, backgrounds_dir=paths.assets_backgrounds,
                      num_images=int(opts.synth_num), img_size=int(opts.synth_img_size), seed=int(opts.seed),
                      workers=1 if opts.smoke else 0)
    log.info("synthetic set %s missing: generating %d images at %dpx -> %s", spec, cfg.num_images, cfg.img_size, root)
    manifest = generate(cfg, fetch=fetch, progress=not opts.smoke)
    return {"spec": spec, "root": str(root), "generated": True, "images": manifest.get("images"), "crops": manifest.get("crops"),
            "card_source": manifest.get("card_source")}


def run_all(opts: TrainOptions, mcfg: ModelConfig, paths: Paths, run_dir: Path, fetch: bool = True) -> Dict[str, Any]:
    """Synthetic check -> detector -> classifier; prints the artefact table and writes ``<run_dir>/summary.json``."""
    t0 = time.time()
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    synth_steps = []
    for spec in list(dict.fromkeys(list(opts.datasets) + list(opts.crops))):
        if spec.startswith(SYNTHETIC_PREFIX):
            synth_steps.append(ensure_synthetic(spec, paths, opts, fetch=fetch))
    det_cfg = ModelConfig(task="detector", arch=mcfg.arch, pretrained=mcfg.pretrained, imgsz=mcfg.imgsz,
                          class_space=mcfg.class_space, weights=mcfg.weights)
    det = train_yolo("detector", opts, det_cfg, paths, run_dir / "detector")
    cls_cfg = ModelConfig(task="classifier", pretrained=mcfg.pretrained, imgsz=mcfg.imgsz, backbone=mcfg.backbone,
                          crop_size=mcfg.crop_size, dropout=mcfg.dropout)
    clf = train_classifier(opts, cls_cfg, paths, run_dir / "classifier")
    summary = {
        "task": "all", "name": run_dir.name, "run_dir": str(run_dir), "created": now_iso(), "smoke": bool(opts.smoke),
        "elapsed_s": round(time.time() - t0, 1), "config": {"train": opts.to_dict(), "model": asdict(mcfg)},
        "steps": {"synthetic": synth_steps, "detector": det, "classifier": clf},
        "metrics": {"detector": det.get("metrics", {}).get("val", {}), "classifier": clf.get("metrics", {}).get("best", {})},
        "artefacts": {"summary_detector": str(Path(det["run_dir"]) / SUMMARY_NAME), "summary_classifier": str(Path(clf["run_dir"]) / SUMMARY_NAME)},
        "errors": list(det.get("errors", [])) + list(clf.get("errors", [])),
    }
    write_summary(run_dir, summary)
    table = format_table(artefact_rows(summary))
    log.info("all done in %.1fs; summary %s", summary["elapsed_s"], run_dir / SUMMARY_NAME)
    print(table)
    return summary


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    d = TrainConfig()
    m = ModelConfig()
    common = argparse.ArgumentParser(add_help=False)
    g = common.add_argument_group("data")
    g.add_argument("--datasets", default=",".join(d.datasets),
                   help="comma-separated synthetic:<name>, registry keys or dataset dirs (default %(default)s)")
    g.add_argument("--box-semantics", choices=("corner", "card"), default=d.box_semantics,
                   help="what the boxes must cover; datasets with another semantics are refused (default %(default)s)")
    g.add_argument("--class-space", choices=DETECTOR_SPACES, default=m.class_space, help="detector / obb label space")
    g.add_argument("--crops", default=",".join(d.crops), help="classifier crop sets (datasets with crops/labels.csv)")
    g.add_argument("--crops-from", default="", help="corner-semantics datasets to cut classifier crops from (comma list)")
    g.add_argument("--data-root", default=None, help=f"dataset root (default {Paths().root}; env RUMMY_VISION_DATA)")
    g.add_argument("--runs-root", default=None, help=f"run output root (default {Paths().runs}; env RUMMY_VISION_RUNS)")
    g = common.add_argument_group("model")
    g.add_argument("--model", default=m.arch, help="yolo11n | yolo11n.yaml (random init) | path/to/best.pt; classifier: a backbone name")
    g.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=m.pretrained,
                   help="start from COCO / ImageNet weights (downloaded on first use)")
    g.add_argument("--backbone", choices=("mobilenet_v3_small", "mobilenet_v3_large", "resnet18"), default=None,
                   help=f"classifier backbone (default {m.backbone})")
    g.add_argument("--crop-size", type=int, default=m.crop_size, help="classifier input size (default %(default)s)")
    g.add_argument("--dropout", type=float, default=m.dropout)
    g = common.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=None, help=f"detector epochs (default {d.epochs}; smoke 1)")
    g.add_argument("--batch", type=int, default=None, help=f"detector batch (default {d.batch}; smoke 8)")
    g.add_argument("--imgsz", type=int, default=None, help=f"detector image size (default {d.imgsz}; smoke 320)")
    g.add_argument("--device", default=None, help="auto | cpu | 0 | 0,1 (default auto; smoke cpu)")
    g.add_argument("--workers", type=int, default=None, help=f"dataloader workers (default {d.workers}; smoke 0)")
    g.add_argument("--fraction", type=float, default=None, help=f"fraction of the training images (default {d.fraction}; smoke <= 0.05)")
    g.add_argument("--lr0", type=float, default=d.lr0)
    g.add_argument("--patience", type=int, default=d.patience)
    g.add_argument("--seed", type=int, default=d.seed)
    g.add_argument("--name", default=None, help="run name under --runs-root (default <task>_v1)")
    g.add_argument("--export", default=None, help="comma-separated export formats (default onnx,tflite; smoke: none unless given)")
    g.add_argument("--smoke", action="store_true", help="tiny CPU run: 1 epoch, imgsz 320, batch 8, <=5%% of the images, no downloads")
    g.add_argument("--no-plots", action="store_true", help="disable ultralytics plots")
    g = common.add_argument_group("classifier")
    g.add_argument("--cls-epochs", type=int, default=d.cls_epochs)
    g.add_argument("--cls-batch", type=int, default=d.cls_batch)
    g.add_argument("--cls-lr", type=float, default=d.cls_lr)
    g.add_argument("--label-smoothing", type=float, default=0.05)
    g.add_argument("--weight-decay", type=float, default=0.01)
    g.add_argument("--val-ratio", type=float, default=0.1, help="hold-out when a crop set has no val crops")
    g.add_argument("--max-crops", type=int, default=None, help="cap on training crops")
    g.add_argument("--no-cache-crops", action="store_true", help="read crops from disk every epoch")
    g = common.add_argument_group("all")
    g.add_argument("--synth-num", type=int, default=500, help="images generated when the synthetic set is missing")
    g.add_argument("--synth-img-size", type=int, default=640)
    g.add_argument("--no-fetch", action="store_true", help="never download card art for the generated synthetic set")
    common.add_argument("--verbose", "-v", action="store_true")

    p = argparse.ArgumentParser(prog="python -m vision.train", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="examples:\n  python -m vision.train detector --datasets synthetic:rummy_v1 --epochs 50\n"
                                       "  python -m vision.train classifier --crops synthetic:rummy_v1\n"
                                       "  python -m vision.train all --smoke")
    sub = p.add_subparsers(dest="task", metavar="TASK")
    sub.required = True
    helps = {"detector": "52/56-class corner-index detector", "localizer": "single-class CARD detector (stage 1)",
             "obb": "oriented-box detector on views/obb", "classifier": "rank/suit crop classifier (stage 2)",
             "all": "synthetic check + detector + classifier"}
    for task in TASKS:
        sub.add_parser(task, parents=[common], help=helps[task], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    return p


def build_options(args: argparse.Namespace) -> Tuple[TrainOptions, ModelConfig, Paths]:
    """``argparse`` namespace -> ``(TrainOptions, ModelConfig, Paths)`` with smoke overrides applied."""
    paths = Paths()
    if args.data_root:
        paths.root = Path(args.data_root)
    if args.runs_root:
        paths.runs = Path(args.runs_root)
    d = TrainConfig()
    task = args.task
    opts = TrainOptions(
        task=task, datasets=split_csv(args.datasets), box_semantics=args.box_semantics,
        epochs=d.epochs if args.epochs is None else int(args.epochs), batch=d.batch if args.batch is None else int(args.batch),
        imgsz=d.imgsz if args.imgsz is None else int(args.imgsz), device=args.device or d.device,
        workers=d.workers if args.workers is None else int(args.workers), lr0=float(args.lr0), patience=int(args.patience),
        seed=int(args.seed), fraction=d.fraction if args.fraction is None else float(args.fraction),
        name=args.name or ("rummy_v1" if task == "all" else f"{task}_v1"),
        export=split_csv(args.export) if args.export is not None else list(DEFAULT_EXPORT), smoke=bool(args.smoke),
        crops=split_csv(args.crops), cls_lr=float(args.cls_lr), cls_batch=int(args.cls_batch), cls_epochs=int(args.cls_epochs),
        class_space=args.class_space, crops_from=split_csv(args.crops_from), max_crops=args.max_crops,
        val_ratio=float(args.val_ratio), label_smoothing=float(args.label_smoothing), weight_decay=float(args.weight_decay),
        cache_crops=not args.no_cache_crops, synth_num=int(args.synth_num), synth_img_size=int(args.synth_img_size),
        plots=not args.no_plots, verbose=bool(args.verbose),
    )
    if args.smoke:
        # explicit CLI values that are *smaller* than the smoke defaults are kept (tests use imgsz 160)
        apply_smoke(opts)
        if args.export is None:
            opts.export = []
    mcfg = make_model_config(task, args.model, bool(args.pretrained), smoke=bool(args.smoke), imgsz=opts.imgsz,
                             class_space=args.class_space, backbone=args.backbone, crop_size=int(args.crop_size),
                             dropout=float(args.dropout))
    return opts, mcfg, paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        opts, mcfg, paths = build_options(args)
        run_dir = Path(paths.runs) / opts.name
        log.info("task=%s name=%s run_dir=%s smoke=%s", opts.task, opts.name, run_dir, opts.smoke)
        if opts.task in YOLO_TASKS:
            summary = train_yolo(opts.task, opts, mcfg, paths, run_dir)
        elif opts.task == "classifier":
            summary = train_classifier(opts, mcfg, paths, run_dir)
        elif opts.task == "all":
            summary = run_all(opts, mcfg, paths, run_dir, fetch=not args.no_fetch and not args.smoke)
        else:  # pragma: no cover - argparse restricts the choices
            log.error("unknown task %r", opts.task)
            return 1
    except KeyboardInterrupt:
        log.error("interrupted")
        return 130
    except DatasetError as exc:
        log.error("%s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        log.error("%s: %s", type(exc).__name__, exc, exc_info=args.verbose)
        return 1
    if opts.task != "all":
        print(format_table(artefact_rows(summary)))
    if summary.get("errors"):
        for e in summary["errors"]:
            log.warning("run finished with a problem: %s", e)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
