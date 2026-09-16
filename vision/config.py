"""Paths, dataset registry and configuration dataclasses for the vision pipeline.

Everything on disk lives under ``Paths.root`` (default ``data/vision``, git-ignored)::

    data/vision/
        raw/<dataset>/            untouched downloads (zips, pkls, roboflow exports ...)
        datasets/<dataset>/       converted to the canonical layout (see labels.py)
        synthetic/<name>/         output of generate_synthetic.py
        assets/cards/             52 (+jokers/back) clean card PNGs used by the generator
        assets/backgrounds/       background textures (DTD) for the generator
    runs/vision/<run>/            training outputs, exports, merged data.yaml files
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- paths
REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Paths:
    root: Path = field(default_factory=lambda: Path(os.environ.get("RUMMY_VISION_DATA", REPO_ROOT / "data" / "vision")))
    runs: Path = field(default_factory=lambda: Path(os.environ.get("RUMMY_VISION_RUNS", REPO_ROOT / "runs" / "vision")))

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def synthetic(self) -> Path:
        return self.root / "synthetic"

    @property
    def assets_cards(self) -> Path:
        return self.root / "assets" / "cards"

    @property
    def assets_backgrounds(self) -> Path:
        return self.root / "assets" / "backgrounds"

    def dataset(self, key: str) -> Path:
        return self.datasets / key

    def mkdirs(self) -> "Paths":
        for p in (self.raw, self.datasets, self.synthetic, self.assets_cards, self.assets_backgrounds, self.runs):
            p.mkdir(parents=True, exist_ok=True)
        return self


# --------------------------------------------------------------------------- dataset registry
@dataclass(frozen=True)
class DatasetSpec:
    key: str
    source: str                      # huggingface | roboflow | kaggle | github_raw | url
    ref: str                         # repo id / workspace/project / kaggle slug / url
    description: str
    kind: str = "detection"          # detection | backgrounds | card_art
    version: Optional[int] = None    # roboflow version (None = latest)
    label_style: str = "yolo"        # yolo | jackfurby_pkl | none
    name_style: str = "english"      # cards.NAME_STYLES
    class_space: str = "cards52"     # cards.CLASS_SPACES the converted labels live in
    box_semantics: str = "corner"    # labels.BOX_SEMANTICS  (verify with download_datasets --preview)
    provides: Tuple[str, ...] = ("det",)   # det | obb | parts | piles | backgrounds | card_art
    requires_env: Tuple[str, ...] = ()
    license: str = "see source page"
    notes: str = ""


DATASETS: Dict[str, DatasetSpec] = {
    "jackfurby": DatasetSpec(
        key="jackfurby", source="huggingface", ref="JackFurby/playing-cards",
        description="40k synthetic images (4 x 10k subsets) with exact 4-corner coordinates per card -> OBB training.",
        label_style="jackfurby_pkl", class_space="cards52", box_semantics="card", provides=("det", "obb"),
        notes=("train.pkl/val.pkl dicts: {id: {img_path, class_label, concept_label, "
               "card_points: [[(TL),(TR),(BL),(BR)], label]}}. Card label index is rank-major: "
               "2C=0 2D=1 2H=2 2S=3 3C=4 ... AC=48 AD=49 AH=50 AS=51 (see vision/configs/jackfurby_card_classes.txt). "
               "Corner order is TL,TR,BL,BR (not clockwise). Images are square (out_len)."),
    ),
    "augstartups": DatasetSpec(
        key="augstartups", source="roboflow", ref="augmented-startups/playing-cards-ow27d", version=4,
        description="Augmented Startups playing cards (YOLO), cards blended into varied textures; 52 classes named like 10C/AS.",
        box_semantics="corner", requires_env=("ROBOFLOW_API_KEY",),
        notes="Boxes are expected to cover the corner index; confirm with --preview (auto-detected area heuristic is recorded in manifest).",
    ),
    "cardsbgop7": DatasetSpec(
        key="cardsbgop7", source="roboflow", ref="cards-bgop7/cards-gzetw", version=None,
        description="~2.5k images with rank and suit annotated as SEPARATE objects (robust to partial occlusion).",
        class_space="parts", box_semantics="parts", provides=("parts",), requires_env=("ROBOFLOW_API_KEY",),
        notes="Converted into the 17-class PART_CLASSES space (RANK_A..RANK_K, SUIT_C..SUIT_S).",
    ),
    "pcc3": DatasetSpec(
        key="pcc3", source="roboflow", ref="deep-learning-for-image-and-video-processing/pcc3.0-yolov8", version=None,
        description="pcc3.0 (Deep Learning for Image and Video Processing): 55 classes incl. pile-face-down / pile-face-up.",
        name_style="pcc_dutch", class_space="all", box_semantics="card", provides=("det", "piles"),
        requires_env=("ROBOFLOW_API_KEY",),
        notes="Dutch class letters: h=hearts k=clubs r=diamonds s=spades; a=Ace b=Jack v=Queen h=King; j=joker.",
    ),
    "andy8744": DatasetSpec(
        key="andy8744", source="kaggle", ref="andy8744/playing-cards-object-detection-dataset",
        description="20k synthetic 416x416 images (YOLOv5 format) built with the geaxgx pipeline over DTD textures; 52 classes.",
        box_semantics="corner", requires_env=("KAGGLE_USERNAME", "KAGGLE_KEY"),
        notes="Boxes cover the corner index (geaxgx-style hulls). Credentials may also come from ~/.kaggle/kaggle.json or KAGGLE_API_TOKEN.",
    ),
    "dtd": DatasetSpec(
        key="dtd", source="url", ref="https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz",
        description="Describable Textures Dataset (5,640 texture images) used as synthetic backgrounds.",
        kind="backgrounds", label_style="none", provides=("backgrounds",), license="research use, see DTD page",
    ),
    "cardart": DatasetSpec(
        key="cardart", source="github_raw", ref="hayeah/playing-cards-assets",
        description="Public-domain vector deck rendered to 222x323 PNGs (png/<rank>_of_<suit>.png, red_joker, black_joker, back).",
        kind="card_art", label_style="none", provides=("card_art",), license="public domain (vector-playing-cards)",
        notes="Fetched from raw.githubusercontent.com/hayeah/playing-cards-assets/master/png/<file>.png",
    ),
}

DATASET_GROUPS: Dict[str, Tuple[str, ...]] = {
    "all": tuple(DATASETS),
    "detection": ("jackfurby", "augstartups", "cardsbgop7", "pcc3", "andy8744"),
    "corner": ("augstartups", "andy8744"),
    "card": ("jackfurby", "pcc3"),
    "assets": ("dtd", "cardart"),
    "free": ("jackfurby", "dtd", "cardart"),   # no API key needed
}


# --------------------------------------------------------------------------- synthetic generation
@dataclass
class SynthConfig:
    out_dir: Path = field(default_factory=lambda: Paths().synthetic / "rummy_v1")
    cards_dir: Path = field(default_factory=lambda: Paths().assets_cards)
    backgrounds_dir: Optional[Path] = field(default_factory=lambda: Paths().assets_backgrounds)
    num_images: int = 20000
    val_ratio: float = 0.1
    img_size: int = 640
    seed: int = 0
    workers: int = 0                            # 0 -> os.cpu_count()
    # scenario mix (weights are normalised). 500 rummy specifics: hands of up to 13 cards,
    # and the discard pile is SPREAD so every card is visible ("spread" scenario).
    scenario_weights: Dict[str, float] = field(default_factory=lambda: {"single": 0.10, "hand": 0.40, "spread": 0.30, "pile": 0.20})
    hand_cards: Tuple[int, int] = (2, 13)       # cards per fanned hand
    spread_cards: Tuple[int, int] = (2, 12)     # cards per spread discard row
    pile_cards: Tuple[int, int] = (2, 8)        # cards per chaotic pile
    card_scale: Tuple[float, float] = (0.55, 1.15)   # relative to a nominal card height of ~0.35 * img_size
    max_perspective: float = 0.08               # fraction of card size for corner jitter
    finger_prob: float = 0.5                    # chance to draw skin-tone occluders on a hand
    glare_prob: float = 0.35
    shadow_prob: float = 0.5
    blur_prob: float = 0.3
    noise_prob: float = 0.4
    min_corner_visibility: float = 0.6          # keep a corner box only if >= this fraction of its hull is unoccluded
    min_card_visibility: float = 0.15           # keep full-card / OBB label only if >= this fraction of the card is visible
    write_obb: bool = True
    write_card_view: bool = True
    write_crops: bool = True
    crop_size: int = 96
    jpeg_quality: Tuple[int, int] = (70, 95)
    copy_views: bool = False                    # copy instead of symlink for views/ (Windows)

    def to_dict(self) -> Dict:
        d = asdict(self)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}


# --------------------------------------------------------------------------- model / training
@dataclass
class ModelConfig:
    task: str = "detector"          # detector | localizer | obb | classifier
    arch: str = "yolo11n"           # ultralytics scale: yolo11n/s/m ... (ignored for classifier)
    pretrained: bool = True         # start from COCO weights (downloads <arch>.pt) when True
    imgsz: int = 640
    class_space: str = "cards52"    # detector: cards52 | all ; localizer: card ; obb: cards52
    # classifier only
    backbone: str = "mobilenet_v3_small"   # mobilenet_v3_small | mobilenet_v3_large | resnet18
    crop_size: int = 96
    dropout: float = 0.2
    weights: Optional[str] = None   # explicit checkpoint path (.pt) to load instead of arch/pretrained


@dataclass
class TrainConfig:
    task: str = "detector"
    datasets: List[str] = field(default_factory=lambda: ["synthetic:rummy_v1", "augstartups", "andy8744"])
    box_semantics: str = "corner"   # datasets with a different manifest semantics are refused
    epochs: int = 50
    batch: int = 32
    imgsz: int = 640
    device: str = "auto"            # auto | cpu | 0 | 0,1 ...
    workers: int = 4
    lr0: float = 0.01
    patience: int = 20
    seed: int = 0
    fraction: float = 1.0           # fraction of the training set to use (smoke runs)
    name: str = "detector_v1"
    export: List[str] = field(default_factory=lambda: ["onnx", "tflite"])
    smoke: bool = False
    # classifier
    crops: List[str] = field(default_factory=lambda: ["synthetic:rummy_v1"])
    cls_lr: float = 1e-3
    cls_batch: int = 128
    cls_epochs: int = 15
