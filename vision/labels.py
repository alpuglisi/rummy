"""YOLO label IO and dataset-layout helpers shared by the vision scripts.

Canonical on-disk layout (used for downloaded, converted and synthetic data)::

    <dataset_root>/
        data.yaml                  # ultralytics dataset file, names == canonical names
        manifest.json              # provenance: source, box_semantics, class mapping, counts
        images/{train,val[,test]}/*.jpg|png
        labels/{train,val[,test]}/*.txt

Ultralytics locates a label by replacing the last ``/images/`` in an image path
with ``/labels/``, so alternative label sets for the same images live in
"views" whose ``images`` entry is a symlink (or copy) of the real image dir::

    <dataset_root>/views/<name>/images -> ../../images
    <dataset_root>/views/<name>/labels/{train,val}/*.txt

Label formats
-------------
* detection: ``cls cx cy w h`` (all normalised to [0, 1])
* OBB (ultralytics format): ``cls x1 y1 x2 y2 x3 y3 x4 y4`` (normalised)

``box_semantics`` (recorded in every manifest) says what a detection box covers:
``corner`` = the rank+suit index in one corner of the card (tight, what a fanned
hand shows), ``card`` = the whole card, ``parts`` = rank and suit as separate
objects.  Never mix semantics inside one training run.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import yaml

IMG_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLITS: Tuple[str, ...] = ("train", "val", "test")
BOX_SEMANTICS: Tuple[str, ...] = ("corner", "card", "parts")

PathLike = Union[str, os.PathLike]
Point = Tuple[float, float]


# --------------------------------------------------------------------------- geometry
def clip01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else float(v)


def points_aabb(points: Sequence[Point]) -> Tuple[float, float, float, float]:
    """Axis-aligned bounding box ``(x1, y1, x2, y2)`` of a point list."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def polygon_area(points: Sequence[Point]) -> float:
    """Shoelace area (always >= 0)."""
    n = len(points)
    if n < 3:
        return 0.0
    a = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def order_quad(points: Sequence[Point]) -> List[Point]:
    """Return the 4 points ordered top-left, top-right, bottom-right, bottom-left."""
    pts = [tuple(map(float, p)) for p in points]
    if len(pts) != 4:
        raise ValueError("order_quad expects exactly 4 points")
    s = [p[0] + p[1] for p in pts]
    d = [p[0] - p[1] for p in pts]
    tl = pts[s.index(min(s))]
    br = pts[s.index(max(s))]
    tr = pts[d.index(max(d))]
    bl = pts[d.index(min(d))]
    return [tl, tr, br, bl]


# --------------------------------------------------------------------------- label records
@dataclass
class Box:
    """One YOLO detection box, normalised ``cx cy w h``."""
    cls: int
    cx: float
    cy: float
    w: float
    h: float

    @classmethod
    def from_xyxy(cls, c: int, x1: float, y1: float, x2: float, y2: float, img_w: float, img_h: float, clip: bool = True) -> Optional["Box"]:
        """Build from pixel corners; clipped to the image.  Returns ``None`` if empty."""
        if clip:
            x1, x2 = max(0.0, min(x1, x2)), min(float(img_w), max(x1, x2))
            y1, y2 = max(0.0, min(y1, y2)), min(float(img_h), max(y1, y2))
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return None
        return cls(int(c), (x1 + x2) / 2.0 / img_w, (y1 + y2) / 2.0 / img_h, w / img_w, h / img_h)

    @classmethod
    def from_points(cls, c: int, points: Sequence[Point], img_w: float, img_h: float) -> Optional["Box"]:
        return cls.from_xyxy(c, *points_aabb(points), img_w=img_w, img_h=img_h)

    def to_xyxy(self, img_w: float, img_h: float) -> Tuple[float, float, float, float]:
        return ((self.cx - self.w / 2) * img_w, (self.cy - self.h / 2) * img_h,
                (self.cx + self.w / 2) * img_w, (self.cy + self.h / 2) * img_h)

    def area(self) -> float:
        return self.w * self.h

    def to_line(self) -> str:
        return f"{self.cls} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"

    @classmethod
    def parse(cls, line: str) -> Optional["Box"]:
        parts = line.split()
        if len(parts) < 5:
            return None
        c = int(float(parts[0]))
        cx, cy, w, h = (float(v) for v in parts[1:5])
        return cls(c, cx, cy, w, h)


@dataclass
class Quad:
    """One oriented box as 4 normalised points (ultralytics OBB format)."""
    cls: int
    pts: List[Point] = field(default_factory=list)

    @classmethod
    def from_pixels(cls, c: int, points: Sequence[Point], img_w: float, img_h: float) -> "Quad":
        return cls(int(c), [(clip01(x / img_w), clip01(y / img_h)) for x, y in order_quad(points)])

    def to_pixels(self, img_w: float, img_h: float) -> List[Point]:
        return [(x * img_w, y * img_h) for x, y in self.pts]

    def aabb(self) -> Optional[Box]:
        x1, y1, x2, y2 = points_aabb(self.pts)
        if x2 <= x1 or y2 <= y1:
            return None
        return Box(self.cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)

    def to_line(self) -> str:
        return f"{self.cls} " + " ".join(f"{clip01(x):.6f} {clip01(y):.6f}" for x, y in self.pts)

    @classmethod
    def parse(cls, line: str) -> Optional["Quad"]:
        parts = line.split()
        if len(parts) < 9:
            return None
        vals = [float(v) for v in parts[1:9]]
        return cls(int(float(parts[0])), [(vals[i], vals[i + 1]) for i in range(0, 8, 2)])


# --------------------------------------------------------------------------- label file IO
def read_boxes(path: PathLike) -> List[Box]:
    p = Path(path)
    if not p.exists():
        return []
    out: List[Box] = []
    for line in p.read_text().splitlines():
        if line.strip():
            b = Box.parse(line)
            if b is not None:
                out.append(b)
    return out


def write_boxes(path: PathLike, boxes: Iterable[Box]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(b.to_line() + "\n" for b in boxes))


def read_quads(path: PathLike) -> List[Quad]:
    p = Path(path)
    if not p.exists():
        return []
    out: List[Quad] = []
    for line in p.read_text().splitlines():
        if line.strip():
            q = Quad.parse(line)
            if q is not None:
                out.append(q)
    return out


def write_quads(path: PathLike, quads: Iterable[Quad]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(q.to_line() + "\n" for q in quads))


def remap_boxes(boxes: Iterable[Box], mapping: Dict[int, Optional[int]]) -> Tuple[List[Box], int]:
    """Apply ``{old_cls: new_cls_or_None}``; returns ``(kept_boxes, dropped_count)``."""
    kept: List[Box] = []
    dropped = 0
    for b in boxes:
        new = mapping.get(b.cls)
        if new is None:
            dropped += 1
            continue
        kept.append(Box(new, b.cx, b.cy, b.w, b.h))
    return kept, dropped


# --------------------------------------------------------------------------- layout helpers
def split_dirs(root: PathLike, split: str) -> Tuple[Path, Path]:
    root = Path(root)
    return root / "images" / split, root / "labels" / split


def ensure_layout(root: PathLike, splits: Sequence[str] = ("train", "val")) -> Path:
    root = Path(root)
    for s in splits:
        (root / "images" / s).mkdir(parents=True, exist_ok=True)
        (root / "labels" / s).mkdir(parents=True, exist_ok=True)
    return root


def list_images(directory: PathLike) -> List[Path]:
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)


def label_path_for(image_path: PathLike, labels_dir: Optional[PathLike] = None) -> Path:
    """Label file for an image.  Mirrors ultralytics' ``images`` -> ``labels`` substitution."""
    ip = Path(image_path)
    if labels_dir is not None:
        return Path(labels_dir) / (ip.stem + ".txt")
    parts = list(ip.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def iter_split(root: PathLike, split: str) -> Iterator[Tuple[Path, Path]]:
    """Yield ``(image_path, label_path)`` for a split (label may not exist yet)."""
    img_dir, lbl_dir = split_dirs(root, split)
    for img in list_images(img_dir):
        yield img, lbl_dir / (img.stem + ".txt")


def link_or_copy(src: PathLike, dst: PathLike, copy: bool = False) -> None:
    """Symlink ``dst -> src`` (relative), falling back to a copy when symlinks fail."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if not copy:
        try:
            dst.symlink_to(os.path.relpath(src.resolve(), dst.parent.resolve()), target_is_directory=src.is_dir())
            return
        except (OSError, NotImplementedError):
            pass
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def make_view(root: PathLike, name: str, labels_src: PathLike, copy: bool = False) -> Path:
    """Create ``<root>/views/<name>`` whose ``images`` points at ``<root>/images``
    and whose ``labels`` points at ``labels_src``.  Returns the view root."""
    root = Path(root)
    view = root / "views" / name
    view.mkdir(parents=True, exist_ok=True)
    link_or_copy(root / "images", view / "images", copy=copy)
    link_or_copy(Path(labels_src), view / "labels", copy=copy)
    return view


# --------------------------------------------------------------------------- data.yaml / manifest
def write_data_yaml(path: PathLike, names: Sequence[str], train: Union[str, Sequence[str]], val: Union[str, Sequence[str]],
                    test: Optional[Union[str, Sequence[str]]] = None, root: Optional[PathLike] = None,
                    extra: Optional[Dict] = None) -> Path:
    """Write an ultralytics dataset yaml.  ``train``/``val`` may be a path or a list of paths
    (ultralytics accepts lists, which is how ``train.py`` merges datasets without copying)."""
    data: Dict = {}
    if root is not None:
        data["path"] = str(Path(root).resolve())
    data["train"] = train if isinstance(train, str) else list(map(str, train))
    data["val"] = val if isinstance(val, str) else list(map(str, val))
    if test:
        data["test"] = test if isinstance(test, str) else list(map(str, test))
    data["nc"] = len(names)
    data["names"] = {i: str(n) for i, n in enumerate(names)}
    if extra:
        data.update(extra)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return p


def read_data_yaml(path: PathLike) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def yaml_names_list(data: Dict) -> List[str]:
    """``names`` from a data.yaml as an ordered list, whether stored as list or ``{id: name}``."""
    names = data.get("names", [])
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda k: int(k))]
    return [str(n) for n in names]


def write_manifest(root: PathLike, manifest: Dict) -> Path:
    p = Path(root) / "manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return p


def read_manifest(root: PathLike) -> Dict:
    p = Path(root) / "manifest.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def dataset_stats(root: PathLike, splits: Sequence[str] = SPLITS) -> Dict[str, Dict]:
    """Per-split image / box counts and per-class histogram (from ``labels``)."""
    stats: Dict[str, Dict] = {}
    for s in splits:
        img_dir, lbl_dir = split_dirs(root, s)
        if not img_dir.is_dir():
            continue
        images = list_images(img_dir)
        per_class: Dict[int, int] = {}
        n_boxes = 0
        n_labelled = 0
        for img in images:
            lp = lbl_dir / (img.stem + ".txt")
            if lp.exists():
                n_labelled += 1
                for b in read_boxes(lp):
                    n_boxes += 1
                    per_class[b.cls] = per_class.get(b.cls, 0) + 1
        stats[s] = {"images": len(images), "labelled_images": n_labelled, "boxes": n_boxes,
                    "per_class": dict(sorted(per_class.items()))}
    return stats
