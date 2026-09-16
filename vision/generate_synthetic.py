"""Synthetic training images for the card detector (section B of the vision pipeline).

The generator composites clean card art onto backgrounds the way the geaxgx
"playing-card detection" notebook does, but with numpy + OpenCV only (no imgaug /
shapely) and with scenarios that match how 500 Rummy is actually played:

* ``single``  one card, any rotation / scale / position
* ``hand``    a fanned hand of 2..13 cards (top-left index of every card visible),
              optional skin-tone finger occluders, mild card bend
* ``spread``  a discard row where every card is partially visible (40-75 % overlap)
* ``pile``    a chaotic pile, later cards on top, the top card fully visible

Every card is placed with one 3x3 homography that is applied identically to the RGBA
card image (``cv2.warpPerspective``) and to its keypoints (4 card corners and the
convex hull of the top-left / bottom-right rank+suit index).  Cards are composited
back to front while a per-card visibility mask records what later cards and
occluders cover, so labels can be filtered by how much of each index is still visible.

Labels written (canonical layout, see ``vision/labels.py``)::

    <out>/images/{train,val}/*.jpg           final JPEG images
    <out>/labels/{train,val}/*.txt           PRIMARY: corner-index boxes, 52 classes ("corner" semantics)
    <out>/data.yaml, manifest.json
    <out>/views/card/{images/, labels/, data.yaml}   whole-card AABBs ("card" semantics)
    <out>/views/obb/{images/, labels/, data.yaml}    oriented boxes (ultralytics OBB format)
    <out>/crops/{train,val}/<CANONICAL>/*.jpg + crops/labels.csv  index crops for the stage-2 classifier
    <out>/preview/*.jpg                        annotated images (``--preview N``)

``views/<name>/images/{train,val}`` are real directories holding one relative symlink per image
(a copy with ``--copy-views``).  A single directory symlink would not do: ultralytics resolves
the dataset directories before substituting ``images`` -> ``labels`` in each image path, so it
would silently train the view on the PRIMARY corner labels.

Card art: ``<cards_dir>/<CANONICAL>.png`` (``10C.png``) or ``<rank>_of_<suit>.png``;
missing files are fetched from the public-domain hayeah deck when allowed, and any card
that is still missing is rendered procedurally so generation always works offline.
Backgrounds: images found under ``backgrounds_dir`` (DTD textures) with procedural
felt / wood / noise / gradient / tile textures as fallback and for variety.

Usage::

    python -m vision.generate_synthetic --out data/vision/synthetic/rummy_v1 --num 20000 --workers 4 --preview 20
    python vision/generate_synthetic.py --num 200 --img-size 416 --scenarios hand:0.6,spread:0.4 --seed 1

    from vision.generate_synthetic import generate, load_card_bank, render_scene
    manifest = generate(SynthConfig(out_dir=Path("/tmp/synth"), num_images=100))

Output is deterministic for a given ``seed`` regardless of ``--workers`` (image ``i``
uses ``numpy.random.default_rng([seed, i])``, so datasets made with nearby seeds share no
images).  ``generate`` refuses to write into an ``out_dir`` that already holds outputs of a
previous run unless ``force`` / ``--force`` is given (those outputs are then removed first).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision import cards as C  # noqa: E402
from vision import labels as L  # noqa: E402
from vision.config import SynthConfig  # noqa: E402

log = logging.getLogger("vision.generate_synthetic")

# --------------------------------------------------------------------------- constants
#: Every card is normalised to this canvas (the hayeah art is exactly 222x323, so real art is never resampled).
CARD_W, CARD_H = 222, 323
#: Rectangular zone (fractions of the card) that contains the top-left rank+suit index.
#: Measured on the hayeah deck: the index spans x 1..41, y 4..79 of 222x323; pips / face frames start at x>=29.
CORNER_ZONE_FRAC: Tuple[float, float, float, float] = (0.0, 0.0, 0.215, 0.275)
CARD_ART_URL = "https://raw.githubusercontent.com/hayeah/playing-cards-assets/master/png/"
SCENARIOS: Tuple[str, ...] = ("single", "hand", "spread", "pile")
HULL_CACHE_NAME = "corner_hulls.json"
CROP_PAD = 0.15          # padding around a corner box before cutting a classifier crop (each side)
MIN_BOX_PX = 4           # boxes smaller than this (either side, in pixels) are dropped
NOMINAL_CARD_FRAC = 0.35  # nominal card height as a fraction of img_size at card_scale == 1
STEM_PREFIX = "synth_"

#: Card corners in canvas pixels, TL, TR, BR, BL.
REF_CORNERS = np.array([[0, 0], [CARD_W, 0], [CARD_W, CARD_H], [0, CARD_H]], dtype=np.float32)

# --------------------------------------------------------------------------- data classes


@dataclass
class CardAsset:
    """One card ready for compositing.

    ``rgba`` is the card image on the fixed ``CARD_H x CARD_W`` canvas with an alpha
    channel, stored in OpenCV channel order (BGRA, exactly what ``cv2.imread(...,
    IMREAD_UNCHANGED)`` returns).  ``hull_tl`` / ``hull_br`` are the convex hulls (k, 2)
    of the top-left / bottom-right rank+suit index in canvas pixels, ``corners`` the 4
    card corners TL, TR, BR, BL.
    """
    name: str
    cid: int
    rgba: np.ndarray
    hull_tl: np.ndarray
    hull_br: np.ndarray
    corners: np.ndarray = field(default_factory=lambda: REF_CORNERS.copy())
    hull_ok: bool = True          # False when the fixed-rectangle fallback zone was used
    source: str = "procedural"    # "file" | "procedural"

    @property
    def bgra(self) -> np.ndarray:
        return self.rgba

    @property
    def rank_idx(self) -> int:
        return self.cid % 13

    @property
    def suit_idx(self) -> int:
        return self.cid // 13


@dataclass
class CardBank:
    """The 52 ``CardAsset`` objects the generator samples from."""
    cards: List[CardAsset]
    source: str = "procedural"    # "files" | "procedural" | "mixed"
    cards_dir: Optional[Path] = None

    def __len__(self) -> int:
        return len(self.cards)

    def __iter__(self):
        return iter(self.cards)

    def __getitem__(self, i: int) -> CardAsset:
        return self.cards[i]

    def by_name(self, name: str) -> CardAsset:
        for c in self.cards:
            if c.name == name:
                return c
        raise KeyError(name)

    def sample(self, rng: np.random.Generator, n: int) -> List[CardAsset]:
        """``n`` distinct cards (a single deck never shows the same card twice)."""
        n = max(1, min(int(n), len(self.cards)))
        idx = rng.choice(len(self.cards), size=n, replace=False)
        return [self.cards[int(i)] for i in idx]

    @property
    def hull_fallbacks(self) -> int:
        return sum(1 for c in self.cards if not c.hull_ok)


@dataclass
class CardAnnotation:
    """Ground truth for one card in a rendered scene (all coordinates in scene pixels)."""
    name: str
    cid: int
    corners: np.ndarray     # (4, 2) TL, TR, BR, BL of the card
    hull_tl: np.ndarray     # (k, 2) transformed top-left index hull
    hull_br: np.ndarray     # (k, 2) transformed bottom-right index hull
    vis_card: float         # fraction of the card area still visible (0..1)
    vis_tl: float           # fraction of the top-left hull still visible
    vis_br: float
    z: int                  # draw order (0 = bottom)
    matrix: Optional[np.ndarray] = None  # the 3x3 homography used for this card


@dataclass
class SceneLabels:
    """Label records derived from ``CardAnnotation`` objects for one image."""
    corner: List[L.Box] = field(default_factory=list)
    corner_which: List[str] = field(default_factory=list)   # "tl" / "br", parallel to ``corner``
    card: List[L.Box] = field(default_factory=list)
    obb: List[L.Quad] = field(default_factory=list)
    dropped_corners: int = 0
    dropped_cards: int = 0
    corner_vis: List[float] = field(default_factory=list)   # visibility of the KEPT corner hulls


# --------------------------------------------------------------------------- small geometry helpers
def mat_translate(tx: float, ty: float) -> np.ndarray:
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)


def mat_scale(s: float, cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    return mat_translate(cx, cy) @ np.diag([s, s, 1.0]) @ mat_translate(-cx, -cy)


def mat_rotate(deg: float, cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    """Rotation about (cx, cy) in image coordinates (y down); positive = clockwise on screen."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    return mat_translate(cx, cy) @ r @ mat_translate(-cx, -cy)


def apply_homography(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 matrix to (n, 2) points (divides by w)."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    h = np.concatenate([p, np.ones((len(p), 1))], axis=1) @ M.T
    w = h[:, 2:3]
    w = np.where(np.abs(w) < 1e-9, 1e-9, w)
    return (h[:, :2] / w).astype(np.float32)


def polygon_area(pts: np.ndarray) -> float:
    return L.polygon_area([(float(x), float(y)) for x, y in np.asarray(pts).reshape(-1, 2)])


def corner_zone(w: int = CARD_W, h: int = CARD_H) -> Tuple[int, int, int, int]:
    """Pixel rectangle ``(x0, y0, x1, y1)`` of the top-left index zone on a ``w x h`` canvas."""
    fx0, fy0, fx1, fy1 = CORNER_ZONE_FRAC
    return int(round(fx0 * w)), int(round(fy0 * h)), int(round(fx1 * w)), int(round(fy1 * h))


def rotate_hull_180(hull: np.ndarray, w: int = CARD_W, h: int = CARD_H) -> np.ndarray:
    """Bottom-right index hull = top-left hull rotated by 180 degrees about the card centre."""
    pts = np.asarray(hull, dtype=np.float32).reshape(-1, 2)
    return np.stack([w - pts[:, 0], h - pts[:, 1]], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- corner hull (geaxgx findHull)
def _ink_mask(bgra: np.ndarray) -> np.ndarray:
    """Binary mask (0/1 uint8) of printed ink: composite over white, then dark OR saturated pixels.

    No dilation on purpose: glyphs must stay separated from content that sits 1 px away
    (face-card frames), so the anti-aliased rim is excluded by a slightly strict threshold.
    """
    a = bgra[:, :, 3:4].astype(np.float32) / 255.0
    rgb = (bgra[:, :, :3].astype(np.float32) * a + 255.0 * (1.0 - a)).astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)[:, :, 1]
    return ((gray < 150) | (sat > 110)).astype(np.uint8)


def find_corner_hull(bgra: np.ndarray, zone: Optional[Tuple[int, int, int, int]] = None,
                     min_area_frac: float = 0.002, min_solidity: float = 0.25,
                     area_range: Tuple[float, float] = (0.04, 0.9)) -> Optional[np.ndarray]:
    """Convex hull (k, 2) of the rank+suit index inside ``zone`` (default: ``corner_zone``).

    Reproduces the idea of geaxgx's ``findHull``: binarise the zone, keep the connected
    components that are plausible glyph parts (area, solidity) and take the convex hull
    of their union.  Two adaptations make it work for clean art and unknown decks:
    components that touch the zone border belong to content that continues outside the
    zone (pips, face-card frames, card borders) and are rejected, and a hull whose area
    or extent is implausible (``area_range`` x zone area) returns ``None`` so the caller
    can fall back to a fixed rectangle.  Returned coordinates are in card-canvas pixels.
    """
    h, w = bgra.shape[:2]
    x0, y0, x1, y1 = zone or corner_zone(w, h)
    zone_img = bgra[y0:y1, x0:x1]
    zh, zw = zone_img.shape[:2]
    if zh < 4 or zw < 4:
        return None
    ink = _ink_mask(zone_img)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=4)
    zone_area = float(zw * zh)
    keep = np.zeros_like(ink)
    for i in range(1, n):
        bx, by, bw, bh, area = (int(v) for v in stats[i])
        if area < min_area_frac * zone_area:
            continue
        if bx <= 0 or by <= 0 or bx + bw >= zw or by + bh >= zh:
            continue  # touches the zone border -> content that continues outside the index
        comp = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(np.concatenate(cnts)))
        if hull_area <= 0 or area / hull_area < min_solidity:
            continue
        keep |= comp
    pts = cv2.findNonZero(keep)
    if pts is None:
        return None
    hull = cv2.convexHull(pts).reshape(-1, 2).astype(np.float32)
    hull_area = cv2.contourArea(hull)
    lo, hi = area_range
    if not (lo * zone_area <= hull_area <= hi * zone_area):
        return None
    bx, by, bw, bh = cv2.boundingRect(hull)
    if bh < 0.2 * zh or bw < 0.12 * zw:
        return None
    # the strict ink threshold shaved the anti-aliased rim off the glyphs: grow the hull by ~1 px
    centre = hull.mean(axis=0, keepdims=True)
    hull = centre + (hull - centre) * (1.0 + 1.5 / max(1.0, math.sqrt(hull_area)))
    hull[:, 0] = np.clip(hull[:, 0], 0, zw) + x0
    hull[:, 1] = np.clip(hull[:, 1], 0, zh) + y0
    return hull


def fallback_corner_hull(w: int = CARD_W, h: int = CARD_H) -> np.ndarray:
    """Fixed rectangle used when hull extraction fails (zone inset by 8 %)."""
    x0, y0, x1, y1 = corner_zone(w, h)
    dx, dy = 0.08 * (x1 - x0), 0.08 * (y1 - y0)
    return np.array([[x0 + dx, y0 + dy], [x1 - dx, y0 + dy], [x1 - dx, y1 - dy], [x0 + dx, y1 - dy]], dtype=np.float32)


# --------------------------------------------------------------------------- procedural card art
_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)


def _load_font(size: int):
    from PIL import ImageFont
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _draw_suit(draw, suit: str, cx: float, cy: float, s: float, colour) -> None:
    """Draw a suit glyph of height ~``s`` centred at (cx, cy) with PIL primitives (font independent)."""
    if suit == "D":
        draw.polygon([(cx, cy - s / 2), (cx + s * 0.36, cy), (cx, cy + s / 2), (cx - s * 0.36, cy)], fill=colour)
    elif suit == "H":
        r = s * 0.27
        draw.ellipse([cx - 2 * r, cy - s / 2, cx, cy - s / 2 + 2 * r], fill=colour)
        draw.ellipse([cx, cy - s / 2, cx + 2 * r, cy - s / 2 + 2 * r], fill=colour)
        draw.polygon([(cx - 2 * r, cy - s / 2 + r * 1.05), (cx + 2 * r, cy - s / 2 + r * 1.05), (cx, cy + s / 2)], fill=colour)
    elif suit == "S":
        r = s * 0.27
        top = cy - s / 2
        draw.polygon([(cx, top), (cx - 2 * r, top + s * 0.55), (cx + 2 * r, top + s * 0.55)], fill=colour)
        draw.ellipse([cx - 2 * r, top + s * 0.28, cx, top + s * 0.28 + 2 * r], fill=colour)
        draw.ellipse([cx, top + s * 0.28, cx + 2 * r, top + s * 0.28 + 2 * r], fill=colour)
        draw.polygon([(cx - s * 0.05, top + s * 0.6), (cx + s * 0.05, top + s * 0.6), (cx + s * 0.2, cy + s / 2), (cx - s * 0.2, cy + s / 2)], fill=colour)
    else:  # clubs
        r = s * 0.2
        top = cy - s / 2
        draw.ellipse([cx - r, top, cx + r, top + 2 * r], fill=colour)
        draw.ellipse([cx - 2.05 * r, top + 1.3 * r, cx - 0.05 * r, top + 3.3 * r], fill=colour)
        draw.ellipse([cx + 0.05 * r, top + 1.3 * r, cx + 2.05 * r, top + 3.3 * r], fill=colour)
        draw.polygon([(cx - s * 0.06, top + 1.8 * r), (cx + s * 0.06, top + 1.8 * r), (cx + s * 0.2, cy + s / 2), (cx - s * 0.2, cy + s / 2)], fill=colour)


def _pip_layout(n: int) -> List[Tuple[float, float]]:
    """Approximate pip positions (fractions of the inner face) for number cards 2..10."""
    layouts = {
        2: [(0.5, 0.2), (0.5, 0.8)],
        3: [(0.5, 0.2), (0.5, 0.5), (0.5, 0.8)],
        4: [(0.3, 0.2), (0.7, 0.2), (0.3, 0.8), (0.7, 0.8)],
        5: [(0.3, 0.2), (0.7, 0.2), (0.5, 0.5), (0.3, 0.8), (0.7, 0.8)],
        6: [(0.3, 0.2), (0.7, 0.2), (0.3, 0.5), (0.7, 0.5), (0.3, 0.8), (0.7, 0.8)],
        7: [(0.3, 0.2), (0.7, 0.2), (0.5, 0.35), (0.3, 0.5), (0.7, 0.5), (0.3, 0.8), (0.7, 0.8)],
        8: [(0.3, 0.2), (0.7, 0.2), (0.5, 0.35), (0.3, 0.5), (0.7, 0.5), (0.5, 0.65), (0.3, 0.8), (0.7, 0.8)],
        9: [(0.3, 0.2), (0.7, 0.2), (0.3, 0.4), (0.7, 0.4), (0.5, 0.5), (0.3, 0.6), (0.7, 0.6), (0.3, 0.8), (0.7, 0.8)],
        10: [(0.3, 0.2), (0.7, 0.2), (0.5, 0.3), (0.3, 0.4), (0.7, 0.4), (0.3, 0.6), (0.7, 0.6), (0.5, 0.7), (0.3, 0.8), (0.7, 0.8)],
    }
    return layouts[n]


def procedural_card(name: str, size: Tuple[int, int] = (CARD_W, CARD_H)) -> np.ndarray:
    """Render a clean, borderless playing card ``name`` (e.g. ``"10H"``) as BGRA uint8.

    White rounded rectangle, rank text + suit glyph in the top-left corner and the
    same index rotated 180 degrees in the bottom-right, red/black by suit, simple centre
    pips (number cards), a big suit (aces) or a framed letter (face cards).  The index
    is placed inside ``corner_zone`` so ``find_corner_hull`` works on it.
    """
    from PIL import Image, ImageDraw

    rank, suit = C.parse_card(name)
    w, h = size
    ss = 2  # supersampling
    W, H = w * ss, h * ss
    red = suit in ("D", "H")
    colour = (205, 20, 35, 255) if red else (25, 25, 30, 255)

    card = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(card)
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=int(0.06 * W), fill=(255, 255, 255, 255), outline=(228, 228, 228, 255), width=ss)

    # top-left index on its own layer, later rotated by 180 degrees for the bottom-right one
    index = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    di = ImageDraw.Draw(index)
    font = _load_font(int(0.078 * H))
    text = rank
    tb = font.getbbox(text)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    ix = int(0.03 * W) + (0 if rank == "10" else int(0.02 * W))
    iy = int(0.02 * H) - tb[1]
    di.text((ix, iy), text, font=font, fill=colour)
    gs = 0.068 * H
    cx = ix + tw / 2 if rank != "10" else ix + tw * 0.42
    cy = int(0.02 * H) + th + gs * 0.72
    _draw_suit(di, suit, cx, cy, gs, colour)
    card.alpha_composite(index)
    card.alpha_composite(index.rotate(180))

    # centre of the card
    d = ImageDraw.Draw(card)
    if rank == "A":
        _draw_suit(d, suit, W / 2, H / 2, 0.42 * H, colour)
    elif rank in ("J", "Q", "K"):
        fx0, fy0, fx1, fy1 = int(0.2 * W), int(0.17 * H), int(0.8 * W), int(0.83 * H)
        d.rectangle([fx0, fy0, fx1, fy1], outline=colour, width=2 * ss)
        d.rectangle([fx0 + 4 * ss, fy0 + 4 * ss, fx1 - 4 * ss, fy1 - 4 * ss], outline=(170, 140, 60, 255), width=ss)
        big = _load_font(int(0.36 * H))
        bb = big.getbbox(rank)
        d.text(((W - (bb[2] - bb[0])) / 2 - bb[0], (H - (bb[3] - bb[1])) / 2 - bb[1]), rank, font=big, fill=colour)
        _draw_suit(d, suit, fx0 + 0.11 * W, fy0 + 0.08 * H, 0.08 * H, colour)
        _draw_suit(d, suit, fx1 - 0.11 * W, fy1 - 0.08 * H, 0.08 * H, colour)
    else:
        n = int(rank)
        px0, py0, px1, py1 = 0.2 * W, 0.16 * H, 0.8 * W, 0.84 * H
        for fx, fy in _pip_layout(n):
            _draw_suit(d, suit, px0 + fx * (px1 - px0), py0 + fy * (py1 - py0), 0.11 * H, colour)

    card = card.resize((w, h), Image.LANCZOS)
    rgba = np.asarray(card, dtype=np.uint8)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)


# --------------------------------------------------------------------------- card art on disk / download
def card_art_candidates(cards_dir: Path, name: str) -> List[Path]:
    """Paths that may hold the art of ``name``: ``<CANONICAL>.png`` then ``<rank>_of_<suit>.png``."""
    return [Path(cards_dir) / f"{name}.png", Path(cards_dir) / C.card_asset_filename(name)]


def missing_card_art(cards_dir: Path) -> List[str]:
    return [n for n in C.CARD_CLASSES if not any(p.is_file() for p in card_art_candidates(cards_dir, n))]


def fetch_card_art_fallback(dest_dir: Path, names: Optional[Sequence[str]] = None, retries: int = 3,
                            timeout: float = 20.0) -> int:
    """Best-effort download of the hayeah PNGs into ``dest_dir/<CANONICAL>.png`` (returns #fetched).

    Used only when ``vision.download_datasets.fetch_card_art`` is not importable.
    Network errors are logged, never raised.
    """
    import urllib.error
    import urllib.request

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    names = list(names or C.CARD_CLASSES)
    fetched = 0
    for name in names:
        out = dest_dir / f"{name}.png"
        if out.is_file():
            continue
        url = CARD_ART_URL + C.card_asset_filename(name)
        for attempt in range(1, retries + 1):
            try:
                with urllib.request.urlopen(url, timeout=timeout) as r:
                    data = r.read()
                if not data.startswith(b"\x89PNG"):
                    raise ValueError("not a PNG")
                out.write_bytes(data)
                fetched += 1
                break
            except (urllib.error.URLError, OSError, ValueError) as e:  # noqa: PERF203
                log.debug("fetch %s attempt %d failed: %s", url, attempt, e)
                if attempt == retries:
                    log.warning("could not fetch %s (%s)", url, e)
                else:
                    time.sleep(0.5 * attempt)
    return fetched


def fetch_card_art(dest_dir: Path) -> int:
    """Fetch the missing card PNGs into ``dest_dir``; prefers the downloader's implementation."""
    dest_dir = Path(dest_dir)
    missing = missing_card_art(dest_dir)
    if not missing:
        return 0
    try:
        from vision.download_datasets import fetch_card_art as _dl_fetch  # type: ignore
    except Exception:  # noqa: BLE001 - module may not exist yet
        _dl_fetch = None
    if _dl_fetch is not None:
        try:
            log.info("fetching %d card PNGs via vision.download_datasets.fetch_card_art -> %s", len(missing), dest_dir)
            _dl_fetch(dest_dir)
        except Exception as e:  # noqa: BLE001
            log.warning("download_datasets.fetch_card_art failed (%s); using the built-in fetcher", e)
        still = missing_card_art(dest_dir)
        if not still:
            return len(missing)
    log.info("fetching %d card PNGs from %s -> %s", len(missing_card_art(dest_dir)), CARD_ART_URL, dest_dir)
    return fetch_card_art_fallback(dest_dir, missing_card_art(dest_dir))


def rounded_rect_mask(w: int = CARD_W, h: int = CARD_H, radius_frac: float = 0.055) -> np.ndarray:
    """uint8 mask (255 inside) of a card-shaped rounded rectangle filling the canvas."""
    r = max(1, int(round(radius_frac * w)))
    m = np.zeros((h, w), np.uint8)
    cv2.rectangle(m, (r, 0), (w - 1 - r, h - 1), 255, -1)
    cv2.rectangle(m, (0, r), (w - 1, h - 1 - r), 255, -1)
    for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
        cv2.circle(m, (cx, cy), r, 255, -1, lineType=cv2.LINE_AA)
    return m


def ensure_card_body(bgra: np.ndarray) -> np.ndarray:
    """Give the art a proper opaque card body.

    The public-domain hayeah PNGs are ink-only overlays (the white face is fully
    transparent), so they are composited onto a white rounded-rectangle body.  Fully
    opaque art (no alpha at all) gets its corners rounded through the same mask.  Art
    that already has a sensible alpha (mostly opaque, transparent corners) is returned
    unchanged.
    """
    h, w = bgra.shape[:2]
    a = bgra[:, :, 3]
    opaque_frac = float(np.mean(a == 255))
    if opaque_frac >= 0.5 and a.min() < 255:
        return bgra
    body = rounded_rect_mask(w, h)
    if opaque_frac < 0.5:  # ink-only overlay -> white face under the ink
        af = a.astype(np.float32)[..., None] / 255.0
        rgb = bgra[:, :, :3].astype(np.float32) * af + 255.0 * (1.0 - af)
        out = np.concatenate([np.clip(rgb, 0, 255).astype(np.uint8), body[..., None]], axis=2)
    else:  # fully opaque -> just cut the corners
        out = bgra.copy()
        out[:, :, 3] = np.minimum(out[:, :, 3], body)
    return np.ascontiguousarray(out)


def _normalise_canvas(bgra: np.ndarray) -> np.ndarray:
    """Any image (BGR or BGRA) -> BGRA with an opaque card body on the fixed card canvas."""
    if bgra.ndim == 2:
        bgra = cv2.cvtColor(bgra, cv2.COLOR_GRAY2BGRA)
    elif bgra.shape[2] == 3:
        bgra = cv2.cvtColor(bgra, cv2.COLOR_BGR2BGRA)
    if bgra.shape[:2] != (CARD_H, CARD_W):
        interp = cv2.INTER_AREA if bgra.shape[0] > CARD_H else cv2.INTER_CUBIC
        bgra = cv2.resize(bgra, (CARD_W, CARD_H), interpolation=interp)
    return ensure_card_body(np.ascontiguousarray(bgra))


def _read_hull_cache(cards_dir: Path) -> Dict:
    p = Path(cards_dir) / HULL_CACHE_NAME
    try:
        if p.is_file():
            data = json.loads(p.read_text())
            if data.get("canvas") == [CARD_W, CARD_H] and data.get("zone") == list(CORNER_ZONE_FRAC):
                return data.get("cards", {})
    except (OSError, ValueError) as e:
        log.debug("ignoring hull cache %s: %s", p, e)
    return {}


def _write_hull_cache(cards_dir: Path, entries: Dict) -> None:
    p = Path(cards_dir) / HULL_CACHE_NAME
    try:
        p.write_text(json.dumps({"canvas": [CARD_W, CARD_H], "zone": list(CORNER_ZONE_FRAC), "cards": entries}, indent=1))
    except OSError as e:
        log.debug("could not write hull cache %s: %s", p, e)


def make_card_asset(name: str, bgra: np.ndarray, source: str = "procedural",
                    hull_tl: Optional[np.ndarray] = None) -> CardAsset:
    """Build a ``CardAsset`` (normalising the canvas and extracting the hull when not given)."""
    bgra = _normalise_canvas(bgra)
    ok = True
    if hull_tl is None:
        hull_tl = find_corner_hull(bgra)
        if hull_tl is None:
            ok = False
            hull_tl = fallback_corner_hull()
            log.debug("hull extraction failed for %s (%s); using the fixed zone", name, source)
    hull_tl = np.asarray(hull_tl, dtype=np.float32).reshape(-1, 2)
    return CardAsset(name=name, cid=C.CLASS_TO_ID[name], rgba=bgra, hull_tl=hull_tl,
                     hull_br=rotate_hull_180(hull_tl), corners=REF_CORNERS.copy(), hull_ok=ok, source=source)


def load_card_bank(cards_dir: Optional[Union[str, Path]], fetch: bool = True, cache: bool = True) -> CardBank:
    """Load the 52 cards from ``cards_dir`` (``<CANONICAL>.png`` or ``<rank>_of_<suit>.png``).

    Missing art is fetched when ``fetch`` is true (best effort), and every card that is
    still unavailable is rendered with ``procedural_card`` so this never fails offline.
    Corner hulls are cached in memory (in the assets) and, when ``cache`` is true, in
    ``<cards_dir>/corner_hulls.json``.
    """
    cards_dir = Path(cards_dir) if cards_dir is not None else None
    if cards_dir is not None and fetch and missing_card_art(cards_dir):
        try:
            fetch_card_art(cards_dir)
        except Exception as e:  # noqa: BLE001
            log.warning("card art fetch failed: %s", e)
    cached = _read_hull_cache(cards_dir) if (cards_dir is not None and cache) else {}
    entries: Dict = {}
    assets: List[CardAsset] = []
    n_files = 0
    for name in C.CARD_CLASSES:
        asset = None
        if cards_dir is not None:
            for p in card_art_candidates(cards_dir, name):
                if not p.is_file():
                    continue
                img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if img is None:
                    log.warning("unreadable card art %s", p)
                    continue
                st = p.stat()
                key = f"{p.name}:{st.st_size}:{int(st.st_mtime)}"
                hull = None
                if cached.get(name, {}).get("key") == key and cached[name].get("hull_tl"):
                    hull = np.asarray(cached[name]["hull_tl"], dtype=np.float32)
                asset = make_card_asset(name, img, source="file", hull_tl=hull)
                if hull is None and asset.hull_ok:
                    entries[name] = {"key": key, "hull_tl": asset.hull_tl.astype(float).tolist()}
                elif hull is not None:
                    entries[name] = cached[name]
                n_files += 1
                break
        if asset is None:
            asset = make_card_asset(name, procedural_card(name), source="procedural")
        assets.append(asset)
    if cards_dir is not None and cache and entries and entries != cached:
        _write_hull_cache(cards_dir, entries)
    source = "files" if n_files == 52 else "procedural" if n_files == 0 else "mixed"
    bank = CardBank(cards=assets, source=source, cards_dir=cards_dir)
    log.info("card bank: %d cards (%s%s), %d hull fallback(s)", len(assets), source,
             f" from {cards_dir}" if n_files else "", bank.hull_fallbacks)
    return bank


# --------------------------------------------------------------------------- backgrounds
def _smooth_noise(rng: np.random.Generator, h: int, w: int, cells: int) -> np.ndarray:
    """Low-frequency noise in [0, 1]: a ``cells x cells`` random grid upsampled with cubic interpolation."""
    cells = max(2, int(cells))
    small = rng.random((cells, cells), dtype=np.float32)
    return np.clip(cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC), 0, 1)


def _perlinish(rng: np.random.Generator, h: int, w: int, octaves: int = 4) -> np.ndarray:
    acc = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        acc += amp * _smooth_noise(rng, h, w, 3 * 2 ** o)
        total += amp
        amp *= 0.5
    return acc / total


def _rand_colour(rng: np.random.Generator, lo: int = 20, hi: int = 235) -> np.ndarray:
    return rng.uniform(lo, hi, size=3).astype(np.float32)


def procedural_background(rng: np.random.Generator, size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """Random offline background as BGR uint8 ``(h, w, 3)`` (``size`` = int or ``(h, w)``).

    Kinds: casino felt (green/blue/red/grey cloth with fibre noise), wood grain,
    Perlin-ish noise blended between two colours, linear/radial gradients, tiled
    tablecloth patterns and plain paper.
    """
    h, w = (size, size) if isinstance(size, (int, np.integer)) else (int(size[0]), int(size[1]))
    kind = rng.choice(["felt", "wood", "noise", "gradient", "tiles", "paper"], p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.1])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if kind == "felt":
        base = rng.choice(np.array([[45, 110, 35], [120, 60, 30], [40, 40, 140], [70, 70, 70], [30, 90, 90], [25, 25, 25]], np.float32))
        base = base * rng.uniform(0.7, 1.3) + rng.normal(0, 8, 3)
        img = np.broadcast_to(base, (h, w, 3)).astype(np.float32)
        img = img * (0.8 + 0.4 * _perlinish(rng, h, w, 3))[..., None]
        img += rng.normal(0, rng.uniform(3, 10), (h, w, 1)).astype(np.float32)
    elif kind == "wood":
        c1, c2 = _rand_colour(rng, 40, 120), _rand_colour(rng, 100, 200)
        c1[0] *= 0.6
        c2[0] *= 0.7  # less blue -> brownish
        ang = rng.uniform(0, math.pi)
        u = xx * math.cos(ang) + yy * math.sin(ang)
        v = -xx * math.sin(ang) + yy * math.cos(ang)
        f = rng.uniform(0.05, 0.25)
        grain = 0.5 + 0.5 * np.sin(u * f + 3.0 * _perlinish(rng, h, w, 3) + 0.01 * v)
        grain = grain ** rng.uniform(1.0, 3.0)
        img = c1 * (1 - grain[..., None]) + c2 * grain[..., None]
        img += rng.normal(0, 4, (h, w, 1)).astype(np.float32)
    elif kind == "noise":
        c1, c2 = _rand_colour(rng), _rand_colour(rng)
        t = _perlinish(rng, h, w, 5)[..., None]
        img = c1 * (1 - t) + c2 * t
    elif kind == "gradient":
        c1, c2 = _rand_colour(rng), _rand_colour(rng)
        if rng.random() < 0.5:
            ang = rng.uniform(0, 2 * math.pi)
            t = (xx * math.cos(ang) + yy * math.sin(ang))
        else:
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            t = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        t = (t - t.min()) / max(1e-6, t.max() - t.min())
        img = c1 * (1 - t[..., None]) + c2 * t[..., None]
        img += rng.normal(0, 3, (h, w, 1)).astype(np.float32)
    elif kind == "tiles":
        c1, c2 = _rand_colour(rng), _rand_colour(rng)
        cell = rng.uniform(0.04, 0.2) * max(h, w)
        ang = rng.uniform(-0.4, 0.4)
        u = xx * math.cos(ang) + yy * math.sin(ang)
        v = -xx * math.sin(ang) + yy * math.cos(ang)
        chk = ((np.floor(u / cell) + np.floor(v / cell)) % 2)[..., None]
        if rng.random() < 0.5:  # plaid instead of checker
            chk = (0.5 * ((np.floor(u / cell) % 2) + (np.floor(v / cell) % 2)))[..., None]
        img = c1 * (1 - chk) + c2 * chk
        img += rng.normal(0, 4, (h, w, 1)).astype(np.float32)
    else:  # paper
        base = np.array([rng.uniform(200, 245)] * 3, np.float32) + rng.normal(0, 6, 3)
        img = np.broadcast_to(base, (h, w, 3)).astype(np.float32)
        img = img * (0.92 + 0.08 * _perlinish(rng, h, w, 4))[..., None]
        img += rng.normal(0, 3, (h, w, 1)).astype(np.float32)
    return np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))


def list_background_files(backgrounds_dir: Optional[Union[str, Path]]) -> List[Path]:
    """All image files under ``backgrounds_dir`` (recursive, sorted for determinism)."""
    if backgrounds_dir is None:
        return []
    d = Path(backgrounds_dir)
    if not d.is_dir():
        return []
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in L.IMG_EXTS and p.is_file())


def load_background(rng: np.random.Generator, files: Sequence[Path], size: int, p_file: float = 0.85) -> np.ndarray:
    """Random background as BGR ``size x size``: a random crop of a file (probability ``p_file``) or procedural."""
    if files and rng.random() < p_file:
        path = files[int(rng.integers(len(files)))]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None and img.shape[0] >= 32 and img.shape[1] >= 32:
            h, w = img.shape[:2]
            side = int(min(h, w) * rng.uniform(0.4, 1.0))
            y = int(rng.integers(0, h - side + 1))
            x = int(rng.integers(0, w - side + 1))
            img = cv2.resize(img[y:y + side, x:x + side], (size, size), interpolation=cv2.INTER_AREA)
            k = int(rng.integers(4))
            if k:
                img = np.rot90(img, k)
            if rng.random() < 0.5:
                img = img[:, ::-1]
            return np.ascontiguousarray(img)
        log.debug("unreadable background %s; using a procedural one", path)
    return procedural_background(rng, size)


def prepare_background(rng: np.random.Generator, bg: np.ndarray, size: int) -> np.ndarray:
    """Resize/crop any BGR image to ``size x size`` and apply mild photometric jitter."""
    if bg.ndim == 2:
        bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    elif bg.shape[2] == 4:
        bg = cv2.cvtColor(bg, cv2.COLOR_BGRA2BGR)
    if bg.shape[:2] != (size, size):
        bg = cv2.resize(bg, (size, size), interpolation=cv2.INTER_AREA)
    img = bg.astype(np.float32)
    img = (img - 128.0) * rng.uniform(0.8, 1.15) + 128.0 + rng.uniform(-20, 20)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.5))
    return np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))


# --------------------------------------------------------------------------- photometric augmentation
def _motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    k = np.zeros((length, length), np.float32)
    c = (length - 1) / 2
    a = math.radians(angle_deg)
    dx, dy = math.cos(a), math.sin(a)
    for t in np.linspace(-c, c, length * 2):
        x, y = int(round(c + t * dx)), int(round(c + t * dy))
        k[y, x] = 1.0
    return k / max(1.0, k.sum())


def add_glare(rng: np.random.Generator, img: np.ndarray, strength: Optional[float] = None) -> np.ndarray:
    """Soft white elliptical specular highlight (float32 in/out)."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    centre = (int(rng.uniform(0.1, 0.9) * w), int(rng.uniform(0.1, 0.9) * h))
    axes = (int(rng.uniform(0.08, 0.35) * w), int(rng.uniform(0.04, 0.2) * h))
    cv2.ellipse(mask, centre, axes, float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), rng.uniform(0.03, 0.1) * w)
    s = rng.uniform(0.35, 0.9) if strength is None else strength
    return img + (255.0 - img) * (s * mask)[..., None]


def photometric_augment(rng: np.random.Generator, img: np.ndarray, cfg: SynthConfig) -> np.ndarray:
    """Camera-like degradations on a BGR uint8 image (returns BGR uint8).

    Glare, brightness/contrast, colour temperature + per-channel gain, gamma, gaussian or
    motion blur, gaussian + JPEG noise and a vignette, each with its own probability
    (``cfg.glare_prob``, ``cfg.blur_prob``, ``cfg.noise_prob``).
    """
    x = np.ascontiguousarray(img, dtype=np.float32)
    h, w = x.shape[:2]
    if rng.random() < cfg.glare_prob:
        x = add_glare(rng, x)
    # brightness / contrast
    x = (x - 128.0) * rng.uniform(0.7, 1.3) + 128.0 + rng.uniform(-35, 35)
    # colour temperature (warm <-> cool) and per-channel gain, BGR order
    t = rng.uniform(-0.12, 0.12)
    gains = np.array([1.0 - t, 1.0, 1.0 + t], np.float32) * rng.uniform(0.92, 1.08, 3).astype(np.float32)
    x = x * gains
    # gamma
    x = np.clip(x, 0, 255)
    gamma = rng.uniform(0.7, 1.4)
    x = 255.0 * (x / 255.0) ** gamma
    # blur
    if rng.random() < cfg.blur_prob:
        if rng.random() < 0.6:
            x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.5, 2.2))
        else:
            x = cv2.filter2D(x, -1, _motion_kernel(int(rng.integers(3, 12)), rng.uniform(0, 180)))
    # noise
    if rng.random() < cfg.noise_prob:
        x = x + rng.normal(0, rng.uniform(2, 10), x.shape).astype(np.float32)
        if rng.random() < 0.5:
            q = int(rng.integers(25, 65))
            ok, buf = cv2.imencode(".jpg", np.clip(x, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])
            if ok:
                x = cv2.imdecode(buf, cv2.IMREAD_COLOR).astype(np.float32)
    # vignette
    if rng.random() < 0.5:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / math.sqrt(2)
        x = x * (1.0 - rng.uniform(0.1, 0.5) * r ** 2)[..., None]
    return np.ascontiguousarray(np.clip(x, 0, 255).astype(np.uint8))


# --------------------------------------------------------------------------- scene layouts (card-canvas units)
def _layout_single(rng: np.random.Generator, n: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    return [mat_rotate(rng.uniform(-180, 180), CARD_W / 2, CARD_H / 2)], []


def _layout_hand(rng: np.random.Generator, n: int, finger_prob: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Fanned hand: card i is rotated ``i * dtheta`` about a pivot below the cards and shifted
    right so the top-left index of every card stays uncovered by the next card (drawn on top).
    Returns the per-card matrices and finger-occluder polygons (in the same local units)."""
    W, H = CARD_W, CARD_H
    dtheta = rng.uniform(8.0, 18.0)
    if n > 1:
        dtheta = max(5.0, min(dtheta, 120.0 / (n - 1)))
    d = rng.uniform(0.2, 1.0) * H                    # pivot distance below the bottom edge
    px, py = W / 2 + rng.uniform(-0.3, 0.3) * W, H + d
    # horizontal offset per card so the index column (~0.24 W) is always exposed at the top edge
    need = 0.26 * W
    dx = max(0.04 * W, need - (H + d) * math.sin(math.radians(dtheta))) + rng.uniform(0, 0.04) * W
    mats: List[np.ndarray] = []
    for i in range(n):
        theta = (i - (n - 1) / 2) * dtheta + rng.uniform(-1.0, 1.0)
        shift = i * dx + rng.uniform(-0.01, 0.01) * W
        mats.append(mat_rotate(theta, px, py) @ mat_translate(shift, rng.uniform(-0.015, 0.015) * H))
    polys: List[np.ndarray] = []
    if rng.random() < finger_prob:
        # thumb over the lower part of the front card(s), finger tips along the bottom edge behind
        front = mats[-1]
        thumb = _ellipse_poly(rng.uniform(0.3, 0.55) * W, rng.uniform(0.8, 1.0) * H, rng.uniform(0.16, 0.24) * W,
                              rng.uniform(0.3, 0.45) * H, rng.uniform(-40, 10))
        polys.append(apply_homography(front, thumb))
        for _ in range(int(rng.integers(0, 4))):
            k = int(rng.integers(0, n))
            tip = _ellipse_poly(rng.uniform(0.2, 0.8) * W, rng.uniform(0.98, 1.06) * H, rng.uniform(0.1, 0.16) * W,
                                rng.uniform(0.12, 0.2) * W, rng.uniform(-30, 30))
            polys.append(apply_homography(mats[k], tip))
    return mats, polys


def _layout_spread(rng: np.random.Generator, n: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Discard row: cards overlapped 40-75 % (each later card covers the right part of the previous one)."""
    W, H = CARD_W, CARD_H
    overlap = rng.uniform(0.4, 0.75)
    step = (1.0 - overlap) * W
    mats = []
    for i in range(n):
        jitter_x = rng.uniform(-0.03, 0.03) * W
        jitter_y = rng.uniform(-0.06, 0.06) * H
        rot = rng.uniform(-7.0, 7.0)
        mats.append(mat_translate(i * step + jitter_x, jitter_y) @ mat_rotate(rot, W / 2, H / 2))
    return mats, []


def _layout_pile(rng: np.random.Generator, n: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Chaotic pile: random rotations and centred jitter, later cards on top."""
    W, H = CARD_W, CARD_H
    mats = []
    for _ in range(n):
        jx, jy = rng.uniform(-0.35, 0.35) * W, rng.uniform(-0.3, 0.3) * H
        mats.append(mat_translate(jx, jy) @ mat_rotate(rng.uniform(-180, 180), W / 2, H / 2))
    return mats, []


def _ellipse_poly(cx: float, cy: float, ax: float, ay: float, angle_deg: float, k: int = 28) -> np.ndarray:
    t = np.linspace(0, 2 * math.pi, k, endpoint=False)
    pts = np.stack([ax * np.cos(t), ay * np.sin(t)], axis=1)
    a = math.radians(angle_deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return (pts @ rot.T + [cx, cy]).astype(np.float32)


def sample_num_cards(rng: np.random.Generator, scenario: str, cfg: SynthConfig) -> int:
    if scenario == "single":
        return 1
    lo, hi = {"hand": cfg.hand_cards, "spread": cfg.spread_cards, "pile": cfg.pile_cards}[scenario]
    return int(rng.integers(int(lo), int(hi) + 1))


def _group_transform(rng: np.random.Generator, mats: Sequence[np.ndarray], scenario: str, cfg: SynthConfig) -> np.ndarray:
    """Similarity + mild perspective that places the whole local layout into the image."""
    S = cfg.img_size
    pts = np.concatenate([apply_homography(m, REF_CORNERS) for m in mats])
    bx0, by0, bx1, by1 = pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    scale = rng.uniform(*cfg.card_scale) * NOMINAL_CARD_FRAC * S / CARD_H
    ext = max(bx1 - bx0, by1 - by0) * scale
    limit = {"single": 1.0, "hand": 1.05, "spread": 1.2, "pile": 1.1}[scenario] * S
    if ext > limit:
        scale *= limit / ext
    if scenario == "single":
        angle = 0.0  # the card matrix already holds a full random rotation
        tx, ty = rng.uniform(0.2, 0.8) * S, rng.uniform(0.2, 0.8) * S
    elif scenario == "hand":
        angle = rng.uniform(-30, 30) if rng.random() < 0.75 else rng.uniform(-180, 180)
        tx, ty = S / 2 + rng.uniform(-0.08, 0.08) * S, S / 2 + rng.uniform(-0.05, 0.15) * S
    else:
        angle = rng.uniform(-25, 25) if rng.random() < 0.6 else rng.uniform(-180, 180)
        tx, ty = S / 2 + rng.uniform(-0.15, 0.15) * S, S / 2 + rng.uniform(-0.15, 0.15) * S
    sim = mat_translate(tx, ty) @ mat_rotate(angle) @ mat_scale(scale) @ mat_translate(-cx, -cy)
    # global perspective: jitter the image corners by a few percent
    src = np.array([[0, 0], [S, 0], [S, S], [0, S]], np.float32)
    amt = 0.5 * cfg.max_perspective * S
    dst = src + rng.uniform(-amt, amt, size=(4, 2)).astype(np.float32)
    G = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
    return G @ sim


def _perturb(rng: np.random.Generator, M: np.ndarray, amount: float) -> np.ndarray:
    """Jitter the card's 4 destination corners by ``amount`` x card size (perspective / bend)."""
    if amount <= 0:
        return M
    dst = apply_homography(M, REF_CORNERS)
    size = math.sqrt(max(polygon_area(dst), 1.0))
    dst = dst + rng.uniform(-amount, amount, size=(4, 2)).astype(np.float32) * size
    return cv2.getPerspectiveTransform(REF_CORNERS, dst.astype(np.float32)).astype(np.float64)


# --------------------------------------------------------------------------- compositing
@dataclass
class Layer:
    """One warped layer on the scene canvas: premultiplied BGR float32 + alpha float32 in [0, 1]."""
    rgb: np.ndarray
    alpha: np.ndarray
    is_card: bool


def warp_card_layer(asset: CardAsset, M: np.ndarray, size: int) -> Layer:
    """Warp a card with the 3x3 matrix ``M`` onto a ``size x size`` canvas (premultiplied alpha)."""
    src = asset.rgba.astype(np.float32)
    a = src[:, :, 3:4] / 255.0
    pre = np.concatenate([src[:, :, :3] * a, a], axis=2)
    out = cv2.warpPerspective(pre, M.astype(np.float64), (size, size), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return Layer(rgb=out[:, :, :3], alpha=out[:, :, 3], is_card=True)


_SKIN_TONES = np.array([[150, 180, 230], [120, 160, 220], [95, 135, 200], [70, 105, 160], [50, 75, 120], [170, 200, 240]], np.float32)  # BGR


def finger_layer(rng: np.random.Generator, polys: Sequence[np.ndarray], size: int) -> Optional[Layer]:
    """Skin-tone occluders (soft-edged filled polygons with simple shading) as one layer."""
    if not polys:
        return None
    alpha = np.zeros((size, size), np.float32)
    for p in polys:
        cv2.fillPoly(alpha, [np.round(p).astype(np.int32)], 1.0, lineType=cv2.LINE_AA)
    alpha = cv2.GaussianBlur(alpha, (0, 0), max(0.5, 0.004 * size))
    tone = _SKIN_TONES[int(rng.integers(len(_SKIN_TONES)))] * rng.uniform(0.85, 1.1)
    inner = cv2.GaussianBlur(alpha, (0, 0), max(1.0, 0.03 * size))
    shade = 0.75 + 0.3 * inner  # darker rim, lighter centre
    rgb = np.ascontiguousarray(np.broadcast_to(tone, (size, size, 3))) * shade[..., None]
    rgb = rgb + rng.normal(0, 4, (size, size, 1)).astype(np.float32)
    return Layer(rgb=np.ascontiguousarray(np.clip(rgb, 0, 255) * alpha[..., None]), alpha=alpha, is_card=False)


def composite(rng: np.random.Generator, bg: np.ndarray, layers: Sequence[Layer], cfg: SynthConfig,
              shadow: Optional[bool] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Draw ``layers`` back to front over ``bg`` (BGR uint8) with optional soft drop shadows.

    Returns the composited BGR uint8 image and, for every layer, the boolean mask of the
    pixels where that layer is still visible after everything drawn later (this is what
    the visibility fractions of the labels are computed from).
    """
    size = bg.shape[0]
    canvas = np.ascontiguousarray(bg, dtype=np.float32)
    if shadow is None:
        shadow = rng.random() < cfg.shadow_prob
    if shadow:
        ang = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(0.008, 0.03) * size
        sdx, sdy = dist * math.cos(ang), dist * math.sin(ang)
        s_sigma = rng.uniform(0.006, 0.02) * size
        s_strength = rng.uniform(0.25, 0.55)
        shift = np.array([[1, 0, sdx], [0, 1, sdy]], np.float32)
    for layer in layers:
        a = layer.alpha
        if shadow and layer.is_card:
            sh = cv2.warpAffine(a, shift, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            sh = cv2.GaussianBlur(sh, (0, 0), s_sigma)
            canvas *= (1.0 - s_strength * sh)[..., None]
        canvas = canvas * (1.0 - a)[..., None] + layer.rgb
    covered = np.zeros((size, size), bool)
    vis: List[Optional[np.ndarray]] = [None] * len(layers)
    for i in range(len(layers) - 1, -1, -1):
        hard = layers[i].alpha > 0.5
        vis[i] = hard & ~covered
        covered |= hard
    return np.ascontiguousarray(np.clip(canvas, 0, 255).astype(np.uint8)), vis  # type: ignore[return-value]


def _visible_fraction(vis: np.ndarray, poly: np.ndarray) -> float:
    """Fraction of the (unclipped) polygon area that is visible inside the image."""
    total = polygon_area(poly)
    if total <= 1e-6:
        return 0.0
    size = vis.shape[0]
    x0, y0, x1, y1 = (int(np.floor(poly[:, 0].min())), int(np.floor(poly[:, 1].min())),
                      int(np.ceil(poly[:, 0].max())) + 1, int(np.ceil(poly[:, 1].max())) + 1)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(size, x1), min(size, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(mask, [np.round(poly - [x0, y0]).astype(np.int32)], 1)
    visible = float(np.count_nonzero(mask.astype(bool) & vis[y0:y1, x0:x1]))
    return float(min(1.0, visible / total))


def render_scene(rng: np.random.Generator, scenario: str, cards: Union[CardBank, Sequence[CardAsset]],
                 bg: np.ndarray, cfg: SynthConfig, augment: bool = True) -> Tuple[np.ndarray, List[CardAnnotation]]:
    """Render one ``scenario`` image and return ``(bgr_uint8, annotations)``.

    ``cards`` is either a ``CardBank`` (the number of cards is sampled from ``cfg`` and
    distinct cards are drawn at random) or an explicit sequence of ``CardAsset`` drawn
    in that order (first = bottom).  ``bg`` is any BGR image (resized to ``cfg.img_size``).
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")
    S = int(cfg.img_size)
    if isinstance(cards, CardBank):
        assets = cards.sample(rng, sample_num_cards(rng, scenario, cfg))
    else:
        assets = list(cards)
    n = len(assets)
    if scenario == "single":
        mats, polys = _layout_single(rng, n)
        if n > 1:  # explicit list with several cards: treat like a pile
            mats, polys = _layout_pile(rng, n)
    elif scenario == "hand":
        mats, polys = _layout_hand(rng, n, cfg.finger_prob)
    elif scenario == "spread":
        mats, polys = _layout_spread(rng, n)
    else:
        mats, polys = _layout_pile(rng, n)
    group = _group_transform(rng, mats, scenario, cfg)
    bend = cfg.max_perspective * (0.35 if scenario in ("hand", "spread") else 1.0)
    matrices = [_perturb(rng, group @ m, bend) for m in mats]

    layers = [warp_card_layer(a, M, S) for a, M in zip(assets, matrices)]
    fingers = finger_layer(rng, [apply_homography(group, p) for p in polys], S)
    if fingers is not None:
        layers.append(fingers)
    bg = prepare_background(rng, bg, S)
    img, vis = composite(rng, bg, layers, cfg)

    anns: List[CardAnnotation] = []
    for z, (a, M) in enumerate(zip(assets, matrices)):
        corners = apply_homography(M, a.corners)
        htl = apply_homography(M, a.hull_tl)
        hbr = apply_homography(M, a.hull_br)
        anns.append(CardAnnotation(name=a.name, cid=a.cid, corners=corners, hull_tl=htl, hull_br=hbr,
                                   vis_card=_visible_fraction(vis[z], corners), vis_tl=_visible_fraction(vis[z], htl),
                                   vis_br=_visible_fraction(vis[z], hbr), z=z, matrix=M))
    if augment:
        img = photometric_augment(rng, img, cfg)
    return img, anns


# --------------------------------------------------------------------------- labels / crops
def _hull_box(hull: np.ndarray, size: int) -> Optional[Tuple[float, float, float, float]]:
    """AABB of transformed hull points with a small margin (geaxgx ``extend``), clipped to the image."""
    x0, y0, x1, y1 = L.points_aabb([(float(x), float(y)) for x, y in hull])
    m = 1.0 + 0.03 * max(x1 - x0, y1 - y0)
    x0, y0, x1, y1 = max(0.0, x0 - m), max(0.0, y0 - m), min(float(size), x1 + m), min(float(size), y1 + m)
    if x1 - x0 < MIN_BOX_PX or y1 - y0 < MIN_BOX_PX:
        return None
    return x0, y0, x1, y1


def labels_from_annotations(anns: Sequence[CardAnnotation], size: int, cfg: SynthConfig) -> SceneLabels:
    """Apply the visibility rules and build the corner / card / OBB label records of one image."""
    out = SceneLabels()
    for a in anns:
        for which, hull, vis in (("tl", a.hull_tl, a.vis_tl), ("br", a.hull_br, a.vis_br)):
            if vis < cfg.min_corner_visibility:
                out.dropped_corners += 1
                continue
            xyxy = _hull_box(hull, size)
            box = L.Box.from_xyxy(a.cid, *xyxy, img_w=size, img_h=size) if xyxy else None
            if box is None:
                out.dropped_corners += 1
                continue
            out.corner.append(box)
            out.corner_which.append(which)
            out.corner_vis.append(float(vis))
        if a.vis_card >= cfg.min_card_visibility:
            x0, y0, x1, y1 = L.points_aabb([(float(x), float(y)) for x, y in a.corners])
            x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(size), x1), min(float(size), y1)
            if x1 - x0 >= MIN_BOX_PX and y1 - y0 >= MIN_BOX_PX:
                box = L.Box.from_xyxy(a.cid, x0, y0, x1, y1, img_w=size, img_h=size)
                if box is not None:
                    out.card.append(box)
                    # a.corners is already TL, TR, BR, BL (a homography keeps the cyclic order).  Re-ordering the
                    # points by their x+y / x-y extremes (Quad.from_pixels) emits one vertex twice for cards
                    # rotated near 45 deg (+k*90), which silently corrupted ~5 % of the OBB labels.
                    out.obb.append(L.Quad(a.cid, [(L.clip01(float(x) / size), L.clip01(float(y) / size)) for x, y in a.corners]))
                    continue
        out.dropped_cards += 1
    return out


def cut_crop(img: np.ndarray, box: L.Box, crop_size: int, pad: float = CROP_PAD) -> np.ndarray:
    """Square ``crop_size`` crop around a corner box padded by ``pad`` on each side (reflect padding at the border)."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box.to_xyxy(w, h)
    side = max(x1 - x0, y1 - y0) * (1.0 + 2 * pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    X0, Y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    X1, Y1 = int(round(cx + side / 2)), int(round(cy + side / 2))
    pad_l, pad_t = max(0, -X0), max(0, -Y0)
    pad_r, pad_b = max(0, X1 - w), max(0, Y1 - h)
    if pad_l or pad_t or pad_r or pad_b:
        img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101)
        X0, X1, Y0, Y1 = X0 + pad_l, X1 + pad_l, Y0 + pad_t, Y1 + pad_t
    crop = img[Y0:Y1, X0:X1]
    if crop.size == 0:
        crop = np.zeros((MIN_BOX_PX, MIN_BOX_PX, 3), np.uint8)
    return cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)


def parse_scenarios(spec: str) -> Dict[str, float]:
    """``"single:0.1,hand:0.4,spread:0.3,pile:0.2"`` -> weights dict (also accepts bare names = weight 1)."""
    weights: Dict[str, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, w = item.partition(":")
        name = name.strip()
        if name not in SCENARIOS:
            raise ValueError(f"unknown scenario {name!r}; expected one of {SCENARIOS}")
        weights[name] = float(w) if w.strip() else 1.0
    if not weights or sum(weights.values()) <= 0:
        raise ValueError("scenario weights must contain at least one positive weight")
    return weights


def choose_scenario(rng: np.random.Generator, weights: Dict[str, float]) -> str:
    names = [n for n in SCENARIOS if weights.get(n, 0) > 0]
    p = np.array([weights[n] for n in names], dtype=np.float64)
    return str(names[int(rng.choice(len(names), p=p / p.sum()))])


def val_indices(num_images: int, val_ratio: float, seed: int) -> frozenset:
    """Deterministic set of image indices that go to the ``val`` split."""
    n_val = int(round(num_images * float(val_ratio)))
    if num_images > 1:
        n_val = min(max(n_val, 1 if val_ratio > 0 else 0), num_images - 1)
    perm = np.random.default_rng(seed).permutation(num_images)
    return frozenset(int(i) for i in perm[:n_val])


# --------------------------------------------------------------------------- per-image work (runs in workers)
@dataclass
class _WorkerState:
    cfg: SynthConfig
    bank: CardBank
    bg_files: List[Path]
    val_ids: frozenset


_STATE: Optional[_WorkerState] = None


def _worker_init(cfg: SynthConfig, cards_dir: Optional[str], bg_files: List[str], val_ids: frozenset,
                 log_level: int = logging.WARNING, threads: int = 1) -> None:
    global _STATE
    logging.basicConfig(level=log_level, format="%(levelname)s %(name)s: %(message)s")
    try:
        cv2.setNumThreads(threads)
    except Exception:  # noqa: BLE001
        pass
    bank = load_card_bank(cards_dir, fetch=False)
    _STATE = _WorkerState(cfg=cfg, bank=bank, bg_files=[Path(p) for p in bg_files], val_ids=val_ids)


def render_and_write(idx: int, cfg: SynthConfig, bank: CardBank, bg_files: Sequence[Path], val_ids: frozenset) -> Dict:
    """Render image ``idx`` deterministically (``default_rng([seed, idx])``) and write all its files. Returns a record."""
    # The (seed, idx) pair is hashed by SeedSequence: still independent of the worker count, but unlike
    # ``seed + idx`` two runs with seeds S and S+1 no longer share N-1 byte-identical images (which leaked
    # train/val when such datasets were merged).
    rng = np.random.default_rng([int(cfg.seed), int(idx)])
    scenario = choose_scenario(rng, cfg.scenario_weights)
    S = int(cfg.img_size)
    bg = load_background(rng, bg_files, S)
    img, anns = render_scene(rng, scenario, bank, bg, cfg)
    labs = labels_from_annotations(anns, S, cfg)

    split = "val" if idx in val_ids else "train"
    stem = f"{STEM_PREFIX}{idx:06d}"
    root = Path(cfg.out_dir)
    img_dir, lbl_dir = L.split_dirs(root, split)
    img_path = img_dir / f"{stem}.jpg"
    q = int(rng.integers(int(cfg.jpeg_quality[0]), int(cfg.jpeg_quality[1]) + 1))
    cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, q])
    L.write_boxes(lbl_dir / f"{stem}.txt", labs.corner)
    if cfg.write_card_view:
        _link_view_image(root / "views" / "card", split, img_path, cfg.copy_views)
        L.write_boxes(root / "views" / "card" / "labels" / split / f"{stem}.txt", labs.card)
    if cfg.write_obb:
        _link_view_image(root / "views" / "obb", split, img_path, cfg.copy_views)
        L.write_quads(root / "views" / "obb" / "labels" / split / f"{stem}.txt", labs.obb)
    crops: List[Tuple[str, int, int, int]] = []
    if cfg.write_crops:
        for k, box in enumerate(labs.corner):
            name = C.CARD_CLASSES[box.cls]
            rel = Path("crops") / split / name / f"{stem}_{k}.jpg"
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(root / rel), cut_crop(img, box, int(cfg.crop_size)), [cv2.IMWRITE_JPEG_QUALITY, 92])
            crops.append((rel.as_posix(), box.cls, box.cls % 13, box.cls // 13))
    return {
        "idx": int(idx), "stem": stem, "split": split, "scenario": scenario, "n_cards": len(anns),
        "n_corner": len(labs.corner), "n_card": len(labs.card), "n_obb": len(labs.obb),
        "corner_cls": [int(b.cls) for b in labs.corner],
        "dropped_corners": labs.dropped_corners, "dropped_cards": labs.dropped_cards,
        "corner_vis": labs.corner_vis, "card_vis": [float(a.vis_card) for a in anns],
        "corner_areas": [float(b.area()) for b in labs.corner], "crops": crops,
    }


def _link_view_image(view: Path, split: str, img_path: Path, copy: bool) -> None:
    """``<view>/images/<split>/<name>`` -> the real image, one relative symlink (or copy) per file.

    A single directory symlink ``<view>/images -> ../../images`` does not work with ultralytics:
    ``check_det_dataset`` resolves the split directories, so the ``images`` -> ``labels`` path
    substitution then lands on the PRIMARY corner labels instead of the view's labels.
    """
    L.link_or_copy(img_path, view / "images" / split / img_path.name, copy=copy)


def _render_worker(idx: int) -> Dict:
    assert _STATE is not None, "worker not initialised"
    return render_and_write(idx, _STATE.cfg, _STATE.bank, _STATE.bg_files, _STATE.val_ids)


# --------------------------------------------------------------------------- dataset assembly
#: Sub-directories of ``out_dir`` that ``generate`` writes (and that a re-run must not silently mix with).
OUTPUT_SUBDIRS: Tuple[str, ...] = ("images", "labels", "views", "crops", "preview")


def _has_entries(d: Path) -> bool:
    """True when ``d`` is a file / symlink, or a directory holding any file or symlink (empty dirs do not count)."""
    if d.is_symlink() or d.is_file():
        return True
    if not d.is_dir():
        return False
    for dirpath, dirnames, filenames in os.walk(d):   # os.walk does not descend into symlinked dirs
        if filenames or any(os.path.islink(os.path.join(dirpath, n)) for n in dirnames):
            return True
    return False


def existing_outputs(root: Union[str, Path]) -> List[Path]:
    """Output sub-directories under ``root`` (see ``OUTPUT_SUBDIRS``) that still hold files of a previous run."""
    root = Path(root)
    return [root / n for n in OUTPUT_SUBDIRS if _has_entries(root / n)]


def _stats_from_records(records: Sequence[Dict], splits: Sequence[str] = ("train", "val")) -> Dict[str, Dict]:
    """Per-split counts in the ``labels.dataset_stats`` shape, computed from this run's records (never from
    whatever happens to be on disk)."""
    stats: Dict[str, Dict] = {s: {"images": 0, "labelled_images": 0, "boxes": 0, "per_class": {}} for s in splits}
    for r in records:
        s = stats[r["split"]]
        s["images"] += 1
        s["labelled_images"] += 1        # every image gets a label file (possibly empty = background sample)
        s["boxes"] += len(r["corner_cls"])
        for c in r["corner_cls"]:
            s["per_class"][c] = s["per_class"].get(c, 0) + 1
    for s in stats.values():
        s["per_class"] = dict(sorted(s["per_class"].items()))
    return stats


def _write_yaml_and_views(cfg: SynthConfig) -> Dict[str, str]:
    root = Path(cfg.out_dir)
    names = C.class_names_for_space("cards52")
    L.write_data_yaml(root / "data.yaml", names, "images/train", "images/val", root=root,
                      extra={"box_semantics": "corner", "class_space": "cards52"})
    views: Dict[str, str] = {}
    for name, enabled, sem in (("card", cfg.write_card_view, "card"), ("obb", cfg.write_obb, "obb")):
        if not enabled:
            continue
        view = root / "views" / name
        if (view / "images").is_symlink():      # legacy directory symlink, see _link_view_image
            (view / "images").unlink()
        for split in ("train", "val"):
            (view / "images" / split).mkdir(parents=True, exist_ok=True)
            (view / "labels" / split).mkdir(parents=True, exist_ok=True)
        L.write_data_yaml(view / "data.yaml", names, "images/train", "images/val", root=view,
                          extra={"box_semantics": sem, "class_space": "cards52"})
        views[name] = f"views/{name}"
    return views


def _write_crops_csv(root: Path, records: Sequence[Dict]) -> int:
    rows = [row for r in records for row in r["crops"]]
    p = root / "crops" / "labels.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "card_id", "rank_idx", "suit_idx"])
        w.writerows(rows)
    return len(rows)


def build_manifest(cfg: SynthConfig, records: Sequence[Dict], views: Dict[str, str], bank: CardBank,
                   n_backgrounds: int, elapsed: float, n_crops: int) -> Dict:
    root = Path(cfg.out_dir)
    names = C.class_names_for_space("cards52")
    stats = _stats_from_records(records)
    areas = np.array([a for r in records for a in r["corner_areas"]], dtype=np.float64)
    med = float(np.median(areas)) if len(areas) else 0.0
    detected = "corner" if (len(areas) and med < 0.02) else "card" if med > 0.05 else "unknown"
    scen: Dict[str, int] = {s: 0 for s in SCENARIOS}
    for r in records:
        scen[r["scenario"]] += 1
    corner_vis = np.array([v for r in records for v in r["corner_vis"]], dtype=np.float64)
    card_vis = np.array([v for r in records for v in r["card_vis"]], dtype=np.float64)
    n_hulls = 2 * sum(r["n_cards"] for r in records)
    n_cards = sum(r["n_cards"] for r in records)
    return {
        "key": root.name, "source": "generate_synthetic", "ref": "procedural" if bank.source == "procedural" else str(bank.cards_dir),
        "class_space": "cards52", "box_semantics": "corner", "box_semantics_detected": detected,
        "names": names,
        "class_mapping": {str(i): {"raw": n, "canonical": n, "id": i} for i, n in enumerate(names)},
        "unmapped": [], "stats": stats, "views": views, "created_by": "generate_synthetic",
        "config": cfg.to_dict(),
        "images": {"train": stats.get("train", {}).get("images", 0), "val": stats.get("val", {}).get("images", 0)},
        "scenario_counts": scen,
        "boxes_per_class": _boxes_per_class(stats, names),
        "visibility": {
            "cards_placed": n_cards, "corner_hulls": n_hulls, "corner_boxes_kept": int(len(corner_vis)),
            "corner_boxes_dropped": int(sum(r["dropped_corners"] for r in records)),
            "card_labels_dropped": int(sum(r["dropped_cards"] for r in records)),
            "mean_kept_corner_visibility": float(corner_vis.mean()) if len(corner_vis) else 0.0,
            "mean_card_visibility": float(card_vis.mean()) if len(card_vis) else 0.0,
            "median_corner_box_area": med,
        },
        "crops": n_crops, "card_source": bank.source, "hull_fallbacks": bank.hull_fallbacks,
        "background_files": n_backgrounds, "elapsed_s": round(elapsed, 2),
        "images_per_second": round(len(records) / elapsed, 2) if elapsed > 0 else None,
    }


def _boxes_per_class(stats: Dict, names: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {n: 0 for n in names}
    for s in stats.values():
        for k, v in s.get("per_class", {}).items():
            out[names[int(k)]] += int(v)
    return out


def draw_annotations(img: np.ndarray, corner: Sequence[L.Box], card: Sequence[L.Box] = (), obb: Sequence[L.Quad] = ()) -> np.ndarray:
    """Preview drawing: corner boxes green (+ class name), card boxes blue, OBB polygons yellow."""
    out = img.copy()
    h, w = out.shape[:2]
    for q in obb:
        cv2.polylines(out, [np.array(q.to_pixels(w, h), np.int32)], True, (0, 220, 255), 1)
    for b in card:
        x0, y0, x1, y1 = (int(round(v)) for v in b.to_xyxy(w, h))
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 120, 0), 1)
    for b in corner:
        x0, y0, x1, y1 = (int(round(v)) for v in b.to_xyxy(w, h))
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(out, C.CARD_CLASSES[b.cls], (x0, max(10, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def write_preview(root: Union[str, Path], n: int, seed: int = 0) -> List[Path]:
    """Draw ``n`` annotated images (from the files on disk) into ``<root>/preview``."""
    root = Path(root)
    pairs = [(img, split) for split in ("train", "val") for img, _ in L.iter_split(root, split)]
    if not pairs or n <= 0:
        return []
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(pairs), size=min(n, len(pairs)), replace=False).tolist())
    out_dir = root / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for i in pick:
        img_path, split = pairs[i]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        stem = img_path.stem
        corner = L.read_boxes(root / "labels" / split / f"{stem}.txt")
        card = L.read_boxes(root / "views" / "card" / "labels" / split / f"{stem}.txt")
        obb = L.read_quads(root / "views" / "obb" / "labels" / split / f"{stem}.txt")
        out = out_dir / f"{stem}_{split}.jpg"
        cv2.imwrite(str(out), draw_annotations(img, corner, card, obb), [cv2.IMWRITE_JPEG_QUALITY, 90])
        written.append(out)
    return written


def generate(cfg: SynthConfig, preview: int = 0, fetch: bool = True, progress: bool = True, force: bool = False) -> Dict:
    """Generate the whole dataset described by ``cfg``; returns the manifest dict (also written to disk).

    Images are rendered with ``num_images`` independent per-image seeds so the output is
    identical for any number of workers.  ``cfg.workers`` 0 means ``os.cpu_count()``,
    1 renders in-process (no pool).

    ``cfg.out_dir`` must not already hold outputs of a previous run (``OUTPUT_SUBDIRS``): a
    re-run with another ``num_images`` / ``val_ratio`` / ``seed`` would otherwise silently mix
    stale images, labels and crops into the dataset, its manifest and ``crops/labels.csv``.
    Raises ``FileExistsError`` in that case unless ``force`` is set, which removes them first.
    """
    t0 = time.time()
    root = Path(cfg.out_dir)
    stale = existing_outputs(root)
    if stale:
        if not force:
            raise FileExistsError(f"{root} already contains generated outputs ({', '.join(p.name for p in stale)}); "
                                  "pass --force / force=True to replace them or choose another --out")
        for p in stale:
            log.warning("force: removing outputs of a previous run under %s", p)
            if p.is_symlink() or p.is_file():
                p.unlink()
            else:
                shutil.rmtree(p)
    L.ensure_layout(root, ("train", "val"))
    views = _write_yaml_and_views(cfg)
    bank = load_card_bank(cfg.cards_dir, fetch=fetch)
    bg_files = list_background_files(cfg.backgrounds_dir)
    log.info("backgrounds: %d file(s) under %s%s", len(bg_files), cfg.backgrounds_dir, "" if bg_files else " -> procedural only")
    val_ids = val_indices(cfg.num_images, cfg.val_ratio, cfg.seed)
    n = int(cfg.num_images)
    workers = int(cfg.workers) if cfg.workers and cfg.workers > 0 else (os.cpu_count() or 1)
    workers = max(1, min(workers, n))
    from tqdm import tqdm
    bar = tqdm(total=n, desc="synthetic", unit="img", disable=None if progress else True)
    records: List[Dict] = []
    if workers == 1:
        for idx in range(n):
            records.append(render_and_write(idx, cfg, bank, bg_files, val_ids))
            bar.update(1)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        init_args = (cfg, str(cfg.cards_dir) if cfg.cards_dir is not None else None, [str(p) for p in bg_files],
                     val_ids, log.getEffectiveLevel())
        with ctx.Pool(processes=workers, initializer=_worker_init, initargs=init_args) as pool:
            for rec in pool.imap_unordered(_render_worker, range(n), chunksize=max(1, min(16, n // (workers * 4) or 1))):
                records.append(rec)
                bar.update(1)
    bar.close()
    records.sort(key=lambda r: r["idx"])
    n_crops = _write_crops_csv(root, records) if cfg.write_crops else 0
    manifest = build_manifest(cfg, records, views, bank, len(bg_files), time.time() - t0, n_crops)
    L.write_manifest(root, manifest)
    if preview > 0:
        paths = write_preview(root, preview, seed=cfg.seed)
        log.info("wrote %d preview image(s) to %s", len(paths), root / "preview")
    v = manifest["visibility"]
    log.info("done: %d images (%s) in %.1fs, %d corner boxes kept / %d dropped, %d crops, scenarios %s",
             n, "/".join(f"{k}={val}" for k, val in manifest["images"].items()), manifest["elapsed_s"],
             v["corner_boxes_kept"], v["corner_boxes_dropped"], n_crops, manifest["scenario_counts"])
    return manifest


# --------------------------------------------------------------------------- CLI
def build_config(args: argparse.Namespace) -> SynthConfig:
    cfg = SynthConfig()
    if args.out:
        cfg.out_dir = Path(args.out)
    if args.cards is not None:
        cfg.cards_dir = Path(args.cards)
    if args.backgrounds is not None:
        cfg.backgrounds_dir = Path(args.backgrounds) if args.backgrounds else None
    cfg.num_images = int(args.num)
    cfg.img_size = int(args.img_size)
    cfg.seed = int(args.seed)
    cfg.workers = int(args.workers)
    cfg.val_ratio = float(args.val_ratio)
    if args.scenarios:
        cfg.scenario_weights = parse_scenarios(args.scenarios)
    cfg.write_obb = not args.no_obb
    cfg.write_card_view = not args.no_card_view
    cfg.write_crops = not args.no_crops
    cfg.copy_views = bool(args.copy_views)
    if args.crop_size:
        cfg.crop_size = int(args.crop_size)
    return cfg


def build_parser() -> argparse.ArgumentParser:
    d = SynthConfig()
    p = argparse.ArgumentParser(prog="generate_synthetic", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out", default=str(d.out_dir), help="dataset root to write (canonical layout)")
    p.add_argument("--num", type=int, default=d.num_images, help="number of images")
    p.add_argument("--img-size", type=int, default=d.img_size, help="square output size in pixels")
    p.add_argument("--cards", default=None, help=f"card art dir (<CANONICAL>.png or <rank>_of_<suit>.png); default {d.cards_dir}")
    p.add_argument("--backgrounds", default=None, help=f"background image dir ('' = procedural only); default {d.backgrounds_dir}")
    p.add_argument("--scenarios", default=None, help="scenario weights, e.g. single:0.1,hand:0.4,spread:0.3,pile:0.2")
    p.add_argument("--workers", type=int, default=d.workers, help="worker processes (0 = all CPUs, 1 = in-process)")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--val-ratio", type=float, default=d.val_ratio)
    p.add_argument("--no-obb", action="store_true", help="skip views/obb")
    p.add_argument("--no-card-view", action="store_true", help="skip views/card")
    p.add_argument("--no-crops", action="store_true", help="skip classifier crops")
    p.add_argument("--copy-views", action="store_true", help="copy each image into views/<name>/images instead of symlinking it")
    p.add_argument("--force", action="store_true", help="replace outputs of a previous run that are already under --out")
    p.add_argument("--crop-size", type=int, default=None, help=f"classifier crop size (default {d.crop_size})")
    p.add_argument("--preview", type=int, default=0, metavar="N", help="write N annotated images to <out>/preview")
    p.add_argument("--no-fetch", action="store_true", help="never download card art (procedural fallback only)")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = build_config(args)
    log.info("generating %d images at %dpx -> %s (seed %d, workers %s)", cfg.num_images, cfg.img_size, cfg.out_dir, cfg.seed,
             cfg.workers or "auto")
    try:
        generate(cfg, preview=int(args.preview), fetch=not args.no_fetch, force=bool(args.force))
    except FileExistsError as e:
        log.error("%s", e)
        return 2
    except KeyboardInterrupt:
        log.error("interrupted")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
