"""Cut assets/sprite/cat.png (an emotion-labelled sprite sheet on a baked-in
checkerboard) into transparent per-frame PNGs plus a manifest for play.py.

    python tools/build_sprites.py

Writes assets/sprite/frames/<emotion>_<n>.png and assets/sprite/frames/manifest.json.
The checkerboard is keyed out by colour (near-grey pixels of the two checker
shades); frames are the connected blobs left in each labelled band, merged
where they overlap horizontally (hearts and sparkles float above a frame).
"""
import json
import os
import sys

import numpy as np
from PIL import Image

SHEET = os.path.join("assets", "sprite", "cat.png")
OUT = os.path.join("assets", "sprite", "frames")

# Bands of the sheet (y ranges) and the emotions laid out left to right in
# each, with the x split between groups sharing a band. Text labels sit in
# the gaps above the frames and are dropped by height.
BANDS = [
    ((0, 145), [("joy", 0, 512), ("surprise", 512, 1024)]),
    ((145, 282), [("excitement", 0, 512), ("confusion", 512, 1024)]),
    ((282, 425), [("happiness", 0, 218), ("satisfaction", 218, 512), ("frustration", 512, 1024)]),
    ((425, 572), [("contemplation", 0, 285), ("awe", 285, 512), ("fear", 512, 1024)]),
]
MIN_FRAME_HEIGHT = 70   # full cat frames; shorter blobs are decorations (kept if coloured) or text (dropped)
MAX_FRAME_WIDTH = 100   # wider blobs are two cats whose arms touch; split at the thinnest column


def key_background(rgb):
    r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
    grey = (np.abs(r - g) < 12) & (np.abs(g - b) < 12) & (np.abs(r - b) < 12)
    lum = (r + g + b) / 3
    return grey & (lum > 160) & (lum < 234)


def components(mask):
    """Bounding boxes of 8-connected foreground blobs (simple flood fill)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    boxes = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        x_min = x_max = x0
        y_min = y_max = y0
        n = 0
        while stack:
            y, x = stack.pop()
            n += 1
            x_min, x_max, y_min, y_max = min(x_min, x), max(x_max, x), min(y_min, y), max(y_max, y)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
        boxes.append((x_min, y_min, x_max + 1, y_max + 1, n))
    return boxes


def split_wide(box, fg):
    """Split a blob holding two touching cats at its thinnest interior column."""
    x0, y0, x1, y1 = box
    if x1 - x0 <= MAX_FRAME_WIDTH:
        return [box]
    counts = fg[y0:y1, x0:x1].sum(axis=0)
    lo, hi = int(0.3 * (x1 - x0)), int(0.7 * (x1 - x0))
    cut = x0 + lo + int(np.argmin(counts[lo:hi]))
    return split_wide((x0, y0, cut, y1), fg) + split_wide((cut, y0, x1, y1), fg)


def is_coloured(rgb):
    """True for hearts and sparkles (saturated pixels), false for black label text."""
    sat = rgb.max(axis=-1).astype(int) - rgb.min(axis=-1).astype(int)
    return (sat > 60).mean() > 0.25


def merge_overlapping(boxes, fg, sheet):
    """Attach coloured decorations (hearts, sparkles) to the frame they sit
    over; drop label text; split blobs that hold two touching cats."""
    big = []
    for b in sorted(b for b in boxes if b[3] - b[1] >= MIN_FRAME_HEIGHT):
        big.extend(split_wide(b[:4], fg))
    small = [b for b in boxes if b[3] - b[1] < MIN_FRAME_HEIGHT
             and is_coloured(sheet[b[1]:b[3], b[0]:b[2]])]
    merged = [list(b) for b in big]
    for b in small:
        cx = (b[0] + b[2]) / 2
        best = min(merged, key=lambda m: abs((m[0] + m[2]) / 2 - cx), default=None)
        if best is not None and b[0] < best[2] + 10 and b[2] > best[0] - 10:
            best[0], best[1], best[2], best[3] = min(best[0], b[0]), min(best[1], b[1]), max(best[2], b[2]), max(best[3], b[3])
    return [tuple(m) for m in merged]


def main():
    sys.setrecursionlimit(10000)
    sheet = np.array(Image.open(SHEET).convert("RGB"))
    fg = ~key_background(sheet)
    rgba = np.dstack([sheet, np.where(fg, 255, 0).astype(np.uint8)])
    os.makedirs(OUT, exist_ok=True)
    manifest = {}
    for (y0, y1), groups in BANDS:
        band = fg[y0:y1]
        boxes = [(x0, y0 + by0, x1, y0 + by1) for x0, by0, x1, by1, n in components(band) if n > 40]
        for name, gx0, gx1 in groups:
            frames = merge_overlapping([b for b in boxes if gx0 <= (b[0] + b[2]) // 2 < gx1], fg, sheet)
            manifest[name] = []
            for i, (x0, fy0, x1, fy1) in enumerate(frames):
                crop = rgba[fy0:fy1, x0:x1]
                path = os.path.join(OUT, f"{name}_{i}.png")
                Image.fromarray(crop, "RGBA").save(path)
                manifest[name].append({"file": os.path.basename(path), "w": int(x1 - x0), "h": int(fy1 - fy0)})
            print(f"{name:14s} {len(frames)} frames")
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main()
