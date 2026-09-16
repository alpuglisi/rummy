"""Offline tests for vision/generate_synthetic.py (procedural cards + procedural backgrounds only)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision import cards as C
from vision import generate_synthetic as G
from vision import labels as L
from vision.config import SynthConfig


@pytest.fixture(scope="module")
def bank() -> G.CardBank:
    return G.load_card_bank(None, fetch=False)


def small_cfg(tmp_path: Path, **kw) -> SynthConfig:
    base = dict(out_dir=tmp_path / "ds", cards_dir=tmp_path / "cards_missing", backgrounds_dir=None, num_images=12,
                img_size=256, seed=7, workers=1, val_ratio=0.25,
                scenario_weights={"single": 0.25, "hand": 0.25, "spread": 0.25, "pile": 0.25})
    base.update(kw)
    return SynthConfig(**base)


# --------------------------------------------------------------------------- card art / hulls
def test_procedural_card_and_hull_inside_corner_zone():
    for name in ("10H", "AS", "QC", "7D"):
        rgba = G.procedural_card(name)
        assert rgba.shape == (G.CARD_H, G.CARD_W, 4) and rgba.dtype == np.uint8
        assert rgba[G.CARD_H // 2, G.CARD_W // 2, 3] == 255      # opaque body
        assert rgba[0, 0, 3] == 0                                  # rounded corner is transparent
        hull = G.find_corner_hull(rgba)
        assert hull is not None and hull.ndim == 2 and hull.shape[1] == 2 and len(hull) >= 3
        x0, y0, x1, y1 = G.corner_zone()
        assert hull[:, 0].min() >= x0 and hull[:, 1].min() >= y0
        assert hull[:, 0].max() <= x1 and hull[:, 1].max() <= y1
        area = cv2.contourArea(hull.astype(np.float32))
        zone_area = (x1 - x0) * (y1 - y0)
        assert 0.04 * zone_area <= area <= 0.9 * zone_area
        # the hull must contain the printed index: every ink pixel of the index column (x < 42; the
        # face-card frame at x >= 44 is content that continues outside the zone) lies inside it
        ink = G._ink_mask(rgba[y0:y1, x0:x1])
        ink[:, 42 - x0:] = 0
        ys, xs = np.nonzero(ink)
        assert len(xs) > 100
        inside = [cv2.pointPolygonTest(hull.astype(np.float32), (float(x + x0), float(y + y0)), False) >= 0
                  for x, y in zip(xs[::5], ys[::5])]
        assert np.mean(inside) > 0.99


def test_bottom_right_hull_is_180_rotation(bank: G.CardBank):
    a = bank.by_name("KD")
    assert a.hull_ok and a.cid == C.CLASS_TO_ID["KD"]
    expected = np.stack([G.CARD_W - a.hull_tl[:, 0], G.CARD_H - a.hull_tl[:, 1]], axis=1)
    assert np.allclose(a.hull_br, expected)
    assert np.allclose(a.corners, G.REF_CORNERS)


def test_bank_has_52_distinct_cards_with_engine_ids(bank: G.CardBank):
    assert len(bank) == 52 and bank.source == "procedural" and bank.hull_fallbacks == 0
    assert [c.cid for c in bank] == list(range(52))
    assert [c.name for c in bank] == list(C.CARD_CLASSES)
    rng = np.random.default_rng(0)
    picked = bank.sample(rng, 13)
    assert len({c.name for c in picked}) == 13


def test_ensure_card_body_repairs_ink_only_art():
    # hayeah-style overlay: transparent everywhere except a black glyph
    art = np.zeros((G.CARD_H, G.CARD_W, 4), np.uint8)
    art[20:60, 10:30] = (0, 0, 0, 255)
    fixed = G.ensure_card_body(art)
    assert fixed[G.CARD_H // 2, G.CARD_W // 2].tolist() == [255, 255, 255, 255]
    assert fixed[0, 0, 3] == 0 and fixed[40, 20].tolist() == [0, 0, 0, 255]
    # art with a sensible alpha is left alone
    good = G.procedural_card("2C")
    assert np.array_equal(G.ensure_card_body(good), good)


def test_load_card_bank_from_files_and_hull_cache(tmp_path: Path):
    d = tmp_path / "cards"
    d.mkdir()
    cv2.imwrite(str(d / "AC.png"), G.procedural_card("AC"))                          # canonical name
    cv2.imwrite(str(d / C.card_asset_filename("10H")), G.procedural_card("10H"))     # <rank>_of_<suit>.png
    b = G.load_card_bank(d, fetch=False)
    assert b.source == "mixed" and len(b) == 52
    assert b.by_name("AC").source == "file" and b.by_name("10H").source == "file" and b.by_name("2C").source == "procedural"
    cache = d / G.HULL_CACHE_NAME
    assert cache.is_file()
    data = json.loads(cache.read_text())
    assert set(data["cards"]) == {"AC", "10H"}
    b2 = G.load_card_bank(d, fetch=False)
    assert np.allclose(b2.by_name("AC").hull_tl, b.by_name("AC").hull_tl)
    assert G.missing_card_art(d) == [n for n in C.CARD_CLASSES if n not in ("AC", "10H")]


def test_fetch_fallback_is_best_effort(tmp_path: Path, monkeypatch):
    import urllib.request

    def boom(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setattr(G.time, "sleep", lambda s: None)
    assert G.fetch_card_art_fallback(tmp_path, names=["AC", "KS"], retries=2) == 0
    assert not list(tmp_path.glob("*.png"))


# --------------------------------------------------------------------------- backgrounds / augmentation
def test_procedural_background_shapes_and_variety():
    seen = set()
    for seed in range(12):
        rng = np.random.default_rng(seed)
        bg = G.procedural_background(rng, 64)
        assert bg.shape == (64, 64, 3) and bg.dtype == np.uint8
        assert bg.std() > 0.5  # never a flat constant
        seen.add(bg.tobytes())
    assert len(seen) == 12
    assert G.procedural_background(np.random.default_rng(1), (48, 80)).shape == (48, 80, 3)
    assert G.load_background(np.random.default_rng(2), [], 32).shape == (32, 32, 3)


def test_photometric_augment_keeps_shape_and_range():
    cfg = SynthConfig(glare_prob=1.0, blur_prob=1.0, noise_prob=1.0)
    img = G.procedural_background(np.random.default_rng(3), 64)
    for seed in range(4):
        out = G.photometric_augment(np.random.default_rng(seed), img, cfg)
        assert out.shape == img.shape and out.dtype == np.uint8


# --------------------------------------------------------------------------- scene rendering
@pytest.mark.parametrize("scenario", G.SCENARIOS)
def test_render_scene_every_scenario(bank: G.CardBank, scenario: str):
    cfg = SynthConfig(img_size=256, finger_prob=1.0)
    rng = np.random.default_rng(11)
    bg = G.procedural_background(rng, 200)  # deliberately not img_size: must be resized
    img, anns = G.render_scene(rng, scenario, bank, bg, cfg)
    assert img.shape == (256, 256, 3) and img.dtype == np.uint8
    assert len(anns) >= 1 and (scenario != "single" or len(anns) == 1)
    for a in anns:
        assert 0 <= a.cid < 52 and a.name == C.CARD_CLASSES[a.cid]
        assert a.corners.shape == (4, 2) and a.hull_tl.shape[1] == 2 and a.hull_br.shape[1] == 2
        assert 0.0 <= a.vis_card <= 1.0 and 0.0 <= a.vis_tl <= 1.0 and 0.0 <= a.vis_br <= 1.0
        assert a.matrix is not None and a.matrix.shape == (3, 3)
    top = anns[-1]
    if scenario in ("single", "pile"):
        assert top.vis_card > 0.9  # top card fully visible (unless it leaves the image)


def test_hand_keeps_every_top_left_index_visible(bank: G.CardBank):
    """The fan layout never lets a card cover the previous card's top-left index (indices that
    leave the image frame are legitimately dropped, so only in-frame hulls are checked)."""
    cfg = SynthConfig(img_size=320, finger_prob=0.0)
    total = ok = in_frame = 0
    for seed in range(8):
        rng = np.random.default_rng(100 + seed)
        img, anns = G.render_scene(rng, "hand", bank, G.procedural_background(rng, 320), cfg, augment=False)
        assert img.flags["C_CONTIGUOUS"]
        for a in anns:
            total += 1
            inside = (a.hull_tl.min() >= 0) and (a.hull_tl.max() <= 320)
            if inside:
                in_frame += 1
                ok += a.vis_tl >= cfg.min_corner_visibility
    assert total >= 8 and in_frame / total >= 0.8
    assert ok == in_frame


def test_spread_leaves_every_card_partially_visible(bank: G.CardBank):
    cfg = SynthConfig(img_size=320, spread_cards=(6, 6))
    rng = np.random.default_rng(5)
    _, anns = G.render_scene(rng, "spread", bank, G.procedural_background(rng, 320), cfg, augment=False)
    assert len(anns) == 6
    assert all(a.vis_card >= cfg.min_card_visibility for a in anns)
    assert sum(a.vis_tl >= cfg.min_corner_visibility for a in anns) >= 5


def test_visibility_filtering_drops_fully_covered_corners(bank: G.CardBank):
    cfg = SynthConfig(img_size=256)
    rng = np.random.default_rng(0)
    bg = G.procedural_background(rng, 256)
    a, b = bank.by_name("AS"), bank.by_name("KH")
    M = G.mat_translate(60, 20) @ G.mat_scale(0.6)
    layers = [G.warp_card_layer(a, M, 256), G.warp_card_layer(b, M, 256)]  # identical placement, b on top
    img, vis = G.composite(rng, bg, layers, cfg, shadow=False)
    assert img.shape == (256, 256, 3)
    assert vis[0].sum() == 0 and vis[1].sum() > 1000
    anns = []
    for z, asset in enumerate((a, b)):
        anns.append(G.CardAnnotation(asset.name, asset.cid, G.apply_homography(M, asset.corners),
                                     G.apply_homography(M, asset.hull_tl), G.apply_homography(M, asset.hull_br),
                                     G._visible_fraction(vis[z], G.apply_homography(M, asset.corners)),
                                     G._visible_fraction(vis[z], G.apply_homography(M, asset.hull_tl)),
                                     G._visible_fraction(vis[z], G.apply_homography(M, asset.hull_br)), z))
    assert anns[0].vis_tl == 0.0 and anns[0].vis_card == 0.0 and anns[1].vis_tl > 0.95 and anns[1].vis_card > 0.95
    labs = G.labels_from_annotations(anns, 256, cfg)
    assert [bx.cls for bx in labs.corner] == [b.cid, b.cid] and labs.corner_which == ["tl", "br"]
    assert labs.dropped_corners == 2 and [bx.cls for bx in labs.card] == [b.cid] and labs.dropped_cards == 1
    assert len(labs.obb) == 1 and len(labs.obb[0].pts) == 4
    # partial cover: shift the top card so that it covers only the bottom-right index of the lower one
    M2 = G.mat_translate(100, 90) @ G.mat_scale(0.6)
    layers = [G.warp_card_layer(a, M, 256), G.warp_card_layer(b, M2, 256)]
    _, vis = G.composite(rng, bg, layers, cfg, shadow=False)
    tl = G._visible_fraction(vis[0], G.apply_homography(M, a.hull_tl))
    br = G._visible_fraction(vis[0], G.apply_homography(M, a.hull_br))
    assert tl > 0.95 and br < 0.05


def test_obb_quad_keeps_four_distinct_corners_in_card_order(bank: G.CardBank):
    """Regression: ordering the transformed corners by their x+y / x-y extremes (Quad.from_pixels) emitted one
    vertex twice for cards rotated near 45 deg (+k*90).  The OBB must be the card's own TL, TR, BR, BL."""
    cfg = SynthConfig(img_size=256)
    a = bank.by_name("AS")
    for deg in (45.0, 135.0, 225.0, 315.0, 10.0, -80.0):
        M = G.mat_translate(128, 128) @ G.mat_rotate(deg) @ G.mat_scale(0.5) @ G.mat_translate(-G.CARD_W / 2, -G.CARD_H / 2)
        corners = G.apply_homography(M, a.corners)
        ann = G.CardAnnotation(a.name, a.cid, corners, G.apply_homography(M, a.hull_tl), G.apply_homography(M, a.hull_br),
                               1.0, 1.0, 1.0, 0)
        labs = G.labels_from_annotations([ann], 256, cfg)
        assert len(labs.obb) == 1 and len(labs.card) == 1
        q = labs.obb[0]
        assert q.cls == a.cid and len(q.pts) == 4
        assert len({(round(x, 6), round(y, 6)) for x, y in q.pts}) == 4, (deg, q.pts)
        assert np.allclose(np.array(q.pts) * 256, corners, atol=1e-3)
        assert abs(L.polygon_area(q.pts) * 256 * 256 - 0.25 * G.CARD_W * G.CARD_H) < 1.0


def test_homography_helpers_match_opencv():
    M = G.mat_translate(5, -3) @ G.mat_rotate(30, 10, 10) @ G.mat_scale(1.5)
    pts = np.array([[0, 0], [10, 0], [10, 20], [0, 20]], np.float32)
    ours = G.apply_homography(M, pts)
    ref = cv2.perspectiveTransform(pts.reshape(1, -1, 2).astype(np.float64), M).reshape(-1, 2)
    assert np.allclose(ours, ref, atol=1e-4)
    # positive rotation is clockwise on screen (y down): +x axis tips towards +y
    p = G.apply_homography(G.mat_rotate(10), np.array([[1.0, 0.0]]))
    assert p[0, 1] > 0


# --------------------------------------------------------------------------- dataset generation
def test_generate_small_dataset_layout_and_labels(tmp_path: Path):
    cfg = small_cfg(tmp_path)
    manifest = G.generate(cfg, preview=2, fetch=False, progress=False)
    root = cfg.out_dir
    # layout
    assert (root / "data.yaml").is_file() and (root / "manifest.json").is_file()
    assert (root / "views" / "card" / "data.yaml").is_file() and (root / "views" / "obb" / "data.yaml").is_file()
    assert (root / "views" / "card" / "images" / "train").is_dir()
    imgs = {s: L.list_images(root / "images" / s) for s in ("train", "val")}
    assert len(imgs["train"]) + len(imgs["val"]) == 12 and len(imgs["val"]) == 3
    d = L.read_data_yaml(root / "data.yaml")
    assert d["nc"] == 52 and L.yaml_names_list(d) == list(C.CARD_CLASSES) and d["train"] == "images/train"
    dv = L.read_data_yaml(root / "views" / "obb" / "data.yaml")
    assert dv["path"] == str((root / "views" / "obb").resolve()) and L.yaml_names_list(dv) == list(C.CARD_CLASSES)
    # labels
    n_corner = n_card = n_obb = 0
    for split, paths in imgs.items():
        for img in paths:
            arr = cv2.imread(str(img))
            assert arr is not None and arr.shape == (256, 256, 3)
            for b in L.read_boxes(root / "labels" / split / f"{img.stem}.txt"):
                n_corner += 1
                assert 0 <= b.cls < 52
                x0, y0, x1, y1 = b.to_xyxy(1, 1)
                assert -1e-6 <= x0 < x1 <= 1 + 1e-6 and -1e-6 <= y0 < y1 <= 1 + 1e-6
                assert b.w * 256 >= G.MIN_BOX_PX - 1e-3 and b.h * 256 >= G.MIN_BOX_PX - 1e-3
                assert b.area() < 0.05  # corner boxes are small
            for b in L.read_boxes(root / "views" / "card" / "labels" / split / f"{img.stem}.txt"):
                n_card += 1
                assert 0 <= b.cls < 52 and 0 <= b.cx <= 1 and 0 <= b.cy <= 1 and 0 < b.w <= 1 and 0 < b.h <= 1
            for q in L.read_quads(root / "views" / "obb" / "labels" / split / f"{img.stem}.txt"):
                n_obb += 1
                assert 0 <= q.cls < 52 and len(q.pts) == 4
                assert all(0 <= x <= 1 and 0 <= y <= 1 for x, y in q.pts)
    assert n_corner > 0 and n_card > 0 and n_obb == n_card
    # crops
    with open(root / "crops" / "labels.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == n_corner and rows[0].keys() == {"path", "card_id", "rank_idx", "suit_idx"}
    for r in rows[:20]:
        p = root / r["path"]
        assert p.is_file() and p.parent.name == C.CARD_CLASSES[int(r["card_id"])]
        assert int(r["card_id"]) == int(r["suit_idx"]) * 13 + int(r["rank_idx"])
        crop = cv2.imread(str(p))
        assert crop.shape == (cfg.crop_size, cfg.crop_size, 3)
    # manifest
    m = L.read_manifest(root)
    assert m == json.loads(json.dumps(manifest))
    assert m["created_by"] == "generate_synthetic" and m["box_semantics"] == "corner" and m["class_space"] == "cards52"
    assert m["box_semantics_detected"] == "corner" and m["names"] == list(C.CARD_CLASSES)
    assert m["views"] == {"card": "views/card", "obb": "views/obb"}
    assert sum(m["scenario_counts"].values()) == 12 and set(m["scenario_counts"]) == set(G.SCENARIOS)
    assert m["stats"]["train"]["boxes"] + m["stats"]["val"]["boxes"] == n_corner
    assert m["visibility"]["corner_boxes_kept"] == n_corner and m["crops"] == n_corner
    assert m["config"]["img_size"] == 256 and m["card_source"] == "procedural"
    assert sum(m["boxes_per_class"].values()) == n_corner
    # preview
    previews = sorted((root / "preview").glob("*.jpg"))
    assert len(previews) == 2 and cv2.imread(str(previews[0])).shape == (256, 256, 3)


def test_generate_respects_disabled_outputs(tmp_path: Path):
    cfg = small_cfg(tmp_path, num_images=3, img_size=128, write_obb=False, write_card_view=False, write_crops=False)
    m = G.generate(cfg, fetch=False, progress=False)
    root = cfg.out_dir
    assert not (root / "views").exists() and not (root / "crops").exists()
    assert m["views"] == {} and m["crops"] == 0
    assert len(L.list_images(root / "images" / "train")) + len(L.list_images(root / "images" / "val")) == 3


@pytest.mark.parametrize("copy_views", [False, True])
def test_view_images_are_per_file_links_that_resolve_inside_the_view(tmp_path: Path, copy_views: bool):
    """Regression: a directory symlink views/<name>/images -> ../../images made ultralytics (which resolves the
    split dirs before substituting images -> labels in each path) train the view on the PRIMARY corner labels."""
    from ultralytics.data.utils import check_det_dataset

    cfg = small_cfg(tmp_path, num_images=4, img_size=96, write_crops=False, copy_views=copy_views)
    G.generate(cfg, fetch=False, progress=False)
    root = cfg.out_dir
    for name in ("card", "obb"):
        view = root / "views" / name
        assert (view / "images").is_dir() and not (view / "images").is_symlink()
        for split in ("train", "val"):
            img_dir = view / "images" / split
            assert img_dir.is_dir() and not img_dir.is_symlink()
            links = L.list_images(img_dir)
            originals = L.list_images(root / "images" / split)
            assert links and [p.name for p in links] == [p.name for p in originals]
            for p, o in zip(links, originals):
                assert p.is_symlink() == (not copy_views)
                assert p.resolve() == o.resolve() if not copy_views else p.read_bytes() == o.read_bytes()
                lbl = L.label_path_for(p)     # ultralytics-style images -> labels substitution
                assert lbl.parent == view / "labels" / split and lbl.is_file()
        d = check_det_dataset(str(view / "data.yaml"))
        for split in ("train", "val"):
            assert Path(d[split]).resolve() == (view / "images" / split).resolve()
            assert str(Path(d[split]).resolve()).startswith(str(view.resolve()) + "/")


def test_generate_refuses_stale_outputs_unless_forced(tmp_path: Path):
    """Regression: re-running into the same out_dir silently mixed a previous run's images, labels and crops into
    the dataset, its manifest and crops/labels.csv (the CLI default --out is fixed, so this was easy to hit)."""
    cfg1 = small_cfg(tmp_path, num_images=6, img_size=96, val_ratio=0.34)
    G.generate(cfg1, preview=1, fetch=False, progress=False)
    root = cfg1.out_dir
    assert L.read_manifest(root)["images"] == {"train": 4, "val": 2}
    cfg2 = small_cfg(tmp_path, num_images=3, img_size=96, val_ratio=0.34, seed=9)
    with pytest.raises(FileExistsError, match="force"):
        G.generate(cfg2, fetch=False, progress=False)
    assert L.read_manifest(root)["images"] == {"train": 4, "val": 2}     # the refused run touched nothing
    m = G.generate(cfg2, fetch=False, progress=False, force=True)
    assert m["images"] == {"train": 2, "val": 1}
    imgs = {s: [p.name for p in L.list_images(root / "images" / s)] for s in ("train", "val")}
    assert len(imgs["train"]) == 2 and len(imgs["val"]) == 1 and not set(imgs["train"]) & set(imgs["val"])
    assert m["stats"] == L.dataset_stats(root, splits=("train", "val"))
    assert sum(m["boxes_per_class"].values()) == m["stats"]["train"]["boxes"] + m["stats"]["val"]["boxes"]
    with open(root / "crops" / "labels.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == m["crops"] == len(list((root / "crops").rglob("*.jpg")))
    assert not (root / "preview").exists()
    for name in ("card", "obb"):
        for s in ("train", "val"):
            assert [p.name for p in L.list_images(root / "views" / name / "images" / s)] == imgs[s]
            assert sorted(p.stem for p in (root / "views" / name / "labels" / s).glob("*.txt")) == sorted(Path(n).stem for n in imgs[s])
    # an existing but empty canonical layout is not "stale"
    empty = L.ensure_layout(tmp_path / "empty")
    G.generate(small_cfg(tmp_path, out_dir=empty, num_images=2, img_size=96), fetch=False, progress=False)
    assert G.existing_outputs(tmp_path / "nowhere") == [] and G.existing_outputs(empty) != []


def _snapshot(root: Path) -> dict:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in (".txt", ".jpg", ".csv") and "preview" not in p.parts:
            out[str(p.relative_to(root))] = p.read_bytes()
    return out


def test_generation_is_deterministic_across_seeds_and_workers(tmp_path: Path):
    cfg1 = small_cfg(tmp_path, out_dir=tmp_path / "a", num_images=8, img_size=160, seed=3, workers=1)
    cfg2 = small_cfg(tmp_path, out_dir=tmp_path / "b", num_images=8, img_size=160, seed=3, workers=2)
    cfg3 = small_cfg(tmp_path, out_dir=tmp_path / "c", num_images=8, img_size=160, seed=4, workers=1)
    G.generate(cfg1, fetch=False, progress=False)
    G.generate(cfg2, fetch=False, progress=False)
    G.generate(cfg3, fetch=False, progress=False)
    s1, s2, s3 = _snapshot(tmp_path / "a"), _snapshot(tmp_path / "b"), _snapshot(tmp_path / "c")
    assert s1 and s1 == s2
    labels1 = {k: v for k, v in s1.items() if k.endswith(".txt")}
    labels3 = {k: v for k, v in s3.items() if k.endswith(".txt")}
    assert labels1 != labels3
    # regression: with per-image seed ``seed + idx`` seeds 3 and 4 shared 7 of these 8 images byte for byte
    # (idx i of one run == idx i-1 of the other), which leaked train/val when such datasets were merged
    images1 = {v for k, v in s1.items() if k.endswith(".jpg") and k.startswith("images/")}
    images3 = {v for k, v in s3.items() if k.endswith(".jpg") and k.startswith("images/")}
    assert len(images1) == 8 and not images1 & images3


def test_val_split_and_scenario_helpers():
    ids = G.val_indices(100, 0.1, 0)
    assert len(ids) == 10 and ids == G.val_indices(100, 0.1, 0) and ids != G.val_indices(100, 0.1, 1)
    assert G.val_indices(1, 0.5, 0) == frozenset() and len(G.val_indices(5, 0.0, 0)) == 0 and len(G.val_indices(5, 0.01, 0)) == 1
    w = G.parse_scenarios("single:0.1,hand:0.4,spread:0.3,pile:0.2")
    assert w == {"single": 0.1, "hand": 0.4, "spread": 0.3, "pile": 0.2}
    assert G.parse_scenarios("hand,pile:2") == {"hand": 1.0, "pile": 2.0}
    with pytest.raises(ValueError):
        G.parse_scenarios("fan:1")
    rng = np.random.default_rng(0)
    picks = {G.choose_scenario(rng, {"hand": 1.0, "single": 0.0}) for _ in range(20)}
    assert picks == {"hand"}


# --------------------------------------------------------------------------- CLI
def test_cli_help_and_run(tmp_path: Path, capsys):
    with pytest.raises(SystemExit) as e:
        G.main(["--help"])
    assert e.value.code == 0
    assert "--scenarios" in capsys.readouterr().out
    out = tmp_path / "cli"
    rc = G.main(["--out", str(out), "--num", "4", "--img-size", "128", "--cards", str(tmp_path / "nocards"),
                 "--backgrounds", "", "--workers", "1", "--seed", "1", "--scenarios", "hand:1,spread:1",
                 "--no-fetch", "--preview", "1", "--no-crops"])
    assert rc == 0
    m = L.read_manifest(out)
    assert m["scenario_counts"]["single"] == 0 and m["scenario_counts"]["pile"] == 0
    assert m["config"]["write_crops"] is False and not (out / "crops").exists()
    assert len(list((out / "preview").glob("*.jpg"))) == 1
    again = ["--out", str(out), "--num", "2", "--img-size", "128", "--cards", str(tmp_path / "nocards"), "--backgrounds", "",
             "--workers", "1", "--no-fetch", "--no-crops"]
    assert G.main(again) == 2 and L.read_manifest(out) == m          # refused, nothing touched
    assert G.main(again + ["--force"]) == 0
    assert L.read_manifest(out)["images"] == {"train": 1, "val": 1} and not (out / "preview").exists()
