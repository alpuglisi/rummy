import logging
import math
import os
import random
import shutil
from pathlib import Path

from vision import labels


def test_box_roundtrip(tmp_path: Path):
    b = labels.Box.from_xyxy(3, 10, 20, 110, 220, 640, 480)
    assert b is not None
    x1, y1, x2, y2 = b.to_xyxy(640, 480)
    assert abs(x1 - 10) < 1e-6 and abs(y2 - 220) < 1e-6
    labels.write_boxes(tmp_path / "l.txt", [b])
    rb = labels.read_boxes(tmp_path / "l.txt")
    assert len(rb) == 1 and rb[0].cls == 3 and abs(rb[0].cx - b.cx) < 1e-5
    assert labels.read_boxes(tmp_path / "missing.txt") == []


def test_box_clipping_and_empty():
    b = labels.Box.from_xyxy(0, -50, -50, 100, 100, 200, 200)
    assert b is not None
    x1, y1, x2, y2 = b.to_xyxy(200, 200)
    assert x1 == 0 and y1 == 0 and abs(x2 - 100) < 1e-6
    assert labels.Box.from_xyxy(0, 300, 300, 400, 400, 200, 200) is None
    assert labels.Box.from_xyxy(0, 10, 10, 10, 50, 200, 200) is None


def test_quad_order_and_roundtrip(tmp_path: Path):
    pts = [(100, 20), (20, 20), (20, 120), (100, 120)]  # arbitrary order
    q = labels.Quad.from_pixels(7, pts, 200, 200)
    assert q.pts[0] == (0.1, 0.1) and q.pts[1] == (0.5, 0.1) and q.pts[2] == (0.5, 0.6) and q.pts[3] == (0.1, 0.6)
    labels.write_quads(tmp_path / "q.txt", [q])
    rq = labels.read_quads(tmp_path / "q.txt")
    assert rq[0].cls == 7 and len(rq[0].pts) == 4
    aabb = q.aabb()
    assert aabb is not None and abs(aabb.w - 0.4) < 1e-6 and abs(aabb.h - 0.5) < 1e-6
    assert abs(labels.polygon_area(pts) - 8000) < 1e-6


def test_order_quad_keeps_four_distinct_corners_for_rotated_quads():
    # regression: a rotated + perspective card used to collapse to 3 distinct points (TL == TR)
    pts = [(526.4, 303.9), (674.8, 462.9), (492.6, 716.7), (322.8, 571.0)]
    q = labels.order_quad(pts)
    assert len(set(q)) == 4 and sorted(q) == sorted(pts)
    assert abs(labels.polygon_area(q) - labels.polygon_area(pts)) < 1e-6
    # the generator's geometry: 222x323 card, any rotation, 8 % corner jitter, arbitrary input order
    rng = random.Random(0)
    w, h = 222.0, 323.0
    rect = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]   # TL, TR, BR, BL
    for _ in range(500):
        a = rng.uniform(-math.pi, math.pi)
        c, s = math.cos(a), math.sin(a)
        orig = [(400 + x * c - y * s + rng.uniform(-0.08 * w, 0.08 * w),
                 400 + x * s + y * c + rng.uniform(-0.08 * h, 0.08 * h)) for x, y in rect]
        shuffled = orig[:]
        rng.shuffle(shuffled)
        q = labels.order_quad(shuffled)
        assert len(set(q)) == 4 and sorted(q) == sorted(orig)
        assert any(q == orig[i:] + orig[:i] for i in range(4)), "not the same cycle as the true corner order"
        assert abs(labels.polygon_area(q) - labels.polygon_area(orig)) < 1e-6
        assert q[0][0] + q[0][1] == min(x + y for x, y in orig)


def test_remap_boxes():
    boxes = [labels.Box(0, .5, .5, .1, .1), labels.Box(1, .5, .5, .1, .1), labels.Box(2, .5, .5, .1, .1)]
    kept, dropped = labels.remap_boxes(boxes, {0: 10, 1: None})
    assert [b.cls for b in kept] == [10] and dropped == 2


def test_layout_yaml_manifest_stats_and_view(tmp_path: Path):
    root = labels.ensure_layout(tmp_path / "ds")
    img_dir, lbl_dir = labels.split_dirs(root, "train")
    (img_dir / "a.jpg").write_bytes(b"x")
    (img_dir / "b.png").write_bytes(b"x")
    (img_dir / "notes.txt").write_text("ignore")
    labels.write_boxes(lbl_dir / "a.txt", [labels.Box(5, .5, .5, .2, .2), labels.Box(5, .2, .2, .1, .1)])
    assert [p.name for p in labels.list_images(img_dir)] == ["a.jpg", "b.png"]
    assert labels.label_path_for(img_dir / "a.jpg") == lbl_dir / "a.txt"
    pairs = list(labels.iter_split(root, "train"))
    assert pairs[0][1] == lbl_dir / "a.txt"
    st = labels.dataset_stats(root)
    assert st["train"]["images"] == 2 and st["train"]["labelled_images"] == 1 and st["train"]["boxes"] == 2
    assert st["train"]["per_class"] == {5: 2}

    y = labels.write_data_yaml(root / "data.yaml", ["AC", "2C"], "images/train", "images/val", root=root)
    d = labels.read_data_yaml(y)
    assert d["nc"] == 2 and labels.yaml_names_list(d) == ["AC", "2C"] and d["path"] == str(root.resolve())
    y2 = labels.write_data_yaml(root / "merged.yaml", ["AC"], ["/a/images/train", "/b/images/train"], "/a/images/val")
    assert labels.read_data_yaml(y2)["train"] == ["/a/images/train", "/b/images/train"]

    labels.write_manifest(root, {"box_semantics": "corner", "n": 1})
    assert labels.read_manifest(root)["box_semantics"] == "corner"
    assert labels.read_manifest(tmp_path / "nothing") == {}

    alt = root / "labels_card" / "train"
    labels.write_boxes(alt / "a.txt", [labels.Box(1, .5, .5, .9, .9)])
    view = labels.make_view(root, "card", root / "labels_card")
    assert (view / "images" / "train" / "a.jpg").exists()
    assert labels.read_boxes(view / "labels" / "train" / "a.txt")[0].cls == 1
    # ultralytics-style substitution works through the view path
    assert labels.label_path_for(view / "images" / "train" / "a.jpg") == view / "labels" / "train" / "a.txt"
    # ... and still after ultralytics resolves the split directory: real dirs of per-file links,
    # never one images -> ../../images directory symlink (which would resolve to the primary labels)
    assert not (view / "images").is_symlink()
    resolved = (view / "images" / "train").resolve()
    assert view.resolve() in resolved.parents
    assert labels.label_path_for(resolved / "a.jpg") == view.resolve() / "labels" / "train" / "a.txt"
    assert [p.name for p in labels.list_images(view / "images" / "train")] == ["a.jpg", "b.png"]
    assert (view / "images" / "train" / "a.jpg").is_symlink()
    assert (view / "images" / "train" / "a.jpg").resolve() == (img_dir / "a.jpg").resolve()
    assert (view / "images" / "val").is_dir() and not (view / "images" / "test").exists()


def test_link_or_copy_and_make_view_replace_stale_links(tmp_path: Path, caplog):
    root = labels.ensure_layout(tmp_path / "ds")
    img = root / "images" / "train" / "a.jpg"
    img.write_bytes(b"x")
    labels.write_boxes(root / "labels_card" / "train" / "a.txt", [labels.Box(0, .5, .5, .9, .9)])
    labels.write_boxes(root / "labels_other" / "train" / "a.txt", [labels.Box(7, .5, .5, .9, .9)])
    view = labels.make_view(root, "card", root / "labels_card")
    assert labels.read_boxes(view / "labels" / "train" / "a.txt")[0].cls == 0
    # same source again: the existing links are kept (not recreated)
    inodes = (os.lstat(view / "labels").st_ino, os.lstat(view / "images" / "train" / "a.jpg").st_ino)
    labels.make_view(root, "card", root / "labels_card")
    assert (os.lstat(view / "labels").st_ino, os.lstat(view / "images" / "train" / "a.jpg").st_ino) == inodes
    # a different source: the stale link is replaced
    labels.make_view(root, "card", root / "labels_other")
    assert labels.read_boxes(view / "labels" / "train" / "a.txt")[0].cls == 7
    # a dangling link is replaced too
    shutil.rmtree(root / "labels_other")
    assert (view / "labels").is_symlink() and not (view / "labels").exists()
    labels.make_view(root, "card", root / "labels_card")
    assert labels.read_boxes(view / "labels" / "train" / "a.txt")[0].cls == 0
    # the older layout (one images -> ../../images directory symlink) is replaced by real split dirs
    shutil.rmtree(view / "images")
    (view / "images").symlink_to(os.path.join("..", "..", "images"))
    labels.make_view(root, "card", root / "labels_card")
    assert not (view / "images").is_symlink() and (view / "images" / "train" / "a.jpg").is_symlink()
    assert not (root / "images" / "train" / "a.jpg").is_symlink()   # the primary image was not touched
    # copy=True replaces a correct symlink by a real copy
    labels.link_or_copy(img, tmp_path / "c.jpg")
    assert (tmp_path / "c.jpg").is_symlink()
    labels.link_or_copy(img, tmp_path / "c.jpg", copy=True)
    assert not (tmp_path / "c.jpg").is_symlink() and (tmp_path / "c.jpg").read_bytes() == b"x"
    # a real file at dst is kept, with a warning
    (tmp_path / "real.txt").write_text("keep")
    with caplog.at_level(logging.WARNING, logger="vision.labels"):
        labels.link_or_copy(img, tmp_path / "real.txt")
    assert (tmp_path / "real.txt").read_text() == "keep" and "keeping" in caplog.text
