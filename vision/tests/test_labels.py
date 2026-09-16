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
