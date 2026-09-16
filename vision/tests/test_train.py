"""Offline, CPU-only tests for ``vision/train.py``.

Fast tests exercise dataset resolution / merging / view materialisation / crop handling / augmentation with
fake manifests and 64x64 images.  The ``slow``-marked tests run real (tiny) ultralytics and classifier smoke
trainings on a 16-image procedural synthetic set at 160 px (``-m "not slow"`` skips them).
"""
from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("YOLO_VERBOSE", "False")

from vision import cards as C  # noqa: E402
from vision import labels as L  # noqa: E402
from vision import train as T  # noqa: E402
from vision.config import ModelConfig, Paths, SynthConfig  # noqa: E402


# --------------------------------------------------------------------------- fixtures / helpers
@pytest.fixture(scope="module")
def synth_root(tmp_path_factory) -> Path:
    """16 procedural synthetic images at 160 px (corner labels, card/obb views, crops)."""
    from vision.generate_synthetic import generate

    base = tmp_path_factory.mktemp("synth")
    cfg = SynthConfig(out_dir=base / "rummy_tiny", cards_dir=base / "cards_missing", backgrounds_dir=None, num_images=16,
                      img_size=160, seed=5, workers=1, val_ratio=0.2,
                      scenario_weights={"single": 0.3, "hand": 0.3, "spread": 0.2, "pile": 0.2})
    generate(cfg, fetch=False, progress=False)
    return cfg.out_dir


def make_fake_dataset(root: Path, key: str, box_semantics: str = "corner", class_space: str = "cards52", n: int = 2,
                      names=None, card_view: bool = False, obb_view: bool = False, splits=("train", "val")) -> Path:
    """Canonical layout with 64x64 JPEGs, one label per image, a manifest and optional card/obb views."""
    names = list(names) if names is not None else C.class_names_for_space(class_space)
    L.ensure_layout(root, splits)
    for split in splits:
        for i in range(n):
            img = np.full((64, 64, 3), 200, np.uint8)
            cv2.imwrite(str(root / "images" / split / f"{key}_{split}_{i}.jpg"), img)
            cls = i % len(names)
            L.write_boxes(root / "labels" / split / f"{key}_{split}_{i}.txt", [L.Box(cls, 0.5, 0.5, 0.2, 0.2), L.Box(len(names) - 1, 0.3, 0.3, 0.1, 0.1)])
            if card_view:
                L.write_boxes(root / "views" / "card" / "labels" / split / f"{key}_{split}_{i}.txt", [L.Box(cls, 0.5, 0.5, 0.8, 0.9)])
            if obb_view:
                L.write_quads(root / "views" / "obb" / "labels" / split / f"{key}_{split}_{i}.txt",
                              [L.Quad.from_pixels(cls, [(5, 5), (60, 8), (58, 60), (3, 55)], 64, 64)])
    views = {}
    for name, enabled in (("card", card_view), ("obb", obb_view)):
        if enabled:
            L.link_or_copy(root / "images", root / "views" / name / "images")
            views[name] = f"views/{name}"
    L.write_data_yaml(root / "data.yaml", names, "images/train", "images/val", root=root)
    L.write_manifest(root, {"key": key, "source": "test", "class_space": class_space, "box_semantics": box_semantics,
                            "box_semantics_detected": box_semantics, "names": names, "views": views, "created_by": "test"})
    return root


def fake_paths(tmp_path: Path) -> Paths:
    return Paths(root=tmp_path / "data", runs=tmp_path / "runs")


# --------------------------------------------------------------------------- pure helpers
def test_split_csv_and_dataset_key():
    assert T.split_csv("a, b,,c") == ["a", "b", "c"]
    assert T.split_csv(["a,b", "c"]) == ["a", "b", "c"]
    assert T.split_csv(None) == []
    assert T.dataset_key("synthetic:rummy_v1") == "synthetic_rummy_v1"
    assert T.dataset_key("augstartups") == "augstartups"
    assert T.dataset_key("/some/where/ds1") == "ds1"


def test_resolve_dataset(tmp_path: Path):
    paths = fake_paths(tmp_path)
    assert T.resolve_dataset("synthetic:rummy_v1", paths) == paths.synthetic / "rummy_v1"
    assert T.resolve_dataset("augstartups", paths) == paths.datasets / "augstartups"
    d = tmp_path / "plain"
    d.mkdir()
    assert T.resolve_dataset(str(d), paths) == d.resolve()
    with pytest.raises(T.DatasetError):
        T.resolve_dataset("synthetic:", paths)
    with pytest.raises(T.DatasetError):
        T.load_dataset_info("synthetic:missing", paths)


def test_load_dataset_info_requires_manifest(tmp_path: Path):
    paths = fake_paths(tmp_path)
    root = make_fake_dataset(paths.datasets / "ds", "ds")
    (root / "manifest.json").unlink()
    with pytest.raises(T.DatasetError, match="manifest"):
        T.load_dataset_info("ds", paths)
    # data.yaml extras are an accepted fallback
    L.write_data_yaml(root / "data.yaml", C.CARD_CLASSES, "images/train", "images/val", root=root,
                      extra={"box_semantics": "corner", "class_space": "cards52"})
    info = T.load_dataset_info("ds", paths)
    assert info.box_semantics == "corner" and info.class_space == "cards52" and info.splits() == ["train", "val"]


def test_make_model_config(tmp_path: Path, monkeypatch):
    cfg = T.make_model_config("detector", "yolo11n.yaml", pretrained=True)
    assert cfg.arch == "yolo11n" and cfg.pretrained is False and cfg.weights is None
    cfg = T.make_model_config("obb", "yolo11s", pretrained=True)
    assert cfg.arch == "yolo11s" and cfg.pretrained is True and cfg.task == "obb"
    w = tmp_path / "best.pt"
    w.write_bytes(b"x")
    cfg = T.make_model_config("detector", str(w), pretrained=False)
    assert cfg.weights == str(w)
    cfg = T.make_model_config("detector", "yolo11m.pt", pretrained=False)
    assert cfg.arch == "yolo11m" and cfg.pretrained is True
    monkeypatch.setattr(T, "local_weights_available", lambda source: False)
    cfg = T.make_model_config("detector", "yolo11n", pretrained=True, smoke=True)
    assert cfg.pretrained is False, "smoke must not download weights"
    monkeypatch.setattr(T, "local_weights_available", lambda source: True)
    cfg = T.make_model_config("detector", "yolo11n", pretrained=True, smoke=True)
    assert cfg.pretrained is True
    cfg = T.make_model_config("classifier", "resnet18", pretrained=False, crop_size=64)
    assert cfg.task == "classifier" and cfg.backbone == "resnet18" and cfg.crop_size == 64
    cfg = T.make_model_config("classifier", "yolo11n", pretrained=False, backbone="mobilenet_v3_large")
    assert cfg.backbone == "mobilenet_v3_large"


def test_apply_smoke_and_device():
    opts = T.TrainOptions(epochs=50, imgsz=640, batch=32, fraction=1.0, workers=4, device="auto", cls_epochs=15, synth_num=500)
    T.apply_smoke(opts)
    assert (opts.epochs, opts.imgsz, opts.batch, opts.fraction, opts.workers, opts.device) == (1, 320, 8, 0.05, 0, "cpu")
    assert opts.cls_epochs == 1 and opts.max_crops == 512 and opts.synth_num == 24 and opts.plots is False
    small = T.apply_smoke(T.TrainOptions(imgsz=160, batch=4, fraction=0.01))
    assert small.imgsz == 160 and small.batch == 4 and small.fraction == 0.01
    assert T.resolve_device("cpu") == "cpu" and T.resolve_device("0") == "0"
    assert T.resolve_device("auto") in ("cpu", "0")
    assert T.cosine_with_warmup(0, 100, 10) == pytest.approx(0.1)
    assert T.cosine_with_warmup(9, 100, 10) == pytest.approx(1.0)
    assert T.cosine_with_warmup(100, 100, 10) == pytest.approx(0.0, abs=1e-9)
    assert T.cosine_with_warmup(5, 0, 0) == 1.0


# --------------------------------------------------------------------------- source selection / merging
def test_select_sources_refuses_mismatched_semantics_and_space(tmp_path: Path):
    paths = fake_paths(tmp_path)
    make_fake_dataset(paths.datasets / "corner_ds", "corner_ds", "corner", "cards52")
    make_fake_dataset(paths.datasets / "card_ds", "card_ds", "card", "cards52")
    make_fake_dataset(paths.datasets / "all_ds", "all_ds", "corner", "all")
    make_fake_dataset(paths.datasets / "parts_ds", "parts_ds", "parts", "parts")
    infos = [T.load_dataset_info(k, paths) for k in ("corner_ds", "card_ds", "all_ds", "parts_ds")]
    with pytest.raises(T.DatasetError) as exc:
        T.select_sources("detector", infos, box_semantics="corner", class_space="cards52")
    msg = str(exc.value)
    assert "card_ds" in msg and "all_ds" in msg and "parts_ds" in msg and "corner_ds" not in msg.split("\n", 1)[1]
    ok = T.select_sources("detector", infos[:1], "corner", "cards52")
    assert len(ok) == 1 and ok[0].view is None and ok[0].remap is None and not ok[0].needs_materialising
    # the "all" space is fine when requested
    ok = T.select_sources("detector", [infos[2]], "corner", "all")
    assert ok[0].info.spec == "all_ds"
    # card semantics: card-labelled datasets use their primary labels, corner ones need views/card
    ok = T.select_sources("detector", [infos[1]], "card", "cards52")
    assert ok[0].view is None and ok[0].semantics == "card"
    with pytest.raises(T.DatasetError, match="no views/card"):
        T.select_sources("detector", [infos[0]], "card", "cards52")
    with pytest.raises(T.DatasetError):
        T.select_sources("detector", infos[:1], "parts", "cards52")


def test_card_semantics_uses_card_view_and_materialises(tmp_path: Path):
    paths = fake_paths(tmp_path)
    make_fake_dataset(paths.datasets / "cv", "cv", "corner", "cards52", card_view=True)
    info = T.load_dataset_info("cv", paths)
    src = T.select_sources("detector", [info], "card", "cards52")[0]
    assert src.view == "card" and src.needs_materialising
    run = tmp_path / "runs" / "r"
    prepared = T.prepare_sources([src], run)
    dirs = prepared[0]["dirs"]
    assert set(dirs) == {"train", "val"}
    img_dir = dirs["train"]
    assert img_dir.is_dir() and not img_dir.is_symlink(), "the images dir must be a real directory (ultralytics resolves symlinks)"
    files = L.list_images(img_dir)
    assert len(files) == 2 and all(f.is_symlink() for f in files)
    lbl = L.label_path_for(files[0])
    assert lbl.exists() and L.read_boxes(lbl)[0].w == pytest.approx(0.8)  # card view boxes, not the corner ones
    y = T.write_merged_yaml(run / "data.yaml", prepared, C.CARD_CLASSES, extra={"box_semantics": "card"})
    d = L.read_data_yaml(y)
    assert d["train"] == [str(img_dir.resolve())] and d["val"] == [str(dirs["val"].resolve())]
    assert d["nc"] == 52 and L.yaml_names_list(d)[0] == "AC" and d["box_semantics"] == "card"


def test_localizer_view_creation_and_remap(tmp_path: Path):
    paths = fake_paths(tmp_path)
    make_fake_dataset(paths.datasets / "c52", "c52", "corner", "cards52")
    make_fake_dataset(paths.datasets / "allsp", "allsp", "corner", "all")      # last class = CARD_BACK -> dropped
    make_fake_dataset(paths.datasets / "single", "single", "corner", "card")
    make_fake_dataset(paths.datasets / "parts", "parts", "corner", "parts")
    infos = [T.load_dataset_info(k, paths) for k in ("c52", "allsp", "single", "parts")]
    with pytest.raises(T.DatasetError, match="parts"):
        T.select_sources("localizer", infos, "corner")
    sources = T.select_sources("localizer", infos[:3], "corner")
    assert [s.out_view for s in sources] == ["localizer"] * 3 and all(s.remap is not None for s in sources)
    assert sources[1].remap[C.CLASS_TO_ID["JOKER"]] == 0 and sources[1].remap[C.CLASS_TO_ID["PILE_FACE_DOWN"]] is None
    run = tmp_path / "runs" / "loc"
    prepared = T.prepare_sources(sources, run)
    for p in prepared:
        for split, d in p["dirs"].items():
            for img in L.list_images(d):
                boxes = L.read_boxes(L.label_path_for(img))
                assert boxes and all(b.cls == 0 for b in boxes)
    # the 'all' dataset lost its CARD_BACK boxes, the cards52 one kept both boxes
    allsp_img = L.list_images(prepared[1]["dirs"]["train"])[0]
    assert len(L.read_boxes(L.label_path_for(allsp_img))) == 1
    c52_img = L.list_images(prepared[0]["dirs"]["train"])[0]
    assert len(L.read_boxes(L.label_path_for(c52_img))) == 2
    y = T.write_merged_yaml(run / "data.yaml", prepared, ["CARD"])
    d = L.read_data_yaml(y)
    assert len(d["train"]) == 3 and d["nc"] == 1 and L.yaml_names_list(d) == ["CARD"]
    assert T.count_images(prepared) == {"train": 6, "val": 6}


def test_materialize_view_is_idempotent_and_rebuilds_on_change(tmp_path: Path):
    paths = fake_paths(tmp_path)
    make_fake_dataset(paths.datasets / "ds", "ds", "corner", "cards52", obb_view=True)
    info = T.load_dataset_info("ds", paths)
    src = T.select_sources("obb", [info])[0]
    out = tmp_path / "view"
    dirs1 = T.materialize_view(src, out)
    stamp = out / T.VIEW_STAMP
    mtime = stamp.stat().st_mtime_ns
    dirs2 = T.materialize_view(src, out)
    assert dirs1 == dirs2 and stamp.stat().st_mtime_ns == mtime
    (out / "labels" / "train.cache").write_bytes(b"stale")
    # a new image in the source changes the stamp -> rebuild (stale cache removed, new image linked)
    cv2.imwrite(str(info.root / "images" / "train" / "ds_train_9.jpg"), np.zeros((64, 64, 3), np.uint8))
    dirs3 = T.materialize_view(src, out)
    assert len(L.list_images(dirs3["train"])) == 3 and not (out / "labels" / "train.cache").exists()
    q = L.read_quads(L.label_path_for(L.list_images(dirs3["train"])[0]))
    assert len(q) == 1 and len(q[0].pts) == 4


def test_remapped_view_rebuilds_when_source_labels_change(tmp_path: Path):
    paths = fake_paths(tmp_path)
    make_fake_dataset(paths.datasets / "c", "c", "corner", "cards52")
    info = T.load_dataset_info("c", paths)
    src = T.select_sources("localizer", [info], "corner")[0]
    assert src.remap is not None
    out = tmp_path / "view"
    dirs = T.materialize_view(src, out)
    stamp = out / T.VIEW_STAMP
    mtime = stamp.stat().st_mtime_ns
    view_lbl = L.label_path_for(dirs["train"] / "c_train_0.jpg")
    assert not view_lbl.is_symlink() and len(L.read_boxes(view_lbl)) == 2   # rewritten copies, not links
    T.materialize_view(src, out)
    assert stamp.stat().st_mtime_ns == mtime, "unchanged source -> view left alone"
    # the source label is regenerated with a different box but the same file name / image count
    L.write_boxes(info.root / "labels" / "train" / "c_train_0.txt", [L.Box(3, 0.4, 0.4, 0.3, 0.3)])
    dirs = T.materialize_view(src, out)
    boxes = L.read_boxes(L.label_path_for(dirs["train"] / "c_train_0.jpg"))
    assert len(boxes) == 1 and boxes[0].cls == 0 and boxes[0].w == pytest.approx(0.3)
    assert stamp.stat().st_mtime_ns != mtime


def test_same_basename_datasets_get_distinct_view_dirs(tmp_path: Path):
    paths = fake_paths(tmp_path)
    a = make_fake_dataset(tmp_path / "a" / "cards", "a", "corner", "cards52", card_view=True)
    b = make_fake_dataset(tmp_path / "b" / "cards", "b", "corner", "cards52", card_view=True)
    infos = [T.load_dataset_info(str(p), paths) for p in (a, b)]
    assert infos[0].key == infos[1].key == "cards"
    sources = T.select_sources("detector", infos, "card", "cards52")
    run = tmp_path / "runs" / "r"
    prepared = T.prepare_sources(sources, run)
    d0, d1 = prepared[0]["dirs"]["train"], prepared[1]["dirs"]["train"]
    assert d0 != d1 and d0.parent.parent == T.view_dir(run, sources[0]) and d1.parent.parent == T.view_dir(run, sources[1])
    assert {f.name for f in L.list_images(d0)} == {"a_train_0.jpg", "a_train_1.jpg"}
    assert {f.name for f in L.list_images(d1)} == {"b_train_0.jpg", "b_train_1.jpg"}
    targets = {f.resolve() for p in prepared for f in L.list_images(p["dirs"]["train"])}
    assert len(targets) == 4 and T.count_images(prepared) == {"train": 4, "val": 4}
    d = L.read_data_yaml(T.write_merged_yaml(run / "data.yaml", prepared, C.CARD_CLASSES))
    assert len(set(d["train"])) == 2
    T.prepare_sources(sources, run)   # idempotent: the second view never wipes the first
    assert len(L.list_images(d0)) == 2 and len(L.list_images(d1)) == 2


def test_torch_device_accepts_multi_gpu_list(caplog):
    torch = pytest.importorskip("torch")
    assert T.torch_device("cpu") == torch.device("cpu")
    assert T.torch_device("0") == torch.device("cuda:0")
    assert T.torch_device("cuda:1") == torch.device("cuda:1")
    with caplog.at_level(logging.INFO, logger="vision.train"):
        assert T.torch_device("0,1") == torch.device("cuda:0")   # the documented multi-GPU form must not crash
    assert any("single GPU" in r.message for r in caplog.records)
    assert T.torch_device("auto").type in ("cpu", "cuda")


def test_obb_skips_datasets_without_view(tmp_path: Path, caplog):
    paths = fake_paths(tmp_path)
    make_fake_dataset(paths.datasets / "with", "with", "corner", "cards52", obb_view=True)
    make_fake_dataset(paths.datasets / "without", "without", "card", "cards52")
    infos = [T.load_dataset_info(k, paths) for k in ("with", "without")]
    with caplog.at_level(logging.WARNING, logger="vision.train"):
        sources = T.select_sources("obb", infos)
    assert [s.info.spec for s in sources] == ["with"] and sources[0].view == "obb"
    assert any("without" in r.message and "views/obb" in r.message for r in caplog.records)
    with pytest.raises(T.DatasetError):
        T.select_sources("obb", infos[1:])
    with pytest.raises(T.DatasetError, match="class_space"):
        make_fake_dataset(paths.datasets / "allobb", "allobb", "corner", "all", obb_view=True)
        T.select_sources("obb", [T.load_dataset_info("allobb", paths)], class_space="cards52")


def test_write_merged_yaml_requires_val(tmp_path: Path):
    paths = fake_paths(tmp_path)
    make_fake_dataset(paths.datasets / "trainonly", "trainonly", splits=("train",))
    info = T.load_dataset_info("trainonly", paths)
    prepared = T.prepare_sources(T.select_sources("detector", [info]), tmp_path / "run")
    with pytest.raises(T.DatasetError, match="validation"):
        T.write_merged_yaml(tmp_path / "run" / "data.yaml", prepared, C.CARD_CLASSES)


# --------------------------------------------------------------------------- crops / augmentation
def test_read_crops_csv_and_split_records(tmp_path: Path):
    root = tmp_path / "ds" / "crops"
    rows = []
    for split, n in (("train", 8), ("val", 3)):
        for i in range(n):
            rel = Path("crops") / split / "AS" / f"x_{split}_{i}.jpg"
            (root.parent / rel).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(root.parent / rel), np.zeros((32, 32, 3), np.uint8))
            rows.append((rel.as_posix(), 39, 0, 3))
    rows.append(("crops/train/JOKER/bad.jpg", 52, 0, 4))   # out-of-range ids are ignored
    with open(root / T.CROPS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "card_id", "rank_idx", "suit_idx"])
        w.writerows(rows)
    recs = T.read_crops_csv(root, source="ds")
    assert len(recs) == 11 and all(r.path.exists() for r in recs) and {r.split for r in recs} == {"train", "val"}
    train, val = T.split_records(recs, 0.1, seed=0)
    assert len(train) == 8 and len(val) == 3
    # no val crops -> deterministic hold-out; caps are honoured
    only_train = [r for r in recs if r.split == "train"]
    t1, v1 = T.split_records(only_train, 0.25, seed=1)
    t2, v2 = T.split_records(only_train, 0.25, seed=1)
    assert len(v1) == 2 and len(t1) == 6 and [r.path for r in v1] == [r.path for r in v2]
    t3, v3 = T.split_records(recs, 0.1, seed=0, max_train=4, max_val=2)
    assert len(t3) == 4 and len(v3) == 2
    with pytest.raises(T.DatasetError):
        T.read_crops_csv(tmp_path / "nowhere")


def test_augment_crop_is_deterministic_and_never_flips():
    img = np.full((64, 64, 3), 40, np.uint8)
    img[:, :24] = 230                        # bright block on the LEFT (chiral pattern)
    a = T.augment_crop(np.random.default_rng(3), img)
    b = T.augment_crop(np.random.default_rng(3), img)
    assert a.shape == img.shape and a.dtype == np.uint8 and np.array_equal(a, b)
    assert not np.array_equal(T.augment_crop(np.random.default_rng(4), img), a)
    cfg = T.AugConfig(cutout_p=0.0, noise_p=0.0, jpeg_p=0.0, gray_p=0.0)
    for seed in range(40):
        out = T.augment_crop(np.random.default_rng(seed), img, cfg).astype(np.float32)
        assert out[:, :20].mean() > out[:, 44:].mean(), f"seed {seed}: the bright side moved (flip?)"


def test_crop_dataset_collate_and_evaluate(synth_root: Path):
    import torch
    from vision import model as M

    recs = T.read_crops_csv(synth_root / "crops", source="synth")
    assert recs and all(r.path.exists() for r in recs)
    ds = T.CropDataset(recs[:6], 96, augment=True, seed=0)
    img, cid = ds[0]
    assert img.shape == (96, 96, 3) and img.dtype == np.uint8 and 0 <= cid < 52
    ds.epoch = 1
    img2, _ = ds[0]
    assert not np.array_equal(img, img2), "augmentation must change with the epoch"
    x, y = T.collate_crops([ds[i] for i in range(4)])
    assert x.shape == (4, 3, 96, 96) and y.dtype == torch.long and y.tolist() == [r.card_id for r in recs[:4]]
    loader = T.make_loader(T.CropDataset(recs[:6], 96, augment=False), 4, False, 0, 0)
    torch.manual_seed(0)
    clf = M.CardClassifier(ModelConfig(task="classifier", pretrained=False)).eval()
    m = T.evaluate_classifier(clf, loader, torch.device("cpu"))
    assert m["n"] == 6 and 0.0 <= m["card_acc"] <= m["rank_acc"] <= 1.0 and np.isfinite(m["loss"])


def test_crops_from_dataset(synth_root: Path, tmp_path: Path):
    out = T.crops_from_dataset(synth_root, tmp_path / "cut", crop_size=48)
    recs = T.read_crops_csv(out, source="cut")
    n_boxes = sum(len(L.read_boxes(lp)) for s in ("train", "val") for _, lp in L.iter_split(synth_root, s))
    assert len(recs) == n_boxes > 0
    img = cv2.imread(str(recs[0].path))
    assert img.shape == (48, 48, 3)
    assert {r.split for r in recs} <= {"train", "val"}
    # idempotent
    assert T.crops_from_dataset(synth_root, tmp_path / "cut", crop_size=48) == out


# --------------------------------------------------------------------------- CLI plumbing
def test_cli_help_and_errors(tmp_path: Path, capsys):
    with pytest.raises(SystemExit) as exc:
        T.main(["--help"])
    assert exc.value.code == 0
    with pytest.raises(SystemExit) as exc:
        T.main(["detector", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--box-semantics" in out and "--smoke" in out
    assert T.main(["detector", "--datasets", "synthetic:nope", "--data-root", str(tmp_path / "d"), "--runs-root", str(tmp_path / "r")]) == 1
    assert T.main(["classifier", "--crops", "synthetic:nope", "--data-root", str(tmp_path / "d"), "--runs-root", str(tmp_path / "r")]) == 1


def test_build_options_from_cli(tmp_path: Path):
    args = T.build_parser().parse_args(["detector", "--datasets", "a,b", "--smoke", "--imgsz", "160", "--runs-root", str(tmp_path),
                                        "--model", "yolo11n.yaml", "--export", "onnx"])
    opts, mcfg, paths = T.build_options(args)
    assert opts.datasets == ["a", "b"] and opts.smoke and opts.imgsz == 160 and opts.epochs == 1 and opts.device == "cpu"
    assert opts.export == ["onnx"] and opts.name == "detector_v1" and paths.runs == tmp_path
    assert mcfg.task == "detector" and mcfg.pretrained is False
    args = T.build_parser().parse_args(["all", "--smoke"])
    opts, _, _ = T.build_options(args)
    assert opts.export == [] and opts.name == "rummy_v1" and opts.synth_num == 24
    args = T.build_parser().parse_args(["classifier", "--crops", "synthetic:x", "--backbone", "resnet18", "--crop-size", "64"])
    opts, mcfg, _ = T.build_options(args)
    assert opts.export == list(T.DEFAULT_EXPORT) and mcfg.backbone == "resnet18" and mcfg.crop_size == 64 and opts.cls_epochs == 15


# --------------------------------------------------------------------------- smoke trainings (tiny, real)
def test_classifier_smoke_cli(synth_root: Path, tmp_path: Path):
    from vision import model as M

    rc = T.main(["classifier", "--crops", str(synth_root), "--smoke", "--no-pretrained", "--runs-root", str(tmp_path / "runs"),
                 "--name", "clf", "--cls-batch", "32"])
    assert rc == 0
    run = tmp_path / "runs" / "clf"
    best, last = run / "weights" / "best.pt", run / "weights" / "last.pt"
    assert best.exists() and last.exists() and (run / "results.csv").exists()
    onnx = run / "export" / "card_classifier.onnx"
    assert onnx.exists() and onnx.with_suffix(".json").exists()
    summary = json.loads((run / T.SUMMARY_NAME).read_text())
    assert summary["task"] == "classifier" and summary["errors"] == []
    assert set(summary["metrics"]["best"]) >= {"rank_acc", "suit_acc", "card_acc", "loss"}
    assert summary["crops"]["train"] > 0 and summary["crops"]["val"] > 0 and len(summary["metrics"]["history"]) == 1
    assert summary["artefacts"]["exports"]["onnx"] == str(onnx)
    clf = M.CardClassifier.load(best)
    import torch
    ids = clf.predict(torch.zeros(2, 3, 96, 96))
    assert ids.shape == (2,) and all(0 <= int(i) < 52 for i in ids)


@pytest.mark.slow
def test_detector_smoke_cli(synth_root: Path, tmp_path: Path):
    rc = T.main(["detector", "--datasets", str(synth_root), "--smoke", "--imgsz", "160", "--model", "yolo11n.yaml",
                 "--runs-root", str(tmp_path / "runs"), "--name", "det", "--export", "onnx", "--data-root", str(tmp_path / "data")])
    assert rc == 0
    run = tmp_path / "runs" / "det"
    assert (run / "weights" / "best.pt").exists() and (run / "weights" / "last.pt").exists()
    d = L.read_data_yaml(run / "data.yaml")
    assert isinstance(d["train"], list) and d["train"] == [str((synth_root / "images" / "train").resolve())]
    assert d["nc"] == 52 and L.yaml_names_list(d) == list(C.CARD_CLASSES)
    summary = json.loads((run / T.SUMMARY_NAME).read_text())
    assert summary["task"] == "detector" and summary["smoke"] is True and summary["errors"] == []
    assert summary["artefacts"]["best"] == str(run / "weights" / "best.pt")
    assert "map50" in summary["metrics"]["val"] and summary["metrics"]["val"]["map50"] is not None
    assert summary["config"]["train"]["epochs"] == 1 and summary["config"]["train"]["imgsz"] == 160
    assert sum(summary["datasets"][0]["images"].values()) == 16 and summary["images"]["val"] >= 1
    onnx = summary["artefacts"]["exports"]["onnx"]
    assert onnx and Path(onnx).exists() and Path(onnx).parent == run / "export"


@pytest.mark.slow
def test_localizer_and_obb_smoke_api(synth_root: Path, tmp_path: Path):
    paths = fake_paths(tmp_path)
    opts = T.apply_smoke(T.TrainOptions(datasets=[str(synth_root)], box_semantics="card", imgsz=160, export=[], seed=1))
    mcfg = T.make_model_config("localizer", "yolo11n.yaml", pretrained=False, smoke=True)
    s = T.train_yolo("localizer", opts, mcfg, paths, paths.runs / "loc")
    assert Path(s["artefacts"]["best"]).exists() and s["class_names"] == ["CARD"]
    view = T.view_dir(paths.runs / "loc", T.TrainSource(T.load_dataset_info(str(synth_root), paths), view="card", out_view="localizer"))
    view_imgs = L.list_images(view / "images" / "train")
    assert view_imgs and all(b.cls == 0 for img in view_imgs for b in L.read_boxes(L.label_path_for(img)))
    assert s["datasets"][0]["labels"].endswith("views/card (classes -> 0)")

    opts = T.apply_smoke(T.TrainOptions(datasets=[str(synth_root)], imgsz=160, export=[], seed=1))
    mcfg = T.make_model_config("obb", "yolo11n.yaml", pretrained=False, smoke=True)
    s = T.train_yolo("obb", opts, mcfg, paths, paths.runs / "obb")
    assert Path(s["artefacts"]["best"]).exists() and Path(s["artefacts"]["last"]).exists()
    d = L.read_data_yaml(paths.runs / "obb" / "data.yaml")
    assert d["box_semantics"] == "obb" and d["train"][0].endswith("_obb/images/train")
    assert "map50" in s["metrics"]["val"]


@pytest.mark.slow
def test_all_smoke_cli(tmp_path: Path, capsys):
    rc = T.main(["all", "--smoke", "--imgsz", "160", "--synth-num", "12", "--synth-img-size", "160", "--model", "yolo11n.yaml",
                 "--no-pretrained", "--datasets", "synthetic:tiny", "--crops", "synthetic:tiny",
                 "--data-root", str(tmp_path / "data"), "--runs-root", str(tmp_path / "runs"), "--name", "all"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "detector/best" in out and "classifier/exports/onnx" in out
    synth = tmp_path / "data" / "synthetic" / "tiny"
    assert (synth / "manifest.json").exists() and (synth / "crops" / T.CROPS_CSV).exists()
    run = tmp_path / "runs" / "all"
    summary = json.loads((run / T.SUMMARY_NAME).read_text())
    assert summary["task"] == "all" and summary["errors"] == []
    assert summary["steps"]["synthetic"][0]["generated"] is True
    assert Path(summary["steps"]["detector"]["artefacts"]["best"]).exists()
    assert Path(summary["steps"]["classifier"]["artefacts"]["exports"]["onnx"]).exists()
    assert (run / "detector" / T.SUMMARY_NAME).exists() and (run / "classifier" / T.SUMMARY_NAME).exists()
