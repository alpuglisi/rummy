"""Offline tests for ``vision/download_datasets.py``.

Everything runs against tiny fake raw layouts built in ``tmp_path`` from the text
fixtures in ``vision/tests/fixtures``; the network clients are replaced by fakes via
``download_datasets.FETCHERS`` / ``download_datasets._urlopen``.
"""
from __future__ import annotations

import io
import json
import os
import pickle
import shutil
import sys
import tarfile
import types
import urllib.error
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pytest
from PIL import Image

from vision import cards, labels
from vision import download_datasets as dd
from vision.config import DATASETS, Paths

FIXTURES = Path(__file__).resolve().parent / "fixtures"


# --------------------------------------------------------------------------- helpers
def make_png(path: Path, w: int = 64, h: int = 64, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, "RGB").save(path)
    return path


def png_bytes(w: int = 4, h: int = 4) -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", (w, h), (255, 255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def build_yolo_raw(raw: Path, fixture_dir: Path, splits: Dict[str, Dict[str, Sequence[str]]],
                   style: str = "roboflow", w: int = 64, h: int = 64) -> Path:
    """``splits = {"train": {"a": ["0 0.5 0.5 0.1 0.1", ...], ...}, "valid": {...}}``.

    ``style="roboflow"``: ``<raw>/<split>/images|labels``; ``"ultralytics"``: ``<raw>/images|labels/<split>``.
    """
    raw.mkdir(parents=True, exist_ok=True)
    for f in fixture_dir.iterdir():
        shutil.copy(f, raw / f.name)
    for split, files in splits.items():
        for i, (stem, lines) in enumerate(files.items()):
            if style == "roboflow":
                img_dir, lbl_dir = raw / split / "images", raw / split / "labels"
            else:
                img_dir, lbl_dir = raw / "images" / split, raw / "labels" / split
            make_png(img_dir / f"{stem}.png", w, h, seed=i)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            (lbl_dir / f"{stem}.txt").write_text("".join(l + "\n" for l in lines))
    return raw


def build_jackfurby_raw(raw: Path) -> Path:
    """deck_a: train.pkl + val.pkl (from the json fixtures); deck_b: train.json only (1 sample)."""
    deck_a = raw / "deck_a"
    for split in ("train", "val"):
        data = json.loads((FIXTURES / "jackfurby" / f"{split}.json").read_text())
        deck_a.mkdir(parents=True, exist_ok=True)
        with open(deck_a / f"{split}.pkl", "wb") as f:
            pickle.dump(data, f)
        for sample in data.values():
            make_png(deck_a / sample["img_path"], 120, 100, seed=1)
    deck_b = raw / "deck_b"
    deck_b.mkdir(parents=True)
    sample = {"img_path": "imgs/deck_b/0.png", "class_label": 0, "concept_label": [1],
              "card_points": [[[[20, 20], [50, 20], [20, 60], [50, 60]], 48]]}   # 48 = AC -> id 0
    (deck_b / "train.json").write_text(json.dumps({"0": sample}))
    make_png(deck_b / "imgs" / "deck_b" / "0.png", 120, 100, seed=2)
    return raw


class FakeResponse:
    """Minimal urlopen-like response for ``download_file``."""

    def __init__(self, payload: bytes, status: int = 200):
        self._buf = io.BytesIO(payload)
        self.status = status
        self.headers = {"Content-Length": str(len(payload))}

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def no_network(monkeypatch):
    """Fail loudly if anything tries to reach the network."""
    def boom(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(dd, "_urlopen", boom)
    for src in list(dd.FETCHERS):
        monkeypatch.setitem(dd.FETCHERS, src, boom)
    return boom


@pytest.fixture
def no_creds(monkeypatch, tmp_path):
    for var in ("ROBOFLOW_API_KEY", "KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "no_kaggle_dir"))


# --------------------------------------------------------------------------- selection / plan
def test_resolve_dataset_keys():
    assert dd.resolve_dataset_keys("free") == ["jackfurby", "dtd", "cardart"]
    assert dd.resolve_dataset_keys("cardart,jackfurby, cardart") == ["cardart", "jackfurby"]
    assert dd.resolve_dataset_keys("all") == list(DATASETS)
    assert dd.resolve_dataset_keys("corner,dtd") == ["augstartups", "andy8744", "dtd"]
    with pytest.raises(ValueError, match="banana"):
        dd.resolve_dataset_keys("free,banana")


def test_dry_run_prints_plan_without_touching_network_or_disk(tmp_path, capsys, no_network, no_creds):
    root = tmp_path / "root"
    assert dd.main(["--dry-run", "--datasets", "all", "--root", str(root)]) == 0
    out = capsys.readouterr().out
    for key in DATASETS:
        assert f"[{key}]" in out
    assert "JackFurby/playing-cards" in out and "huggingface.co" in out
    assert "ROBOFLOW_API_KEY" in out and "KAGGLE_USERNAME" in out and "MISSING" in out
    assert "dtd-r1.0.1.tar.gz" in out and "raw.githubusercontent.com" in out
    assert "WILL BE SKIPPED" in out
    assert str(root / "raw" / "jackfurby") in out and str(root / "datasets" / "jackfurby") in out
    assert not root.exists()  # dry run creates nothing


def test_credentials_status(monkeypatch, tmp_path, no_creds):
    ok, note = dd.credentials_status(DATASETS["augstartups"])
    assert not ok and "ROBOFLOW_API_KEY" in note
    assert dd.credentials_status(DATASETS["augstartups"], roboflow_key="abc")[0]
    monkeypatch.setenv("ROBOFLOW_API_KEY", "xyz")
    assert dd.credentials_status(DATASETS["augstartups"])[0]
    ok, note = dd.credentials_status(DATASETS["andy8744"])
    assert not ok and "KAGGLE_USERNAME" in note
    kdir = tmp_path / "kaggle_home"
    kdir.mkdir()
    (kdir / "kaggle.json").write_text("{}")
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(kdir))
    assert dd.credentials_status(DATASETS["andy8744"]) == (True, str(kdir / "kaggle.json"))
    assert dd.credentials_status(DATASETS["cardart"])[0] and dd.credentials_status(DATASETS["dtd"])[0]


# --------------------------------------------------------------------------- generic YOLO converter
def test_convert_yolo_english_roboflow_export(tmp_path):
    raw = build_yolo_raw(tmp_path / "raw" / "augstartups", FIXTURES / "yolo_english", {
        "train": {"a": ["0 0.5 0.5 0.05 0.05", "4 0.2 0.2 0.05 0.05", "5 0.8 0.8 0.05 0.05"],
                  "b": ["2 0.3 0.3 0.04 0.06", "9 0.1 0.1 0.1 0.1"]},
        "valid": {"c": ["3 0.5 0.5 0.06 0.06"]},
        "test": {"d": []},
    })
    out = tmp_path / "datasets" / "augstartups"
    manifest = dd.convert_yolo(DATASETS["augstartups"], raw, out)

    assert (out / "images" / "train" / "a.png").is_symlink()
    assert (out / "images" / "train" / "a.png").resolve() == (raw / "train" / "images" / "a.png").resolve()
    assert [b.cls for b in labels.read_boxes(out / "labels" / "train" / "a.txt")] == [cards.CLASS_TO_ID["10C"]]
    assert [b.cls for b in labels.read_boxes(out / "labels" / "train" / "b.txt")] == [cards.CLASS_TO_ID["AS"]]
    assert [b.cls for b in labels.read_boxes(out / "labels" / "val" / "c.txt")] == [cards.CLASS_TO_ID["KH"]]
    assert (out / "labels" / "test" / "d.txt").read_text() == ""

    assert manifest["names"] == list(cards.CARD_CLASSES) and manifest["class_space"] == "cards52"
    assert manifest["class_mapping"]["0"] == {"raw": "10C", "canonical": "10C", "id": 9}
    assert manifest["class_mapping"]["4"] == {"raw": "joker", "canonical": "JOKER", "id": None}
    assert manifest["unmapped"] == ["joker", "banana"]
    assert manifest["box_semantics"] == "corner" and manifest["box_semantics_detected"] == "corner"
    assert manifest["stats"]["train"] == {"images": 2, "labelled_images": 2, "boxes": 2, "per_class": {"9": 1, "39": 1}} \
        or manifest["stats"]["train"]["boxes"] == 2
    assert manifest["config"]["dropped_boxes"] == 3 and manifest["config"]["unknown_class_ids"] == {"9": 1}
    assert manifest["created_by"] == "download_datasets" and manifest["views"] == {}

    y = labels.read_data_yaml(out / "data.yaml")
    assert y["nc"] == 52 and labels.yaml_names_list(y)[:2] == ["AC", "2C"] and y["test"] == "images/test"
    on_disk = labels.read_manifest(out)
    assert on_disk["class_mapping"]["2"]["id"] == cards.CLASS_TO_ID["AS"]


def test_convert_yolo_pcc_dutch_all_space(tmp_path):
    raw = build_yolo_raw(tmp_path / "raw" / "pcc3", FIXTURES / "yolo_pcc", {
        "train": {"p1": ["0 0.5 0.5 0.3 0.4", "23 0.4 0.4 0.3 0.4"], "p2": ["52 0.5 0.5 0.3 0.4", "53 0.5 0.5 0.5 0.5"]},
        "valid": {"p3": ["12 0.5 0.5 0.3 0.4"]},
    })
    out = tmp_path / "datasets" / "pcc3"
    manifest = dd.convert_yolo(DATASETS["pcc3"], raw, out)
    assert manifest["unmapped"] == [] and len(manifest["class_mapping"]) == 55
    assert [b.cls for b in labels.read_boxes(out / "labels" / "train" / "p1.txt")] == [cards.CLASS_TO_ID["10H"], cards.CLASS_TO_ID["JC"]]
    assert [b.cls for b in labels.read_boxes(out / "labels" / "train" / "p2.txt")] == [52, 53]  # JOKER, PILE_FACE_DOWN
    assert [b.cls for b in labels.read_boxes(out / "labels" / "val" / "p3.txt")] == [cards.CLASS_TO_ID["QH"]]
    assert manifest["names"] == list(cards.ALL_CLASSES) and manifest["box_semantics_detected"] == "card"
    assert labels.read_data_yaml(out / "data.yaml")["nc"] == 56


def test_convert_yolo_parts_from_classes_txt_and_val_carving(tmp_path):
    files = {f"img{i}": ["0 0.5 0.5 0.02 0.03", "15 0.2 0.2 0.02 0.02", "9 0.7 0.7 0.02 0.02", "17 0.5 0.5 0.9 0.9"]
             for i in range(4)}
    raw = build_yolo_raw(tmp_path / "raw" / "cardsbgop7", FIXTURES / "yolo_parts", {"train": files}, style="ultralytics")
    out = tmp_path / "datasets" / "cardsbgop7"
    layout = dd.find_yolo_layout(raw)
    assert layout.data_yaml is None and layout.names[:2] == ["ace", "2"] and layout.names[-1] == "card"
    assert set(layout.splits) == {"train"}

    manifest = dd.convert_yolo(DATASETS["cardsbgop7"], raw, out, val_ratio=0.25)
    assert manifest["names"] == list(cards.PART_CLASSES) and manifest["unmapped"] == ["card"]
    assert manifest["class_mapping"]["0"] == {"raw": "ace", "canonical": "RANK_A", "id": 0}
    assert manifest["class_mapping"]["15"] == {"raw": "hearts", "canonical": "SUIT_H", "id": 15}
    assert manifest["class_mapping"]["9"]["canonical"] == "RANK_10" and manifest["class_mapping"]["9"]["id"] == 9
    assert manifest["stats"]["train"]["images"] == 3 and manifest["stats"]["val"]["images"] == 1  # every 4th -> val
    assert manifest["config"]["val_carved_every"] == 4
    val_imgs = labels.list_images(out / "images" / "val")
    assert [p.name for p in val_imgs] == ["img0.png"]
    boxes = labels.read_boxes(out / "labels" / "val" / "img0.txt")
    assert sorted(b.cls for b in boxes) == [0, 9, 15]  # the big "card" box was dropped


def test_parse_label_line_polygon_to_aabb():
    b = dd.parse_label_line("3 0.1 0.2 0.5 0.2 0.5 0.6 0.1 0.6")
    assert b is not None and b.cls == 3
    assert abs(b.cx - 0.3) < 1e-9 and abs(b.cy - 0.4) < 1e-9 and abs(b.w - 0.4) < 1e-9 and abs(b.h - 0.4) < 1e-9
    assert dd.parse_label_line("1 0.5 0.5 0.1 0.1").w == pytest.approx(0.1)
    assert dd.parse_label_line("garbage") is None and dd.parse_label_line("1 2 3") is None


def test_infer_names_from_readme(tmp_path):
    (tmp_path / "README.dataset.txt").write_text("Exported by fixture\nnames: ['10C', 'AS']\n")
    assert dd.infer_names(tmp_path) == ["10C", "AS"]
    (tmp_path / "obj.names").write_text("0: ace of spades\n1: two of hearts\n")
    assert dd.infer_names(tmp_path) == ["ace of spades", "two of hearts"]


def test_detect_box_semantics(tmp_path):
    root = labels.ensure_layout(tmp_path / "ds")
    assert dd.detect_box_semantics(root) == "unknown"
    labels.write_boxes(root / "labels" / "train" / "a.txt", [labels.Box(0, .5, .5, .1, .1), labels.Box(0, .5, .5, .05, .05)])
    assert dd.detect_box_semantics(root) == "corner"
    labels.write_boxes(root / "labels" / "train" / "b.txt", [labels.Box(0, .5, .5, .4, .4)] * 3)
    assert dd.detect_box_semantics(root) == "card"
    labels.write_boxes(root / "labels" / "train" / "b.txt", [labels.Box(0, .5, .5, .2, .2)] * 3)
    assert dd.detect_box_semantics(root) == "unknown"


# --------------------------------------------------------------------------- JackFurby converter
def test_jackfurby_quad_reordering():
    tl, tr, bl, br = (10, 10), (60, 12), (8, 80), (62, 82)
    assert dd.jackfurby_quad_points([tl, tr, bl, br]) == [tl, tr, br, bl]
    boxes, quads, dropped = dd.jackfurby_points_to_labels([[[tl, tr, bl, br], 0], [[tl, tr, bl, br], 999]],
                                                          dd.load_jackfurby_classes(), 120, 100)
    assert dropped == 1 and len(boxes) == len(quads) == 1
    assert boxes[0].cls == cards.CLASS_TO_ID["2C"] == 1
    x1, y1, x2, y2 = boxes[0].to_xyxy(120, 100)
    assert (round(x1), round(y1), round(x2), round(y2)) == (8, 10, 62, 82)
    q = quads[0]
    assert q.pts[0] == (10 / 120, 10 / 100) and q.pts[1] == (60 / 120, 12 / 100)
    assert q.pts[2] == (62 / 120, 82 / 100) and q.pts[3] == (8 / 120, 80 / 100)   # BR before BL
    with pytest.raises(dd.ConversionError):
        dd.jackfurby_quad_points([tl, tr, bl])


def test_load_jackfurby_classes():
    classes = dd.load_jackfurby_classes()
    assert classes[0] == "2C" and classes[51] == "AS" and classes[32] == "10C" and len(classes) == 52


def test_convert_jackfurby_pkl_and_json_subsets(tmp_path):
    raw = build_jackfurby_raw(tmp_path / "raw" / "jackfurby")
    out = tmp_path / "datasets" / "jackfurby"
    manifest = dd.convert_jackfurby(DATASETS["jackfurby"], raw, out)

    found = dd.find_jackfurby_annotations(raw)
    assert [(s, sp, p.name) for s, sp, p in found] == [("deck_a", "train", "train.pkl"), ("deck_a", "val", "val.pkl"),
                                                       ("deck_b", "train", "train.json")]
    assert sorted(p.name for p in labels.list_images(out / "images" / "train")) == ["deck_a_0.png", "deck_a_1.png", "deck_b_0.png"]
    assert [p.name for p in labels.list_images(out / "images" / "val")] == ["deck_a_2.png"]
    assert (out / "images" / "train" / "deck_a_0.png").is_symlink()

    b0 = labels.read_boxes(out / "labels" / "train" / "deck_a_0.txt")
    assert [b.cls for b in b0] == [cards.CLASS_TO_ID["2C"], cards.CLASS_TO_ID["AS"]]
    assert b0[0].cx == pytest.approx((8 + 62) / 2 / 120, abs=1e-5) and b0[0].w == pytest.approx(54 / 120, abs=1e-5)
    assert b0[0].h == pytest.approx(72 / 100, abs=1e-5)
    assert [b.cls for b in labels.read_boxes(out / "labels" / "train" / "deck_a_1.txt")] == [cards.CLASS_TO_ID["10C"]]
    assert [b.cls for b in labels.read_boxes(out / "labels" / "val" / "deck_a_2.txt")] == [cards.CLASS_TO_ID["6C"]]
    assert [b.cls for b in labels.read_boxes(out / "labels" / "train" / "deck_b_0.txt")] == [cards.CLASS_TO_ID["AC"]]

    obb = out / "views" / "obb"
    q = labels.read_quads(obb / "labels" / "train" / "deck_a_0.txt")
    assert len(q) == 2 and len(q[0].pts) == 4
    assert q[0].pts[2] == pytest.approx((62 / 120, 82 / 100), abs=1e-5) and q[0].pts[3] == pytest.approx((8 / 120, 80 / 100), abs=1e-5)
    assert (obb / "images" / "train" / "deck_a_0.png").exists()
    assert labels.label_path_for(obb / "images" / "train" / "deck_a_0.png") == obb / "labels" / "train" / "deck_a_0.txt"
    # per-file links in real split directories: ultralytics resolves the split dir, which must stay inside the view
    assert not (obb / "images").is_symlink() and not (obb / "images" / "train").is_symlink()
    assert (obb / "images" / "train" / "deck_a_0.png").is_symlink()
    resolved_split = (obb / "images" / "train").resolve()
    assert resolved_split == (obb / "images" / "train") and labels.label_path_for(resolved_split / "deck_a_0.png") == \
        obb / "labels" / "train" / "deck_a_0.txt"
    assert (obb / "images" / "train" / "deck_a_0.png").resolve() == (out / "images" / "train" / "deck_a_0.png").resolve()
    assert sorted(p.name for p in labels.list_images(obb / "images" / "val")) == ["deck_a_2.png"]
    assert labels.read_data_yaml(obb / "data.yaml")["path"] == str(obb.resolve())

    assert manifest["views"] == {"obb": "views/obb"} and manifest["box_semantics"] == "card"
    assert manifest["box_semantics_detected"] == "card"
    assert manifest["class_mapping"]["0"] == {"raw": "2C", "canonical": "2C", "id": 1}
    assert manifest["class_mapping"]["51"] == {"raw": "AS", "canonical": "AS", "id": 39}
    assert manifest["unmapped"] == [] and manifest["stats"]["train"]["images"] == 3 and manifest["stats"]["val"]["boxes"] == 1
    assert manifest["config"]["missing_images"] == 0


def test_convert_jackfurby_missing_image_and_no_annotations(tmp_path):
    raw = tmp_path / "raw" / "jackfurby"
    raw.mkdir(parents=True)
    with pytest.raises(dd.ConversionError, match="no train/val"):
        dd.convert_jackfurby(DATASETS["jackfurby"], raw, tmp_path / "out")
    sub = raw / "deck"
    sub.mkdir()
    with open(sub / "train.pkl", "wb") as f:
        pickle.dump({0: {"img_path": "imgs/deck/0.png", "card_points": [[[[0, 0], [1, 0], [0, 1], [1, 1]], 0]]}}, f)
    with pytest.raises(dd.ConversionError, match="no images"):
        dd.convert_jackfurby(DATASETS["jackfurby"], raw, tmp_path / "out")
    make_png(sub / "somewhere" / "else" / "0.png", 30, 30)   # found via the name index fallback
    manifest = dd.convert_jackfurby(DATASETS["jackfurby"], raw, tmp_path / "out")
    assert manifest["stats"]["train"]["images"] == 1


# --------------------------------------------------------------------------- previews
def test_write_previews(tmp_path):
    raw = build_jackfurby_raw(tmp_path / "raw" / "jackfurby")
    out = tmp_path / "datasets" / "jackfurby"
    dd.convert_jackfurby(DATASETS["jackfurby"], raw, out)
    written = dd.write_previews(out, 2)
    assert len(written) == 2 and all(p.suffix == ".jpg" and p.parent == out / "preview" for p in written)
    import cv2
    img = cv2.imread(str(written[0]))
    assert img is not None and img.shape[:2] == (100, 120)


# --------------------------------------------------------------------------- network layer (mocked)
def test_download_file_retries_then_succeeds(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(url, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("temporary")
        return FakeResponse(b"hello")

    monkeypatch.setattr(dd, "_urlopen", fake_urlopen)
    monkeypatch.setattr(dd.time, "sleep", lambda s: None)
    dest = tmp_path / "f.bin"
    assert dd.download_file("https://example/x", dest, retries=3) == dest
    assert dest.read_bytes() == b"hello" and calls["n"] == 3 and not dest.with_name("f.bin.part").exists()

    monkeypatch.setattr(dd, "_urlopen", lambda url, timeout: (_ for _ in ()).throw(urllib.error.URLError("down")))
    with pytest.raises(dd.DownloadError, match="giving up"):
        dd.download_file("https://example/y", tmp_path / "g.bin", retries=2)


def test_fetch_card_art_mocked(tmp_path, monkeypatch):
    urls: List[str] = []

    def fake_urlopen(url, timeout):
        urls.append(url)
        return FakeResponse(png_bytes())

    monkeypatch.setattr(dd, "_urlopen", fake_urlopen)
    dest = tmp_path / "cards"
    paths = dd.fetch_card_art(dest)
    assert len(paths) == 55 and all(p.exists() for p in paths)
    names = sorted(p.name for p in dest.glob("*.png"))
    assert names == sorted([f"{n}.png" for n in cards.CARD_CLASSES] + ["JOKER_RED.png", "JOKER_BLACK.png", "BACK.png"])
    assert dd.CARD_ART_BASE_URL + "10_of_clubs.png" in urls and dd.CARD_ART_BASE_URL + "red_joker.png" in urls
    assert len(urls) == 55
    assert dd.missing_card_art(dest) == []

    dd.fetch_card_art(dest)               # idempotent: nothing re-fetched
    assert len(urls) == 55
    (dest / "AS.png").unlink()
    dd.fetch_card_art(dest)
    assert len(urls) == 56 and urls[-1].endswith("ace_of_spades.png")
    dd.fetch_card_art(dest, force=True)
    assert len(urls) == 111

    monkeypatch.setattr(dd, "_urlopen", lambda url, timeout: FakeResponse(b"<html>blocked</html>"))
    with pytest.raises(dd.DownloadError, match="did not return a PNG"):
        dd.fetch_card_art(tmp_path / "bad")
    assert not list((tmp_path / "bad").glob("*.png"))


def test_looks_blocked_and_describe():
    spec = DATASETS["jackfurby"]
    exc = ConnectionError("HTTPSConnectionPool: Max retries exceeded (Tunnel connection failed: 403 Forbidden)")
    assert dd.looks_blocked(exc)
    msg = dd.describe_fetch_error(spec, exc, Path("/raw/jackfurby"))
    assert "huggingface.co" in msg and "blocked" in msg and "/raw/jackfurby" in msg
    assert not dd.looks_blocked(ValueError("bad pickle"))
    assert "fetch failed" in dd.describe_fetch_error(spec, ValueError("bad pickle"), Path("/x"))


# --------------------------------------------------------------------------- archives / backgrounds
def test_extract_archives_zip_tar_and_traversal(tmp_path):
    z = tmp_path / "raw" / "bundle.zip"
    z.parent.mkdir(parents=True)
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("sub/a.txt", "A")
        zf.writestr("../evil.txt", "E")
    t = tmp_path / "raw" / "textures.tar.gz"
    with tarfile.open(t, "w:gz") as tf:
        data = b"B"
        info = tarfile.TarInfo("d/b.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
        bad = tarfile.TarInfo("/abs.txt")
        bad.size = 0
        tf.addfile(bad, io.BytesIO(b""))
    done = dd.extract_archives(tmp_path / "raw")
    assert sorted(p.name for p in done) == ["bundle", "textures"]
    assert (tmp_path / "raw" / "bundle" / "sub" / "a.txt").read_text() == "A"
    assert (tmp_path / "raw" / "textures" / "d" / "b.txt").read_text() == "B"
    assert not (tmp_path / "evil.txt").exists() and not (tmp_path / "raw" / "evil.txt").exists()
    assert (tmp_path / "raw" / "bundle" / ".extracted_ok").exists()
    assert dd.extract_archives(tmp_path / "raw") == []   # markers make it a no-op
    assert dd.raw_has_content(tmp_path / "raw") and not dd.raw_has_content(tmp_path / "nothing")


def test_collect_backgrounds(tmp_path):
    raw = tmp_path / "raw" / "dtd"
    make_png(raw / "dtd" / "images" / "banded" / "x.jpg", 8, 8)
    make_png(raw / "dtd" / "images" / "woven" / "x.jpg", 8, 8)
    make_png(raw / "dtd" / "imdb" / "no.jpg", 8, 8)   # not under an images/ dir
    dest = tmp_path / "assets" / "backgrounds"
    assert dd.collect_backgrounds(raw, dest) == 2
    assert sorted(p.name for p in labels.list_images(dest)) == ["banded_x.jpg", "woven_x.jpg"]
    assert (dest / "banded_x.jpg").is_symlink()
    assert dd.collect_backgrounds(raw, dest) == 2   # idempotent


# --------------------------------------------------------------------------- orchestration
def test_missing_credentials_are_reported_not_raised(tmp_path, capsys, no_network, no_creds, caplog):
    root = tmp_path / "root"
    rc = dd.main(["--datasets", "augstartups,andy8744", "--root", str(root)])
    out = capsys.readouterr().out
    assert rc == 2                                    # nothing at all could be done
    assert out.count("no-credentials") == 2 and "ROBOFLOW_API_KEY" in out and "KAGGLE_USERNAME" in out
    assert "2 skipped (credentials)" in out and "0 failed" in out


def test_blocked_host_is_reported_and_others_continue(tmp_path, capsys, no_network, no_creds, monkeypatch):
    def blocked(spec, raw_dir, opts):
        raise OSError("HTTPSConnectionPool(host='huggingface.co'): Tunnel connection failed: 403 Forbidden")

    def fake_card_art(spec, raw_dir, opts):
        for stem in dd.card_art_files():
            (opts.paths.assets_cards / f"{stem}.png").write_bytes(png_bytes())
        return opts.paths.assets_cards

    monkeypatch.setitem(dd.FETCHERS, "huggingface", blocked)
    monkeypatch.setitem(dd.FETCHERS, "github_raw", fake_card_art)
    root = tmp_path / "root"
    rc = dd.main(["--datasets", "jackfurby,cardart,augstartups", "--root", str(root)])
    out = capsys.readouterr().out
    assert rc == 2
    lines = {l.split()[0]: l for l in out.splitlines() if l and l.split()[0] in DATASETS}
    assert "failed" in lines["jackfurby"] and "huggingface.co" in lines["jackfurby"] and "blocked" in lines["jackfurby"]
    assert "ok" in lines["cardart"] and "no-credentials" in lines["augstartups"]
    assert len(list((root / "assets" / "cards").glob("*.png"))) == 55
    assert not (root / "datasets" / "jackfurby").exists()


def test_exit_code_rules():
    ok = dd.DatasetResult("a", "x", fetch="ok", convert="ok")
    nocred = dd.DatasetResult("b", "x", fetch="no-credentials", convert="skipped")
    failed = dd.DatasetResult("c", "x", fetch="ok", convert="failed")
    assert dd.exit_code([ok]) == 0 and dd.exit_code([ok, nocred]) == 0
    assert dd.exit_code([nocred]) == 2 and dd.exit_code([ok, failed]) == 2 and dd.exit_code([]) == 1
    assert "1 ok, 1 skipped (credentials), 1 failed" in dd.format_summary([ok, nocred, failed])


def test_main_end_to_end_with_fake_fetchers(tmp_path, capsys, no_network, no_creds, monkeypatch):
    def fake_hf(spec, raw_dir, opts):
        build_jackfurby_raw(raw_dir)
        return raw_dir

    def fake_card_art(spec, raw_dir, opts):
        for stem in dd.card_art_files():
            (opts.paths.assets_cards / f"{stem}.png").write_bytes(png_bytes())
        return opts.paths.assets_cards

    def fake_dtd(spec, raw_dir, opts):
        make_png(raw_dir / "dtd" / "images" / "banded" / "b1.jpg", 8, 8)
        make_png(raw_dir / "dtd" / "images" / "dotted" / "d1.jpg", 8, 8)
        return raw_dir

    monkeypatch.setitem(dd.FETCHERS, "huggingface", fake_hf)
    monkeypatch.setitem(dd.FETCHERS, "github_raw", fake_card_art)
    monkeypatch.setitem(dd.FETCHERS, "url", fake_dtd)
    root = tmp_path / "root"
    args = ["--datasets", "free", "--root", str(root), "--preview", "1"]

    assert dd.main(args) == 0
    out = capsys.readouterr().out
    paths = Paths(root=root)
    manifest = labels.read_manifest(paths.dataset("jackfurby"))
    assert manifest["key"] == "jackfurby" and manifest["stats"]["train"]["images"] == 3
    assert len(list((paths.dataset("jackfurby") / "preview").glob("*.jpg"))) == 1
    assert len(labels.list_images(paths.assets_backgrounds)) == 2
    assert len(dd.missing_card_art(paths.assets_cards)) == 0
    assert "3 ok" in out and "card/card" in out

    # second run: everything cached / up-to-date, no fetcher is called
    for src in ("huggingface", "github_raw", "url"):
        monkeypatch.setitem(dd.FETCHERS, src, no_network)
    assert dd.main(args) == 0
    out = capsys.readouterr().out
    row = next(l for l in out.splitlines() if l.startswith("jackfurby"))
    assert "cached" in row and "up-to-date" in row

    # --convert-only --force re-converts from raw without fetching
    assert dd.main(["--datasets", "jackfurby", "--root", str(root), "--convert-only", "--force"]) == 0
    row = next(l for l in capsys.readouterr().out.splitlines() if l.startswith("jackfurby"))
    assert "skipped" in row and " ok " in row

    # --skip-convert only fetches (cached here)
    assert dd.main(["--datasets", "jackfurby", "--root", str(root), "--skip-convert"]) == 0
    row = next(l for l in capsys.readouterr().out.splitlines() if l.startswith("jackfurby"))
    assert "cached" in row and "skipped" in row

    # --convert-only on a dataset that was never fetched fails cleanly with exit 2
    assert dd.main(["--datasets", "augstartups", "--root", str(root), "--convert-only"]) == 2
    row = next(l for l in capsys.readouterr().out.splitlines() if l.startswith("augstartups"))
    assert "failed" in row and "raw dir missing" in row

    # --copy produces real files instead of symlinks
    assert dd.main(["--datasets", "jackfurby", "--root", str(root), "--convert-only", "--force", "--copy"]) == 0
    img = paths.dataset("jackfurby") / "images" / "train" / "deck_a_0.png"
    assert img.exists() and not img.is_symlink()


def test_unknown_dataset_is_usage_error(tmp_path, capsys, no_network):
    assert dd.main(["--datasets", "nope", "--root", str(tmp_path)]) == 1


def test_process_dataset_yolo_with_fake_roboflow(tmp_path, no_creds, monkeypatch):
    def fake_roboflow(spec, raw_dir, opts):
        build_yolo_raw(raw_dir, FIXTURES / "yolo_english", {"train": {"a": ["0 0.5 0.5 0.05 0.05"]}, "valid": {"b": ["2 0.5 0.5 0.05 0.05"]}})
        return raw_dir

    monkeypatch.setitem(dd.FETCHERS, "roboflow", fake_roboflow)
    opts = dd.Options(paths=Paths(root=tmp_path / "root"), roboflow_key="k", preview=1)
    opts.paths.mkdirs()
    res = dd.process_dataset(DATASETS["augstartups"], opts)
    assert res.fetch == "ok" and res.convert == "ok" and res.images == 2 and res.boxes == 2
    assert res.semantics == "corner/corner" and not res.failed
    assert (opts.paths.dataset("augstartups") / "preview" / "train_a.jpg").exists()


# --------------------------------------------------------------------------- interrupted / partial fetches
def test_raw_state_helpers_ignore_partial_downloads(tmp_path):
    raw = tmp_path / "raw" / "dtd"
    raw.mkdir(parents=True)
    (raw / "dtd-r1.0.1.tar.gz.part").write_bytes(b"\x00" * 10)      # Ctrl-C mid download_file
    assert not dd.raw_has_content(raw) and dd.raw_in_progress(raw) and not dd.raw_is_fetched(raw)

    raw = tmp_path / "raw" / "jackfurby"                              # interrupted snapshot_download
    (raw / "deck_a").mkdir(parents=True)
    (raw / "deck_a" / "train.pkl").write_bytes(b"x")
    assert dd.raw_has_content(raw) and dd.raw_is_fetched(raw)         # complete-looking, e.g. placed by hand
    inc = raw / ".cache" / "huggingface" / "download" / "deck_b" / "train.pkl.incomplete"
    inc.parent.mkdir(parents=True)
    inc.write_bytes(b"y")
    assert dd.raw_has_content(raw) and dd.raw_in_progress(raw) and not dd.raw_is_fetched(raw)
    marker = dd.mark_fetched(raw)
    assert marker == raw / dd.FETCHED_MARKER and dd.raw_is_fetched(raw)
    assert not dd.raw_is_fetched(tmp_path / "nothing")


def test_download_file_discards_stale_part(tmp_path, monkeypatch):
    dest = tmp_path / "f.bin"
    part = dest.with_name("f.bin.part")
    part.write_bytes(b"stale")
    monkeypatch.setattr(dd.time, "sleep", lambda s: None)
    monkeypatch.setattr(dd, "_urlopen", lambda url, timeout: (_ for _ in ()).throw(urllib.error.URLError("down")))
    with pytest.raises(dd.DownloadError):
        dd.download_file("https://example/x", dest, retries=1)
    assert not part.exists() and not dest.exists()
    part.write_bytes(b"stale")
    monkeypatch.setattr(dd, "_urlopen", lambda url, timeout: FakeResponse(b"fresh"))
    assert dd.download_file("https://example/x", dest, retries=1) == dest
    assert dest.read_bytes() == b"fresh" and not part.exists()


def test_interrupted_fetch_is_fetched_again_not_cached(tmp_path, no_creds, no_network, monkeypatch):
    calls: List[str] = []

    def fake_dtd(spec, raw_dir, opts):
        calls.append("url")
        make_png(raw_dir / "dtd" / "images" / "banded" / "b1.jpg", 8, 8)
        return raw_dir

    def fake_hf(spec, raw_dir, opts):
        calls.append("huggingface")
        build_jackfurby_raw(raw_dir)
        return raw_dir

    monkeypatch.setitem(dd.FETCHERS, "url", fake_dtd)
    monkeypatch.setitem(dd.FETCHERS, "huggingface", fake_hf)
    opts = dd.Options(paths=Paths(root=tmp_path / "root"))
    opts.paths.mkdirs()

    dtd_raw, hf_raw = opts.paths.raw / "dtd", opts.paths.raw / "jackfurby"
    dtd_raw.mkdir()
    (dtd_raw / "dtd-r1.0.1.tar.gz.part").write_bytes(b"\x00" * 16)
    (hf_raw / "deck_a").mkdir(parents=True)
    (hf_raw / "deck_a" / "train.pkl").write_bytes(b"partial")
    inc = hf_raw / ".cache" / "huggingface" / "download" / "deck_a" / "val.pkl.incomplete"
    inc.parent.mkdir(parents=True)
    inc.write_bytes(b"...")

    res = dd.process_dataset(DATASETS["dtd"], opts)
    assert res.fetch == "ok" and res.convert == "ok" and calls == ["url"]
    assert (dtd_raw / dd.FETCHED_MARKER).is_file()
    res = dd.process_dataset(DATASETS["jackfurby"], opts)
    assert res.fetch == "ok" and res.convert == "ok" and calls == ["url", "huggingface"]
    assert (hf_raw / dd.FETCHED_MARKER).is_file() and res.images == 4

    # complete now: the next run is served from the marker, no fetcher runs
    monkeypatch.setitem(dd.FETCHERS, "url", no_network)
    monkeypatch.setitem(dd.FETCHERS, "huggingface", no_network)
    assert dd.process_dataset(DATASETS["dtd"], opts).fetch == "cached"
    assert dd.process_dataset(DATASETS["jackfurby"], opts).fetch == "cached"
    assert all(e.raw_present for e in dd.build_plan(["dtd", "jackfurby"], opts))

    # --force drops the marker before fetching (so an interrupted --force run is not "cached" either)
    def failing(spec, raw_dir, opts):
        raise OSError("connection reset")

    monkeypatch.setitem(dd.FETCHERS, "url", failing)
    res = dd.process_dataset(DATASETS["dtd"], dd.Options(paths=opts.paths, force=True))
    assert res.fetch == "failed" and not (dtd_raw / dd.FETCHED_MARKER).exists()


def test_fetch_roboflow_downloads_into_precreated_location(tmp_path, no_creds, monkeypatch):
    """roboflow's Version.download() returns early when ``location`` exists and overwrite is False."""
    downloads: List[Dict] = []

    class FakeVersion:
        def __init__(self, v):
            self.version = v

        def download(self, model_format, location=None, overwrite=False):
            downloads.append({"format": model_format, "location": location, "overwrite": overwrite})
            if os.path.exists(location) and not overwrite:
                return None                                   # exactly what roboflow 1.3.x does
            build_yolo_raw(Path(location), FIXTURES / "yolo_english",
                           {"train": {"a": ["0 0.5 0.5 0.05 0.05"]}, "valid": {"b": ["2 0.5 0.5 0.05 0.05"]}})

    class FakeProject:
        def versions(self):
            return [FakeVersion("ws/proj/3"), FakeVersion("ws/proj/4")]

        def version(self, v):
            return FakeVersion(v)

    class FakeRoboflow:
        def __init__(self, api_key):
            assert api_key == "k"

        def workspace(self, ws):
            assert ws == "augmented-startups"
            return self

        def project(self, proj):
            return FakeProject()

    monkeypatch.setitem(sys.modules, "roboflow", types.SimpleNamespace(Roboflow=FakeRoboflow))
    opts = dd.Options(paths=Paths(root=tmp_path / "root"), roboflow_key="k")
    opts.paths.mkdirs()
    raw = opts.paths.raw / "augstartups"
    raw.mkdir()                                               # e.g. left behind by an earlier failed run
    res = dd.process_dataset(DATASETS["augstartups"], opts)
    assert res.fetch == "ok" and res.convert == "ok" and res.images == 2, res.message
    assert downloads == [{"format": "yolov8", "location": str(raw), "overwrite": True}]
    assert (raw / dd.FETCHED_MARKER).is_file()

    # library use on a complete dir without --force keeps roboflow's own skip
    dd.fetch_roboflow(DATASETS["augstartups"], raw, opts)
    assert downloads[-1]["overwrite"] is False
    dd.fetch_roboflow(DATASETS["augstartups"], raw, dd.Options(paths=opts.paths, roboflow_key="k", force=True))
    assert downloads[-1]["overwrite"] is True


def test_kaggle_authenticate_exit_is_reported_as_missing_credentials(tmp_path, capsys, no_creds, no_network, monkeypatch):
    """kaggle >= 2 calls exit(1) from authenticate(); that must not abort the whole run."""
    class FakeKaggleApi:
        def authenticate(self):
            raise SystemExit(1)

        def dataset_download_files(self, *a, **k):  # pragma: no cover - never reached
            raise AssertionError("download attempted without authentication")

    for name in ("kaggle", "kaggle.api"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "kaggle.api.kaggle_api_extended", types.SimpleNamespace(KaggleApi=FakeKaggleApi))
    monkeypatch.setattr(dd.shutil, "which", lambda name: "/usr/bin/kaggle")     # CLI present but never consulted
    monkeypatch.setattr(dd.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("CLI fallback used")))
    kdir = tmp_path / "kaggle_home"
    kdir.mkdir()
    (kdir / "kaggle.json").write_text("{}")                   # exists, but holds no usable credentials
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(kdir))
    monkeypatch.setitem(dd.FETCHERS, "kaggle", dd.fetch_kaggle)

    opts = dd.Options(paths=Paths(root=tmp_path / "root"))
    opts.paths.mkdirs()
    res = dd.process_dataset(DATASETS["andy8744"], opts)
    assert res.fetch == "no-credentials" and res.convert == "skipped"
    assert "KAGGLE_USERNAME" in res.message and "kaggle.json" in res.message
    assert not (opts.paths.raw / "andy8744" / dd.FETCHED_MARKER).exists()

    def fake_hf(spec, raw_dir, opts):
        build_jackfurby_raw(raw_dir)
        return raw_dir

    monkeypatch.setitem(dd.FETCHERS, "huggingface", fake_hf)
    rc = dd.main(["--datasets", "andy8744,jackfurby", "--root", str(tmp_path / "root")])
    out = capsys.readouterr().out
    assert rc == 0 and "1 ok, 1 skipped (credentials), 0 failed" in out


def test_process_dataset_survives_system_exit_from_a_fetcher(tmp_path, no_creds, no_network, monkeypatch):
    def exits(spec, raw_dir, opts):
        raise SystemExit(3)

    monkeypatch.setitem(dd.FETCHERS, "huggingface", exits)
    opts = dd.Options(paths=Paths(root=tmp_path / "root"))
    opts.paths.mkdirs()
    res = dd.process_dataset(DATASETS["jackfurby"], opts)
    assert res.fetch == "failed" and res.convert == "skipped" and "SystemExit" in res.message


@pytest.mark.skipif(not os.environ.get("RUMMY_VISION_NETWORK_TESTS"), reason="set RUMMY_VISION_NETWORK_TESTS=1 to hit raw.githubusercontent.com")
def test_fetch_card_art_real_network(tmp_path):
    paths = dd.fetch_card_art(tmp_path / "cards", retries=2, timeout=30)
    assert len(paths) == 55
    with Image.open(tmp_path / "cards" / "AS.png") as im:
        assert im.size == (222, 323) and im.mode == "RGBA"
