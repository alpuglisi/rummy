"""Offline, CPU-only tests for ``vision/model.py`` (detector factories from yaml, classifier, recogniser, export)."""
from __future__ import annotations

import importlib.util
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from vision import cards
from vision import model as M
from vision.config import ModelConfig

os.environ.setdefault("YOLO_VERBOSE", "False")


# --------------------------------------------------------------------------- helpers / fakes
class FakeBoxes:
    """numpy stand-in for ``ultralytics.engine.results.Boxes``."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)


class FakeOBB(FakeBoxes):
    def __init__(self, xyxy, conf, cls, xyxyxyxy):
        super().__init__(xyxy, conf, cls)
        self.xyxyxyxy = np.asarray(xyxyxyxy, dtype=np.float32).reshape(-1, 4, 2)


class FakeResult:
    def __init__(self, boxes=None, names=None, obb=None):
        self.boxes, self.obb, self.names = boxes, obb, names or {}


class FakeDetector:
    """Duck-typed YOLO: ``predict`` returns canned results and records the kwargs it was called with."""

    task = "detect"

    def __init__(self, results):
        self.results = results
        self.calls = []

    def predict(self, image, **kwargs):
        self.calls.append(kwargs)
        return self.results


@pytest.fixture(scope="module")
def classifier() -> M.CardClassifier:
    torch.manual_seed(0)
    return M.CardClassifier(ModelConfig(task="classifier", pretrained=False)).eval()


@pytest.fixture(scope="module")
def yaml_localizer():
    return M.build_localizer(ModelConfig(task="localizer", pretrained=False))


# --------------------------------------------------------------------------- pure helpers
def test_yolo_source_and_num_classes():
    assert M.yolo_source(ModelConfig(pretrained=True)) == "yolo11n.pt"
    assert M.yolo_source(ModelConfig(pretrained=False)) == "yolo11n.yaml"
    assert M.yolo_source(ModelConfig(arch="yolo11s", pretrained=False), "-obb") == "yolo11s-obb.yaml"
    assert M.yolo_source(ModelConfig(arch="yolo11s-obb.pt", pretrained=True), "-obb") == "yolo11s-obb.pt"
    assert M.yolo_source(ModelConfig(arch="yolo11n.pt", pretrained=False)) == "yolo11n.yaml"
    assert M.yolo_source(ModelConfig(weights="runs/x/best.pt", pretrained=True)) == "runs/x/best.pt"
    assert M.num_classes(ModelConfig(task="detector")) == 52
    assert M.num_classes(ModelConfig(task="detector", class_space="all")) == 56
    assert M.num_classes(ModelConfig(task="localizer", class_space="cards52")) == 1
    assert M.class_names(ModelConfig(task="localizer")) == ["CARD"]
    assert M.num_classes(ModelConfig(task="obb")) == 52


def test_card_id_from_class_name():
    assert M.card_id_from_class_name("AC") == 0
    assert M.card_id_from_class_name("10c") == 9
    assert M.card_id_from_class_name("ace of spades") == 39
    assert M.card_id_from_class_name("JOKER") == 52
    assert M.card_id_from_class_name("7") == 7            # numeric names -> ids
    assert M.card_id_from_class_name("99") is None
    assert M.card_id_from_class_name("banana") is None


# --------------------------------------------------------------------------- YOLO factories (yaml, offline)
def test_build_detector_from_yaml():
    m = M.build_detector(ModelConfig(task="detector", arch="yolo11n", pretrained=False))
    assert m.task == "detect"
    assert type(m.model).__name__ == "DetectionModel"
    assert m.model.nc == 52 and m.model.yaml["nc"] == 52
    assert m.model.model[-1].nc == 52                       # Detect head rebuilt for 52 classes
    assert len(m.names) == 52 and m.names[0] == "AC" and m.names[51] == "KS"
    with torch.no_grad():
        out = m.model(torch.zeros(1, 3, 64, 64))
    assert out is not None


def test_build_detector_all_space_from_yaml():
    m = M.build_detector(ModelConfig(task="detector", pretrained=False, class_space="all"))
    assert m.model.nc == 56 and m.names[52] == "JOKER" and m.model.model[-1].nc == 56


def test_build_localizer_from_yaml(yaml_localizer):
    m = yaml_localizer
    assert m.task == "detect"
    assert m.model.nc == 1 and m.model.model[-1].nc == 1
    assert list(m.names.values()) == ["CARD"]


def test_build_obb_from_yaml():
    m = M.build_obb(ModelConfig(task="obb", pretrained=False))
    assert m.task == "obb"
    assert type(m.model).__name__ == "OBBModel" and type(m.model.model[-1]).__name__ == "OBB"
    assert m.model.nc == 52 and m.model.model[-1].nc == 52 and m.names[13] == "AD"


def test_build_model_dispatch_and_summary():
    m = M.build_model(ModelConfig(task="localizer", pretrained=False))
    s = M.model_summary(m, imgsz=64)
    assert "task=detect" in s and "classes=1" in s and "CARD" in s and "params=" in s
    clf = M.build_model(ModelConfig(task="classifier", pretrained=False))
    assert isinstance(clf, M.CardClassifier)
    s2 = M.model_summary(clf)
    assert "CardClassifier" in s2 and "mobilenet_v3_small" in s2 and "rank(13)+suit(4)" in s2
    with pytest.raises(ValueError):
        M.build_model(ModelConfig(task="nope"))


def test_build_model_classifier_honours_weights(tmp_path: Path, capsys):
    """Regression: ``build_model`` ignored ``cfg.weights`` for the classifier task and returned a random model."""
    torch.manual_seed(1)
    src = M.CardClassifier(ModelConfig(task="classifier", pretrained=False), crop_size=64).train()
    ckpt = src.save(tmp_path / "best.pt")
    m = M.build_model(ModelConfig(task="classifier", pretrained=False, weights=str(ckpt)))
    assert isinstance(m, M.CardClassifier) and m.crop_size == 64 and not m.training
    assert m.state_dict().keys() == src.state_dict().keys()
    assert all(torch.equal(a, b) for a, b in zip(src.state_dict().values(), m.state_dict().values()))
    # no weights -> fresh model from the config, as before
    fresh = M.build_classifier(ModelConfig(task="classifier", pretrained=False))
    assert fresh.crop_size == ModelConfig().crop_size and not torch.equal(fresh.rank_head.weight, src.rank_head.weight)
    # the CLI summarises the checkpoint, not a random network
    assert M.main(["--task", "classifier", "--weights", str(ckpt), "--summary"]) == 0
    assert "crop_size=64" in capsys.readouterr().out


# --------------------------------------------------------------------------- classifier
def test_classifier_forward_predict_loss(classifier):
    x = torch.randn(2, 3, 96, 96)
    out = classifier(x)
    assert set(out) == {"rank", "suit"}
    assert out["rank"].shape == (2, 13) and out["suit"].shape == (2, 4)
    ids = classifier.predict(x)
    assert ids.shape == (2,) and ids.dtype == torch.long and bool(((ids >= 0) & (ids < 52)).all())
    rank_idx, suit_idx = M.CardClassifier.split_card_ids(ids)
    assert torch.equal(suit_idx * 13 + rank_idx, ids)
    assert torch.equal(ids, out["suit"].argmax(1) * 13 + out["rank"].argmax(1))
    ids2, conf = classifier.predict_with_conf(x)
    assert torch.equal(ids, ids2) and bool(((conf > 0) & (conf <= 1)).all())
    loss = classifier.loss(out, torch.tensor([0, 12]), torch.tensor([3, 1]))
    assert loss.ndim == 0 and torch.isfinite(loss)
    parts = classifier.loss_components(out, torch.tensor([0, 12]), torch.tensor([3, 1]))
    assert torch.isclose(parts["rank"] + parts["suit"], loss)
    # engine convention: id 39 == AS == suit 3, rank 0
    r, s = M.CardClassifier.split_card_ids(torch.tensor([39]))
    assert r.item() == 0 and s.item() == 3 and cards.card_name(39) == "AS"


@pytest.mark.parametrize("backbone", M.BACKBONES)
def test_classifier_backbones(backbone):
    clf = M.CardClassifier(ModelConfig(task="classifier", pretrained=False, backbone=backbone, crop_size=64)).eval()
    out = clf(torch.randn(1, 3, 64, 64))
    assert out["rank"].shape == (1, 13) and out["suit"].shape == (1, 4)
    assert clf.crop_size == 64 and clf.backbone_name == backbone


def test_classifier_unknown_backbone():
    with pytest.raises(ValueError):
        M.CardClassifier(ModelConfig(task="classifier", pretrained=False, backbone="vit_huge"))


def test_classifier_pretrained_download_failure_is_graceful(monkeypatch, caplog):
    """When ImageNet weights cannot be fetched (download.pytorch.org blocked) the model still builds."""
    import torchvision.models as tvm

    real = tvm.mobilenet_v3_small

    def failing(weights=None, **kw):
        if weights is not None:
            raise OSError("simulated: download.pytorch.org unreachable")
        return real(weights=None, **kw)

    monkeypatch.setattr(tvm, "mobilenet_v3_small", failing)
    with caplog.at_level(logging.WARNING, logger="vision.model"):
        clf = M.CardClassifier(ModelConfig(task="classifier", pretrained=True))
    assert any("ImageNet weights" in r.getMessage() and "unavailable" in r.getMessage() for r in caplog.records)
    assert clf(torch.randn(1, 3, 96, 96))["rank"].shape == (1, 13)


def test_classifier_save_load_roundtrip(tmp_path: Path, classifier):
    p = classifier.save(tmp_path / "clf.pt", extra={"epoch": 3})
    ck = torch.load(p, map_location="cpu", weights_only=False)
    assert ck["format"] == M.CLASSIFIER_CHECKPOINT_FORMAT and ck["config"]["backbone"] == "mobilenet_v3_small"
    assert ck["extra"]["epoch"] == 3 and ck["class_names"] == list(cards.CARD_CLASSES)
    clf2 = M.CardClassifier.load(p)
    assert not clf2.training and clf2.crop_size == classifier.crop_size
    x = torch.randn(3, 3, 96, 96)
    with torch.no_grad():
        a, b = classifier(x), clf2(x)
    assert torch.allclose(a["rank"], b["rank"]) and torch.allclose(a["suit"], b["suit"])


# --------------------------------------------------------------------------- preprocessing / crops
def test_preprocess_crops_normalisation_and_shapes():
    gray = np.full((40, 60, 3), 128, np.uint8)               # BGR == RGB for a gray image
    t = M.preprocess_crops([gray, gray[:20, :20]], 32)
    assert t.shape == (2, 3, 32, 32) and t.dtype == torch.float32
    expected = (128 / 255.0 - np.asarray(M.IMAGENET_MEAN)) / np.asarray(M.IMAGENET_STD)
    assert np.allclose(t[0, :, 0, 0].numpy(), expected, atol=1e-5)
    # BGR -> RGB channel swap
    bgr = np.zeros((32, 32, 3), np.uint8)
    bgr[..., 0] = 255                                          # blue in BGR
    t = M.preprocess_crops([bgr], 32)
    assert t[0, 2].mean() > t[0, 0].mean()                     # ends up in the R=0, B=1 layout
    assert M.preprocess_crops([], 32).shape == (0, 3, 32, 32)


def test_square_crop_borders_and_size():
    img = np.random.randint(0, 255, (50, 80, 3), np.uint8)
    for box in [(0, 0, 10, 20), (70, 40, 80, 50), (30, 10, 50, 40), (-5, -5, 3, 3)]:
        c = M.square_crop(img, box, 24)
        assert c.shape == (24, 24, 3) and c.dtype == np.uint8
    # a centred box keeps its centre pixel: paint a bright dot and check it lands in the middle
    img2 = np.zeros((100, 100, 3), np.uint8)
    img2[50, 50] = 255
    c = M.square_crop(img2, (40, 40, 60, 60), 26, pad=0.15)
    assert c[13, 13].max() > 0


def test_square_crop_and_preprocess_match_training_resize():
    """Regression: inference crops must be pixel-identical to ``generate_synthetic.cut_crop`` (INTER_AREA even when
    enlarging); the old AREA-if-downscaling / LINEAR-if-upscaling rule diverged on every enlarged crop."""
    from vision import generate_synthetic as G
    from vision import labels as L

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    for x1, y1, x2, y2 in [(50, 60, 80, 110), (0, 0, 20, 30), (170, 150, 200, 200),   # enlarged (side < 96)
                           (10, 10, 150, 160), (20, 40, 180, 120)]:                      # shrunk (side > 96)
        box = L.Box.from_xyxy(0, x1, y1, x2, y2, w, h)
        ours, train = M.square_crop(img, box.to_xyxy(w, h), 96), G.cut_crop(img, box, 96)
        assert ours.shape == train.shape == (96, 96, 3)
        assert np.array_equal(ours, train), (x1, y1, x2, y2)
    # preprocess_crops resizes an odd-sized crop the same way (and a bilinear upscale would not match)
    c = rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)
    t = M.preprocess_crops([c], 96, bgr=False)
    got = t[0].numpy().transpose(1, 2, 0) * np.asarray(M.IMAGENET_STD, np.float32) + np.asarray(M.IMAGENET_MEAN, np.float32)
    area = cv2.resize(c, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    linear = cv2.resize(c, (96, 96), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    assert np.allclose(got, area, atol=1e-5) and not np.allclose(got, linear, atol=1e-3)


# --------------------------------------------------------------------------- recogniser
def test_recognizer_two_stage_with_mock_detector(classifier):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (128, 160, 3), dtype=np.uint8)
    boxes = [[100, 10, 130, 50], [5, 20, 30, 60], [60, 60, 80, 90]]
    det = FakeDetector([FakeResult(FakeBoxes(boxes, [0.9, 0.8, 0.7], [0, 0, 0]), names={0: "CARD"})])
    rec = M.CardRecognizer("two_stage", det, classifier, conf=0.3, imgsz=320, dedupe=False)
    dets = rec.recognize(img)
    assert len(dets) == 3 and all(isinstance(d, M.Detection) for d in dets)
    assert det.calls[0]["conf"] == 0.3 and det.calls[0]["imgsz"] == 320
    # sorted left-to-right, fields consistent with the engine vocabulary
    assert [d.xyxy[0] for d in dets] == [5.0, 60.0, 100.0]
    for d, det_conf in zip(dets, [0.8, 0.7, 0.9]):
        assert 0 <= d.card_id < 52 and d.name == cards.CARD_CLASSES[d.card_id]
        assert d.points == cards.points_500_rummy(d.name)
        assert 0.0 < d.conf <= det_conf + 1e-6                # detector conf * classifier conf
        assert d.is_card and d.quad is None and set(d.to_dict()) >= {"card_id", "name", "conf", "xyxy", "points"}
    ids = M.CardRecognizer.to_engine_ids(dets)
    assert ids == [d.card_id for d in dets]
    # the classifier actually decided: same crops -> same ids
    crops = [M.square_crop(img, b, classifier.crop_size) for b in [boxes[1], boxes[2], boxes[0]]]
    expect = classifier.predict(M.preprocess_crops(crops, classifier.crop_size)).tolist()
    assert ids == expect
    # localizer boxes labelled as an extra class are passed through unclassified
    det2 = FakeDetector([FakeResult(FakeBoxes([[0, 0, 20, 20]], [0.6], [1]), names={0: "CARD", 1: "JOKER"})])
    d2 = M.CardRecognizer("two_stage", det2, classifier).recognize(img)
    assert len(d2) == 1 and d2[0].card_id == 52 and d2[0].points == 0 and M.CardRecognizer.to_engine_ids(d2) == []


def test_recognizer_single_stage_with_mock_detector():
    img = np.zeros((100, 200, 3), np.uint8)
    names = {i: n for i, n in enumerate(cards.ALL_CLASSES)}
    det = FakeDetector([FakeResult(FakeBoxes([[100, 10, 130, 50], [5, 20, 30, 60], [50, 50, 70, 70]],
                                             [0.9, 0.8, 0.5], [39, 9, 52]), names=names)])
    rec = M.CardRecognizer("single_stage", det)
    dets = rec.recognize(img)
    assert [(d.card_id, d.name, d.points) for d in dets] == [(9, "10C", 10), (52, "JOKER", 0), (39, "AS", 15)]
    assert abs(dets[0].conf - 0.8) < 1e-6 and dets[0].xyxy == (5.0, 20.0, 30.0, 60.0)
    assert M.CardRecognizer.to_engine_ids(dets) == [9, 39]
    # unknown class names are ignored, torch tensors are accepted, numeric names map to ids
    det2 = FakeDetector([FakeResult(FakeBoxes(torch.tensor([[0., 0., 10., 10.], [20., 0., 30., 10.]]),
                                              torch.tensor([0.5, 0.4]), torch.tensor([0., 1.])), names={0: "banana", 1: "12"})])
    d2 = M.CardRecognizer("single_stage", det2).recognize(img)
    assert [d.card_id for d in d2] == [12]
    # OBB results (boxes=None, obb=...) yield the polygon
    obb = FakeOBB([[10, 10, 30, 30]], [0.7], [0], [[[10, 10], [30, 10], [30, 30], [10, 30]]])
    d3 = M.CardRecognizer("single_stage", FakeDetector([FakeResult(boxes=None, obb=obb, names={0: "AC"})])).recognize(img)
    assert len(d3) == 1 and d3[0].card_id == 0 and len(d3[0].quad) == 4 and d3[0].quad[1] == (30.0, 10.0)
    # empty result
    assert M.CardRecognizer("single_stage", FakeDetector([FakeResult(FakeBoxes([], [], []), names=names)])).recognize(img) == []


def test_recognizer_argument_validation(classifier):
    with pytest.raises(ValueError):
        M.CardRecognizer("three_stage", FakeDetector([]))
    with pytest.raises(ValueError):
        M.CardRecognizer("single_stage", None)
    with pytest.raises(ValueError):
        M.CardRecognizer("two_stage", FakeDetector([]), None)


def test_dedupe_detections():
    D = M.Detection
    a = D(0, "AC", 0.9, (0, 0, 10, 10), 15)
    a_dup = D(1, "2C", 0.8, (1, 1, 11, 11), 5)             # overlaps a -> dropped (class-agnostic NMS)
    b = D(0, "AC", 0.7, (50, 50, 60, 60), 15)              # same card far away -> dropped
    c = D(5, "6C", 0.6, (80, 80, 90, 90), 5)               # kept
    j1 = D(52, "JOKER", 0.5, (0, 50, 10, 60), 0)
    j2 = D(52, "JOKER", 0.4, (30, 50, 40, 60), 0)          # extras are not de-duplicated by id
    kept = M.dedupe_detections([c, b, a_dup, a, j1, j2], iou_thr=0.6)
    assert [(d.card_id, d.conf) for d in kept] == [(0, 0.9), (5, 0.6), (52, 0.5), (52, 0.4)]
    assert len(M.dedupe_detections([c, b, a_dup, a], same_card=False)) == 3
    assert M.iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0 and M.iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_recognizer_with_real_yaml_localizer(classifier, yaml_localizer):
    """End-to-end through ultralytics predict (random weights): must not crash, output well-formed."""
    img = np.random.default_rng(1).integers(0, 255, (96, 128, 3), dtype=np.uint8)
    rec = M.CardRecognizer("two_stage", yaml_localizer, classifier, conf=0.0, imgsz=64, max_det=5)
    dets = rec.recognize(img)
    assert all(0 <= d.card_id < 52 and d.name == cards.CARD_CLASSES[d.card_id] for d in dets)
    assert len(M.CardRecognizer.to_engine_ids(dets)) == len(dets)


# --------------------------------------------------------------------------- export
def test_export_classifier_onnx_matches_torch(tmp_path: Path, classifier):
    import onnx
    import onnxruntime as ort

    path = M.export_classifier(classifier, tmp_path, 96)
    assert path == tmp_path / "card_classifier.onnx" and path.stat().st_size > 1000
    meta = (tmp_path / "card_classifier.json")
    assert meta.exists()
    m = onnx.load(str(path))
    assert [i.name for i in m.graph.input] == ["images"]
    assert [o.name for o in m.graph.output] == ["rank", "suit"]
    assert any(op.version == 17 for op in m.opset_import if op.domain in ("", "ai.onnx"))
    assert m.graph.input[0].type.tensor_type.shape.dim[0].dim_param == "batch"   # dynamic batch
    assert M.check_onnx_classifier(path, classifier, 96) < 1e-4
    # dynamic batch really works and matches torch on a batch of 3
    x = torch.randn(3, 3, 96, 96)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rank, suit = sess.run(None, {"images": x.numpy()})
    with torch.no_grad():
        ref = classifier(x)
    assert rank.shape == (3, 13) and suit.shape == (3, 4)
    assert np.abs(rank - ref["rank"].numpy()).max() < 1e-4 and np.abs(suit - ref["suit"].numpy()).max() < 1e-4
    assert classifier.training is False  # eval mode preserved


def test_export_and_check_preserve_training_mode(tmp_path: Path):
    """Regression: ``export_classifier(check=True)`` left a training-mode classifier in eval mode (the onnxruntime
    check called ``.eval()`` on the wrapper and never undid it), so a mid-training export would freeze BN/dropout."""
    torch.manual_seed(0)
    clf = M.CardClassifier(ModelConfig(task="classifier", pretrained=False)).train()
    path = M.export_classifier(clf, tmp_path, 32, check=True)
    assert clf.training is True and all(m.training for m in clf.modules())
    assert M.check_onnx_classifier(path, clf, 32) < 1e-4       # the standalone check restores the mode too
    assert clf.training is True and all(m.training for m in clf.modules())
    meta = json.loads((tmp_path / "card_classifier.json").read_text())
    assert meta["input"]["resize_interpolation"] == "area" and meta["input"]["crop_pad"] == M.CROP_PAD


def test_export_classifier_embedded_preprocess(tmp_path: Path, classifier):
    import onnxruntime as ort

    path = M.export_classifier(classifier, tmp_path, 96, name="clf01", embed_preprocess=True)
    x01 = torch.rand(2, 3, 96, 96)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rank, _ = sess.run(None, {"images": x01.numpy()})
    with torch.no_grad():
        ref = classifier(classifier.normalize(x01))["rank"].numpy()
    assert np.abs(rank - ref).max() < 1e-4


def test_export_detector_skips_missing_deps_without_calling_export(tmp_path: Path, monkeypatch, caplog):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None)   # nothing is installed
    assert M.missing_export_modules("tflite") and M.missing_export_modules("litert")
    assert M.missing_export_modules("saved_model") == ["tensorflow", "onnx2tf"]
    assert M.missing_export_modules("unknown_fmt") == []

    class NeverExport:
        task = "detect"

        def export(self, **kw):
            raise AssertionError("export must not be called when dependencies are missing")

    with caplog.at_level(logging.WARNING, logger="vision.model"):
        res = M.export_detector(NeverExport(), tmp_path, formats=("tflite",), imgsz=64)
    assert res == {"tflite": None}
    assert any('ultralytics[export]' in r.getMessage() for r in caplog.records)


def test_export_detector_failure_is_logged_not_raised(tmp_path: Path, caplog):
    class Boom:
        task = "detect"

        def export(self, **kw):
            raise ImportError("No module named 'tensorflow'", name="tensorflow")

    with caplog.at_level(logging.ERROR, logger="vision.model"):
        res = M.export_detector(Boom(), tmp_path, formats=("onnx",), imgsz=64)
    assert res == {"onnx": None}
    assert any("export failed" in r.getMessage() and "tensorflow" in r.getMessage() for r in caplog.records)


def test_export_detector_onnx_from_yaml(tmp_path: Path, yaml_localizer):
    import onnx

    cwd_before = {p.name for p in Path.cwd().iterdir()}
    res = M.export_detector(yaml_localizer, tmp_path / "export", formats=("onnx", "tflite"), imgsz=64)
    assert res["onnx"] is not None and res["onnx"].parent == tmp_path / "export" and res["onnx"].suffix == ".onnx"
    assert res["onnx"].stat().st_size > 10_000
    m = onnx.load(str(res["onnx"]))
    assert [i.name for i in m.graph.input] == ["images"]
    if M.missing_export_modules("tflite"):
        assert res["tflite"] is None
    assert {p.name for p in Path.cwd().iterdir()} == cwd_before   # nothing leaked into the CWD


# --------------------------------------------------------------------------- CLI
def test_cli_help_and_summary(capsys):
    with pytest.raises(SystemExit) as e:
        M.main(["--help"])
    assert e.value.code == 0
    assert M.main(["--task", "localizer", "--arch", "yolo11n", "--no-pretrained", "--summary"]) == 0
    out = capsys.readouterr().out
    assert "task=detect" in out and "classes=1" in out
    assert M.main(["--task", "classifier", "--no-pretrained", "--backbone", "resnet18"]) == 0
    assert "resnet18" in capsys.readouterr().out


def test_cli_export_classifier_demo(tmp_path: Path, capsys):
    rc = M.main(["--export-classifier-demo", "--no-pretrained", "--crop-size", "64", "--out", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "card_classifier.onnx").exists() and (tmp_path / "card_classifier.json").exists()
    assert "classifier ONNX" in capsys.readouterr().out
