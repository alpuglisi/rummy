# Card vision pipeline

Scripts under `vision/` that download public playing-card datasets, generate synthetic
training images, define the detection / classification models and train + export them.

## Purpose

The target is an Android app for 500 Rummy that looks at the table and reports

* the cards in the player's fanned hand, and
* the cards in the discard pile, which in 500 Rummy is **spread** so every card is visible,

then hands those cards to the existing C++/PPO rummy engine (`rummy_env.h`, `train.py`,
`trainer.py` at the repo root; the vision code does not touch them).

Card ids are the engine's ids, no remapping needed:

```
card_id    = suit_index * 13 + rank_index
rank_index = 0 -> Ace, 1..8 -> 2..9, 9 -> 10, 10 -> Jack, 11 -> Queen, 12 -> King
suit_index = 0 -> Clubs, 1 -> Diamonds, 2 -> Hearts, 3 -> Spades
```

`vision/cards.py` is the single source of truth: `CARD_CLASSES[i]` is the class name of engine
card `i` (`AC`, `2C`, ..., `KC`, `AD`, ..., `KS`), `points_500_rummy(name)` gives Ace 15,
10/J/Q/K 10, others 5. A detector trained in the `all` label space additionally reports
`JOKER`, `PILE_FACE_DOWN`, `PILE_FACE_UP`, `CARD_BACK` as ids 52..55; these carry 0 points
and are excluded from `CardRecognizer.to_engine_ids`.

## Architecture

Three model flavours share one data layout and one vocabulary. Pick one per run.

| Flavour | Model | Labels it trains on | When to use |
|---|---|---|---|
| **Single-stage corner-index detector** (primary) | ultralytics YOLO11 (`yolo11n/s/m...`), 52 classes (`cards52`) or 56 (`all`) | `labels/` with `box_semantics: corner` | The app path: one network, boxes on the rank+suit index in the card corner |
| **Two-stage** | stage 1: single-class `CARD` localizer (YOLO11); stage 2: `CardClassifier` (torchvision `mobilenet_v3_small` by default, `mobilenet_v3_large` or `resnet18`; two heads, rank 13 + suit 4; 96 px RGB crops, ImageNet-normalised) | stage 1: any card-like dataset, rewritten to class 0; stage 2: `crops/labels.csv` | When you want to add card designs without retraining the detector, or when the localizer generalises better than 52 fine classes |
| **OBB variant** | `yolo11<scale>-obb` | `views/obb` (4-point oriented boxes) | Whole-card orientation (angle, corners) is needed |

**Why corner boxes.** In a fanned hand or an overlapped discard row only a thin strip of each
card is visible, and that strip always contains the top-left (or, rotated 180 degrees, the
bottom-right) rank+suit index. A box on that index stays fully visible under heavy overlap,
fingers and other cards, while whole-card boxes overlap each other almost completely and
teach the detector nothing distinctive. The generator follows the geaxgx method: it computes
the convex hull of the index glyphs on the clean card art, warps the hull with the same
homography as the card, tracks how much of it later cards / occluders cover, and only writes
a box when at least `min_corner_visibility` (0.6) of the hull is still visible.

**Box-semantics rule.** Every dataset root carries a `manifest.json` whose `box_semantics`
is one of

* `corner` - the box covers the rank+suit index in one corner of the card,
* `card` - the box covers the whole card,
* `parts` - rank and suit are separate objects (17-class `parts` space).

A detector cannot learn a mixture: **never mix corner, card and parts labels in one training
run.** `vision/train.py` enforces this: `--box-semantics corner|card` (default `corner`)
refuses every dataset whose manifest says otherwise and prints the offenders. The one
sanctioned bridge is a `views/card` label set inside a corner-labelled dataset, which
`train.py` uses automatically when you ask for `card` semantics. `download_datasets.py
--preview N` draws annotated images so you can confirm what a downloaded dataset's boxes
actually cover; the auto-detected value (`box_semantics_detected`, median box area / image
area < 0.02 -> corner, > 0.05 -> card) is recorded in the manifest.

**Inference wrapper.** `vision.model.CardRecognizer(mode, detector, classifier)` with
`mode="single_stage"` or `"two_stage"` turns a BGR image into `Detection(card_id, name, conf,
xyxy, points, quad)` records, de-duplicates the same card reported twice (keeps the highest
confidence), and `CardRecognizer.to_engine_ids(dets)` returns the engine ids sorted
left-to-right. `CardRecognizer.from_paths(mode, detector_weights, classifier_weights)` loads
`best.pt` files directly.

## Setup

```bash
cd /path/to/rummy
python3.11 -m venv .venv-vision
source .venv-vision/bin/activate
pip install -r vision/requirements.txt
```

`vision/requirements.txt` pins torch, torchvision, ultralytics, opencv-python-headless, numpy,
pillow, pyyaml, tqdm, scipy, huggingface_hub, roboflow, kaggle, onnx, onnxruntime and pytest.
Everything runs on CPU; that is what the tests and `--smoke` runs assume.

**CUDA.** The default `pip install torch` may give you a CPU-only wheel. For GPU training
install the CUDA build first, then the requirements file (its comment shows the command):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r vision/requirements.txt
```

`train.py --device auto` (the default) picks GPU `0` when `torch.cuda.is_available()` and
`cpu` otherwise; `--device cpu`, `--device 0` or `--device 0,1` force it. Mixed precision is
used for the classifier only on CUDA.

All commands below are run from the repo root with the venv active. Every script also runs
as `python vision/<script>.py ...`.

## Credentials

| Variable | Needed by | Notes |
|---|---|---|
| `ROBOFLOW_API_KEY` (or `--roboflow-key K`) | `augstartups`, `cardsbgop7`, `pcc3` | app.roboflow.com -> Settings -> API |
| `KAGGLE_USERNAME` + `KAGGLE_KEY`, or `KAGGLE_API_TOKEN`, or `~/.kaggle/kaggle.json` (`KAGGLE_CONFIG_DIR` is honoured) | `andy8744` | kaggle.com/settings -> API |
| `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` (optional) | `jackfurby` | The dataset is public; a token only raises rate limits |

Missing credentials never abort a run: the dataset is skipped with a message naming the
variable to set, and the other datasets still run (see exit codes below).

## Datasets

Registry: `vision/config.py` (`DATASETS`). Sizes and class counts are what the registry
records from the source pages; licences other than the two stated below are whatever the
source page says, so check it before redistributing anything derived from them.

| Key | Source / slug | Content | Box semantics | Class space | Creds | Licence / attribution |
|---|---|---|---|---|---|---|
| `jackfurby` | Hugging Face dataset `JackFurby/playing-cards` | 40k synthetic images (4 x 10k subsets) with exact 4-corner coordinates per card (pickle annotations, rank-major labels 2C=0 ... AS=51, see `vision/configs/jackfurby_card_classes.txt`) | `card` (+ `views/obb`) | `cards52` | none | see source page |
| `augstartups` | Roboflow `augmented-startups/playing-cards-ow27d`, version 4 | Cards blended into varied textures, 52 classes named like `10C`/`AS` | `corner` (confirm with `--preview`) | `cards52` | Roboflow | see source page |
| `cardsbgop7` | Roboflow `cards-bgop7/cards-gzetw`, latest version | ~2.5k images with rank and suit as separate objects | `parts` | `parts` (17 classes) | Roboflow | see source page |
| `pcc3` | Roboflow `deep-learning-for-image-and-video-processing/pcc3.0-yolov8`, latest | 55 classes incl. `pile-face-down` / `pile-face-up`; Dutch class letters (h=hearts k=clubs r=diamonds s=spades; a=Ace b=Jack v=Queen h=King; j=joker) | `card` | `all` (56 ids) | Roboflow | see source page |
| `andy8744` | Kaggle `andy8744/playing-cards-object-detection-dataset` | 20k synthetic 416x416 images (YOLOv5 format) built with the geaxgx pipeline over DTD textures, 52 classes | `corner` | `cards52` | Kaggle | see source page |
| `dtd` | `https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz` | Describable Textures Dataset, 5,640 texture images used as synthetic backgrounds | - | - | none | research use, see the DTD page |
| `cardart` | `raw.githubusercontent.com/hayeah/playing-cards-assets/master/png/<name>.png` | 52 cards + red/black joker + back, 222x323 RGBA PNGs, borders removed | - | - | none | public domain (vector-playing-cards) |

Groups accepted by `--datasets`: `all`, `detection` (the five labelled sets), `corner`
(`augstartups`, `andy8744`), `card` (`jackfurby`, `pcc3`), `assets` (`dtd`, `cardart`),
`free` (`jackfurby`, `dtd`, `cardart`; no API key; the default). Comma-separated keys also
work.

The synthetic generator (below) is the sixth data source and needs nothing but the card art
(and falls back to procedurally drawn cards if even that is unavailable).

## The four scripts

Run them in this order. Each has `--help`; the flags shown are the real ones.

### 1. Download and convert: `vision/download_datasets.py`

```bash
# plan only: sources, target dirs, which credentials are present. No network or disk I/O.
python -m vision.download_datasets --dry-run --datasets all

# default group "free": JackFurby + DTD backgrounds + card art
python -m vision.download_datasets

# Roboflow sets, with 8 annotated preview images each so you can check box semantics
python -m vision.download_datasets --datasets augstartups,pcc3 --roboflow-key "$ROBOFLOW_API_KEY" --preview 8

# Kaggle set (credentials from the environment or ~/.kaggle/kaggle.json)
python -m vision.download_datasets --datasets andy8744

# re-run only the conversion of something already in raw/
python -m vision.download_datasets --datasets jackfurby --convert-only --force
```

Options: `--datasets`, `--root DIR` (default `data/vision` or `$RUMMY_VISION_DATA`),
`--dry-run`, `--convert-only`, `--skip-convert`, `--copy` (copy images instead of
symlinking), `--preview N`, `--roboflow-key K`, `--force`, `--retries N` (default 3),
`--timeout S` (default 60), `--val-ratio F` (val carved from train when an export has no val
split), `--verbose`.

Two idempotent stages: **fetch** into `data/vision/raw/<key>/` (untouched), then **convert**
into `data/vision/datasets/<key>/` in the canonical layout. Conversion remaps every class name
onto `vision/cards.py` (English compact names, `ace of spades`, Dutch pcc names, separate
rank/suit objects), drops boxes of unknown classes and lists them under `unmapped` in the
manifest, writes `data.yaml` and `manifest.json`, and for JackFurby also writes whole-card
boxes plus a `views/obb` label set from the four exact corners. Card art goes straight to
`data/vision/assets/cards/<CANONICAL>.png` (`10C.png`, `AS.png`, `JOKER_RED.png`,
`JOKER_BLACK.png`, `BACK.png`); DTD images are linked into `data/vision/assets/backgrounds/`.

A summary table is printed at the end. Exit codes: `0` everything requested succeeded (sets
skipped for missing credentials are tolerated as long as something else succeeded); `2` any
requested set failed (network error, blocked host, conversion error) or every requested set
was skipped for missing credentials; `1` usage error.

### 2. Synthetic data: `vision/generate_synthetic.py`

```bash
# the default output the train script expects: data/vision/synthetic/rummy_v1
python -m vision.generate_synthetic --out data/vision/synthetic/rummy_v1 --num 20000 --workers 4 --preview 20

# a quick small set with a custom scenario mix
python -m vision.generate_synthetic --out data/vision/synthetic/quick --num 200 --img-size 416 \
    --scenarios single:0.1,hand:0.5,spread:0.3,pile:0.1 --seed 1 --preview 10

# fully offline: never try to fetch card art, draw cards procedurally if the PNGs are missing
python -m vision.generate_synthetic --out data/vision/synthetic/offline --num 100 --no-fetch
```

Options: `--out DIR`, `--num N` (default 20000), `--img-size PX` (default 640), `--cards DIR`
(`<CANONICAL>.png` or `<rank>_of_<suit>.png`; default `data/vision/assets/cards`),
`--backgrounds DIR` (`''` = procedural backgrounds only; default
`data/vision/assets/backgrounds`), `--scenarios single:0.1,hand:0.4,spread:0.3,pile:0.2`,
`--workers N` (0 = all CPUs, 1 = in-process), `--seed S`, `--val-ratio F` (default 0.1),
`--no-obb`, `--no-card-view`, `--no-crops`, `--copy-views`, `--force` (replace outputs of a
previous run under `--out`; otherwise the script refuses), `--crop-size PX` (default 96),
`--preview N`, `--no-fetch`, `--verbose`.

Scenarios: `single` (one card, any pose), `hand` (2..13 cards fanned so every top-left index
stays visible, optional skin-tone finger occluders, mild bend), `spread` (2..12 cards in a
row overlapped 40-75 %, the 500 Rummy discard pile), `pile` (2..8 cards dropped on top of
each other, top card fully visible). Photometric augmentation (brightness, colour
temperature, gamma, blur, noise, JPEG, glare, shadows, vignette) is applied per image.
Output is deterministic for a given `--seed` regardless of `--workers`.

Card art is loaded from `--cards`; missing files are fetched from the hayeah deck unless
`--no-fetch`; anything still missing is drawn procedurally so generation always works
offline. Corner hulls are cached in `<cards_dir>/corner_hulls.json`.

Python API: `generate(cfg: SynthConfig, preview=0, fetch=True, progress=True, force=False) -> manifest`,
`render_scene(rng, scenario, cards, bg, cfg)`, `load_card_bank(cards_dir, fetch=True)`,
`procedural_card(name)`, `procedural_background(rng, size)`.

### 3. Model summary and export: `vision/model.py`

```bash
# build the detector head for 52 classes from yolo11n.yaml (no download) and print a summary
python -m vision.model --task detector --arch yolo11n --no-pretrained --summary
```

Output in this checkout:

```
YOLO task=detect source=yolo11n.yaml
  type=DetectionModel head=Detect head.nc=52 classes=52 [AC, 2C, 3C, 4C, 5C, 6C, ...]
  params=2.60M GFLOPs@640=6.55 stride=tensor([ 8., 16., 32.])
```

```bash
# other flavours
python -m vision.model --task localizer --no-pretrained --summary
python -m vision.model --task obb --no-pretrained --summary
python -m vision.model --task classifier --backbone mobilenet_v3_small --summary

# build an untrained CardClassifier, export it to ONNX and check it with onnxruntime
python -m vision.model --export-classifier-demo --out runs/vision/demo

# export a trained detector
python -m vision.model --task detector --weights runs/vision/detector_v1/weights/best.pt \
    --export-detector --formats onnx,tflite --out runs/vision/detector_v1/export
```

Options: `--task {detector,localizer,obb,classifier}`, `--arch` (default `yolo11n`),
`--pretrained | --no-pretrained`, `--weights PATH`, `--class-space {cards52,all,parts,card}`,
`--backbone {mobilenet_v3_small,mobilenet_v3_large,resnet18}`, `--crop-size PX`, `--imgsz PX`,
`--summary` (default action), `--export-classifier-demo`, `--export-detector`, `--formats`
(default `onnx,tflite`), `--out DIR` (default `runs/vision/model_exports`), `--half`, `--int8`,
`--nms` (bake NMS into the detector export), `--seed`, `--verbose`.

Python API: `build_detector(cfg)`, `build_localizer(cfg)`, `build_obb(cfg)`,
`build_classifier(cfg)` / `CardClassifier`, `CardRecognizer`, `export_detector(model, out_dir,
formats, imgsz, half, int8, nms)`, `export_classifier(model, out_dir, crop_size)`,
`model_summary(model)`, `num_classes(cfg)`; `cfg` is `vision.config.ModelConfig`.

### 4. Train: `vision/train.py`

Sub-commands `detector`, `localizer`, `obb`, `classifier`, `all`; every sub-command accepts
the same flag set (irrelevant groups are ignored).

```bash
# primary corner-index detector on synthetic + the two corner-labelled public sets
python -m vision.train detector --datasets synthetic:rummy_v1,augstartups,andy8744 --epochs 50 --name detector_v1

# 56-class variant that also learns jokers / piles / backs (needs an "all"-space dataset, e.g. pcc3 has card boxes, so use --box-semantics card)
python -m vision.train detector --datasets pcc3 --class-space all --box-semantics card --name detector_all_v1

# two-stage: localizer on whole-card boxes (synthetic views/card + JackFurby), then the classifier on crops
python -m vision.train localizer --datasets synthetic:rummy_v1,jackfurby --box-semantics card --name localizer_v1
python -m vision.train classifier --crops synthetic:rummy_v1 --cls-epochs 15 --name classifier_v1

# classifier with extra crops cut from a corner-labelled public dataset
python -m vision.train classifier --crops synthetic:rummy_v1 --crops-from andy8744 --name classifier_v2

# oriented boxes (datasets without views/obb are skipped with a warning)
python -m vision.train obb --datasets synthetic:rummy_v1,jackfurby --name obb_v1

# everything: generate a small synthetic set if missing, detector, classifier, one summary
python -m vision.train all --name all_v1
```

**Smoke runs** (CPU, minutes, no downloads) - use these first to check the pipeline:

```bash
python -m vision.train all --smoke --name smoke_all
python -m vision.train detector --smoke --model yolo11n.yaml --datasets synthetic:rummy_v1 --name smoke_det
python -m vision.train localizer --smoke --datasets synthetic:rummy_v1 --box-semantics card --name smoke_loc
python -m vision.train obb --smoke --datasets synthetic:rummy_v1 --name smoke_obb
python -m vision.train classifier --smoke --crops synthetic:rummy_v1 --name smoke_cls
```

`--smoke` forces 1 epoch, imgsz 320, batch 8, at most 5 % of the training images, `workers 0`,
CPU, plots off, no export unless `--export` is given; the classifier trains 1 epoch on at most
512 crops; `all` generates 24 images at 320 px when the synthetic set is missing. If the
pretrained `<arch>.pt` is not already on disk the model is built from the yaml instead of
downloading it.

Data flags: `--datasets` (comma list of `synthetic:<name>`, registry keys or plain dataset
directories; default `synthetic:rummy_v1,augstartups,andy8744`), `--box-semantics
{corner,card}`, `--class-space {cards52,all}`, `--crops` (crop sets with `crops/labels.csv`;
default `synthetic:rummy_v1`), `--crops-from` (corner-semantics datasets to cut crops from),
`--data-root DIR`, `--runs-root DIR`.
Model flags: `--model` (`yolo11n` = COCO weights when `--pretrained`; `yolo11n.yaml` = random
init; `path/to/best.pt` = start from that checkpoint's weights; a backbone name for the classifier),
`--pretrained | --no-pretrained`, `--backbone`, `--crop-size`, `--dropout`.
Training flags: `--epochs`, `--batch`, `--imgsz`, `--device`, `--workers`, `--fraction`,
`--lr0`, `--patience`, `--seed`, `--name` (default `<task>_v1`), `--export` (default
`onnx,tflite`), `--smoke`, `--no-plots`.
Classifier flags: `--cls-epochs`, `--cls-batch`, `--cls-lr`, `--label-smoothing`,
`--weight-decay`, `--val-ratio`, `--max-crops`, `--no-cache-crops`.
`all` flags: `--synth-num` (default 500), `--synth-img-size`, `--no-fetch`.

How merging works: every dataset's manifest is checked against the task (semantics and class
space, offenders listed in one error; `obb` checks class space only and ignores
`--box-semantics`); `<runs-root>/<name>/data.yaml` then lists the absolute
image directories of all accepted datasets, so nothing is copied. Views (`views/card`,
`views/obb`, and the class-0 `localizer` rewrite) are materialised under
`<run>/views/<dataset>_<hash>_<view>/` as per-file symlinks because ultralytics resolves
symlinks before deriving label paths. The classifier uses numpy/OpenCV augmentation only:
rotation +-15 degrees, scale, shift, colour jitter, blur, noise, cutout, and **no flips** (rank
glyphs are chiral). AdamW + cosine LR with warm-up; validation reports `rank_acc`,
`suit_acc` and joint `card_acc`.

## Data layout on disk

Everything lives under `data/vision/` (override with `RUMMY_VISION_DATA`) and
`runs/vision/` (`RUMMY_VISION_RUNS`); both are git-ignored, as are `*.pt`, `*.onnx`,
`*.tflite` and `*.pkl`.

```
data/vision/
  raw/<key>/                     untouched downloads (zips, pkls, roboflow exports)
  datasets/<key>/                converted public datasets, canonical layout (below)
  synthetic/<name>/              output of generate_synthetic.py, canonical layout + crops
  assets/cards/                  10C.png ... KS.png, JOKER_RED.png, JOKER_BLACK.png, BACK.png, corner_hulls.json
  assets/backgrounds/            DTD textures (symlinks/copies named <category>_<file>)

<dataset root>/                  canonical layout (labels.py)
  data.yaml                      ultralytics dataset file, names == canonical names
  manifest.json                  key, source, ref, class_space, box_semantics, box_semantics_detected,
                                 names, class_mapping, unmapped, stats, views, created_by, config
  images/{train,val[,test]}/     *.jpg / *.png
  labels/{train,val[,test]}/     *.txt  "cls cx cy w h" normalised (PRIMARY labels)
  views/<name>/                  alternative label sets for the same images:
    images/{train,val}/          one relative symlink (or copy) per image
    labels/{train,val}/          card: whole-card AABBs; obb: "cls x1 y1 x2 y2 x3 y3 x4 y4"
    data.yaml
  crops/                         synthetic only (or cut by train.py --crops-from)
    {train,val}/<CANONICAL>/*.jpg  square corner crops, 96 px, 15 % padding
    labels.csv                   path,card_id,rank_idx,suit_idx
  preview/                       annotated images from --preview N
```

Synthetic sets are written with `box_semantics: corner`, `class_space: cards52`, plus
`views/card`, `views/obb` and `crops/` unless disabled.

## Outputs, exports and loading them on Android

Each training run writes `runs/vision/<name>/`:

```
runs/vision/<name>/
  data.yaml, args.yaml           merged dataset file and the ultralytics arguments
  weights/best.pt, last.pt       checkpoints
  results.csv                    per-epoch metrics
  val/                           ultralytics validation outputs
  views/                         materialised views (detector/localizer/obb only)
  export/                        onnx / tflite exports when --export was active
  summary.json                   task, datasets, metrics (train_final + val mAP50 / mAP50-95 / precision / recall), artefacts, config
runs/vision/<name>/              for "classifier"
  weights/best.pt, last.pt       CardClassifier.save payload: {format, config, state_dict, class_names, ranks, suits, extra}
  results.csv                    epoch,lr,train_loss,val_loss,rank_acc,suit_acc,card_acc,time_s
  export/card_classifier.onnx    + card_classifier.json (input contract)
  summary.json
runs/vision/<name>/              for "all"
  detector/ ... classifier/ ... summary.json
```

**Detector (ONNX).** `export/*.onnx` comes from `YOLO.export(format="onnx")` at the
training `--imgsz` (`--half`, `--int8`, `--nms` via `vision.model`). Feed it as with any
ultralytics export: RGB, float32 scaled to 0-1, NCHW, letterboxed to `imgsz`. Without `--nms`
the output is the raw ultralytics head tensor and you run NMS on-device; with `--nms` boxes
come out post-processed. Class index `k` in the output *is* engine card id `k` for a
`cards52` model (the class list is written to the run's `data.yaml` and `summary.json`
`class_names`); ids 52..55 are the extras in an `all`-space model. Run it with ONNX Runtime
for Android, or convert to TFLite below.

**Detector (TFLite / LiteRT).** `tflite` is in the default `--export` list but needs the
TensorFlow / LiteRT stack, which `vision/requirements.txt` does **not** install. Without it
the export is skipped with a logged hint and the run still succeeds (`summary.json` shows
`"tflite": null`). To enable it:

```bash
pip install "ultralytics[export]"          # or: pip install tensorflow onnx2tf   /   pip install litert-torch ai-edge-litert
python -m vision.model --task detector --weights runs/vision/detector_v1/weights/best.pt --export-detector --formats tflite --out runs/vision/detector_v1/export
```

`vision.model.export_detector` checks the required modules before calling ultralytics so its
auto-`pip install` never fires; the skip message names the packages.

**Classifier (ONNX, two-stage only).** `export/card_classifier.onnx` (opset 17, dynamic
batch) with `card_classifier.json` next to it describing the contract:

* input `images`: NCHW float32 `[-1, 3, 96, 96]`, RGB, ImageNet-normalised
  (`mean 0.485/0.456/0.406`, `std 0.229/0.224/0.225`); with
  `export_classifier(..., embed_preprocess=True)` from Python the graph takes 0-1 RGB and
  normalises internally;
* the crop is a square around the localizer box with side `max(w, h) * (1 + 2 * 0.15)`,
  reflect-padded at image borders, resized with area interpolation (`vision.model.square_crop`
  does exactly this; match it on-device or accuracy drops);
* outputs `rank` `[N, 13]` (A, 2..10, J, Q, K) and `suit` `[N, 4]` (C, D, H, S);
  `card_id = argmax(suit) * 13 + argmax(rank)`.

The ONNX classifier is checked against torch with onnxruntime at export time (max abs diff
<= 1e-3 or the export raises). It can be run with ONNX Runtime for Android, or converted to
TFLite with onnx2tf if you prefer one runtime for both models.

**On-device pipeline** = the Python `CardRecognizer`: detect (or localize + classify),
de-duplicate the same card reported twice keeping the higher confidence, sort by x, map
to engine ids, feed the engine.

## Tests

```bash
source .venv-vision/bin/activate
python -m pytest vision/tests -q -m "not slow"     # offline, CPU; 167 passed, 1 skipped in ~17 s here
python -m pytest vision/tests -q -m slow           # 3 real (tiny) trainings: detector smoke CLI, localizer + obb, "all"
python -m pytest vision/tests -q                   # everything
```

Test files: `test_cards.py`, `test_labels.py`, `test_download_datasets.py` (fake raw
layouts and mocked clients; fixtures in `vision/tests/fixtures/`), `test_generate_synthetic.py`
(procedural cards and backgrounds only), `test_model.py` (yaml-built models, classifier
forward/ONNX round-trip, mocked two-stage recognizer), `test_train.py` (merge / refusal logic
plus the slow smoke trainings). Nothing in the suite needs network access or a GPU.

## Troubleshooting

**Blocked or unreachable hosts.** The downloader recognises proxy denials and timeouts and
reports `host <name> unreachable or blocked by the network/proxy policy`; the run continues
with the other datasets and exits `2`. Hosts contacted: `huggingface.co` (jackfurby),
`api.roboflow.com` (Roboflow sets), `www.kaggle.com` (andy8744), `www.robots.ox.ac.uk`
(DTD), `raw.githubusercontent.com` (card art), and `github.com` release assets for the
pretrained `yolo11*.pt` (fetched by ultralytics on first use). Use `--dry-run` to see the
plan without touching the network. Everything downstream works without any of them:
`generate_synthetic --no-fetch` draws cards procedurally and uses procedural backgrounds,
and `train.py --smoke` or `--model yolo11n.yaml` / `--no-pretrained` builds models from yaml
instead of downloading weights. Never disable TLS verification to get around a proxy.

**Missing credentials.** A dataset that needs `ROBOFLOW_API_KEY` or Kaggle credentials is
skipped with a message naming what to set (also visible in `--dry-run` as `credentials:
MISSING`). Exit code is `0` if something else succeeded and `2` if every requested dataset was
skipped. Kaggle: set `KAGGLE_USERNAME` + `KAGGLE_KEY`, or `KAGGLE_API_TOKEN`, or place
`kaggle.json` in `~/.kaggle` (or `$KAGGLE_CONFIG_DIR`).

**Symlinks on Windows.** Converted datasets, views and materialised training views are
relative symlinks by default. All linking goes through `labels.link_or_copy`, which falls
back to copying when a symlink cannot be created (no Developer Mode / privilege), so runs
still work but use more disk. To avoid the dance entirely pass `--copy` to
`download_datasets.py` and `--copy-views` to `generate_synthetic.py`. Do not replace a view's
`images/` directory with a single directory symlink: ultralytics resolves it and would train
on the primary labels.

**CPU-only machines.** Everything runs on CPU, slowly. Use `--smoke` to validate a pipeline,
then `--workers 0` (or a small number), `--batch 8`, `--imgsz 320..416` and `--fraction 0.1`
for longer CPU runs; `--device cpu` is implied when CUDA is absent. The default detector
(`yolo11n`) is the smallest scale; the classifier's `mobilenet_v3_small` trains in minutes on
CPU. `generate_synthetic --workers 0` uses every core.

**`generate_synthetic` refuses to run.** The `--out` directory already holds outputs of a
previous run; pass `--force` to replace them or choose another `--out`. Mixing runs with
different `--num` / `--seed` in one directory would corrupt the manifest and `crops/labels.csv`.

**`train.py` refuses a dataset.** The error lists each dataset with its manifest
`box_semantics` / `class_space` versus what the task wants. Either pick datasets with the same
semantics, switch `--box-semantics`, or (for corner-labelled sets used as `card`) rely on the
dataset's `views/card`. `--crops-from` accepts corner-semantics datasets only.

**`tflite` export "skipped: missing python module(s)".** Expected without the TensorFlow /
LiteRT stack; see the export section above. ONNX exports are unaffected.
