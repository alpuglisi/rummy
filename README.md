# rummy
# rummy

## Card vision pipeline

`vision/` holds the playing-card detection pipeline for the Android app: it downloads public
card datasets, generates synthetic fanned-hand / spread-pile images, and trains + exports the
corner-index detector, the two-stage localizer + rank/suit classifier and an OBB variant.
Detections come out as this engine's card ids (`suit * 13 + rank`, Ace = 0).

Setup, credentials, dataset table, the four scripts (`download_datasets`,
`generate_synthetic`, `model`, `train`), on-disk layout, exports for Android and
troubleshooting are documented in [vision/README.md](vision/README.md).

```bash
python3.11 -m venv .venv-vision && source .venv-vision/bin/activate
pip install -r vision/requirements.txt
python -m vision.train all --smoke --name smoke_all      # CPU end-to-end check, no downloads
```
