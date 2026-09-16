Small offline fixtures for `vision/tests/test_download_datasets.py`.

* `yolo_english/`  roboflow-style `data.yaml` with english compact names (+ two non-card classes)
* `yolo_pcc/`      pcc3.0-style `data.yaml` with the 55 Dutch class names
* `yolo_parts/`    `classes.txt` only (rank / suit annotated separately, no data.yaml)
* `jackfurby/`     `train.json` / `val.json` in the JackFurby generator format
                   (`card_points: [[[TL, TR, BL, BR], label], ...]`, labels per
                   `vision/configs/jackfurby_card_classes.txt`); the tests pickle them
                   to `train.pkl` / `val.pkl` because `*.pkl` is git-ignored.

Images and label files are generated at test time in `tmp_path` (no binary blobs here).
