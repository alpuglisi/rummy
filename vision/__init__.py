"""Computer-vision pipeline for the 500 Rummy assistant.

Modules
-------
cards               canonical 52-card vocabulary (matches the C++ rummy engine ids)
labels              YOLO detection / OBB label IO and dataset-layout helpers
config              path layout, dataset registry, and config dataclasses
download_datasets   (A) fetch public datasets + assets and convert to the canonical layout
generate_synthetic  (B) 2-D compositing generator for fanned hands / spread & chaotic piles
model               (C) detector / localizer / OBB / rank-suit classifier factories + export
train               (D) dataset merging, training loops, evaluation and export
"""
