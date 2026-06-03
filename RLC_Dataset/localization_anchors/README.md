# Localization Anchors

Anchor-based vehicle localization annotations for lane change detection.

## Structure

- `images/train/`, `images/val/`, `images/test/` — Cropped vehicle images
- `annotations/train.csv`, `val.csv`, `test.csv`, `all.csv` — Bounding box annotations (YOLO normalized format: class, x_center, y_center, width, height)
- `splits/train.txt`, `val.txt`, `test.txt` — Sample name lists per split

## Annotation Format

Each CSV row: `image, class, x_center, y_center, width, height`
- Coordinates are normalized to [0, 1] relative to image dimensions
- Class `-1` indicates uncertain/no-anchor; `0` and `1` indicate anchor classes
