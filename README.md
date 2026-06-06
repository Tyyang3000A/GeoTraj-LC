# GeoTraj-LC: Geometry Corrected Trajectory Modeling for Lane Change Detection in Fixed Roadside Surveillance Videos

## Overview

Lane-change detection from long-duration fixed roadside surveillance videos is
challenging because oblique camera views make image-plane detection boxes
inconsistent with the road contact positions needed for lateral motion
reasoning. GeoTraj-LC takes MOT-style vehicle tracks with refined lane IDs,
lane geometry, homography matrices, and sequence-specific scale factors as
input, and outputs event-level lane-change intervals together with precision,
recall, F1, temporal IoU, and latency metrics. The core workflow corrects
vehicle bottom anchors, projects trajectories onto the calibrated road plane,
constructs lane-reference lateral motion features, smooths lateral states with a
Kalman filter, and decodes frame probabilities into lane-change events with a
causal Temporal Convolutional Network (TCN). The repository also releases the
RLC Dataset, a highway lane-change event dataset built from fixed roadside
surveillance videos; in the associated manuscript, GeoTraj-LC achieves 0.86
precision, 0.93 recall, 0.90 F1, and 0.48 s average detection latency.

## Key Features

- Geometry-aware trajectory representation using homography-based road-plane
  projection.
- Lane-centered lateral displacement and lateral velocity features.
- Causal TCN event decoder for online lane-change recognition.
- Scene-level train/validation/test split to avoid camera-scene leakage.
- Event-level evaluation protocol with overlap matching and latency reporting.
- Optional bottom-anchor correction module for refined vehicle positions.

## Dataset

The expected dataset root is `RLC_Dataset/`.

```text
RLC_Dataset/
├── annotations/
│   ├── frame_labels/            # frame-level lane-change labels
│   └── events/                  # event-level annotations, if available
├── maps/
│   ├── centerlines/             # lane centerline geometry
│   ├── lane_boundaries/         # lane boundary geometry
│   ├── homography/              # image-to-road homography matrices
│   └── scale_factors.json       # scene-specific metric scale factors
├── tracks/
│   └── refined/                 # MOT tracks with refined lane IDs
└── videos/                      # source or sample videos
```

Input track files use MOT-style rows with one additional lane ID column:

```text
frame_id, track_id, x1, y1, w, h, score, cls, lane_id
```

Training labels add the frame-level driving state:

```text
frame_id, track_id, x1, y1, w, h, score, cls, lane_id, gt_state, eval_tag
```

`gt_state` is encoded as `0` for lane keeping, `1` for left lane change, and
`2` for right lane change. Complete and incomplete lane-change events are both
present in the annotations; incomplete events may start before entering the
camera view or continue after leaving it.

More details are provided in [docs/DATASET.md](docs/DATASET.md).

## Environment

GeoTraj-LC requires Python 3.8 or later. Install dependencies in a virtual
environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

The main dependencies include PyTorch, NumPy, pandas, OpenCV, SciPy,
scikit-learn, tqdm, and Pillow. See [requirements.txt](requirements.txt) and
[pyproject.toml](pyproject.toml) for the complete dependency specification.

## Quick Start

Evaluate the default test split with pretrained weights:

```bash
python run.py --split test
```

Evaluate selected clips:

```bash
python run.py --sequences G0512-K111_clip_0000 --no-progress
```

The default evaluation expects the following files:

```text
pretrained/best/tcn_model.pth
pretrained/best/normalizer.pkl
```

## Training

Train the TCN lane-change detector:

```bash
python train.py tcn
```

Rebuild cached trajectory features before training:

```bash
python train.py tcn --rebuild-cache
```

Training saves the best checkpoint and feature normalizer to
`pretrained/best/`.

## Detection-Box Correction

The optional correction stage refines the detection-box bottom anchor before
road-plane projection. It requires a correction config, input detections, and a
trained regressor:

```bash
python correct_dets.py --config path/to/correct_dets.json --model pretrained/anchor_correction/regressor_best.pth --output-dir outputs/corrected_dets
```

## Evaluation Protocol

Evaluation is performed at the event level. Predicted lane-change intervals are
matched to ground-truth intervals by temporal overlap. Reported metrics include:

- Precision, recall, and F1.
- Mean temporal IoU of matched events.
- Detection latency relative to the annotated event start.
- Per-scene and overall summaries.

For paper experiments, record the git commit, environment versions, split
definition, configuration file, and whether incomplete lane changes are included
in the reported numbers. Additional guidance is available in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Qualitative Visualization

For a paper-facing release, qualitative visualization is recommended. Useful
figures include:

- Projected road-plane trajectories over lane centerlines and boundaries.
- Lateral displacement `d` and lateral velocity `v` curves with predicted and
  annotated lane-change intervals.
- Video-frame overlays showing the event start, target-lane entry, and final
  matched interval.

No generated visualization assets are committed by default. Recommended
locations are `docs/assets/` for paper figures and `outputs/visualizations/` for
generated qualitative results.

## Configuration

Default parameters are implemented in [geotraj_lc/config.py](geotraj_lc/config.py)
and mirrored by the snapshot in [configs/default.yaml](configs/default.yaml).
The default frame rate is 20 FPS. Sequence-specific scale factors convert
homography road-plane units to meters.

## Citation

The manuscript associated with this repository is currently under review. Until
an accepted or public preprint version is available, cite it as a submitted
manuscript and update the entry after publication:

```bibtex
@unpublished{geotrajlc2026,
  title  = {GeoTraj-LC: Geometry-Rectified Trajectory Learning for Lane-Change Detection},
  author = {{GeoTraj-LC Authors}},
  year   = {2026},
  note   = {Manuscript submitted for publication}
}
```

## Acknowledgements

This repository builds on standard open-source scientific Python and PyTorch
components. We thank the contributors and maintainers of these ecosystems.
