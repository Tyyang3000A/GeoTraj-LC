# GeoTraj-LC: Road Contact Anchor Correction and Trajectory Modeling Relative to Lane Geometry for Lane Change Event Detection in Fixed Roadside Video

<p align="center">
  <a href="#overview">Overview</a> |
  <a href="#dataset">Dataset</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#pipeline">Pipeline</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.9%2B-red?style=flat-square&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/Dataset-RLC-yellow?style=flat-square" alt="Dataset">
</p>

## Overview

Lane-change detection from long-duration fixed roadside surveillance videos is
challenging because oblique camera views make generic detector boxes
inconsistent with the actual road contact positions needed for lateral motion
reasoning. GeoTraj-LC improves this localization anchor problem by correcting
vehicle bottom anchors, mapping trajectories onto the calibrated road plane, and
modeling lane-referenced lateral motion for event-level lane-change detection.
The repository also releases the RLC Dataset, which, to the best of our
knowledge, is the first lane-change detection dataset built for highway fixed
roadside surveillance views.

## Key Features

- Geometry-aware trajectory representation using homography-based road-plane
  projection.
- Lane-centered lateral displacement and lateral velocity features.
- Causal TCN event decoder for online lane-change recognition.
- Scene-level train/validation/test split to avoid camera-scene leakage.
- Event-level evaluation protocol with overlap matching and latency reporting.
- Optional bottom-anchor correction module for refined vehicle positions.

## Pipeline

GeoTraj-LC starts from MOT-style tracks with refined lane IDs, corrects vehicle
bottom anchors, projects trajectories to the road plane, and constructs
lane-referenced lateral motion features. Smoothed trajectory states are then
processed by a causal TCN and decoded into lane-change event intervals.

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

**RLC Dataset Access:** To request access to the RLC Dataset, please complete
the following form:

- [Download RLC Video Data](https://docs.google.com/forms/d/e/1FAIpQLSe82Kc17H0slxKrjgtgakVpVSJeijC4zTFzkqn-LZr5lJvamA/viewform)

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

Representative lane-change detection cases from the RLC Dataset are shown
below.

<p align="center">
  <img src="docs/assets/lc_case.png" width="900" alt="Representative lane-change detection cases">
</p>

Additional generated visualizations can be stored in `outputs/visualizations/`.

## Configuration

Default parameters are implemented in [geotraj_lc/config.py](geotraj_lc/config.py)
and mirrored by the snapshot in [configs/default.yaml](configs/default.yaml).
The default frame rate is 20 FPS. Sequence-specific scale factors convert
homography road-plane units to meters.

## Contact

For questions about the code or data format, please open an issue in this
repository.

For video data access requests, please use the request form above.

Email: corfyi@csust.edu.cn

## Acknowledgements

This repository builds on standard open-source scientific Python and PyTorch
components. We thank the contributors and maintainers of these ecosystems.
