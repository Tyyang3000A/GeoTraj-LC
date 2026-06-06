# GeoTraj-LC

GeoTraj-LC is a geometry-rectified trajectory framework for lane-change event
detection in fixed-view highway surveillance videos. The project is designed as
the reference implementation for a paper submission and provides the data
layout, training entry points, evaluation protocol, and reproducibility notes
needed to benchmark lane-change detection from multi-object tracking results.

## Overview

The pipeline starts from MOT-style vehicle tracks with refined lane IDs. Vehicle
positions are represented by the bottom-center point of each corrected detection
box, projected from the image plane to the road plane with sequence-specific
homography matrices, and normalized to metric coordinates with per-scene scale
factors. GeoTraj-LC then computes lane-referenced lateral motion features,
smooths lateral position and velocity over time, and decodes online
lane-change events with a causal Temporal Convolutional Network (TCN).

A lane-change event is evaluated as a temporal interval: it starts when the
vehicle begins crossing the lane boundary and ends when the vehicle fully
enters the target lane. Predictions are matched to annotated events by temporal
overlap and summarized with event-level precision, recall, F1, temporal IoU,
and detection latency.

## Key Features

- Geometry-aware trajectory representation using homography-based road-plane
  projection.
- Lane-centered lateral displacement and lateral velocity features.
- Causal TCN event decoder for online lane-change recognition.
- Scene-level train/validation/test split to avoid camera-scene leakage.
- Event-level evaluation protocol with overlap matching and latency reporting.
- Optional bottom-anchor correction module for refined vehicle positions.

## Repository Layout

```text
GeoTraj-LC/
|-- geotraj_lc/              # core implementation
|-- RLC_Dataset/             # tracks, annotations, maps and sample video
|-- configs/                 # configuration snapshot
|-- docs/                    # dataset and reproducibility notes
|-- outputs/                 # generated outputs, ignored by Git
|-- pretrained/              # model weights, ignored by Git
|-- correct_dets.py          # optional detection-box correction entry
|-- run.py                   # evaluation entry
|-- train.py                 # training entry
|-- requirements.txt
`-- pyproject.toml
```

## Dataset

The expected dataset root is `RLC_Dataset/`.

```text
RLC_Dataset/
|-- annotations/frame_labels/    # frame-level lane-change labels
|-- annotations/events/          # event-level annotations, if available
|-- maps/centerlines/            # lane centerline geometry
|-- maps/lane_boundaries/        # lane boundary geometry
|-- maps/homography/             # image-to-road homography matrices
|-- maps/scale_factors.json      # scene-specific metric scale factors
|-- tracks/refined/              # MOT tracks with refined lane IDs
`-- videos/                      # source or sample videos
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

## Data Split

The default protocol splits data by camera scene rather than by individual
tracks:

| Split | Scene prefixes | Clips | Tracks | Left LC | Right LC | Total LC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | G4202-K32, S2-K188_0, S2-K188_1, S3-K316 | 38 | 9,102 | 482 | 308 | 790 |
| val | G4215-K12, S2-K201 | 17 | 3,126 | 98 | 137 | 235 |
| test | G0512-K111, S2-K152 | 21 | 10,509 | 389 | 351 | 740 |
| total | 8 scenes | 76 | 22,737 | 969 | 796 | 1,765 |

This protocol evaluates generalization to unseen fixed-camera scenes.

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

If you use this code or dataset in your research, please cite the corresponding
paper. The BibTeX entry will be added after publication.

```bibtex
@misc{geotrajlc2026,
  title  = {GeoTraj-LC: Geometry-Rectified Trajectory Learning for Lane-Change Detection},
  author = {GeoTraj-LC Contributors},
  year   = {2026},
  note   = {Code repository}
}
```

## License

The source code is released under the MIT License. See [LICENSE](LICENSE).

Dataset files, videos, annotations, and map assets may be subject to additional
terms. Please check the dataset release notes before redistribution or external
use.

## Acknowledgements

This repository builds on standard open-source scientific Python and PyTorch
components. We thank the contributors and maintainers of these ecosystems.
