# GeoTraj-LC

GeoTraj-LC is a geometry-rectified trajectory framework for online lane-change
event detection in fixed roadside highway surveillance videos.

The project is built for long-duration monitoring from oblique fixed cameras.
It uses MOT-style vehicle tracks with lane IDs, maps vehicle positions from the
image plane to a calibrated road plane, builds lane-referenced lateral-motion
features, smooths the lateral state with a Kalman filter, and detects
lane-change intervals with a lightweight causal TCN and event decoder.

In the paper, GeoTraj-LC is evaluated on the RLC Dataset and reports event-level
precision 0.86, recall 0.93, F1 0.90, and average detection latency 0.48 s under
20 fps roadside surveillance streams.

## What This Repository Provides

- RLC Dataset files used by this implementation: tracks, frame labels, maps,
  homography matrices, scale factors, and a sample video.
- Trajectory-based lane-change detection from corrected MOT-style tracks.
- Road-plane geometric projection and lane-centerline lateral-offset features.
- Temporal smoothing of lateral offset and lateral velocity.
- Causal TCN training and event-level evaluation.
- Optional detection-box anchor correction with an OffsetRegressor.

This repository expects detection and tracking results as input. It does not
currently provide a complete YOLO/ByteTrack video-to-track pipeline. Detection
and tracking are upstream stages; GeoTraj-LC consumes their MOT-style outputs
with lane IDs.

## Repository Layout

```text
GeoTraj-LC/
|-- geotraj_lc/
|   |-- anchor_correction/   # bottom-anchor regression and detection correction
|   |-- geometry/            # homography and lane-centerline geometry
|   |-- trajectory/          # data loading, layout, feature extraction, normalizer
|   |-- tcn/                 # TCN model, sample generation, TCN training
|   |-- event_decoding/      # online frame-to-event decoding
|   |-- evaluation/          # event matching and metrics
|   |-- config.py            # default FPS, scales, split and model parameters
|   |-- evaluate.py          # implementation behind run.py
|   `-- train.py             # implementation behind train.py
|-- RLC_Dataset/             # dataset files and maps
|-- configs/                 # paper-facing configuration snapshot
|-- docs/                    # dataset and reproducibility notes
|-- outputs/                 # generated cache and run outputs
|-- pretrained/              # place trained model weights here
|-- run.py                   # lane-change evaluation entry
|-- train.py                 # training entry
|-- correct_dets.py          # detection-box correction entry
|-- requirements.txt
`-- pyproject.toml
```

Generated files under `outputs/` and large model artifacts under
`pretrained/` are ignored by Git by default.

## Environment

The project was developed with Python, PyTorch, OpenCV, NumPy, SciPy,
scikit-learn, pandas, Pillow and tqdm. Install dependencies with:

```powershell
D:\pycharmproj\roslearn\.venv\Scripts\python.exe -m pip install -r requirements.txt
D:\pycharmproj\roslearn\.venv\Scripts\python.exe -m pip install -e .
```

If you use another Python environment, replace the interpreter path in the
commands below.

## Dataset

The expected dataset root is:

```text
RLC_Dataset/
|-- annotations/
|   |-- frame_labels/*.txt
|   `-- events/lane_change_events.csv
|-- tracks/
|   |-- mot_raw/*.txt
|   `-- refined/*.txt
|-- maps/
|   |-- centerlines/{sequence}.json
|   |-- lane_boundaries/{sequence}.json
|   |-- homography/{sequence}.npy
|   `-- scale_factors.json
|-- localization_anchors/
`-- videos/sample.mp4
```

Track files use MOT-style columns plus lane ID:

```text
frame_id, track_id, x1, y1, w, h, score, cls, lane_id
```

Frame-label files add lane-change state and evaluation tag:

```text
frame_id, track_id, x1, y1, w, h, score, cls, lane_id, gt_state, eval_tag
```

`gt_state` is `0` for lane keeping, `1` for left lane change, and `2` for right
lane change. The current TCN pipeline trains a binary detector, mapping both
left and right lane changes to one lane-change class for event detection.

More details are in [docs/DATASET.md](docs/DATASET.md).

## Data Split

The split is scene-based, not random over trajectories:

| Split | Scene prefixes |
| --- | --- |
| train | G4202-K32, S2-K188_0, S2-K188_1, S3-K316 |
| val | S2-K152, S2-K201 |
| test | G0512-K111, G4215-K12 |

This avoids training and testing on clips from the same fixed camera scene.

## Run Lane-Change Evaluation

Evaluation requires trained TCN files:

```text
pretrained/best/tcn_model.pth
pretrained/best/normalizer.pkl
```

Run the default test split:

```powershell
D:\pycharmproj\roslearn\.venv\Scripts\python.exe run.py --split test
```

Run one or more clips:

```powershell
D:\pycharmproj\roslearn\.venv\Scripts\python.exe run.py --sequences G0512-K111_clip_0000 --no-progress
```

Available split choices are `train`, `val`, `test`, and `all`.

## Train the TCN Lane-Change Detector

Train with the default scene split and parameters in
[geotraj_lc/config.py](geotraj_lc/config.py):

```powershell
D:\pycharmproj\roslearn\.venv\Scripts\python.exe train.py tcn
```

Training writes generated feature caches to `outputs/cache/` and saves the best
TCN checkpoint and feature normalizer to `pretrained/best/`.

To rebuild cached features:

```powershell
D:\pycharmproj\roslearn\.venv\Scripts\python.exe train.py tcn --rebuild-cache
```

To change how often event-level validation is run during training:

```powershell
D:\pycharmproj\roslearn\.venv\Scripts\python.exe train.py tcn --train-event-eval-interval 5
```

## Detection-Box Anchor Correction

The paper uses an OffsetRegressor to correct the bottom anchors of detection
boxes before road-plane projection. This repository includes the correction
model and utilities, but the correction stage needs:

- a correction config JSON containing video source, detection path, crop
  channels, output path, and model path;
- trained regressor weights, usually `pretrained/anchor_correction/regressor_best.pth`;
- input detections in MOT-style text files.

Run correction:

```powershell
D:\pycharmproj\roslearn\.venv\Scripts\python.exe correct_dets.py --config path\to\correct_dets.json --model pretrained\anchor_correction\regressor_best.pth --output-dir outputs\corrected_dets
```

Process selected sequences only:

```powershell
D:\pycharmproj\roslearn\.venv\Scripts\python.exe correct_dets.py --config path\to\correct_dets.json --seqs G4202-K32_clip_0000
```

## Train the Anchor Regressor

The regressor trainer expects an exported patch dataset:

```text
exported/train/
|-- images/*.jpg or *.png
`-- labels/*.txt
```

Each label file should contain normalized bottom-anchor coordinates:

```text
left_x left_y right_x right_y [optional_side_label]
```

Train the regressor:

```powershell
D:\pycharmproj\roslearn\.venv\Scripts\python.exe train.py regressor --data-dir exported/train --save-dir pretrained/anchor_correction
```

Resume from existing weights:

```powershell
D:\pycharmproj\roslearn\.venv\Scripts\python.exe train.py regressor --data-dir exported/train --save-dir pretrained/anchor_correction --resume pretrained\anchor_correction\regressor_best.pth
```

## Metrics

The evaluation is event-level. Predicted lane-change intervals are matched to
ground-truth intervals using temporal IoU. The reported metrics include:

- precision, recall and F1;
- mean temporal IoU;
- mean and standard deviation of detection latency;
- per-scene and overall summaries.

Left and right lane changes are treated as one lane-change event class in the
current binary TCN detector.

## Reproducibility

The paper-facing parameter snapshot is [configs/default.yaml](configs/default.yaml).
The active implementation defaults are in [geotraj_lc/config.py](geotraj_lc/config.py).

Recommended checks:

```powershell
D:\pycharmproj\roslearn\.venv\Scripts\python.exe -m compileall geotraj_lc run.py train.py correct_dets.py
D:\pycharmproj\roslearn\.venv\Scripts\python.exe run.py --help
D:\pycharmproj\roslearn\.venv\Scripts\python.exe train.py --help
D:\pycharmproj\roslearn\.venv\Scripts\python.exe correct_dets.py --help
```

For experiment reporting, record the Git commit, Python/PyTorch/CUDA versions,
GPU model, split protocol, model weights, and whether incomplete lane-change
events are included.
