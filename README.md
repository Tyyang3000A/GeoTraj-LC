# GeoTraj-LC

Trajectory-based lane-change detection for fixed-view highway surveillance videos.

The code follows the actual algorithm pipeline: corrected detections and lane IDs
are projected to road-plane trajectories, lateral offset is computed against lane
centerlines, temporal smoothing is applied, and lane-change events are evaluated
with IoU, latency, precision, recall and F1.

## Directory Layout

```text
GeoTraj-LC/
|-- src/geotraj_lc/          # source code
|   |-- core/                # geometry, temporal filter, neural models
|   |-- data/                # file readers, path layout, feature extraction
|   |-- evaluation/          # event matching and metrics
|   |-- training/            # TCN sample generation and training
|   |-- tools/               # detection-box refinement utility
|   |-- config.py            # default parameters, scales and scene splits
|   |-- evaluate.py          # implementation behind run.py
|   `-- train.py             # implementation behind train.py
|-- RLC_Dataset/             # dataset: tracks, annotations, maps
|-- checkpoints/best/        # trained TCN weights and feature normalizer
|-- cache/                   # generated feature/split cache; safe to delete
|-- configs/                 # static paper configuration snapshot
|-- docs/                    # dataset and reproducibility notes
|-- run.py                   # evaluation entry
|-- train.py                 # training entry
|-- correct_dets.py          # detection correction entry
`-- pyproject.toml           # package metadata and console commands
```

There are only two data-like directories:

- `RLC_Dataset/`: real input data and maps.
- `cache/`: generated acceleration cache. Delete it any time; training will rebuild it.

## Install

```powershell
D:\pycharmproj\roslearn\.venv\Scripts\python.exe -m pip install -r requirements.txt
D:\pycharmproj\roslearn\.venv\Scripts\python.exe -m pip install -e .
```

## Dataset Format

Track files:

```text
RLC_Dataset/tracks/refined/*.txt
frame_id, track_id, x1, y1, w, h, score, cls, lane_id
```

Frame labels:

```text
RLC_Dataset/annotations/frame_labels/*.txt
frame_id, track_id, x1, y1, w, h, score, cls, lane_id, gt_state, eval_tag
```

Maps:

```text
RLC_Dataset/maps/centerlines/{sequence}.json
RLC_Dataset/maps/homography/{sequence}.npy
RLC_Dataset/maps/lane_boundaries/{sequence}.json
```

More detail is in [docs/DATASET.md](docs/DATASET.md).

## Run Evaluation

```powershell
D:\pycharmproj\roslearn\.venv\Scripts\python.exe run.py --split test
```

Single clip:

```powershell
D:\pycharmproj\roslearn\.venv\Scripts\python.exe run.py --sequences G4202-K32_clip_0000 --no-progress
```

## Train

```powershell
D:\pycharmproj\roslearn\.venv\Scripts\python.exe train.py tcn
```

Rebuild feature cache:

```powershell
D:\pycharmproj\roslearn\.venv\Scripts\python.exe train.py tcn --rebuild-cache
```

## Reproducibility

Default scene split:

| Split | Scene prefixes |
| --- | --- |
| train | G0512-K111, S2-K152, S2-K188_0, S2-K188_1, S2-K201 |
| val | G4215-K12, S3-K316 |
| test | G4202-K32 |

Main parameters are in [src/geotraj_lc/config.py](src/geotraj_lc/config.py).
The paper-facing snapshot is [configs/default.yaml](configs/default.yaml).

## Check

```powershell
D:\pycharmproj\roslearn\.venv\Scripts\python.exe -m compileall src run.py train.py correct_dets.py
D:\pycharmproj\roslearn\.venv\Scripts\python.exe run.py --help
```
