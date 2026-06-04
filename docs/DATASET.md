# Dataset

GeoTraj-LC expects fixed-camera highway clips with tracking results, lane IDs,
lane maps and frame-level lane-change labels.

## Input Tracks

`RLC_Dataset/tracks/refined/*.txt`

Each row uses MOT-style boxes plus the refined lane ID:

```text
frame_id, track_id, x1, y1, w, h, score, cls, lane_id
```

The detector uses the bottom-center point of each box as the vehicle trajectory
point. The point is projected to the road plane by homography and scaled to
metric coordinates with sequence-specific scale factors.

## Frame Labels

`RLC_Dataset/annotations/frame_labels/*.txt`

The first 10 columns are used:

```text
frame_id, track_id, x1, y1, w, h, score, cls, lane_id, gt_state
```

`gt_state` values:

- `0`: stable lane keeping
- `1`: left lane change
- `2`: right lane change

The current binary TCN pipeline maps `1` and `2` to the lane-change class during
sample generation and evaluates event intervals.

## Maps

```text
RLC_Dataset/maps/
|-- centerlines/{sequence}.json
|-- lane_boundaries/{sequence}.json
`-- homography/{sequence}.npy
```

The code resolves clip names such as `G4202-K32_clip_0000` to the base sequence
`G4202-K32` when loading maps and metric scale factors.

## Split Protocol

Splits are defined by scene prefix rather than random trajectory sampling:

- train: `G4202-K32`, `S2-K188_0`, `S2-K188_1`, `S3-K316`
- val: `G4215-K12`, `S2-K201`
- test: `G0512-K111`, `S2-K152`

This avoids training and testing on clips from the same camera scene.
