# GeoTraj-LC

GeoTraj-LC is a geometry-rectified trajectory framework for online lane-change
event detection in fixed roadside highway surveillance videos.

The code takes MOT-style vehicle tracks with lane IDs as input, projects vehicle
positions onto the road plane, builds lane-referenced lateral-motion features,
and detects lane-change events with a causal TCN decoder.

## Repository Layout

```text
GeoTraj-LC/
|-- geotraj_lc/              # core implementation
|-- RLC_Dataset/             # tracks, annotations, maps and sample video
|-- configs/                 # configuration snapshot
|-- docs/                    # dataset and reproducibility notes
|-- outputs/                 # generated outputs, ignored by Git
|-- pretrained/              # model weights, ignored by Git
|-- run.py                   # evaluation entry
|-- train.py                 # training entry
|-- correct_dets.py          # optional detection-box correction entry
|-- requirements.txt
`-- pyproject.toml
```

## Installation

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Use a virtual environment. The commands below assume `python` points to that
environment.

## Dataset

The expected dataset root is `RLC_Dataset/`. It contains MOT-style refined
tracks, frame-level annotations, lane maps, homography matrices, scale factors
and a sample video.

Track files are expected under `RLC_Dataset/tracks/refined/`. Frame labels for
training are expected under `RLC_Dataset/annotations/frame_labels/`.

More details are in [docs/DATASET.md](docs/DATASET.md).

## Split

The default split is scene-based:

| Split | Scene prefixes |
| --- | --- |
| train | G4202-K32, S2-K188_0, S2-K188_1, S3-K316 |
| val | G4215-K12, S2-K201 |
| test | G0512-K111, S2-K152 |

## Evaluation

Evaluation requires:

```text
pretrained/best/tcn_model.pth
pretrained/best/normalizer.pkl
```

Run the default test split:

```bash
python run.py --split test
```

Run selected clips:

```bash
python run.py --sequences G0512-K111_clip_0000 --no-progress
```

## Training

Train the TCN detector:

```bash
python train.py tcn
```

Rebuild cached features before training:

```bash
python train.py tcn --rebuild-cache
```

Training saves the best checkpoint and feature normalizer to `pretrained/best/`.

## Optional Detection-Box Correction

`correct_dets.py` applies the optional bottom-anchor correction stage used
before road-plane projection. It requires a correction config, input detections
and trained regressor weights.

```bash
python correct_dets.py --config path/to/correct_dets.json --model pretrained/anchor_correction/regressor_best.pth --output-dir outputs/corrected_dets
```

## Metrics

Evaluation is event-level. Predicted and ground-truth lane-change intervals are
matched by temporal overlap, and the reported metrics include precision, recall,
F1, temporal IoU and detection latency.

## Reproducibility

Default parameters are in [geotraj_lc/config.py](geotraj_lc/config.py), with a
configuration snapshot in [configs/default.yaml](configs/default.yaml).

Additional reproduction notes are in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).
