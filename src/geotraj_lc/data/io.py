from pathlib import Path

import pandas as pd


DETECTION_COLUMNS = [
    "frame_id",
    "track_id",
    "x1",
    "y1",
    "w",
    "h",
    "score",
    "cls",
    "lane_id",
]

GT_COLUMNS = DETECTION_COLUMNS + ["gt_state"]


def load_corrected_tracks(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None).iloc[:, :9]
    df.columns = DETECTION_COLUMNS
    return df


def load_lane_change_gt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None).iloc[:, :10]
    df.columns = GT_COLUMNS
    return df

