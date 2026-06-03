import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from geotraj_lc.config import Config, get_data_splits
from geotraj_lc.trajectory.io import load_lane_change_gt
from geotraj_lc.trajectory.feature_extractor import FeatureExtractor


def load_or_generate_splits(splits_path: Path, layout, config: Config):
    if splits_path.exists():
        print(f"Loading dataset splits from: {splits_path}")
        with open(splits_path, "rb") as f:
            splits = pickle.load(f)
        print(f"Loaded: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
        return splits["train"], splits["val"], splits["test"]

    print("Dataset splits not found, generating...")
    train_seqs, val_seqs, test_seqs = get_data_splits(layout, config)

    splits = {
        "train": train_seqs,
        "val": val_seqs,
        "test": test_seqs,
    }
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_path, "wb") as f:
        pickle.dump(splits, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved dataset splits to: {splits_path}")

    return train_seqs, val_seqs, test_seqs


def build_split_samples(extractor: FeatureExtractor, sequences, layout, split_name, cache_dir=None, use_cache=True, force_rebuild=False):
    cache_path = None
    if cache_dir is not None:
        seq_key = "|".join(sorted(sequences))
        key_src = f"v2_feature12|{split_name}|{seq_key}"
        cache_name = f"{split_name}_{hashlib.sha1(key_src.encode('utf-8')).hexdigest()[:16]}.pkl"
        cache_path = cache_dir / cache_name

    if use_cache and cache_path is not None and cache_path.exists() and not force_rebuild:
        print(f"Loading cached {split_name} samples from: {cache_path}")
        with open(cache_path, "rb") as f:
            cache_data = pickle.load(f)
        return cache_data["X_list"], cache_data["y_list"], cache_data["meta_list"], cache_data["gt_df_all"]

    X_list = []
    y_list = []
    meta_list = []
    gt_parts = []

    for seq in sequences:
        print(f"Processing {split_name} sequence: {seq}")
        paths = layout.sequence_paths(seq)
        gt_df = load_lane_change_gt(paths.lane_change_gt)
        gt_parts.append(gt_df[["frame_id", "track_id", "gt_state"]].copy())

        track_feats = extractor.extract_track_features(seq, layout)
        print(f"  Found {len(track_feats)} tracks")

        for _, frames in track_feats.items():
            if len(frames) < 5:
                continue

            feat_arr = np.zeros((len(frames), extractor.feature_dim), dtype=np.float32)
            label_arr = np.zeros(len(frames), dtype=np.int64)
            meta_arr = np.zeros((len(frames), 2), dtype=np.int64)

            for t, f in enumerate(frames):
                feat_arr[t] = [
                    f["filtered_d"],
                    f["filtered_vd"],
                    f["noise"],
                    f["lane_id"],
                    f["rx"],
                    f["ry"],
                    f["relative_frame"],
                    f.get("cum_d_change", 0.0),
                    f.get("vd_sign_consistency", 0.0),
                    f.get("dist_left_boundary", 0.0),
                    f.get("dist_right_boundary", 0.0),
                    f.get("heading_dev", 0.0),
                ]
                label_arr[t] = 1 if f["gt_state"] in (1, 2) else 0
                meta_arr[t, 0] = int(f["frame_id"])
                meta_arr[t, 1] = int(f["track_id"])

            X_list.append(feat_arr)
            y_list.append(label_arr)
            meta_list.append(meta_arr)

    gt_df_all = (
        pd.concat(gt_parts, ignore_index=True)
        if gt_parts
        else pd.DataFrame(columns=["frame_id", "track_id", "gt_state"])
    )

    if use_cache and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(
                {"X_list": X_list, "y_list": y_list, "meta_list": meta_list, "gt_df_all": gt_df_all},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"Saved cached {split_name} samples to: {cache_path}")

    return X_list, y_list, meta_list, gt_df_all

