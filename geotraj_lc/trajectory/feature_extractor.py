import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque

from geotraj_lc.config import Config
from geotraj_lc.geometry.geometry import GeometryEngine
from geotraj_lc.trajectory.kalman import LateralKalmanFilter
from .io import load_corrected_tracks, load_lane_change_gt


def _wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


class _TrackState:
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.initial_lane_id: int = -1
        self.kf: Optional[LateralKalmanFilter] = None
        self.frames: List[Dict] = []
        self.prev_rx: Optional[float] = None
        self.prev_ry: Optional[float] = None
        self.d_window = deque(maxlen=10)
        self.vd_sign_window = deque(maxlen=10)


class FeatureExtractor:
    def __init__(self, config: Config):
        self.config = config
        self.feature_dim = config.tcn_feature_dim

    def _build_feature_vector(self, f: Dict) -> List[float]:
        return [
            f.get("filtered_d", 0.0),
            f.get("filtered_vd", 0.0),
            f.get("noise", 0.0),
            f.get("lane_id", -1),
            f.get("rx", 0.0),
            f.get("ry", 0.0),
            f.get("relative_frame", 0.0),
            f.get("cum_d_change", 0.0),
            f.get("vd_sign_consistency", 0.0),
            f.get("dist_left_boundary", 0.0),
            f.get("dist_right_boundary", 0.0),
            f.get("heading_dev", 0.0),
        ]

    def extract_track_features(self, sequence: str, layout) -> Dict[int, List[Dict]]:
        paths = layout.sequence_paths(sequence)
        corrected_df = load_corrected_tracks(paths.corrected_tracks)
        gt_df = load_lane_change_gt(paths.lane_change_gt)

        scale = self.config.scale_for_sequence(sequence)
        geometry = GeometryEngine(paths.homography, paths.centerline, scale=scale)

        tracks: Dict[int, _TrackState] = {}

        grouped = corrected_df.groupby("frame_id")
        all_frames = sorted(grouped.groups.keys())
        first_frame_id = min(all_frames) if all_frames else 0

        for frame_id in all_frames:
            frame_data = grouped.get_group(frame_id)
            for row in frame_data.itertuples(index=False):
                track_id = int(row.track_id)
                lane_id = int(row.lane_id)

                if track_id not in tracks:
                    tracks[track_id] = _TrackState(track_id)
                state = tracks[track_id]

                gt_state = 0
                gt_match = gt_df[(gt_df["frame_id"] == frame_id) & (gt_df["track_id"] == track_id)]
                if not gt_match.empty:
                    gt_state = int(gt_match.iloc[0]["gt_state"])

                if lane_id != -1:
                    cx = float(row.x1) + float(row.w) / 2.0
                    cy = float(row.y1) + float(row.h)
                    rx, ry = geometry.pixel_to_metric(cx, cy)

                    if state.initial_lane_id == -1:
                        state.initial_lane_id = lane_id

                    raw_d = geometry.get_frenet_d_from_ref_lane(rx, ry, state.initial_lane_id)
                    filtered_d = None
                    filtered_vd = None

                    if raw_d is not None:
                        if state.kf is None:
                            state.kf = LateralKalmanFilter(
                                init_d=raw_d,
                                dt=self.config.dt,
                                process_noise_q=self.config.process_noise_q,
                                measurement_noise_r=self.config.measurement_noise_r,
                            )
                        state.kf.predict()
                        filtered_d, filtered_vd = state.kf.update(raw_d)

                    if filtered_d is not None and filtered_vd is not None:
                        state.d_window.append(float(filtered_d))
                        cum_d_change = float(state.d_window[-1] - state.d_window[0]) if len(state.d_window) > 1 else 0.0

                        vd_sign = 0
                        if filtered_vd > 1e-4:
                            vd_sign = 1
                        elif filtered_vd < -1e-4:
                            vd_sign = -1
                        state.vd_sign_window.append(vd_sign)

                        non_zero_signs = [s for s in state.vd_sign_window if s != 0]
                        if non_zero_signs:
                            pos_ratio = sum(1 for s in non_zero_signs if s > 0) / float(len(non_zero_signs))
                            neg_ratio = sum(1 for s in non_zero_signs if s < 0) / float(len(non_zero_signs))
                            vd_sign_consistency = float(max(pos_ratio, neg_ratio))
                        else:
                            vd_sign_consistency = 0.0

                        dist_left_boundary, dist_right_boundary = geometry.get_lane_boundary_distances(rx, ry, lane_id)

                        heading_dev = 0.0
                        lane_heading = geometry.get_lane_tangent_angle(ry, lane_id)
                        if state.prev_rx is not None and state.prev_ry is not None and lane_heading is not None:
                            dx = rx - state.prev_rx
                            dy = ry - state.prev_ry
                            if abs(dx) + abs(dy) > 1e-6:
                                vehicle_heading = float(np.arctan2(dx, dy))
                                heading_dev = _wrap_angle(vehicle_heading - lane_heading)

                        frame_feature = {
                            "frame_id": int(frame_id),
                            "track_id": int(track_id),
                            "filtered_d": filtered_d,
                            "filtered_vd": filtered_vd,
                            "noise": filtered_d - raw_d,
                            "lane_id": lane_id,
                            "rx": rx,
                            "ry": ry,
                            "relative_frame": frame_id - first_frame_id,
                            "cum_d_change": cum_d_change,
                            "vd_sign_consistency": vd_sign_consistency,
                            "dist_left_boundary": dist_left_boundary,
                            "dist_right_boundary": dist_right_boundary,
                            "heading_dev": heading_dev,
                            "gt_state": gt_state,
                        }
                        state.frames.append(frame_feature)
                        state.prev_rx = rx
                        state.prev_ry = ry
                else:
                    state.frames.append({
                        "frame_id": int(frame_id),
                        "track_id": int(track_id),
                        "filtered_d": 0.0,
                        "filtered_vd": 0.0,
                        "noise": 0.0,
                        "lane_id": -1,
                        "rx": 0.0,
                        "ry": 0.0,
                        "relative_frame": frame_id - first_frame_id,
                        "cum_d_change": 0.0,
                        "vd_sign_consistency": 0.0,
                        "dist_left_boundary": 0.0,
                        "dist_right_boundary": 0.0,
                        "heading_dev": 0.0,
                        "gt_state": gt_state,
                    })

        return {tid: t.frames for tid, t in tracks.items()}

