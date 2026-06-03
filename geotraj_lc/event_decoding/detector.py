from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from geotraj_lc.trajectory.kalman import LateralKalmanFilter


@dataclass
class _TCNTrackState:
    track_id: int
    initial_lane_id: int = -1
    kf: Optional[LateralKalmanFilter] = None
    frame_buffer: deque = None
    first_frame_id: Optional[int] = None
    prev_rx: Optional[float] = None
    prev_ry: Optional[float] = None
    d_window: deque = None
    vd_sign_window: deque = None
    tcn_trigger_counter: int = 0
    tcn_gap_frames: int = 0
    is_in_event: bool = False
    is_confirmed: bool = False


class TCNDetector:
    def __init__(self, geometry, config, model, processor):
        self.geometry = geometry
        self.config = config
        self.tracks = {}
        self.cumulative_window = 10
        self.sign_window = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.feature_means = processor.means
        self.feature_stds = processor.stds
        self.min_trigger_frames = config.tcn_min_trigger_frames
        self.max_gap_frames = config.max_gap_frames
        self.confidence_threshold = config.tcn_confidence_threshold

    def normalize_features(self, features):
        if self.feature_means is None or self.feature_stds is None:
            return features
        return (features - self.feature_means) / self.feature_stds

    @staticmethod
    def _wrap_angle(angle):
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    def process_one_frame(self, frame_df):
        results = {}
        for row in frame_df.itertuples(index=False):
            track_id = int(row.track_id)
            lane_id = int(row.lane_id)

            if track_id not in self.tracks:
                self.tracks[track_id] = _TCNTrackState(
                    track_id=track_id,
                    frame_buffer=deque(maxlen=100),
                    d_window=deque(maxlen=self.cumulative_window),
                    vd_sign_window=deque(maxlen=self.sign_window),
                )
            state = self.tracks[track_id]
            current_pred = 0

            if lane_id != -1:
                cx = float(row.x1) + float(row.w) / 2.0
                cy = float(row.y1) + float(row.h)
                rx, ry = self.geometry.pixel_to_metric(cx, cy)

                if state.initial_lane_id == -1:
                    state.initial_lane_id = lane_id
                if state.first_frame_id is None:
                    state.first_frame_id = int(row.frame_id)

                raw_d = self.geometry.get_frenet_d_from_ref_lane(rx, ry, state.initial_lane_id)
                filtered_d = 0.0
                filtered_vd = 0.0
                noise = 0.0

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
                    noise = filtered_d - raw_d

                relative_frame = int(row.frame_id) - state.first_frame_id if state.first_frame_id else 0

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

                dist_left_boundary, dist_right_boundary = self.geometry.get_lane_boundary_distances(rx, ry, lane_id)

                heading_dev = 0.0
                lane_heading = self.geometry.get_lane_tangent_angle(ry, lane_id)
                if state.prev_rx is not None and state.prev_ry is not None and lane_heading is not None:
                    dx = rx - state.prev_rx
                    dy = ry - state.prev_ry
                    if abs(dx) + abs(dy) > 1e-6:
                        vehicle_heading = float(np.arctan2(dx, dy))
                        heading_dev = self._wrap_angle(vehicle_heading - lane_heading)

                frame_feature = [
                    filtered_d, filtered_vd, noise, lane_id, rx, ry,
                    relative_frame, cum_d_change, vd_sign_consistency,
                    dist_left_boundary, dist_right_boundary, heading_dev,
                ]
                state.frame_buffer.append(frame_feature)
                state.prev_rx = rx
                state.prev_ry = ry

                tcn_pred_raw = 0
                if len(state.frame_buffer) >= 5:
                    seq_array = np.array(list(state.frame_buffer), dtype=np.float32)
                    seq_array = self.normalize_features(seq_array)
                    seq_tensor = torch.from_numpy(seq_array).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        logits = self.model(seq_tensor)
                        current_logits = logits[:, :, -1]
                        probs = F.softmax(current_logits, dim=1)
                        tcn_pred_raw = 1 if float(probs[0, 1].item()) > self.confidence_threshold else 0

                is_tcn_trigger = tcn_pred_raw == 1

                if is_tcn_trigger:
                    state.tcn_trigger_counter += 1
                    state.tcn_gap_frames = 0
                else:
                    state.tcn_trigger_counter = 0
                    if state.is_in_event:
                        state.tcn_gap_frames += 1

                if not state.is_in_event and state.tcn_trigger_counter >= self.min_trigger_frames:
                    state.is_in_event = True
                    state.is_confirmed = False
                    state.tcn_gap_frames = 0

                if state.is_in_event:
                    state.is_confirmed = True
                    if state.tcn_gap_frames > self.max_gap_frames:
                        state.is_in_event = False
                        state.is_confirmed = False
                        state.tcn_gap_frames = 0
                        state.tcn_trigger_counter = 0

                if state.is_confirmed:
                    current_pred = 1

            results[track_id] = current_pred
        return results


def build_tcn_detector(geometry, config, model, processor):
    return TCNDetector(geometry, config, model, processor)

