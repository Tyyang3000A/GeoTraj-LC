import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from scipy.interpolate import interp1d


class GeometryEngine:
    def __init__(self, homography_path: Path, centerline_path: Path, scale: float):
        self.homography = np.load(str(homography_path))
        self.centerlines = self._load_centerlines(centerline_path)
        self.scale = float(scale)

    def _load_centerlines(self, path: Path) -> Dict[int, interp1d]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        functions = {}
        for lane_id, points in enumerate(data):
            points_array = np.array(points, dtype=np.float32)
            if len(points_array) < 2:
                continue
            functions[lane_id] = interp1d(
                points_array[:, 1],
                points_array[:, 0],
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
        return functions

    def pixel_to_metric(self, x: float, y: float):
        pts = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pts, self.homography)
        rx = float(transformed[0][0][0]) / self.scale
        ry = float(transformed[0][0][1]) / self.scale
        return rx, ry

    def get_frenet_d(self, rx: float, ry: float, lane_id: int):
        if lane_id == -1 or lane_id not in self.centerlines:
            return None
        return float(rx - self.centerlines[lane_id](ry))

    def get_frenet_d_from_ref_lane(self, rx: float, ry: float, ref_lane_id: int) -> Optional[float]:
        if ref_lane_id == -1 or ref_lane_id not in self.centerlines:
            return None
        return float(rx - self.centerlines[ref_lane_id](ry))

    def get_original_lane_frenet_d(self, rx: float, ry: float, lane_id: int, ref_lane_id: int) -> Optional[Tuple[float, float]]:
        if ref_lane_id not in self.centerlines or lane_id not in self.centerlines:
            return None
        ref_centerline = self.centerlines[ref_lane_id](ry)
        current_centerline = self.centerlines[lane_id](ry)
        d_ref = rx - ref_centerline
        lane_offset = current_centerline - ref_centerline
        return float(d_ref), float(lane_offset)

    def get_lane_center_x(self, ry: float, lane_id: int) -> Optional[float]:
        if lane_id == -1 or lane_id not in self.centerlines:
            return None
        return float(self.centerlines[lane_id](ry))

    def get_lane_tangent_angle(self, ry: float, lane_id: int, delta_ry: float = 0.5) -> Optional[float]:
        center_func = self.centerlines.get(lane_id)
        if center_func is None:
            return None
        x_prev = float(center_func(ry - delta_ry))
        x_next = float(center_func(ry + delta_ry))
        dx_dy = (x_next - x_prev) / (2.0 * delta_ry)
        return float(np.arctan2(dx_dy, 1.0))

    def get_lane_boundary_distances(self, rx: float, ry: float, lane_id: int) -> Tuple[float, float]:
        if lane_id == -1 or lane_id not in self.centerlines:
            return 0.0, 0.0

        lane_centers = []
        for lid, centerline in self.centerlines.items():
            lane_centers.append((float(centerline(ry)), int(lid)))

        lane_centers.sort(key=lambda item: item[0])
        lane_ids = [lid for _, lid in lane_centers]
        if lane_id not in lane_ids:
            return 0.0, 0.0

        idx = lane_ids.index(lane_id)
        x_curr = lane_centers[idx][0]

        x_left = lane_centers[idx - 1][0] if idx > 0 else None
        x_right = lane_centers[idx + 1][0] if idx < len(lane_centers) - 1 else None

        half_width = 1.75
        if x_left is not None and x_right is not None:
            half_width = 0.25 * ((x_curr - x_left) + (x_right - x_curr))
        elif x_left is not None:
            half_width = 0.5 * (x_curr - x_left)
        elif x_right is not None:
            half_width = 0.5 * (x_right - x_curr)

        left_boundary_x = (x_left + x_curr) * 0.5 if x_left is not None else x_curr - half_width
        right_boundary_x = (x_curr + x_right) * 0.5 if x_right is not None else x_curr + half_width

        dist_left = float(rx - left_boundary_x)
        dist_right = float(right_boundary_x - rx)
        return dist_left, dist_right

