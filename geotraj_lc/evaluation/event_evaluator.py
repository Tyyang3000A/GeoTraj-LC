from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


class EventEvaluator:
    def __init__(
        self,
        gt_df: pd.DataFrame,
        pred_dict: Dict[Tuple[int, int], int],
        fps: float = 20.0,
        iou_threshold: float = 0.3,
    ):
        self.gt_df = gt_df
        self.pred_dict = pred_dict
        self.fps = fps
        self.dt = 1.0 / fps
        self.iou_threshold = iou_threshold
        self.merged_df = self._merge_data()

    def _merge_data(self) -> pd.DataFrame:
        data = []
        for row in self.gt_df.itertuples(index=False):
            key = (int(row.frame_id), int(row.track_id))
            gt_bin = 1 if int(row.gt_state) > 0 else 0
            pred_bin = int(self.pred_dict.get(key, 0))
            data.append(
                {
                    "track_id": int(row.track_id),
                    "frame_id": int(row.frame_id),
                    "gt": gt_bin,
                    "pred": pred_bin,
                }
            )
        return pd.DataFrame(data)

    @staticmethod
    def _extract_events(df_group: pd.DataFrame, column: str) -> List[Tuple[int, int]]:
        events: List[Tuple[int, int]] = []
        frames = df_group["frame_id"].to_numpy()
        states = df_group[column].to_numpy()

        diffs = np.diff(states, prepend=0)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        if len(ends) < len(starts):
            ends = np.append(ends, len(states))

        for start_idx, end_idx in zip(starts, ends):
            end_frame_idx = end_idx - 1 if end_idx - 1 < len(frames) else len(frames) - 1
            events.append((int(frames[start_idx]), int(frames[end_frame_idx])))

        return events

    @staticmethod
    def _compute_iou(event_a: Tuple[int, int], event_b: Tuple[int, int]) -> float:
        start_a, end_a = event_a
        start_b, end_b = event_b

        intersection = max(0, min(end_a, end_b) - max(start_a, start_b) + 1)
        union = max(end_a, end_b) - min(start_a, start_b) + 1
        return float(intersection / union) if union > 0 else 0.0

    def evaluate(self) -> Dict[str, object]:
        total_pred = 0
        total_gt = 0

        true_positive = 0
        false_positive = 0
        false_negative = 0

        ious: List[float] = []
        latencies: List[float] = []

        for _, group in self.merged_df.groupby("track_id"):
            group = group.sort_values("frame_id")
            gt_events = self._extract_events(group, "gt")
            pred_events = self._extract_events(group, "pred")

            total_gt += len(gt_events)
            total_pred += len(pred_events)

            matched_gt: Set[int] = set()
            matched_pred: Set[int] = set()
            candidate_matches: List[Tuple[float, int, int]] = []

            for gt_idx, gt_event in enumerate(gt_events):
                for pred_idx, pred_event in enumerate(pred_events):
                    iou = self._compute_iou(gt_event, pred_event)
                    if iou >= self.iou_threshold:
                        candidate_matches.append((iou, gt_idx, pred_idx))

            candidate_matches.sort(reverse=True)

            for iou, gt_idx, pred_idx in candidate_matches:
                if gt_idx in matched_gt or pred_idx in matched_pred:
                    continue

                matched_gt.add(gt_idx)
                matched_pred.add(pred_idx)
                true_positive += 1

                gt_event = gt_events[gt_idx]
                pred_event = pred_events[pred_idx]
                latency = (pred_event[0] - gt_event[0]) * self.dt

                ious.append(float(iou))
                latencies.append(float(latency))

            false_positive += len(pred_events) - len(matched_pred)
            false_negative += len(gt_events) - len(matched_gt)

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        f1_score = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        mean_iou = float(np.mean(ious)) if ious else 0.0
        mean_latency = float(np.mean(latencies)) if latencies else 0.0
        std_latency = float(np.std(latencies)) if latencies else 0.0

        return {
            "total_gt": total_gt,
            "total_pred": total_pred,
            "tp_pred": true_positive,
            "tp_gt": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "mean_iou": mean_iou,
            "mean_latency": mean_latency,
            "std_latency": std_latency,
            "all_ious": ious,
            "all_latencies": latencies,
        }

