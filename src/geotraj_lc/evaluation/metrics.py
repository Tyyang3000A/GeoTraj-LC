import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

from geotraj_lc.config import Config
from geotraj_lc.core.geometry import GeometryEngine
from geotraj_lc.data.io import load_corrected_tracks, load_lane_change_gt
from geotraj_lc.data.layout import DataLayout
from geotraj_lc.evaluation.event_evaluator import EventEvaluator
from geotraj_lc.evaluation.tcn_detector import build_tcn_detector


def evaluate_frame_metrics(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y, _, _, _ in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_X)
            _, predicted = logits.max(1)

            valid_mask = batch_y != -1
            total += valid_mask.sum().item()
            correct += (predicted == batch_y)[valid_mask].sum().item()

            all_preds.extend(predicted[valid_mask].cpu().numpy())
            all_labels.extend(batch_y[valid_mask].cpu().numpy())

    acc = correct / total if total > 0 else 0.0
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    return acc, prec, rec, f1, cm


def evaluate_event_f1_with_detector_pipeline(model, processor, sequences, layout, config):
    if not sequences:
        return {
            "total_gt": 0,
            "total_pred": 0,
            "tp_pred": 0,
            "tp_gt": 0,
            "fp": 0,
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "mean_iou": 0.0,
            "mean_latency": 0.0,
            "std_latency": 0.0,
            "all_ious": [],
            "all_latencies": [],
        }

    total_gt = 0
    total_pred = 0
    tp_pred = 0
    tp_gt = 0
    fp = 0
    fn = 0
    all_ious = []
    all_latencies = []

    for sequence in sequences:
        paths = layout.sequence_paths(sequence)
        corrected_df = load_corrected_tracks(paths.corrected_tracks)
        gt_df = load_lane_change_gt(paths.lane_change_gt)

        scale = config.scale_for_sequence(sequence)
        geometry = GeometryEngine(paths.homography, paths.centerline, scale=scale)

        detector = build_tcn_detector(geometry, config, model, processor)

        predictions = {}
        grouped_frames = corrected_df.groupby("frame_id")
        all_frames = sorted(grouped_frames.groups.keys())
        for frame_id in all_frames:
            frame_df = grouped_frames.get_group(frame_id)
            frame_result = detector.process_one_frame(frame_df)
            for track_id, state in frame_result.items():
                predictions[(int(frame_id), int(track_id))] = int(state)

        metrics = EventEvaluator(gt_df, predictions, fps=config.fps).evaluate()
        total_gt += int(metrics["total_gt"])
        total_pred += int(metrics["total_pred"])
        tp_pred += int(metrics["tp_pred"])
        tp_gt += int(metrics["tp_gt"])
        fp += int(metrics["fp"])
        fn += int(metrics["fn"])
        all_ious.extend(metrics.get("all_ious", []))
        all_latencies.extend(metrics.get("all_latencies", []))

    precision = tp_pred / total_pred if total_pred > 0 else 0.0
    recall = tp_gt / total_gt if total_gt > 0 else 0.0
    f1_score = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    mean_latency = float(np.mean(all_latencies)) if all_latencies else 0.0
    std_latency = float(np.std(all_latencies)) if all_latencies else 0.0

    return {
        "total_gt": total_gt,
        "total_pred": total_pred,
        "tp_pred": tp_pred,
        "tp_gt": tp_gt,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mean_iou": mean_iou,
        "mean_latency": mean_latency,
        "std_latency": std_latency,
        "all_ious": all_ious,
        "all_latencies": all_latencies,
    }


def run_event_eval(
    sequences: Iterable[str],
    layout: DataLayout,
    config: Config,
    model_path: Path,
    normalizer_path: Path,
    show_progress: bool = True,
) -> Tuple[List[Dict[str, object]], List[Tuple[str, str]]]:
    from geotraj_lc.core.tcn import LaneChangeTCN
    from geotraj_lc.data.normalizer import Normalizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LaneChangeTCN(
        input_size=config.tcn_input_size,
        output_size=config.tcn_output_size,
        num_channels=list(config.tcn_num_channels),
        kernel_size=config.tcn_kernel_size,
        dropout=config.tcn_dropout,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    normalizer = Normalizer()
    normalizer.load(normalizer_path)

    class _Processor:
        means = normalizer.means
        stds = normalizer.stds

    processor = _Processor()

    results: List[Dict[str, object]] = []
    failed: List[Tuple[str, str]] = []

    for sequence in sequences:
        try:
            paths = layout.sequence_paths(sequence)
            tracks_df = load_corrected_tracks(paths.corrected_tracks)
            gt_df = load_lane_change_gt(paths.lane_change_gt)

            scale = config.scale_for_sequence(sequence)
            geometry = GeometryEngine(paths.homography, paths.centerline, scale=scale)
            detector = build_tcn_detector(geometry, config, model, processor)

            predictions: Dict[Tuple[int, int], int] = {}
            grouped_frames = tracks_df.groupby("frame_id")
            frame_ids = sorted(grouped_frames.groups.keys())

            iterator = tqdm(frame_ids, desc=f"Processing {sequence}", unit="frame", disable=not show_progress)
            for frame_id in iterator:
                frame_result = detector.process_one_frame(grouped_frames.get_group(frame_id))
                for track_id, state in frame_result.items():
                    predictions[(int(frame_id), int(track_id))] = int(state)

            metrics = EventEvaluator(gt_df, predictions, fps=config.fps).evaluate()
            metrics["sequence"] = sequence
            metrics["scale"] = scale
            results.append(metrics)
        except Exception as exc:
            failed.append((sequence, str(exc)))

    return results, failed


def aggregate_event_metrics(results: List[Dict[str, object]]) -> Dict[str, object]:
    if not results:
        return {
            "clip_count": 0,
            "total_gt": 0,
            "total_pred": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "mean_iou": 0.0,
            "mean_latency": 0.0,
            "std_latency": 0.0,
        }

    total_gt = sum(int(item["total_gt"]) for item in results)
    total_pred = sum(int(item["total_pred"]) for item in results)
    tp = sum(int(item["tp_pred"]) for item in results)
    fp = sum(int(item.get("fp", 0)) for item in results)
    fn = sum(int(item.get("fn", 0)) for item in results)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1_score = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    all_ious: List[float] = []
    all_latencies: List[float] = []
    for item in results:
        all_ious.extend(item.get("all_ious", []))
        all_latencies.extend(item.get("all_latencies", []))

    return {
        "clip_count": len(results),
        "total_gt": total_gt,
        "total_pred": total_pred,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mean_iou": float(np.mean(all_ious)) if all_ious else 0.0,
        "mean_latency": float(np.mean(all_latencies)) if all_latencies else 0.0,
        "std_latency": float(np.std(all_latencies)) if all_latencies else 0.0,
    }


def _base_sequence(sequence: str) -> str:
    match = re.match(r"^(.*?)_clip_\d+$", sequence)
    return match.group(1) if match else sequence


def group_by_base_sequence(results: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for item in results:
        grouped[_base_sequence(str(item["sequence"]))].append(item)
    return dict(grouped)


def format_event_metrics(title: str, metrics: Dict[str, object]) -> str:
    return "\n".join(
        [
            f"\n{'=' * 60}",
            f"  {title}",
            f"{'=' * 60}",
            f"  Clips:        {metrics['clip_count']}",
            f"  GT events:    {metrics['total_gt']}",
            f"  Pred events:  {metrics['total_pred']}",
            f"  TP: {metrics['tp']}  FP: {metrics['fp']}  FN: {metrics['fn']}",
            f"  Precision:    {metrics['precision']:.4f}",
            f"  Recall:       {metrics['recall']:.4f}",
            f"  F1-score:     {metrics['f1_score']:.4f}",
            f"  Mean IoU:     {metrics['mean_iou']:.4f}",
            f"  Mean Latency: {metrics['mean_latency']:.4f}s  (std={metrics['std_latency']:.4f})",
            f"{'=' * 60}",
        ]
    )

