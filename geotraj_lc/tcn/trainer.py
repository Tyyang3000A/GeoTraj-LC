import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from geotraj_lc.config import Config
from geotraj_lc.trajectory.layout import default_experiment, default_layout
from geotraj_lc.trajectory.dataset import VarLenDataset, collate_fn
from geotraj_lc.evaluation.metrics import evaluate_event_f1_with_detector_pipeline, evaluate_frame_metrics
from geotraj_lc.tcn.model import LaneChangeTCN
from geotraj_lc.trajectory.feature_extractor import FeatureExtractor
from geotraj_lc.trajectory.normalizer import Normalizer
from geotraj_lc.tcn.samples import build_split_samples, load_or_generate_splits


def train(config: Config = None, rebuild_cache: bool = False):
    if config is None:
        config = Config()

    experiment = default_experiment()
    experiment.ensure_dirs()

    layout = default_layout()

    train_seqs, val_seqs, test_seqs = load_or_generate_splits(experiment.splits_path, layout, config)

    print(f"Train sequences ({len(train_seqs)}): {train_seqs}")
    print(f"Val sequences ({len(val_seqs)}): {val_seqs}")
    print(f"Test sequences ({len(test_seqs)}): {test_seqs}")

    extractor = FeatureExtractor(config)

    print("\nGenerating dataset samples...")
    t0 = time.time()
    train_X, train_y, train_meta, _ = build_split_samples(
        extractor, train_seqs, layout, "train",
        cache_dir=experiment.cache_dir, use_cache=True, force_rebuild=rebuild_cache,
    )
    val_X, val_y, _, _ = build_split_samples(
        extractor, val_seqs, layout, "val",
        cache_dir=experiment.cache_dir, use_cache=True, force_rebuild=rebuild_cache,
    )
    test_X, test_y, test_meta, _ = build_split_samples(
        extractor, test_seqs, layout, "test",
        cache_dir=experiment.cache_dir, use_cache=True, force_rebuild=rebuild_cache,
    )
    print(f"Sample generation finished in {time.time() - t0:.1f}s")

    if not train_X:
        print("No training samples generated.")
        return 1

    normalizer = Normalizer()
    train_X = normalizer.fit_transform(train_X)
    val_X = [normalizer.transform(x) for x in val_X]
    test_X = [normalizer.transform(x) for x in test_X]
    normalizer.save(experiment.best_normalizer_path)

    print(f"\nTrain samples: {len(train_X)}, Val samples: {len(val_X)}, Test samples: {len(test_X)}")
    print(f"Feature shape: {train_X[0].shape[1:]}")

    train_total_frames = sum(len(y) for y in train_y)
    train_class_0 = sum(np.sum(y == 0) for y in train_y)
    train_class_1 = sum(np.sum(y == 1) for y in train_y)
    print(f"\nTrain frame distribution: 0(stable)={train_class_0}, 1(lane_change)={train_class_1}")

    val_class_0 = sum(np.sum(y == 0) for y in val_y)
    val_class_1 = sum(np.sum(y == 1) for y in val_y)
    print(f"Val frame distribution: 0(stable)={val_class_0}, 1(lane_change)={val_class_1}")

    test_class_0 = sum(np.sum(y == 0) for y in test_y)
    test_class_1 = sum(np.sum(y == 1) for y in test_y)
    print(f"Test frame distribution: 0(stable)={test_class_0}, 1(lane_change)={test_class_1}")

    train_dataset = VarLenDataset(train_X, train_y, train_meta)
    test_dataset = VarLenDataset(test_X, test_y, test_meta)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    model = LaneChangeTCN(
        input_size=config.tcn_input_size,
        output_size=config.tcn_output_size,
        num_channels=list(config.tcn_num_channels),
        kernel_size=config.tcn_kernel_size,
        dropout=config.tcn_dropout,
    ).to(device)

    positive_weight_factor = config.positive_weight_factor
    class_weights = torch.tensor(
        [
            train_total_frames / (2.0 * train_class_0),
            (train_total_frames / (2.0 * train_class_1)) * positive_weight_factor,
        ],
        dtype=torch.float32,
    ).to(device)
    print(f"Class weights: {class_weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_val_event_f1 = 0.0
    best_epoch = 0

    print("\nStarting training...")

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0

        for batch_X, batch_y, _, _, _ in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()

            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_max_norm)
            optimizer.step()

            total_loss += loss.item()

        val_event_metrics = evaluate_event_f1_with_detector_pipeline(model, normalizer, val_seqs, layout, config)
        val_event_f1 = float(val_event_metrics["f1_score"])
        epoch_loss = total_loss / len(train_loader)

        scheduler.step(val_event_f1)

        is_best = val_event_f1 > best_val_event_f1
        if is_best:
            best_val_event_f1 = val_event_f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), experiment.best_model_path)

        best_suffix = " [BEST MODEL SAVED]" if is_best else ""
        print(f"\nEpoch {epoch + 1}{best_suffix}")
        print(f"Loss: {epoch_loss:.4f}")
        print(
            f"Val(eval) - Event Prec: {val_event_metrics['precision']:.4f}, "
            f"Event Rec: {val_event_metrics['recall']:.4f}, Event F1: {val_event_f1:.4f}, "
            f"Mean IoU: {val_event_metrics['mean_iou']:.4f}"
        )

    if experiment.best_model_path.exists():
        model.load_state_dict(torch.load(experiment.best_model_path, map_location=device))

    test_acc, test_prec, test_rec, test_f1_frame, _ = evaluate_frame_metrics(model, test_loader, device)
    test_event_metrics = evaluate_event_f1_with_detector_pipeline(model, normalizer, test_seqs, layout, config)

    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"Best epoch (by val event F1): {best_epoch}")
    print(f"Best val event F1: {best_val_event_f1:.4f}")
    print("\nTest Results (best checkpoint):")
    print(f"Frame - Acc: {test_acc:.4f}, Prec: {test_prec:.4f}, Rec: {test_rec:.4f}, F1: {test_f1_frame:.4f}")
    print(
        f"Event - Prec: {test_event_metrics['precision']:.4f}, Rec: {test_event_metrics['recall']:.4f}, "
        f"F1: {test_event_metrics['f1_score']:.4f}, Mean IoU: {test_event_metrics['mean_iou']:.4f}"
    )
    print(f"{'=' * 60}")

    print(f"\nModel saved to: {experiment.best_checkpoint_dir}")
    return 0

