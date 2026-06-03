import argparse

from geotraj_lc.config import Config
from geotraj_lc.trajectory.layout import default_experiment, default_layout
from geotraj_lc.evaluation.metrics import (
    aggregate_event_metrics,
    format_event_metrics,
    group_by_base_sequence,
    run_event_eval,
)
from geotraj_lc.tcn.samples import load_or_generate_splits


def run_tcn(sequences, layout, config, no_progress=False):
    experiment = default_experiment()
    model_path = experiment.existing_best_model_path()
    normalizer_path = experiment.existing_best_normalizer_path()

    for label, path in [("model", model_path), ("normalizer", normalizer_path)]:
        if not path.exists():
            print(f"Error: {label} not found: {path}")
            return 1

    print(f"Model: {model_path}")
    print(f"Normalizer: {normalizer_path}")

    results, failed = run_event_eval(
        sequences,
        layout=layout,
        config=config,
        model_path=model_path,
        normalizer_path=normalizer_path,
        show_progress=not no_progress,
    )

    if failed:
        print(f"\nFailed sequences ({len(failed)}):")
        for sequence, reason in failed:
            print(f"  - {sequence}: {reason}")

    if not results:
        print("Error: no successful evaluation results.")
        return 1

    for base_name, group in sorted(group_by_base_sequence(results).items()):
        print(format_event_metrics(base_name, aggregate_event_metrics(group)))

    print(format_event_metrics("OVERALL", aggregate_event_metrics(results)))
    return 0


def get_sequences(split, layout, config, experiment):
    train_seqs, val_seqs, test_seqs = load_or_generate_splits(experiment.splits_path, layout, config)
    split = split.lower()
    if split == "all":
        return train_seqs + val_seqs + test_seqs
    if split == "train":
        return train_seqs
    if split == "val":
        return val_seqs
    return test_seqs


def main():
    parser = argparse.ArgumentParser(description="Lane change detection evaluation")
    parser.add_argument("--split", choices=["test", "val", "train", "all"], default="test")
    parser.add_argument("--sequences", nargs="+", default=None, help="Specific sequences to process")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    args = parser.parse_args()

    layout = default_layout()
    layout.ensure_dirs()
    config = Config()

    if args.sequences:
        sequences = args.sequences
    else:
        experiment = default_experiment()
        sequences = get_sequences(args.split, layout, config, experiment)

    if not sequences:
        print("No sequences selected.")
        return 1

    print(f"Running TCN detector on {len(sequences)} sequences ({args.split} split)")
    return run_tcn(sequences, layout, config, no_progress=args.no_progress)


if __name__ == "__main__":
    raise SystemExit(main())
