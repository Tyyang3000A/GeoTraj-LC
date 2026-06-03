import argparse

from geotraj_lc.anchor_correction.trainer import train_regressor
from geotraj_lc.config import Config
from geotraj_lc.tcn.trainer import train as train_tcn


def main():
    parser = argparse.ArgumentParser(description="Train models")
    subparsers = parser.add_subparsers(dest="command", help="Model to train")

    tcn_parser = subparsers.add_parser("tcn", help="Train TCN lane change detector")
    tcn_parser.add_argument("--rebuild-cache", action="store_true", help="Force rebuilding cached samples")
    tcn_parser.add_argument("--train-event-eval-interval", type=int, default=None)

    reg_parser = subparsers.add_parser("regressor", help="Train offset regressor")
    reg_parser.add_argument("--data-dir", "--data_dir", dest="data_dir", type=str, required=True, help="Path to exported dataset/train folder")
    reg_parser.add_argument("--save-dir", "--save_dir", dest="save_dir", type=str, default="pretrained/anchor_correction", help="Folder to save trained weights")
    reg_parser.add_argument("--resume", type=str, default=None, help="Path to pretrained .pth")
    reg_parser.add_argument("--epochs", type=int, default=50)
    reg_parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=32)
    reg_parser.add_argument("--lr", type=float, default=1e-4)
    reg_parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()

    if args.command == "tcn":
        config = Config()
        if args.train_event_eval_interval is not None:
            config.train_event_eval_interval = args.train_event_eval_interval
        return train_tcn(config=config, rebuild_cache=args.rebuild_cache)

    elif args.command == "regressor":
        return train_regressor(
            data_dir=args.data_dir,
            save_dir=args.save_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            resume=args.resume,
            num_workers=args.num_workers,
        )

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

