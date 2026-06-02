import argparse
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from geotraj_lc.config import Config
from geotraj_lc.core.regressor import OffsetRegressor
from geotraj_lc.training.trainer import train


class CarPatchDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(data_dir, "images", "*.jpg"))
        if not self.image_paths:
            self.image_paths = glob.glob(os.path.join(data_dir, "images", "*.png"))
        print(f"Found {len(self.image_paths)} images in {data_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = img_path.replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt")

        image = Image.open(img_path).convert("RGB")

        with open(label_path, "r") as f:
            line = f.readline().strip()
            parts = line.split()
            coords = [float(x) for x in parts[:4]]

        target = torch.tensor(coords, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, target


def train_regressor(data_dir, save_dir, epochs=50, batch_size=32, lr=1e-4, resume=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CarPatchDataset(data_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = OffsetRegressor(pretrained=True)
    if resume and os.path.exists(resume):
        print(f"Loading pretrained weights from {resume}")
        model.load_state_dict(torch.load(resume, map_location=device))

    model.to(device)
    model.train()

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    best_loss = float("inf")

    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            pred_coords, _ = model(images)
            loss = criterion(pred_coords, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Average Loss: {epoch_loss:.6f}")

        scheduler.step()

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_path = os.path.join(save_dir, "regressor_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

    print("Training finished.")


def main():
    parser = argparse.ArgumentParser(description="Train models")
    subparsers = parser.add_subparsers(dest="command", help="Model to train")

    tcn_parser = subparsers.add_parser("tcn", help="Train TCN lane change detector")
    tcn_parser.add_argument("--rebuild-cache", action="store_true", help="Force rebuilding cached samples")
    tcn_parser.add_argument("--train-event-eval-interval", type=int, default=None)

    reg_parser = subparsers.add_parser("regressor", help="Train offset regressor")
    reg_parser.add_argument("--data_dir", type=str, required=True, help="Path to exported dataset/train folder")
    reg_parser.add_argument("--save_dir", type=str, default="./weights", help="Folder to save trained weights")
    reg_parser.add_argument("--resume", type=str, default=None, help="Path to pretrained .pth")
    reg_parser.add_argument("--epochs", type=int, default=50)
    reg_parser.add_argument("--batch_size", type=int, default=32)
    reg_parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    if args.command == "tcn":
        config = Config()
        if args.train_event_eval_interval is not None:
            config.train_event_eval_interval = args.train_event_eval_interval
        return train(config=config, rebuild_cache=args.rebuild_cache)

    elif args.command == "regressor":
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        train_regressor(
            data_dir=args.data_dir,
            save_dir=args.save_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            resume=args.resume,
        )
        return 0

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

