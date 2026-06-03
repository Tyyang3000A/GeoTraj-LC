import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from geotraj_lc.anchor_correction.dataset import CarPatchDataset
from geotraj_lc.anchor_correction.model import OffsetRegressor


def train_regressor(data_dir, save_dir, epochs=50, batch_size=32, lr=1e-4, resume=None, num_workers=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CarPatchDataset(data_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = OffsetRegressor(pretrained=True)
    if resume and os.path.exists(resume):
        print(f"Loading pretrained weights from {resume}")
        model.load_state_dict(torch.load(resume, map_location=device))

    os.makedirs(save_dir, exist_ok=True)
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
    return 0
