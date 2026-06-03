import glob
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


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
