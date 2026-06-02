from pathlib import Path
from typing import List, Optional

import numpy as np
import pickle


class Normalizer:
    def __init__(self):
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None

    def fit(self, all_features: List[np.ndarray]) -> None:
        all_data = np.concatenate(all_features, axis=0)
        self.means = np.mean(all_data, axis=0)
        self.stds = np.std(all_data, axis=0)
        self.stds[self.stds < 1e-8] = 1.0

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.means is None or self.stds is None:
            return features
        return (features - self.means) / self.stds

    def fit_transform(self, all_features: List[np.ndarray]) -> List[np.ndarray]:
        self.fit(all_features)
        return [self.transform(x) for x in all_features]

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"means": self.means, "stds": self.stds}, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.means = data["means"]
            self.stds = data["stds"]

