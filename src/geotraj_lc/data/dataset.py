import numpy as np
import torch
from torch.utils.data import Dataset


class VarLenDataset(Dataset):
    def __init__(self, X_list, y_list, meta_list):
        self.X_list = X_list
        self.y_list = y_list
        self.meta_list = meta_list

    def __len__(self):
        return len(self.X_list)

    def __getitem__(self, idx):
        return self.X_list[idx], self.y_list[idx], self.meta_list[idx]


def collate_fn(batch):
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    X_list, y_list, meta_list = zip(*batch)

    max_seq_len = max(x.shape[0] for x in X_list)
    feature_dim = X_list[0].shape[1]
    batch_size = len(X_list)

    X_batch = np.zeros((batch_size, max_seq_len, feature_dim), dtype=np.float32)
    y_batch = np.full((batch_size, max_seq_len), -1, dtype=np.int64)
    meta_batch = np.full((batch_size, max_seq_len, 2), -1, dtype=np.int64)
    mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)

    lengths = []
    for i, (x, y, m) in enumerate(zip(X_list, y_list, meta_list)):
        seq_len = x.shape[0]
        X_batch[i, :seq_len, :] = x
        y_batch[i, :seq_len] = y
        meta_batch[i, :seq_len, :] = m
        mask_batch[i, :seq_len] = 1.0
        lengths.append(seq_len)

    return (
        torch.from_numpy(X_batch),
        torch.from_numpy(y_batch),
        torch.from_numpy(meta_batch),
        torch.from_numpy(mask_batch),
        torch.tensor(lengths, dtype=torch.int64),
    )

