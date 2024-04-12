import torch
import numpy as np
from typing import Callable
from torch.utils.data import Dataset, Subset


class ADataset(Dataset):
    """
    Dataset for Anomaly Detection.

    Inputs:
        - data_path (str): Filepath of source data, which should end with `.npz` or '.npy`.
        - step_num_in (int): Window length to set.
        - stride (int): Stride of the window.
        - keys (Union[Optional, Iterable]): if data_path ends with `.npy`, it should be None, while
          the arrays of keys for `.npz`.
        - start (int, default 0): The start time point to build dataset.
    """
    def __init__(self, data_path: str, step_num_in: int, stride: int, keys=None, start=0):
        super(ADataset, self).__init__()
        self.step_num_in: int = step_num_in
        self.stride: int = stride
        self.keys = keys

        data = np.load(data_path)
        self.data = data[start:] if keys is None else [data[key][start:] for key in keys]

    def fit_transform(self, transform: Callable):
        if transform is None:
            return
        if self.keys is None:
            self.data = transform(self.data)
        else:
            self.data[0] = transform(self.data[0])

    def __getitem__(self, index):
        start = index * self.stride
        if self.keys is None:
            return self.data[start: start + self.step_num_in]
        return [obj[start: start + self.step_num_in] for obj in self.data]

    def __len__(self):
        data_len = self.data.shape[0] if self.keys is None else self.data[0].shape[0]
        return (data_len - self.step_num_in) // self.stride + 1


def split_dataset(dataset: Dataset, val_ratio=0.1, random=False):
    dataset_len = len(dataset)
    val_use_len = int(dataset_len * val_ratio)

    if random:
        val_indices = torch.randperm(dataset_len)[:val_use_len]
    else:
        start = torch.randint(0, dataset_len - val_use_len, (1,)).item()
        val_indices = torch.arange(start, start + val_use_len)

    val_indices = val_indices.numpy().tolist()
    train_indices = list(set(range(dataset_len)) - set(val_indices))

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    return train_subset, val_subset
