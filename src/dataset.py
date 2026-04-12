from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from data_loader import ASVspoofDataLoader


class ASVspoofDataset(Dataset):
    """
    PyTorch Dataset for ASVspoof 2019.

    Returns:
        waveform: torch.FloatTensor of shape (64000,)
        label: torch.LongTensor (0 or 1)
    """

    def __init__(self, data_dir: str | Path, split: str = "train") -> None:
        self.data_dir = Path(data_dir)
        self.split = split

        # Use your existing data loader
        self.loader = ASVspoofDataLoader(self.data_dir, split=self.split)

    def __len__(self) -> int:
        return len(self.loader)

    def __getitem__(self, idx: int):
        sample = self.loader.get_example(idx)

        waveform = sample["waveform"]  # numpy array (64000,)
        label = sample["label"]        # int (0 or 1)

        # Convert to PyTorch tensors
        waveform = torch.tensor(waveform, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return waveform, label


# Quick test (run this file directly)
if __name__ == "__main__":
    dataset = ASVspoofDataset("../data/LA", split="train")

    print("Total samples:", len(dataset))

    x, y = dataset[0]

    print("Waveform shape:", x.shape)
    print("Waveform dtype:", x.dtype)
    print("Label:", y)