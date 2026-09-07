from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EpochChunks(Dataset):
    """Non-overlapping token windows. One pass = one epoch, no leaked overlap."""

    def __init__(self, path: Path, seq_len: int):
        self.seq_len = seq_len
        path = Path(path)
        if path.suffix == ".npy":
            self.tokens = np.load(path, mmap_mode="r")
        else:
            self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        if self.tokens.ndim != 1:
            raise ValueError(f"{path} must be a 1-D token array")
        self.n = len(self.tokens) // seq_len
        if self.n < 1:
            raise ValueError(f"{path} is shorter than seq_len={seq_len}")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> torch.Tensor:
        start = index * self.seq_len
        window = np.asarray(self.tokens[start : start + self.seq_len], dtype=np.int64)
        return torch.from_numpy(window)


class TokenStream(Dataset):
    """Random overlapping windows from a 1-D token file (uint16/int32 .npy or .bin)."""

    def __init__(self, path: Path, seq_len: int):
        self.seq_len = seq_len
        path = Path(path)
        if path.suffix == ".npy":
            self.tokens = np.load(path, mmap_mode="r")
        else:
            self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        if self.tokens.ndim != 1:
            raise ValueError(f"{path} must be a 1-D token array")
        if len(self.tokens) <= seq_len:
            raise ValueError(f"{path} is shorter than seq_len={seq_len}")

    def __len__(self) -> int:
        return len(self.tokens) - self.seq_len

    def __getitem__(self, index: int) -> torch.Tensor:
        window = np.asarray(self.tokens[index : index + self.seq_len], dtype=np.int64)
        return torch.from_numpy(window)


def iter_jsonl_text(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)["text"]
