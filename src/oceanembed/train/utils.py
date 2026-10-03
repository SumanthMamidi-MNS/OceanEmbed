"""Shared training helpers: seeding, LR schedule, JSON-lines logging, data loaders."""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy, torch and every CUDA device; ask cuDNN for deterministic kernels.

    On CPU the same seed reproduces a run bit for bit. On CUDA it reproduces it closely but not
    exactly: backward passes through nearest-neighbour upsampling and scaled-dot-product attention
    use atomic additions, and fp16 autocast reductions are order dependent.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def use_progress_bars(progress: bool) -> bool:
    """Progress bars only on an interactive terminal (a redirected log would fill with updates)."""
    return bool(progress) and sys.stderr.isatty()


def get_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_warmup(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int):
    """Linear warmup to the base LR, then cosine decay to 5 % of it."""
    total_steps = max(total_steps, 1)

    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class JsonlLogger:
    """Append one JSON object per line (truncates an existing file on creation)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, record: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")


def make_loader(
    ds: Dataset, batch_size: int, shuffle: bool, num_workers: int = 0, seed: int = 0
) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle and len(ds) >= batch_size,
        num_workers=num_workers,
        generator=g if shuffle else None,
        pin_memory=torch.cuda.is_available(),
    )


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def random_crop(batch: dict, size: tuple[int, int] | None, rng: np.random.Generator) -> dict:
    """Same random ``(h, w)`` spatial crop (corner on a multiple of 4) of every spatial tensor.

    Spatial tensors are those with ndim >= 3 (``x``, ``y``, ``mask``, ``sv``); the lat/lon channels
    of ``x`` keep their absolute values. A ``None`` size, or a crop at least as large as the
    batch, returns the batch unchanged.
    """
    if size is None:
        return batch
    if size[0] % 4 or size[1] % 4:
        raise ValueError(f"crop {size} must be a multiple of 4")
    h, w = batch["x"].shape[-2:]
    ch, cw = min(size[0], h), min(size[1], w)
    if (ch, cw) == (h, w):
        return batch
    i = int(rng.integers(0, (h - ch) // 4 + 1)) * 4
    j = int(rng.integers(0, (w - cw) // 4 + 1)) * 4
    return {k: v[..., i : i + ch, j : j + cw] if v.ndim >= 3 else v for k, v in batch.items()}
