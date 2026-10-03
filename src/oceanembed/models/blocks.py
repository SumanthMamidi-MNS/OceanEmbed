"""Small shared conv building blocks."""

from __future__ import annotations

import torch
from torch import nn


def groups_for(channels: int) -> int:
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    """Pre-activation residual block: (GN-SiLU-conv3x3) x 2 with a 1x1 shortcut when needed."""

    def __init__(self, cin: int, cout: int | None = None):
        super().__init__()
        cout = cout or cin
        self.norm1 = nn.GroupNorm(groups_for(cin), cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups_for(cout), cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.act = nn.SiLU()
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class Up(nn.Module):
    """Nearest x2 upsample followed by a 3x3 conv."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))
