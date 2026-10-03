"""Per-pixel MLP baseline (research stage R1).

The non-linear counterpart of the ridge baseline: the same 11 per-pixel features (the 7
standardised surface values, sin / cos day of year, normalised lat / lon), one shared network that
outputs the 15 standardised anomalies. No spatial context, so it measures how much a pixel's own
surface state explains. Like every predictor it maps ``x (B,12,H,W)`` to ``(B,15,H,W)``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from oceanembed.models.baselines import FEATURE_CHANNELS, N_FEATURES


class PixelMLP(nn.Module):
    def __init__(
        self,
        hidden: int = 256,
        layers: int = 3,
        n_features: int = N_FEATURES,
        n_depths: int = 15,
        channels: list[int] | None = None,
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        # input channels used as features; the default is the ridge feature set. A different list
        # (research stage R3: extra channels holding the previous days) fixes ``n_features``.
        self.channels = list(FEATURE_CHANNELS if channels is None else channels)
        if channels is not None:
            n_features = len(self.channels)
        mods: list[nn.Module] = [nn.Linear(n_features, hidden), nn.SiLU()]
        for _ in range(layers - 1):
            mods += [nn.Linear(hidden, hidden), nn.SiLU()]
        mods.append(nn.Linear(hidden, n_depths))
        self.net = nn.Sequential(*mods)
        self.spec = {
            "hidden": hidden,
            "layers": layers,
            "n_features": n_features,
            "n_depths": n_depths,
            **({} if channels is None else {"channels": self.channels}),
        }

    def forward_features(self, f: torch.Tensor) -> torch.Tensor:
        """``(N, n_features)`` -> ``(N, n_depths)``."""
        return self.net(f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x (B,12,H,W)`` -> standardised anomaly ``(B,15,H,W)``."""
        f = x[:, self.channels].permute(0, 2, 3, 1)  # (B,H,W,F)
        return self.net(f).permute(0, 3, 1, 2)


def load_pixel_mlp(path: str | Path, device=None) -> tuple[PixelMLP, dict]:
    """Rebuild a :class:`PixelMLP` (eval mode) from a checkpoint written by ``run_train_mlp``."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = PixelMLP(**ckpt["spec"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    if device is not None:
        model.to(device)
    meta = {
        "path": str(path),
        "epoch": ckpt.get("epoch"),
        "val_rmse": (ckpt.get("metrics") or {}).get("val_rmse"),
        "seed": ckpt.get("seed"),
    }
    return model, meta
