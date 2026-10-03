"""Reconstruction model: encoder + U-Net-style decoder -> 15 standardised temperature anomalies."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from oceanembed.config import ModelConfig
from oceanembed.models.blocks import ResBlock, Up, groups_for
from oceanembed.models.encoder import OceanEncoder


class ReconDecoder(nn.Module):
    """Embedding (H/4) -> up x2 (+ skip at H/2) -> up x2 (+ skip at H) -> 1x1 head."""

    def __init__(self, mc: ModelConfig):
        super().__init__()
        c = mc.stem_channels
        self.inp = ResBlock(mc.emb_dim, 2 * c)
        self.up1 = Up(2 * c, c)
        self.blk1 = ResBlock(c + c, c)  # upsampled + skip2 (c channels)
        self.up2 = Up(c, c // 2)
        self.blk2 = ResBlock(c // 2 + c // 2, c // 2)  # upsampled + skip1 (c//2 channels)
        self.head = nn.Sequential(
            nn.GroupNorm(groups_for(c // 2), c // 2),
            nn.SiLU(),
            nn.Conv2d(c // 2, mc.n_depths, 1),
        )

    def forward(self, emb: torch.Tensor, skip1: torch.Tensor, skip2: torch.Tensor) -> torch.Tensor:
        h = self.inp(emb)
        h = self.blk1(torch.cat([self.up1(h), skip2], dim=1))
        h = self.blk2(torch.cat([self.up2(h), skip1], dim=1))
        return self.head(h)


class ReconModel(nn.Module):
    def __init__(self, mc: ModelConfig):
        super().__init__()
        self.encoder = OceanEncoder(mc)
        self.decoder = ReconDecoder(mc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` (B,12,H,W) -> standardised anomaly ``(B,15,H,W)`` (nothing is hidden)."""
        out = self.encoder(x)
        return self.decoder(out.emb, out.skip1, out.skip2)

    def load_pretrained_encoder(self, path: str | Path, map_location="cpu") -> None:
        self.encoder.load_state_dict(load_encoder_state(path, map_location), strict=True)

    def param_groups(self, lr: float, encoder_lr_scale: float, weight_decay: float) -> list[dict]:
        """AdamW groups: {encoder, decoder} x {decay, no decay}; encoder LR is scaled."""
        groups = []
        for name, module, scale in (
            ("encoder", self.encoder, encoder_lr_scale),
            ("decoder", self.decoder, 1.0),
        ):
            decay = [p for p in module.parameters() if p.ndim >= 2]
            plain = [p for p in module.parameters() if p.ndim < 2]
            groups.append(
                {"params": decay, "lr": lr * scale, "weight_decay": weight_decay, "name": name}
            )
            groups.append({"params": plain, "lr": lr * scale, "weight_decay": 0.0, "name": name})
        return groups


def load_encoder_state(path: str | Path, map_location="cpu") -> dict[str, torch.Tensor]:
    """Encoder weights from a ``pretrain.pt`` or ``recon*.pt`` checkpoint."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    state = ckpt["model"]
    prefix = "encoder."
    return {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
