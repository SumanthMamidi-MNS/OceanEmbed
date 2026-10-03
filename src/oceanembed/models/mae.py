"""Masked surface-reconstruction pretraining (the self-supervised embedding objective).

Random blocks of the 7 surface channels are hidden in the encoder input: hidden pixels are set to
0 and flagged in the encoder's internal "visible" channel. The ocean mask, day-of-year and lat/lon
channels are never hidden. A light conv decoder, fed **only** the embedding map (no stem skips),
reconstructs all 7 surface channels. The loss is the MSE on hidden pixels that are ocean and were
really observed.

Leakage: the encoder only ever sees the masked input, so its stem skip features carry no hidden
information either; the pretraining decoder nevertheless ignores them so that the embedding
itself has to encode the surface state.
"""

from __future__ import annotations

import torch
from torch import nn

from oceanembed.config import ModelConfig, PretrainConfig
from oceanembed.data.dataset import N_SURFACE
from oceanembed.models.blocks import ResBlock, Up
from oceanembed.models.encoder import OceanEncoder


def block_mask(
    batch: int,
    h: int,
    w: int,
    block: int,
    ratio: float,
    device=None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Random hidden-block map ``(B, 1, H, W)`` bool.

    The image is tiled by ``block x block`` blocks (ragged at the border); exactly
    ``round(ratio * n_blocks)`` blocks per sample are hidden.
    """
    nbh, nbw = -(-h // block), -(-w // block)
    n = nbh * nbw
    k = min(n, max(1, round(ratio * n))) if ratio > 0 else 0
    scores = torch.rand(batch, n, device=device, generator=generator)
    rank = scores.argsort(dim=1).argsort(dim=1)
    hidden = (rank < k).reshape(batch, 1, nbh, nbw)
    hidden = hidden.repeat_interleave(block, dim=2).repeat_interleave(block, dim=3)
    return hidden[:, :, :h, :w]


class PretrainDecoder(nn.Module):
    """Embedding map (H/4) -> 7 surface channels (H): two x2 upsampling stages."""

    def __init__(self, mc: ModelConfig, n_out: int = N_SURFACE):
        super().__init__()
        c = mc.stem_channels
        self.net = nn.Sequential(
            ResBlock(mc.emb_dim, c),
            Up(c, c),
            ResBlock(c),
            Up(c, c // 2),
            ResBlock(c // 2),
            nn.GroupNorm(min(8, c // 2), c // 2),
            nn.SiLU(),
            nn.Conv2d(c // 2, n_out, 1),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb)


class MaskedAutoencoder(nn.Module):
    def __init__(self, mc: ModelConfig, pc: PretrainConfig):
        super().__init__()
        self.pc = pc
        self.encoder = OceanEncoder(mc)
        self.decoder = PretrainDecoder(mc)

    def forward(
        self, x: torch.Tensor, sv: torch.Tensor, generator: torch.Generator | None = None
    ) -> dict[str, torch.Tensor]:
        """``x`` (B,12,H,W) dataset input, ``sv`` (B,7,H,W) bool observed-surface map.

        Returns the scalar ``loss`` plus per-channel sums over the loss pixels
        (``sq_err``, ``sq_meanfill`` (error of predicting 0), ``count``), each ``(7,)``.
        """
        b, _, h, w = x.shape
        hidden = block_mask(b, h, w, self.pc.block, self.pc.mask_ratio, x.device, generator)
        x_in = x.clone()
        x_in[:, :N_SURFACE] = x[:, :N_SURFACE].masked_fill(hidden, 0.0)
        emb = self.encoder(x_in, visible=(~hidden).to(x.dtype)).emb
        pred = self.decoder(emb).float()
        target = x[:, :N_SURFACE]
        m = hidden & sv  # (B,7,H,W): hidden, ocean and really observed
        mf = m.float()
        err2 = (pred - target) ** 2 * mf
        sq_err = err2.sum(dim=(0, 2, 3))
        count = mf.sum(dim=(0, 2, 3))
        sq_meanfill = (target**2 * mf).sum(dim=(0, 2, 3))
        loss = sq_err.sum() / count.sum().clamp(min=1.0)
        return {
            "loss": loss,
            "pred": pred,
            "hidden": hidden,
            "sq_err": sq_err.detach(),
            "sq_meanfill": sq_meanfill.detach(),
            "count": count.detach(),
        }
