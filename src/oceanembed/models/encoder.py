"""OceanEmbed encoder -- the "satellite embedding engine".

CNN stem (residual conv blocks, two stride-2 stages -> H/4 x W/4) -> pre-norm Transformer encoder
with fixed 2-D sin-cos positions -> projection to an embedding map ``(emb_dim, H/4, W/4)``.
The stem features at full and half resolution are returned as skips for the reconstruction
decoder.

The encoder appends one internal "visible" channel to the ``in_channels`` dataset channels.
Pretraining sets it to 0 inside hidden blocks; everywhere else (in particular downstream, where
nothing is hidden) it is 1, so the interface is identical in both stages. Because the stem skips
are computed from the same (masked) input, they cannot leak hidden pixels.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from oceanembed.config import ModelConfig
from oceanembed.models.blocks import ResBlock


class EncoderOutput(NamedTuple):
    emb: torch.Tensor  # (B, emb_dim, H/4, W/4)
    skip1: torch.Tensor  # (B, stem_channels // 2, H, W)
    skip2: torch.Tensor  # (B, stem_channels, H/2, W/2)


def sincos_2d(dim: int, h: int, w: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Fixed 2-D sin-cos position embedding ``(h*w, dim)``; ``dim`` must be divisible by 4."""
    if dim % 4:
        raise ValueError("transformer dim must be divisible by 4 for 2-D sin-cos positions")
    q = dim // 4
    omega = 1.0 / (10000.0 ** (torch.arange(q, dtype=torch.float64) / q))
    ys = torch.arange(h, dtype=torch.float64)[:, None] * omega[None]  # (h, q)
    xs = torch.arange(w, dtype=torch.float64)[:, None] * omega[None]  # (w, q)
    ey = torch.cat([ys.sin(), ys.cos()], dim=1)[:, None, :].expand(h, w, 2 * q)
    ex = torch.cat([xs.sin(), xs.cos()], dim=1)[None, :, :].expand(h, w, 2 * q)
    pos = torch.cat([ey, ex], dim=2).reshape(h * w, dim)
    return pos.to(device=device, dtype=dtype)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).unbind(2)
        o = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )  # (B, heads, N, hd)
        return self.proj(o.transpose(1, 2).reshape(b, n, c))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


def unet_blocks(mc: ModelConfig) -> int:
    """Width-``dim`` residual conv blocks that replace the Transformer in the U-Net variant.

    A Transformer block has about ``(4 + 2 * mlp_ratio) * dim^2`` weights and a ResBlock
    ``18 * dim^2``; the count keeps the parameter budget of the two variants within a few percent
    (4 blocks instead of 6 at the default size).
    """
    return max(1, round(mc.depth * (4 + 2 * mc.mlp_ratio) / 18))


class OceanEncoder(nn.Module):
    def __init__(self, mc: ModelConfig):
        super().__init__()
        c = mc.stem_channels
        if c % 2:
            raise ValueError("stem_channels must be even")
        self.cfg = mc
        self.stem0 = nn.Sequential(
            nn.Conv2d(mc.in_channels + 1, c // 2, 3, padding=1), ResBlock(c // 2)
        )
        self.stem1 = nn.Sequential(nn.Conv2d(c // 2, c, 3, stride=2, padding=1), ResBlock(c))
        self.stem2 = nn.Sequential(nn.Conv2d(c, 2 * c, 3, stride=2, padding=1), ResBlock(2 * c))
        self.to_tokens = nn.Conv2d(2 * c, mc.dim, 1)
        if mc.arch == "unet":  # no attention: residual convolutions at the H/4 resolution
            self.blocks = nn.ModuleList(ResBlock(mc.dim) for _ in range(unet_blocks(mc)))
        else:
            self.blocks = nn.ModuleList(
                TransformerBlock(mc.dim, mc.heads, mc.mlp_ratio) for _ in range(mc.depth)
            )
        self.norm = nn.LayerNorm(mc.dim)
        self.proj = nn.Linear(mc.dim, mc.emb_dim)
        self._pos_cache: dict[tuple, torch.Tensor] = {}

    def _pos(self, h: int, w: int, like: torch.Tensor) -> torch.Tensor:
        key = (h, w, like.device, like.dtype)
        if key not in self._pos_cache:
            self._pos_cache[key] = sincos_2d(self.cfg.dim, h, w, like.device, like.dtype)
        return self._pos_cache[key]

    def forward(self, x: torch.Tensor, visible: torch.Tensor | None = None) -> EncoderOutput:
        """``x`` is ``(B, in_channels, H, W)``; ``visible`` an optional ``(B, 1, H, W)`` 0/1 map."""
        b, _, hh, ww = x.shape
        if hh % 4 or ww % 4:
            raise ValueError(f"H and W must be divisible by 4, got {hh}x{ww}")
        if visible is None:
            visible = torch.ones_like(x[:, :1])
        s1 = self.stem0(torch.cat([x, visible.to(x.dtype)], dim=1))
        s2 = self.stem1(s1)
        f = self.stem2(s2)
        t = self.to_tokens(f)
        h, w = t.shape[-2:]
        if self.cfg.arch == "unet":
            for blk in self.blocks:
                t = blk(t)
            tokens = t.flatten(2).transpose(1, 2)
        else:
            tokens = t.flatten(2).transpose(1, 2)
            tokens = tokens + self._pos(h, w, tokens).to(tokens.dtype)[None]
            for blk in self.blocks:
                tokens = blk(tokens)
        emb = self.proj(self.norm(tokens))  # (B, h*w, emb_dim)
        emb = emb.transpose(1, 2).reshape(b, self.cfg.emb_dim, h, w)
        return EncoderOutput(emb, s1, s2)
