"""Masked losses for the reconstruction model (all inputs ``(B, D, H, W)``)."""

from __future__ import annotations

import torch


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over ``mask`` points; exactly 0 (and finite) when nothing is valid.

    Values under the mask never enter the result, not even as NaN/inf (``where`` is used rather
    than a multiplication, so a non-finite masked-out prediction cannot poison the loss).
    """
    m = mask.bool()
    diff = torch.where(m, pred.float() - target.float(), torch.zeros((), device=pred.device))
    return (diff**2).sum() / m.sum().clamp(min=1)


def vertical_gradient_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """MSE between predicted and true differences of adjacent depth levels, where both are valid."""
    m = mask.bool()
    m2 = m[:, 1:] & m[:, :-1]
    dp = pred.float()[:, 1:] - pred.float()[:, :-1]
    dt = target.float()[:, 1:] - target.float()[:, :-1]
    return masked_mse(dp, dt, m2)


def reconstruction_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, vertical_weight: float = 0.05
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(total, mse, vertical_gradient)`` with ``total = mse + vertical_weight * vgrad``."""
    mse = masked_mse(pred, target, mask)
    vg = vertical_gradient_loss(pred, target, mask)
    return mse + vertical_weight * vg, mse, vg
