"""Unit tests for the encoder, masking, losses and decoders (tiny models, CPU)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from oceanembed.config import ModelConfig, PretrainConfig
from oceanembed.models.encoder import OceanEncoder, sincos_2d
from oceanembed.models.mae import MaskedAutoencoder, block_mask
from oceanembed.models.recon import ReconModel
from oceanembed.train.losses import masked_mse, reconstruction_loss, vertical_gradient_loss

MC = ModelConfig(emb_dim=16, dim=32, depth=2, heads=2, stem_channels=16)


@pytest.mark.parametrize("hw", [(16, 24), (8, 12), (20, 36)])
def test_encoder_and_decoder_shapes(hw):
    h, w = hw
    x = torch.randn(2, 12, h, w)
    out = OceanEncoder(MC)(x)
    assert out.emb.shape == (2, 16, h // 4, w // 4)
    assert out.skip1.shape == (2, 8, h, w)
    assert out.skip2.shape == (2, 16, h // 2, w // 2)
    assert ReconModel(MC)(x).shape == (2, 15, h, w)
    assert MaskedAutoencoder(MC, PretrainConfig(block=4))(x, torch.ones(2, 7, h, w).bool())[
        "pred"
    ].shape == (2, 7, h, w)


def test_encoder_rejects_bad_grid():
    with pytest.raises(ValueError):
        OceanEncoder(MC)(torch.randn(1, 12, 10, 12))


def test_sincos_positions_are_distinct():
    pos = sincos_2d(32, 5, 7)
    assert pos.shape == (35, 32)
    assert len({tuple(np.round(r, 5)) for r in pos.numpy()}) == 35
    with pytest.raises(ValueError):
        sincos_2d(30, 4, 4)


@pytest.mark.parametrize("ratio", [0.25, 0.5, 0.75])
def test_block_mask_ratio_and_blockiness(ratio):
    m = block_mask(6, 16, 24, 4, ratio, generator=torch.Generator().manual_seed(0))
    assert m.shape == (6, 1, 16, 24) and m.dtype == torch.bool
    assert m.float().mean().item() == pytest.approx(ratio, abs=1e-6)
    # every 4x4 block is entirely hidden or entirely visible
    blocks = m.reshape(6, 1, 4, 4, 6, 4).float().mean(dim=(3, 5))
    assert set(blocks.unique().tolist()) <= {0.0, 1.0}
    # different samples get different masks
    assert not torch.equal(m[0], m[1])


def test_block_mask_ragged_grid_is_cropped():
    m = block_mask(2, 10, 14, 4, 0.5, generator=torch.Generator().manual_seed(1))
    assert m.shape == (2, 1, 10, 14)


def test_masking_hides_only_surface_channels():
    mae = MaskedAutoencoder(MC, PretrainConfig(block=4, mask_ratio=0.5))
    seen = {}
    orig = mae.encoder.forward

    def spy(x, visible=None):
        seen["x"], seen["visible"] = x.clone(), visible.clone()
        return orig(x, visible)

    mae.encoder.forward = spy
    x = torch.randn(3, 12, 16, 24) + 5.0  # nothing is accidentally 0
    sv = torch.ones(3, 7, 16, 24, dtype=torch.bool)
    out = mae(x, sv)
    hidden = out["hidden"]
    assert hidden.float().mean().item() == pytest.approx(0.5, abs=1e-6)
    # auxiliary channels (mask, doy sin/cos, lat, lon) untouched
    assert torch.equal(seen["x"][:, 7:], x[:, 7:])
    # surface channels are zero exactly where hidden, untouched elsewhere
    assert torch.all(seen["x"][:, :7][hidden.expand(-1, 7, -1, -1)] == 0)
    keep = (~hidden).expand(-1, 7, -1, -1)
    assert torch.equal(seen["x"][:, :7][keep], x[:, :7][keep])
    # the visibility flag is the complement of the hidden map
    assert torch.equal(seen["visible"].bool(), ~hidden)


def test_hidden_values_cannot_influence_the_prediction():
    """Changing the surface values under the hidden blocks must not change the output."""
    torch.manual_seed(0)
    mae = MaskedAutoencoder(MC, PretrainConfig(block=4, mask_ratio=0.5)).eval()
    x = torch.randn(2, 12, 16, 24)
    sv = torch.ones(2, 7, 16, 24, dtype=torch.bool)
    g1, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    o1 = mae(x, sv, generator=g1)
    x2 = x.clone()
    hid = o1["hidden"].expand(-1, 7, -1, -1)
    x2[:, :7][hid] = torch.randn(int(hid.sum()))
    o2 = mae(x2, sv, generator=g2)
    assert torch.equal(o1["hidden"], o2["hidden"])
    assert torch.allclose(o1["pred"], o2["pred"], atol=1e-6)


def test_mae_loss_only_on_hidden_observed_pixels():
    mae = MaskedAutoencoder(MC, PretrainConfig(block=4, mask_ratio=0.5))
    x = torch.randn(2, 12, 16, 24)
    sv = torch.rand(2, 7, 16, 24) > 0.3
    out = mae(x, sv)
    m = (out["hidden"] & sv).float()
    assert out["count"].sum().item() == pytest.approx(m.sum().item())
    expected = ((out["pred"] - x[:, :7]) ** 2 * m).sum() / m.sum()
    assert out["loss"].item() == pytest.approx(expected.item(), rel=1e-5)
    # mean-fill error equals the sum of squares of the targets on those pixels
    assert out["sq_meanfill"].sum().item() == pytest.approx(
        ((x[:, :7] ** 2) * m).sum().item(), rel=1e-5
    )


def test_mae_all_invalid_gives_zero_loss():
    mae = MaskedAutoencoder(MC, PretrainConfig(block=4))
    out = mae(torch.randn(1, 12, 16, 24), torch.zeros(1, 7, 16, 24, dtype=torch.bool))
    assert out["loss"].item() == 0.0


# ----- losses ------------------------------------------------------------------------------


def test_masked_mse_ignores_masked_values():
    torch.manual_seed(0)
    pred = torch.randn(2, 4, 5, 6)
    y = torch.randn(2, 4, 5, 6)
    mask = torch.rand(2, 4, 5, 6) > 0.4
    base = reconstruction_loss(pred, y, mask, 0.1)[0]
    pred2, y2 = pred.clone(), y.clone()
    pred2[~mask] = 1e6
    y2[~mask] = -1e6
    assert reconstruction_loss(pred2, y2, mask, 0.1)[0].item() == pytest.approx(base.item())
    pred2[~mask] = float("nan")
    assert torch.isfinite(masked_mse(pred2, y2, mask))


def test_masked_mse_value_and_zero_safety():
    pred = torch.tensor([[[[1.0, 2.0]]]])
    y = torch.tensor([[[[0.0, 0.0]]]])
    assert masked_mse(pred, y, torch.tensor([[[[True, False]]]])).item() == 1.0
    assert masked_mse(pred, y, torch.zeros_like(y).bool()).item() == 0.0


def test_losses_finite_with_fully_masked_depth_and_gradients_clean():
    pred = torch.randn(2, 5, 4, 4, requires_grad=True)
    y = torch.randn(2, 5, 4, 4)
    mask = torch.ones(2, 5, 4, 4, dtype=torch.bool)
    mask[:, 2] = False  # a whole depth below the sea floor
    total, mse, vg = reconstruction_loss(pred, y, mask, 0.1)
    assert all(torch.isfinite(t) for t in (total, mse, vg))
    total.backward()
    assert torch.isfinite(pred.grad).all()
    assert (pred.grad[:, 2] == 0).all()
    # everything masked -> exactly zero loss and zero gradient
    pred.grad = None
    t0 = reconstruction_loss(pred, y, torch.zeros_like(mask), 0.1)[0]
    t0.backward()
    assert t0.item() == 0.0 and (pred.grad == 0).all()


def test_vertical_gradient_only_where_both_levels_valid():
    y = torch.zeros(1, 3, 1, 1)
    pred = torch.tensor([[[[1.0]], [[5.0]], [[9.0]]]])
    mask = torch.tensor([[[[True]], [[False]], [[True]]]])
    # levels 0-1 and 1-2 each touch the masked level -> nothing valid
    assert vertical_gradient_loss(pred, y, mask).item() == 0.0
    mask2 = torch.tensor([[[[True]], [[True]], [[False]]]])
    assert vertical_gradient_loss(pred, y, mask2).item() == pytest.approx(16.0)  # (5-1)^2


# ----- pretrained loading + overfit --------------------------------------------------------


def test_pretrained_encoder_loads_into_recon(tmp_path):
    torch.manual_seed(0)
    mae = MaskedAutoencoder(MC, PretrainConfig(block=4))
    path = tmp_path / "pretrain.pt"
    torch.save({"model": mae.state_dict(), "config": {}, "epoch": 1, "metrics": {}}, path)
    recon = ReconModel(MC)
    assert not torch.equal(recon.encoder.proj.weight, mae.encoder.proj.weight)
    recon.load_pretrained_encoder(path)
    for (k, a), (_, b) in zip(
        recon.encoder.state_dict().items(), mae.encoder.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b), k
    # decoder untouched by loading, and a forward pass still works
    assert recon(torch.randn(1, 12, 16, 24)).shape == (1, 15, 16, 24)


def test_param_groups_scale_encoder_lr():
    groups = ReconModel(MC).param_groups(1e-3, 0.1, 0.01)
    lrs = {g["name"]: g["lr"] for g in groups}
    assert lrs["encoder"] == pytest.approx(1e-4) and lrs["decoder"] == pytest.approx(1e-3)
    n = sum(len(g["params"]) for g in groups)
    assert n == len(list(ReconModel(MC).parameters()))


def test_overfit_pretraining_batch():
    torch.manual_seed(0)
    mae = MaskedAutoencoder(MC, PretrainConfig(block=4, mask_ratio=0.5))
    x = torch.randn(2, 12, 16, 24)
    sv = torch.ones(2, 7, 16, 24, dtype=torch.bool)
    opt = torch.optim.AdamW(mae.parameters(), lr=3e-3)
    losses = []
    for _ in range(60):
        out = mae(x, sv, generator=torch.Generator().manual_seed(1))  # same hidden blocks
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        losses.append(out["loss"].item())
    assert losses[-1] < 0.5 * losses[0]


def test_overfit_reconstruction_batch():
    torch.manual_seed(0)
    model = ReconModel(MC)
    x = torch.randn(2, 12, 16, 24)
    y = torch.randn(2, 15, 16, 24)
    mask = torch.rand(2, 15, 16, 24) > 0.3
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(60):
        loss, _, _ = reconstruction_loss(model(x), y, mask, 0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
