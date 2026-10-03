"""Dataset preloading, baselines, prediction helpers and CLI smoke tests (tiny config, CPU)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr
from typer.testing import CliRunner

from oceanembed.cli import app
from oceanembed.config import Config
from oceanembed.data.dataset import make_dataset
from oceanembed.infer.predict import load_recon_model, model_predictor, predict_batch
from oceanembed.models.baselines import (
    ClimatologyBaseline,
    RidgeBaseline,
    fit_ridge,
    rmse_per_depth,
)

TINY = Path(__file__).resolve().parents[1] / "configs" / "test_tiny.yaml"
runner = CliRunner()


@pytest.fixture
def outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("OCEANEMBED_OUTPUTS_ROOT", str(tmp_path / "out"))
    return tmp_path / "out"


def test_outputs_root_env_override(tiny_cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("OCEANEMBED_OUTPUTS_ROOT", str(tmp_path / "o"))
    assert tiny_cfg.outputs_dir == tmp_path / "o" / "test_tiny"
    monkeypatch.delenv("OCEANEMBED_OUTPUTS_ROOT")
    assert tiny_cfg.outputs_dir.parts[-2:] == ("outputs", "test_tiny")


def test_preloaded_dataset_matches_lazy(tiny_pipeline: Config):
    lazy = make_dataset(tiny_pipeline, "val")
    fast = make_dataset(tiny_pipeline, "val", preload=True)
    assert fast.preloaded and not lazy.preloaded and len(lazy) == len(fast)
    for i in (0, 3, -1):
        a, b = lazy[i], fast[i]
        assert set(a) == {"x", "y", "mask", "sv", "t", "index"}
        for k in a:
            assert torch.equal(a[k], b[k]), k
    assert a["x"].shape[0] == 12 and a["sv"].shape[0] == 7


def test_ridge_roundtrip_and_shapes(tiny_pipeline: Config, tmp_path):
    train = make_dataset(tiny_pipeline, "train", preload=True)
    ridge = fit_ridge(tiny_pipeline, train)
    assert ridge.coef.shape == (15, 11) and ridge.intercept.shape == (15,)
    path = tmp_path / "ridge.joblib"
    ridge.save(path)
    loaded = RidgeBaseline.load(path)
    x = train[0]["x"]
    one = loaded.predict(x.numpy())
    assert one.shape == (15, *train.shape)
    many = loaded.predict(torch.stack([train[0]["x"], train[1]["x"]]).numpy())
    assert many.shape == (2, 15, *train.shape)
    np.testing.assert_allclose(many[0], one, atol=1e-6)
    np.testing.assert_allclose(ridge.predict(x.numpy()), one, atol=1e-6)
    # linear in its features: matches an explicit matrix product at one pixel
    f = x[[0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11], 3, 4].numpy()
    np.testing.assert_allclose(one[:, 3, 4], loaded.coef @ f + loaded.intercept, atol=1e-5)


def test_ridge_beats_climatology_on_signal_and_handles_sea_floor(tiny_pipeline: Config):
    train = make_dataset(tiny_pipeline, "train", preload=True)
    ridge = fit_ridge(tiny_pipeline, train)
    # every depth was fitted from points valid at that depth only
    n_fit = ridge.meta["n_fit_per_depth"]
    assert n_fit[0] > n_fit[-1] > 0
    r = rmse_per_depth(ridge, train)
    c = rmse_per_depth(ClimatologyBaseline(15), train)
    assert np.isfinite(r).all() and np.isfinite(c).all()
    assert r[:5].mean() < c[:5].mean()  # in-sample, near the surface


def test_climatology_baseline_is_zero_anomaly():
    out = ClimatologyBaseline(15)(torch.randn(2, 12, 8, 12))
    assert out.shape == (2, 15, 8, 12) and out.abs().sum() == 0


def test_predict_batch_roundtrip_and_nan_mask(tiny_pipeline: Config):
    ds = make_dataset(tiny_pipeline, "test", preload=True)
    batch = {k: v[None] if v.ndim == 0 else v[None] for k, v in ds[0].items()}
    batch["t"] = ds[0]["t"].reshape(1)
    temp = predict_batch(lambda x: batch["y"], batch, ds, "cpu")  # perfect predictor
    assert temp.shape == (1, 15, *ds.shape) and temp.dtype == np.float32
    assert np.isnan(temp[0][~ds.mask]).all() and np.isfinite(temp[0][ds.mask]).all()
    truth = xr.open_zarr(tiny_pipeline.zarr_path, consolidated=False)["temp"].isel(
        time=int(ds.indices[0])
    )
    valid = batch["mask"][0].numpy()
    np.testing.assert_allclose(temp[0][valid], truth.values[valid], atol=2e-4)


def _invoke(*args):
    res = runner.invoke(app, [*map(str, args)])
    assert res.exit_code == 0, res.output + str(res.exception)
    return res.output


def test_cli_pretrain_train_baseline_embed(tiny_pipeline: Config, outputs):
    out = _invoke("pretrain", "--config", TINY, "--device", "cpu")
    assert "mean-fill" in out
    ck = torch.load(outputs / "test_tiny/checkpoints/pretrain.pt", weights_only=False)
    assert ck["epoch"] == 1 and "val_loss" in ck["metrics"] and "model" in ck
    assert Config.model_validate(ck["config"]).run_name == "test_tiny"
    log = (outputs / "test_tiny/logs/pretrain.jsonl").read_text().strip().splitlines()
    assert len(log) == 1 and json.loads(log[0])["epoch"] == 1

    out = _invoke("train", "--config", TINY, "--device", "cpu")
    assert "val RMSE" in out
    model, cfg = load_recon_model(outputs / "test_tiny/checkpoints/recon.pt")
    assert cfg.model.dim == 32
    rec = json.loads((outputs / "test_tiny/logs/train.jsonl").read_text().splitlines()[0])
    assert len(rec["val_rmse_per_depth"]) == 15 and rec["val_rmse"] > 0
    # the pretrained encoder really was the starting point (weights differ only by fine-tuning)
    pre = torch.load(outputs / "test_tiny/checkpoints/pretrain.pt", weights_only=False)["model"]
    w0 = pre["encoder.proj.weight"]
    w1 = model.encoder.proj.weight.detach()
    assert (w1 - w0).abs().max() < 0.05

    out = _invoke(
        "train", "--config", TINY, "--no-pretrained", "--tag", "scratch", "--device", "cpu"
    )
    assert (outputs / "test_tiny/checkpoints/recon_scratch.pt").exists()
    assert (outputs / "test_tiny/logs/train_scratch.jsonl").exists()

    out = _invoke("baseline", "--config", TINY)
    assert "ridge" in out
    assert (outputs / "test_tiny/checkpoints/ridge.joblib").exists()

    _invoke("embed", "--config", TINY, "--split", "test", "--device", "cpu")
    ds = xr.open_zarr(outputs / "test_tiny/embeddings/embeddings.zarr", consolidated=False)
    emb = ds["embedding"]
    assert emb.dims == ("time", "emb", "y", "x")
    assert emb.shape == (10, 16, 4, 6)  # 10 test days, emb_dim 16, 16x24 / 4
    assert "lat" in ds.coords and "lon" in ds.coords and "time" in ds.coords
    assert ds["lat"].values[0] == pytest.approx(8.0 + 0.5)  # centre of the first 4 cells
    assert np.isfinite(emb.values).all()

    # predictor helper on the trained model
    test = make_dataset(tiny_pipeline, "test", preload=True)
    batch = {"x": test[0]["x"][None], "t": test[0]["t"].reshape(1)}
    temp = predict_batch(model_predictor(model, amp=False), batch, test, "cpu")
    assert temp.shape == (1, 15, *test.shape) and np.isnan(temp[0][~test.mask]).all()


def test_train_requires_pretrained_checkpoint(tiny_pipeline: Config, outputs):
    res = runner.invoke(app, ["train", "--config", str(TINY), "--device", "cpu"])
    assert res.exit_code != 0 and isinstance(res.exception, FileNotFoundError)


def test_random_crop_is_consistent_and_optional():
    from oceanembed.train.utils import random_crop

    rng = np.random.default_rng(0)
    batch = {
        "x": torch.arange(2 * 3 * 16 * 24, dtype=torch.float32).reshape(2, 3, 16, 24),
        "y": torch.arange(2 * 2 * 16 * 24, dtype=torch.float32).reshape(2, 2, 16, 24),
        "t": torch.tensor([1, 2]),
    }
    assert random_crop(batch, None, rng) is batch
    assert random_crop(batch, (16, 24), rng) is batch
    out = random_crop(batch, (8, 12), rng)
    assert out["x"].shape == (2, 3, 8, 12) and out["y"].shape == (2, 2, 8, 12)
    assert torch.equal(out["t"], batch["t"])
    # x and y were cropped at the same corner
    i, j = (int(out["x"][0, 0, 0, 0]) // 24, int(out["x"][0, 0, 0, 0]) % 24)
    assert i % 4 == 0 and j % 4 == 0
    assert int(out["y"][0, 0, 0, 0]) == i * 24 + j
    with pytest.raises(ValueError):
        random_crop(batch, (6, 12), rng)
