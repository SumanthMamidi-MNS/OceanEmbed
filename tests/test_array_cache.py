"""The optional on-disk (memmap) cache of the standardised arrays: equivalence with the RAM preload,
round-off, resumability, tail views, gathers and training (tiny synthetic config, CPU)."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from oceanembed.config import Config, load_config
from oceanembed.data.dataset import OceanDataset, cache_dir, make_dataset
from oceanembed.data.stats import Stats
from oceanembed.models.baselines import fit_ridge, rmse_per_depth, sample_points
from oceanembed.train.mlp import run_train_mlp
from oceanembed.train.train import run_train

TINY = Path(__file__).resolve().parents[1] / "configs" / "test_tiny.yaml"


def _memmap_cfg(cfg: Config, dtype: str = "float16") -> Config:
    return cfg.model_copy(
        update={"train": cfg.train.model_copy(update={"cache": "memmap", "cache_dtype": dtype})}
    )


def _fresh(cfg: Config, split: str) -> OceanDataset:
    r = cfg.split.get(split)
    return OceanDataset(cfg.zarr_path, Stats.load(cfg.stats_path), r.start, r.end)


# ----------------------------------------------------------------------------------------
# the on-disk cache
# ----------------------------------------------------------------------------------------
def test_default_configs_keep_the_ram_preload(tiny_pipeline: Config):
    assert tiny_pipeline.train.cache == "ram"
    ds = make_dataset(tiny_pipeline, "val", preload=True)
    assert ds.preloaded and not ds.memory_mapped and isinstance(ds.arrays()["y"], np.ndarray)
    assert not isinstance(ds.arrays()["y"], np.memmap)
    long_cfg = load_config(TINY.parent / "poc_long.yaml")
    assert long_cfg.train.cache == "memmap" and long_cfg.train.cache_dtype == "float16"
    assert load_config(TINY.parent / "poc.yaml").train.cache == "ram"


def test_float32_cache_reproduces_the_preloaded_arrays_exactly(tiny_pipeline: Config, tmp_path):
    cfg = tiny_pipeline
    ram = make_dataset(cfg, "train", preload=True)
    mm = _fresh(cfg, "train").use_cache(tmp_path / "c", "float32")
    assert mm.memory_mapped and mm.preloaded
    for k in ("surf", "sv", "y", "valid"):
        assert isinstance(mm.arrays()[k], np.memmap)
        assert np.array_equal(ram.arrays()[k], mm.arrays()[k]), k
    for i in (0, 7, len(ram) - 1):
        a, b = ram[i], mm[i]
        assert all(torch.equal(a[k], b[k]) for k in a)
    meta = json.loads((tmp_path / "c" / "meta.json").read_text())
    assert meta["complete"] is True and meta["dtype"] == "float32" and meta["roundoff"] is None


def test_float16_cache_roundoff_is_negligible_and_recorded(tiny_pipeline: Config, tmp_path):
    cfg = tiny_pipeline
    ram = make_dataset(cfg, "train", preload=True)
    mm = _fresh(cfg, "train").use_cache(tmp_path / "c", "float16")
    meta = json.loads((tmp_path / "c" / "meta.json").read_text())
    ro = meta["roundoff"]
    assert ro["temperature_max_abs_degC"] < 0.02 and ro["temperature_rms_degC"] < 5e-3
    assert ro["surface_max_abs_standardised"] < 0.01
    for k in ("surf", "y"):
        assert mm.arrays()[k].dtype == np.float16
        err = np.abs(mm.arrays()[k].astype(np.float32) - ram.arrays()[k]).max()
        assert err < 0.01, k
    for k in ("sv", "valid"):
        assert np.array_equal(ram.arrays()[k], mm.arrays()[k])
    # the stored round-off equals the one measured here
    sd = ram.stats.anom_std.astype(np.float32)[None, :, None, None]
    ey = np.where(
        ram.arrays()["valid"],
        np.abs(mm.arrays()["y"].astype(np.float32) - ram.arrays()["y"]) * sd,
        0,
    )
    assert meta["roundoff"]["temperature_max_abs_degC"] == pytest.approx(float(ey.max()), rel=1e-5)
    assert mm[0]["x"].dtype == torch.float32 and mm[0]["y"].dtype == torch.float32


def test_cache_is_built_once_and_rebuilt_when_missing_partial_or_stale(
    tiny_pipeline: Config, tmp_path
):
    cfg = tiny_pipeline
    d = tmp_path / "c"
    ds = _fresh(cfg, "val")
    assert not ds.cache_ready(d, "float16")
    ds.use_cache(d, "float16")
    assert ds.cache_ready(d, "float16") and not ds.cache_ready(d, "float32")
    stamp = (d / "y.npy").stat().st_mtime_ns
    _fresh(cfg, "val").use_cache(d, "float16")  # a complete matching cache is reused
    assert (d / "y.npy").stat().st_mtime_ns == stamp
    # deleted file -> rebuilt (the maps of the first dataset are released first: Windows locks them)
    del ds
    gc.collect()
    (d / "y.npy").unlink()
    ds2 = _fresh(cfg, "val")
    assert not ds2.cache_ready(d, "float16")
    ds2.use_cache(d, "float16")
    assert (d / "y.npy").exists() and ds2.cache_ready(d, "float16")
    del ds2
    gc.collect()
    # a build that never finished (no meta.json, leftover .tmp) is never used
    (d / "meta.json").unlink()
    (tmp_path / "c.tmp").mkdir()
    (tmp_path / "c.tmp" / "junk").write_text("x")
    ds3 = _fresh(cfg, "val")
    assert not ds3.cache_ready(d, "float16")
    ds3.use_cache(d, "float16")
    assert ds3.cache_ready(d, "float16") and not (tmp_path / "c.tmp").exists()
    del ds3
    gc.collect()
    # changed statistics (a different normalisation / climatology) invalidate it
    ds4 = _fresh(cfg, "val")
    ds4.stats.input_mean = ds4.stats.input_mean + 1.0
    assert not ds4.cache_ready(d, "float16")
    # a different split range does too
    other = OceanDataset(cfg.zarr_path, Stats.load(cfg.stats_path), "2022-02-10", "2022-02-15")
    assert not other.cache_ready(d, "float16")


def test_make_dataset_uses_the_configured_cache(tiny_pipeline: Config):
    cfg = _memmap_cfg(tiny_pipeline)
    ds = make_dataset(cfg, "val", preload=True)
    assert ds.memory_mapped
    d = cache_dir(cfg, "val")
    assert d.parent.name == cfg.run_name and d.parent.parent.name == "cache"
    assert (d / "meta.json").exists() and "float16" in d.name
    assert not make_dataset(cfg, "val", preload=False).preloaded


def test_tail_view_and_gather_match_the_ram_dataset(tiny_pipeline: Config, tmp_path):
    cfg = tiny_pipeline
    ram = make_dataset(cfg, "train", preload=True)
    mm = _fresh(cfg, "train").use_cache(tmp_path / "c", "float32")
    rng = np.random.default_rng(0)
    t = rng.integers(0, len(ram), 300)
    p = rng.choice(np.flatnonzero(ram.mask[0].reshape(-1)), 300)
    for a, b in zip(ram.gather(t, p), mm.gather(t, p), strict=True):
        assert np.array_equal(a, b)
    assert np.array_equal(ram.gather(t, p, ("surf",))[0], mm.gather(t, p, ("surf",))[0])
    tail = mm.tail(10)
    assert len(tail) == 10 and tail.dates()[0] == ram.dates()[-10]
    assert np.array_equal(tail.arrays()["y"], ram.arrays()["y"][-10:])
    for i in (0, 9):
        assert torch.equal(tail[i]["x"], ram[len(ram) - 10 + i]["x"])
        assert torch.equal(tail[i]["y"], ram[len(ram) - 10 + i]["y"])
    assert len(mm.tail(10_000)) == len(mm)
    # the base dataset is untouched by taking a tail
    assert len(mm) == len(ram)


def test_sampling_ridge_and_rmse_work_on_a_cache_exactly_like_in_ram(
    tiny_pipeline: Config, tmp_path
):
    cfg = tiny_pipeline
    ram = make_dataset(cfg, "train", preload=True)
    mm = _fresh(cfg, "train").use_cache(tmp_path / "c", "float32")
    for x, y in zip(sample_points(ram, 800, 4), sample_points(mm, 800, 4), strict=True):
        assert np.array_equal(x, y)
    r1_ = fit_ridge(cfg, ram)
    r2_ = fit_ridge(cfg, mm)
    assert np.array_equal(r1_.coef, r2_.coef) and np.array_equal(r1_.intercept, r2_.intercept)
    assert np.allclose(rmse_per_depth(r1_, ram), rmse_per_depth(r1_, mm))


def test_training_on_a_float32_cache_is_identical_and_float16_is_close(
    tiny_pipeline: Config, tmp_path
):
    cfg = tiny_pipeline
    ram = (make_dataset(cfg, "train", preload=True), make_dataset(cfg, "val", preload=True))

    def cached(dtype: str):
        return tuple(
            _fresh(cfg, s).use_cache(tmp_path / f"{dtype}_{s}", dtype) for s in ("train", "val")
        )

    def fit(datasets, name):
        best = run_train(
            cfg, pretrained=False, device="cpu", progress=False, seed=1,
            ckpt_path=tmp_path / f"{name}.pt", log_path=tmp_path / f"{name}.jsonl",
            datasets=datasets,
        )  # fmt: skip
        return best, torch.load(tmp_path / f"{name}.pt", weights_only=False)["model"]

    ba, wa = fit(ram, "ram")
    bb, wb = fit(cached("float32"), "f32")
    bc, _ = fit(cached("float16"), "f16")
    assert all(torch.equal(wa[k], wb[k]) for k in wa) and ba["val_rmse"] == bb["val_rmse"]
    assert abs(bc["val_rmse"] - ba["val_rmse"]) < 0.02  # float16 round-off does not matter

    def fit_mlp(datasets, name):
        run_train_mlp(cfg, 1, "cpu", False, ckpt_path=tmp_path / f"{name}.pt", datasets=datasets)
        return torch.load(tmp_path / f"{name}.pt", weights_only=False)["model"]

    ma, mb = fit_mlp(ram, "mlp_ram"), fit_mlp(cached("float32"), "mlp_f32")
    assert all(torch.equal(ma[k], mb[k]) for k in ma)
