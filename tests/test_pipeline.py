"""synthetic -> harmonize -> stats -> dataset integration tests on the tiny config."""

from __future__ import annotations

import pickle
import shutil

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
import zarr
from torch.utils.data import DataLoader

from oceanembed.config import SURFACE_VARS, Config, load_config
from oceanembed.data.dataset import INPUT_CHANNELS, N_INPUT_CHANNELS, make_dataset
from oceanembed.data.harmonize import VAR_ATTRS, open_harmonized
from oceanembed.data.providers.base import ARGO_COLUMNS, raw_file
from oceanembed.data.providers.synthetic import GLORYS_LEVELS, SyntheticProvider, generate_all
from oceanembed.data.stats import Stats, compute_stats, harmonic_design, split_indices
from oceanembed.grid import build_grid

# --------------------------------------------------------------------------- raw products


def test_raw_layout_native_resolutions_and_conventions(tiny_pipeline):
    cfg = tiny_pipeline
    r = cfg.raw_root
    for p in ("sst", "sss", "sla", "currents", "winds", "temp"):
        assert [f.name for f in sorted((r / p).glob("*.nc"))] == [
            f"{p}_202201.nc", f"{p}_202202.nc", f"{p}_202203.nc"
        ]  # fmt: skip
    sst = xr.open_dataset(raw_file(cfg, "sst", pd.Timestamp("2022-01-01")))
    assert np.diff(sst.latitude.values)[0] == pytest.approx(0.05)  # OSTIA 0.05 deg
    assert sst["analysed_sst"].attrs["units"] == "kelvin"
    assert float(sst["analysed_sst"].min()) > 270.0
    assert pd.Timestamp(sst.time.values[0]).hour == 12  # OSTIA-style noon stamp
    sss = xr.open_dataset(raw_file(cfg, "sss", pd.Timestamp("2022-01-01")))
    assert np.diff(sss.latitude.values)[0] == pytest.approx(0.125)
    sla = xr.open_dataset(raw_file(cfg, "sla", pd.Timestamp("2022-01-01")))
    assert np.diff(sla.longitude.values)[0] == pytest.approx(0.25)
    cur = xr.open_dataset(raw_file(cfg, "currents", pd.Timestamp("2022-01-01")))
    assert cur["u"].dims == ("time", "longitude", "latitude")  # OSCAR order
    assert np.all(np.diff(cur.latitude.values) < 0)  # descending latitude
    wind = xr.open_dataset(raw_file(cfg, "winds", pd.Timestamp("2022-01-01")))
    assert wind.sizes["time"] == 31 * 4  # 6-hourly
    assert set(wind.data_vars) == {"uwnd", "vwnd"}
    temp = xr.open_dataset(raw_file(cfg, "temp", pd.Timestamp("2022-01-01")))
    assert np.diff(temp.latitude.values)[0] == pytest.approx(1 / 12, abs=1e-5)
    assert temp.depth.max() > 1000 and temp.sizes["depth"] == len(GLORYS_LEVELS)
    assert not set([5.0, 10.0, 20.0]) & set(temp.depth.values.round(3))  # non-standard levels
    assert np.isnan(temp["thetao"].values).any()  # land / below sea floor


def test_synthetic_argo_schema(tiny_pipeline):
    df = pd.read_parquet(raw_file(tiny_pipeline, "argo", pd.Timestamp("2022-01-01")))
    assert list(df.columns) == ARGO_COLUMNS
    assert df.profile_id.nunique() > 5
    g = tiny_pipeline.grid
    assert df.latitude.between(g.lat_min, g.lat_max).all()
    assert df.longitude.between(g.lon_min, g.lon_max).all()
    assert df.time.min() >= pd.Timestamp("2022-01-01") and df.time.max() < pd.Timestamp(
        "2022-02-01"
    )
    # irregular levels, plausible temperatures that decrease with depth on average
    assert df.groupby("profile_id").pres.apply(lambda s: s.is_monotonic_increasing).all()
    top = df[df.depth < 20].temp.mean()
    deep = df[df.depth > 800].temp.mean()
    assert top > deep + 10


def test_synthetic_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setenv("OCEANEMBED_DATA_ROOT", str(tmp_path / "a"))
    cfg = load_config("configs/test_tiny.yaml")
    SyntheticProvider(cfg).fetch("sla", cfg.time.start, cfg.time.end)
    a = xr.open_dataset(raw_file(cfg, "sla", pd.Timestamp("2022-02-01"))).load()
    monkeypatch.setenv("OCEANEMBED_DATA_ROOT", str(tmp_path / "b"))
    cfg = load_config("configs/test_tiny.yaml")
    SyntheticProvider(cfg).fetch("sla", cfg.time.start, cfg.time.end)
    b = xr.open_dataset(raw_file(cfg, "sla", pd.Timestamp("2022-02-01"))).load()
    xr.testing.assert_identical(a, b)
    other = cfg.model_copy(update={"synthetic": cfg.synthetic.model_copy(update={"seed": 99})})
    monkeypatch.setenv("OCEANEMBED_DATA_ROOT", str(tmp_path / "c"))
    SyntheticProvider(other).fetch("sla", other.time.start, other.time.end)
    c = xr.open_dataset(raw_file(other, "sla", pd.Timestamp("2022-02-01"))).load()
    assert not np.allclose(a["sla"].values, c["sla"].values, equal_nan=True)


# --------------------------------------------------------------------------- harmonised store


def test_zarr_structure_and_attrs(tiny_pipeline):
    cfg = tiny_pipeline
    ds = open_harmonized(cfg)
    g = build_grid(cfg)
    np.testing.assert_allclose(ds.lat.values, g.lat)
    np.testing.assert_allclose(ds.lon.values, g.lon)
    np.testing.assert_allclose(ds.depth.values, g.depth)
    assert len(ds.time) == 60
    assert (np.diff(ds.time.values) == np.timedelta64(1, "D")).all()
    assert ds.time.values[0] == np.datetime64("2022-01-01")
    for v in SURFACE_VARS:
        assert ds[v].dims == ("time", "lat", "lon") and ds[v].dtype == np.float32
        assert ds[v].attrs["units"] == VAR_ATTRS[v]["units"]
        assert ds[v].attrs["standard_name"] == VAR_ATTRS[v]["standard_name"]
    assert ds["temp"].dims == ("time", "depth", "lat", "lon") and ds["temp"].dtype == np.float32
    assert ds["temp"].attrs["units"] == "degC"
    assert ds["mask"].dims == ("depth", "lat", "lon") and ds["mask"].dtype == bool
    # one day per chunk
    assert ds["temp"].encoding["chunks"] == (1, 15, 16, 24)
    assert ds["sst"].encoding["chunks"] == (1, 16, 24)


def test_physical_ranges_and_units(tiny_pipeline):
    ds = open_harmonized(tiny_pipeline)
    sst = ds.sst.values
    assert 20 < np.nanmin(sst) and np.nanmax(sst) < 35  # degC, not kelvin
    assert 30 < np.nanmin(ds.sss.values) and np.nanmax(ds.sss.values) < 38
    assert np.abs(np.nanmax(ds.sla.values)) < 1.0
    assert np.nanmin(ds.temp.values) > 0 and np.nanmax(ds.temp.values) < 35


def test_mask_consistent_with_target_and_land(tiny_pipeline):
    ds = open_harmonized(tiny_pipeline)
    mask = ds.mask.values
    temp = ds.temp.values
    assert 0.2 < mask[0].mean() < 0.95  # both land and ocean in the tiny domain
    assert mask[-1].sum() < mask[0].sum()  # deep levels masked near the coast
    # nesting: a deeper valid point implies the shallower one is valid
    assert not (mask[1:] & ~mask[:-1]).any()
    for d in (0, 5, 14):
        valid_all_days = np.isfinite(temp[:, d]).all(axis=0)
        np.testing.assert_array_equal(valid_all_days, mask[d])
    # land is NaN in every surface field wherever the target is masked at the surface
    land = ~mask[0]
    for v in ("sst", "sla"):
        assert np.isnan(ds[v].values[:, land]).all()


def test_block_mean_matches_independent_calculation(tiny_pipeline):
    """SST: independent re-computation of one target cell from the raw 0.05 deg file."""
    cfg = tiny_pipeline
    ds = open_harmonized(cfg)
    raw = xr.open_dataset(raw_file(cfg, "sst", pd.Timestamp("2022-01-01")))
    day = pd.Timestamp("2022-01-10")
    v = raw["analysed_sst"].sel(time=day + pd.Timedelta(hours=12)).values - 273.15
    # pick an interior all-ocean cell
    sst = ds.sst.sel(time=day).values
    iy, ix = np.argwhere(np.isfinite(sst))[len(np.argwhere(np.isfinite(sst))) // 2]
    lat0, lon0 = ds.lat.values[iy] - 0.125, ds.lon.values[ix] - 0.125
    sel_y = (raw.latitude.values >= lat0) & (raw.latitude.values < lat0 + 0.25)
    sel_x = (raw.longitude.values >= lon0) & (raw.longitude.values < lon0 + 0.25)
    block = v[np.ix_(sel_y, sel_x)]
    assert block.shape == (5, 5)
    assert np.nanmean(block) == pytest.approx(sst[iy, ix], abs=2e-3)


def test_temp_vertical_and_block_mean_against_truth(tiny_pipeline):
    """The harmonised temperature at 100 m equals the block mean of the native field
    interpolated vertically (independent numpy re-computation)."""
    cfg = tiny_pipeline
    ds = open_harmonized(cfg)
    raw = xr.open_dataset(raw_file(cfg, "temp", pd.Timestamp("2022-01-01")))
    day = pd.Timestamp("2022-01-05")
    prof = raw["thetao"].sel(time=day)  # (depth, lat, lon)
    z = raw.depth.values
    k = np.searchsorted(z, 100.0) - 1
    w = (100.0 - z[k]) / (z[k + 1] - z[k])
    t100 = (1 - w) * prof.values[k] + w * prof.values[k + 1]
    out = ds.temp.sel(time=day, depth=100.0).values
    iy, ix = np.argwhere(np.isfinite(out))[10]
    lat0, lon0 = ds.lat.values[iy] - 0.125, ds.lon.values[ix] - 0.125
    sy = (raw.latitude.values >= lat0 - 1e-6) & (raw.latitude.values < lat0 + 0.25 - 1e-6)
    sx = (raw.longitude.values >= lon0 - 1e-6) & (raw.longitude.values < lon0 + 0.25 - 1e-6)
    block = t100[np.ix_(sy, sx)]
    assert block.shape == (3, 3)
    assert np.nanmean(block) == pytest.approx(out[iy, ix], abs=5e-3)


def test_currents_and_winds_handle_conventions(tiny_pipeline):
    """Descending-lat / (time, lon, lat) OSCAR-like currents and 6-hourly winds end up on the
    canonical grid with consistent values: harmonised winds equal the daily mean of raw."""
    cfg = tiny_pipeline
    ds = open_harmonized(cfg)
    wind = xr.open_dataset(raw_file(cfg, "winds", pd.Timestamp("2022-01-01")))
    day = pd.Timestamp("2022-01-07")
    seg = wind["uwnd"].sel(time=slice(day, day + pd.Timedelta(hours=23)))
    assert seg.sizes["time"] == 4
    mean = seg.mean("time").values
    # winds are on the canonical centres already (aligned) -> compare at an ocean cell
    out = ds.uw.sel(time=day).values
    iy, ix = np.argwhere(np.isfinite(out))[7]
    ry = np.argmin(abs(wind.latitude.values - ds.lat.values[iy]))
    rx = np.argmin(abs(wind.longitude.values - ds.lon.values[ix]))
    assert mean[ry, rx] == pytest.approx(out[iy, ix], abs=1e-4)
    # currents: geostrophic -> flow roughly follows SLA gradients; check finite & plausible
    assert np.nanmax(np.abs(ds.uo.values)) < 2.0


@pytest.mark.filterwarnings("ignore:Mean of empty slice")
def test_surface_and_temp_correlated_through_sla(tiny_pipeline):
    """The synthetic world is learnable: subsurface (~100 m) anomaly co-varies with SLA."""
    ds = open_harmonized(tiny_pipeline)
    sla = ds.sla.values
    t = ds.temp.sel(depth=100.0).values
    m = np.isfinite(sla) & np.isfinite(t)
    sa = sla - np.nanmean(sla, axis=0)
    ta = t - np.nanmean(t, axis=0)
    assert np.corrcoef(sa[m], ta[m])[0, 1] > 0.2


# --------------------------------------------------------------------------- statistics


def test_harmonic_design_leap_years_and_continuity():
    d = pd.to_datetime(["2020-02-29", "2021-03-01", "2020-12-31", "2021-01-01", "2020-01-01"])
    x = harmonic_design(d)
    assert x.shape == (5, 5) and np.isfinite(x).all()
    assert x[4] == pytest.approx([1, 1, 0, 1, 0])  # Jan 1 -> phi = 0
    # last day of the year is next to the first day of the next year (periodic)
    assert abs(x[2, 2] - x[3, 2]) < 0.02
    # 2020-02-29 is day 59 of 366: phi = 2 pi * 59 / 366
    phi = 2 * np.pi * 59 / 366
    assert x[0, 1] == pytest.approx(np.cos(phi))


def test_stats_use_train_split_only(tiny_pipeline, tmp_path):
    cfg = tiny_pipeline
    stats = Stats.load(cfg.stats_path)
    assert stats.n_train_days == 40
    assert (stats.train_start, stats.train_end) == ("2022-01-01", "2022-02-09")
    ds = open_harmonized(cfg)
    idx = split_indices(ds.time.values, cfg.split.train.start, cfg.split.train.end)
    assert len(idx) == 40
    ocean0 = ds.mask.values[0]
    for ci, v in enumerate(SURFACE_VARS):
        a = ds[v].isel(time=idx).values
        vals = a[np.isfinite(a) & ocean0[None]]
        assert stats.input_mean[ci] == pytest.approx(vals.mean(), rel=1e-5, abs=1e-6)
        assert stats.input_std[ci] == pytest.approx(vals.std(), rel=1e-4)
    # corrupt val + test days in a copy of the store: the stats must not change
    other = cfg.model_copy(update={"run_name": "tiny_corrupt"})
    shutil.copytree(cfg.zarr_path, other.zarr_path)
    root = zarr.open_group(str(other.zarr_path), mode="r+")
    first_after_train = int(idx[-1]) + 1
    for name in (*SURFACE_VARS, "temp"):
        arr = root[name]
        if arr.ndim == 3:
            arr[first_after_train:] = 999.0
        else:
            arr[first_after_train:] = 999.0
    s2 = compute_stats(other, progress=False)
    np.testing.assert_allclose(s2.input_mean, stats.input_mean)
    np.testing.assert_allclose(s2.input_std, stats.input_std)
    np.testing.assert_allclose(s2.anom_std, stats.anom_std)
    np.testing.assert_allclose(s2.clim_coef, stats.clim_coef, equal_nan=True)


def test_short_train_period_reduces_harmonics(tiny_pipeline):
    stats = Stats.load(tiny_pipeline.stats_path)
    assert stats.n_terms == 1  # 40 days: only the mean is identifiable
    assert (stats.clim_coef[1:][np.isfinite(stats.clim_coef[1:])] == 0).all()


def _write_harmonic_store(cfg: Config, rng):
    """Zarr store with a known harmonic temperature + noise over 2019-2021."""
    g = build_grid(cfg)
    time = pd.date_range(cfg.time.start, cfg.time.end, freq="D")
    h, w, nd = *g.shape, g.n_depth
    coef = rng.normal(size=(5, nd, h, w)) * np.array([5.0, 3.0, 3.0, 1.0, 1.0])[:, None, None, None]
    coef[0] += 15.0
    x = harmonic_design(time.values)
    temp = np.tensordot(x, coef, axes=([1], [0]))
    noise = 0.1 * rng.standard_normal(temp.shape)
    mask = np.ones((nd, h, w), bool)
    mask[10:, :2, :] = False  # some masked points
    temp = np.where(mask[None], temp + noise, np.nan).astype(np.float32)
    data = {}
    for i, v in enumerate(SURFACE_VARS):
        field = rng.normal(size=(len(time), h, w)) * (i + 1) + 10 * i
        data[v] = (("time", "lat", "lon"), field.astype("float32"))
    data["temp"] = (("time", "depth", "lat", "lon"), temp)
    data["mask"] = (("depth", "lat", "lon"), mask)
    ds = xr.Dataset(
        data, coords={"time": time.values, "depth": g.depth, "lat": g.lat, "lon": g.lon}
    )
    cfg.zarr_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(cfg.zarr_path, mode="w", consolidated=False)
    return coef, x


def test_harmonic_fit_recovers_known_coefficients_incl_leap_year(tmp_path, monkeypatch):
    monkeypatch.setenv("OCEANEMBED_DATA_ROOT", str(tmp_path))
    raw = {
        "run_name": "harm",
        "grid": {"lat_min": 8, "lat_max": 9, "lon_min": 70, "lon_max": 71, "resolution": 0.25},
        "time": {"start": "2019-01-01", "end": "2021-12-31"},
        "split": {
            "train": ["2019-01-01", "2020-12-31"],  # includes leap day 2020-02-29
            "val": ["2021-01-01", "2021-06-30"],
            "test": ["2021-07-01", "2021-12-31"],
        },
    }
    cfg = Config.model_validate(raw)
    rng = np.random.default_rng(1)
    coef, _ = _write_harmonic_store(cfg, rng)
    stats = compute_stats(cfg, chunk_days=50, progress=False)
    assert stats.n_terms == 5
    ok = np.isfinite(stats.clim_coef[0])
    np.testing.assert_allclose(stats.clim_coef[:, ok], coef[:, ok], atol=0.05)
    assert np.isnan(stats.clim_coef[0][~ok]).all()
    # climatology is evaluable for any date (leap day, val/test years) and matches the truth
    dates = pd.to_datetime(["2020-02-29", "2021-03-15", "2021-09-01"]).values
    clim = stats.climatology(dates)
    truth = np.tensordot(harmonic_design(dates), coef, axes=([1], [0]))
    np.testing.assert_allclose(clim[:, ok], truth[:, ok], atol=0.05)
    # anomaly std ~ the injected noise (0.1)
    assert stats.anom_std[0] == pytest.approx(0.1, rel=0.1)
    # reload round-trip
    again = Stats.load(cfg.stats_path)
    np.testing.assert_allclose(again.clim_coef, stats.clim_coef, equal_nan=True)


# --------------------------------------------------------------------------- dataset


@pytest.mark.parametrize("split,n", [("train", 40), ("val", 10), ("test", 10)])
def test_dataset_item_contract(tiny_pipeline, split, n):
    ds = make_dataset(tiny_pipeline, split)
    assert len(ds) == n
    it = ds[0]
    assert INPUT_CHANNELS[:7] == SURFACE_VARS and N_INPUT_CHANNELS == 12
    assert it["x"].shape == (12, 16, 24) and it["x"].dtype == torch.float32
    assert it["y"].shape == (15, 16, 24) and it["y"].dtype == torch.float32
    assert it["mask"].shape == (15, 16, 24) and it["mask"].dtype == torch.bool
    assert torch.isfinite(it["x"]).all() and torch.isfinite(it["y"]).all()
    # ocean-mask channel is the surface mask; NaN inputs (land / gaps) become exactly 0
    ocean = it["x"][7].numpy().astype(bool)
    np.testing.assert_array_equal(ocean, it["mask"][0].numpy())
    zs = open_harmonized(tiny_pipeline)
    for ci, v in enumerate(SURFACE_VARS):
        nan = np.isnan(zs[v].isel(time=int(ds.indices[0])).values)
        assert nan.any() and (it["x"][ci].numpy()[nan] == 0).all()
    # y is zero wherever masked
    assert (it["y"].numpy()[~it["mask"].numpy()] == 0).all()
    # seasonal and positional channels
    assert torch.allclose(it["x"][8] ** 2 + it["x"][9] ** 2, torch.ones(16, 24))
    assert it["x"][10].min() == -1 and it["x"][10].max() == 1
    assert it["x"][11].min() == -1 and it["x"][11].max() == 1
    # standardised surface inputs are O(1)
    assert it["x"][:7].abs().max() < 20


def test_dataset_dates_are_split_ranges(tiny_pipeline):
    cfg = tiny_pipeline
    for split in ("train", "val", "test"):
        ds = make_dataset(cfg, split)
        r = cfg.split.get(split)
        d = ds.dates()
        assert d[0] == pd.Timestamp(r.start) and d[-1] == pd.Timestamp(r.end)
    tr, te = make_dataset(cfg, "train"), make_dataset(cfg, "test")
    assert tr.dates().max() < te.dates().min()
    t = torch.stack([tr[i]["t"] for i in range(3)])
    assert (t[1:] - t[:-1] == 1).all()


def test_climatology_plus_anomaly_reconstructs_temperature(tiny_pipeline):
    cfg = tiny_pipeline
    zs = open_harmonized(cfg)
    for split in ("train", "test"):
        ds = make_dataset(cfg, split)
        for i in (0, len(ds) - 1):
            it = ds[i]
            rec = ds.denormalize(it["y"], it["t"])
            truth = zs.temp.isel(time=int(ds.indices[i])).values
            m = it["mask"].numpy()
            np.testing.assert_allclose(rec[m], truth[m], atol=2e-4)
    # batched de-normalisation
    ds = make_dataset(cfg, "val")
    batch = next(iter(DataLoader(ds, batch_size=3, shuffle=False, num_workers=0)))
    rec = ds.denormalize(batch["y"], batch["t"])
    assert rec.shape == (3, 15, 16, 24)
    truth = zs.temp.isel(time=ds.indices[:3]).values
    m = batch["mask"].numpy()
    np.testing.assert_allclose(rec[m], truth[m], atol=2e-4)


def test_dataloader_num_workers_zero_and_pickle(tiny_pipeline):
    ds = make_dataset(tiny_pipeline, "train")
    b = next(iter(DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)))
    assert b["x"].shape == (4, 12, 16, 24) and b["mask"].dtype == torch.bool
    clone = pickle.loads(pickle.dumps(ds))  # what spawn workers do
    assert torch.equal(clone[3]["x"], ds[3]["x"])


def test_dataloader_with_spawned_worker(tiny_pipeline):
    ds = make_dataset(tiny_pipeline, "val")
    dl = DataLoader(
        ds, batch_size=5, num_workers=1, multiprocessing_context="spawn", persistent_workers=False
    )
    batches = list(dl)
    assert len(batches) == 2 and batches[0]["y"].shape == (5, 15, 16, 24)


def test_generate_all_is_idempotent(tiny_pipeline):
    assert all(v == [] for v in generate_all(tiny_pipeline).values())  # nothing rewritten
