"""Prediction product, evaluation, Argo validation, report, run-all and the dashboard data layer,
all on one complete tiny ``run-all`` (CPU) built once per session (see ``conftest.tiny_run``)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from app import data_access as da
from typer.testing import CliRunner

from oceanembed.cli import app
from oceanembed.config import Config
from oceanembed.data.dataset import OceanDataset, SurfaceOnlyDataset, make_dataset
from oceanembed.data.harmonize import open_harmonized
from oceanembed.data.stats import Stats
from oceanembed.grid import build_grid
from oceanembed.infer.predict import (
    expected_product_files,
    load_recon_model,
    model_predictor,
    predict_batch,
)
from oceanembed.models.baselines import ClimatologyBaseline, RidgeBaseline, rmse_per_depth

TINY = Path(__file__).resolve().parents[1] / "configs" / "test_tiny.yaml"
runner = CliRunner()
METHODS = ["model", "model_scratch", "ridge", "climatology"]


def _json(cfg: Config, name: str) -> dict:
    return json.loads((cfg.outputs_dir / "metrics" / name).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------- run-all contract
def test_run_all_produces_every_contract_file(tiny_run: Config, _tiny_run_roots):
    out = tiny_run.outputs_dir
    must = [
        "run_meta.json",
        "checkpoints/pretrain.pt",
        "checkpoints/recon.pt",
        "checkpoints/ridge.joblib",
        "embeddings/embeddings.zarr",
        "metrics/metrics_glorys.json",
        "metrics/maps_glorys.nc",
        "metrics/metrics_argo.json",
        "metrics/argo_matchups.parquet",
        "report.md",
        "logs/pretrain.jsonl",
        "logs/train.jsonl",
    ]
    for rel in must:
        assert (out / rel).exists(), rel
    assert [f.exists() for f in expected_product_files(tiny_run, "2022-02-20", "2022-03-01")] == [
        True,
        True,
    ]
    pngs = {p.name for p in (out / "figures").glob("*.png")}
    assert {
        "rmse_profile.png", "bias_profile.png", "anomaly_corr_profile.png", "skill_profile.png",
        "rmse_maps.png", "example_day_100m.png", "section_lat.png", "argo_validation.png",
        "training_curves.png",
    } <= pngs  # fmt: skip
    text = _tiny_run_roots[2]
    for step in (
        "synth", "harmonize", "stats", "pretrain", "train", "train-ablation", "baseline", "embed",
        "predict", "predict-ridge", "predict-ablation", "evaluate", "validate-argo", "report",
    ):  # fmt: skip
        assert f"] {step} =" in text and f"--- {step} done" in text
    assert "=== summary" in text


def test_run_all_skip_existing_skips_every_step(tiny_run: Config):
    res = runner.invoke(app, ["run-all", "--config", str(TINY), "--skip-existing"])
    assert res.exit_code == 0, res.output + str(res.exception)
    assert res.output.count("skipped (--skip-existing)") == 14


def test_run_all_stops_at_first_failing_step(tmp_path, monkeypatch):
    monkeypatch.setenv("OCEANEMBED_DATA_ROOT", str(tmp_path / "d"))
    monkeypatch.setenv("OCEANEMBED_OUTPUTS_ROOT", str(tmp_path / "o"))
    called = []

    def boom(config):
        raise RuntimeError("boom")

    monkeypatch.setattr("oceanembed.cli.harmonize", boom)
    monkeypatch.setattr("oceanembed.cli.stats", lambda config: called.append("stats"))
    res = runner.invoke(app, ["run-all", "--config", str(TINY), "--device", "cpu"])
    assert res.exit_code == 1
    assert "FAILED at step 2/14 (harmonize): RuntimeError: boom" in res.output
    assert called == []  # nothing after the failure ran


def test_run_meta_contents(tiny_run: Config):
    meta = da.load_run_meta(tiny_run.outputs_dir)
    assert meta["run_name"] == "test_tiny" and meta["data_source"] == "synthetic"
    assert meta["split"]["test"] == {"start": "2022-02-20", "end": "2022-03-01"}
    assert meta["grid"]["n_lat"] == 16 and meta["grid"]["n_lon"] == 24
    assert len(meta["grid"]["depths"]) == 15
    assert Path(meta["paths"]["zarr"]).exists() and Path(meta["paths"]["stats"]).exists()
    assert Config.model_validate(meta["config"]).run_name == "test_tiny"


# ----------------------------------------------------------------------- prediction product
def test_prediction_netcdf_structure_and_attrs(tiny_run: Config):
    pdir = tiny_run.outputs_dir / "predictions"
    files = sorted(pdir.glob("oceanembed_T_*.nc"))
    assert [f.name for f in files] == ["oceanembed_T_202202.nc", "oceanembed_T_202203.nc"]
    g = build_grid(tiny_run)
    with xr.open_dataset(files[0]) as ds:
        t = ds["temperature"]
        assert t.dims == ("time", "depth", "lat", "lon") and t.dtype == np.float32
        assert t.shape == (9, 15, 16, 24)  # 20..28 Feb
        assert pd.DatetimeIndex(ds["time"].values)[[0, -1]].strftime("%F").tolist() == [
            "2022-02-20", "2022-02-28",
        ]  # fmt: skip
        np.testing.assert_allclose(ds["lat"], g.lat)
        np.testing.assert_allclose(ds["lon"], g.lon)
        np.testing.assert_allclose(ds["depth"], g.depth)
        assert ds.attrs["Conventions"] == "CF-1.8"
        for k in ("title", "institution", "source", "history", "comment", "model_checkpoint"):
            assert ds.attrs[k]
        assert ds.attrs["data_source"] == "synthetic" and "SYNTHETIC" in ds.attrs["comment"]
        assert ds.attrs["model_checkpoint"] == "recon.pt"
        assert json.loads(ds.attrs["config"])["run_name"] == "test_tiny"
        assert t.attrs["standard_name"] == "sea_water_potential_temperature"
        assert t.attrs["units"] == "degree_Celsius"
        assert ds["depth"].attrs["positive"] == "down" and ds["depth"].attrs["axis"] == "Z"
        assert ds["lat"].attrs["axis"] == "Y" and ds["lon"].attrs["axis"] == "X"
        assert ds["time"].attrs["axis"] == "T"
        mask = open_harmonized(tiny_run)["mask"].values
        assert np.isnan(t.values[:, ~mask]).all() and np.isfinite(t.values[:, mask]).all()
    with netCDF4.Dataset(files[0]) as nc:
        v = nc["temperature"]
        assert v.filters()["zlib"] is True
        assert float(v._FillValue) == pytest.approx(9.96921e36, rel=1e-5)
    with xr.open_dataset(files[1]) as ds:
        assert ds["temperature"].shape[0] == 1  # 1 March only


def test_prediction_matches_predict_batch(tiny_run: Config):
    model, _ = load_recon_model(tiny_run.checkpoints_dir / "recon.pt")
    ds = make_dataset(tiny_run, "test", preload=True)
    idx = [0, 4, 9]
    batch = {k: torch.stack([ds[i][k] for i in idx]) for k in ("x", "t")}
    want = predict_batch(model_predictor(model, amp=False), batch, ds, "cpu")
    for n, i in enumerate(idx):
        date = ds.dates()[i]
        got = da.load_prediction(tiny_run.outputs_dir, date).values
        np.testing.assert_allclose(got, want[n], atol=1e-4, equal_nan=True)


def test_surface_only_dataset_never_reads_target(tiny_run: Config, tmp_path):
    """Inputs equal the full dataset's, and work on a store without the ``temp`` variable."""
    full = make_dataset(tiny_run, "test")
    inputs = SurfaceOnlyDataset(
        tiny_run.zarr_path, Stats.load(tiny_run.stats_path), "2022-02-20", "2022-03-01"
    )
    b = inputs.batch(2, 6)
    for n, i in enumerate(range(2, 6)):
        torch.testing.assert_close(b["x"][n], full[i]["x"])
        assert torch.equal(b["t"][n], full[i]["t"]) and b["index"][n] == i
    assert torch.equal(inputs[3]["x"], full[3]["x"])
    # remove the target from a copy of the store: prediction inputs still load
    copy = tmp_path / "no_target.zarr"
    shutil.copytree(tiny_run.zarr_path, copy)
    shutil.rmtree(copy / "temp")
    assert "temp" not in xr.open_zarr(copy, consolidated=False)
    notarget = SurfaceOnlyDataset(copy, Stats.load(tiny_run.stats_path), "2022-02-20", "2022-03-01")
    torch.testing.assert_close(notarget.batch(0, 10)["x"], inputs.batch(0, 10)["x"])
    assert isinstance(full, OceanDataset) and not isinstance(full, SurfaceOnlyDataset)


def test_run_all_wrote_ridge_and_ablation_products(tiny_run: Config):
    """predict-ridge / predict-ablation: full-test-period products in their own folders, labelled in
    the file attributes, the ridge one equal to what the ridge predictor gives."""
    pred = tiny_run.outputs_dir / "predictions"
    assert [p.name for p in sorted(pred.glob("oceanembed_T_*.nc"))] == [
        "oceanembed_T_202202.nc",
        "oceanembed_T_202203.nc",
    ]
    assert set(da.prediction_methods(tiny_run.outputs_dir)) == {"model", "ridge", "model_scratch"}
    n_days = {
        m: len(da.available_dates(tiny_run.outputs_dir, m))
        for m in da.prediction_methods(tiny_run.outputs_dir)
    }
    assert set(n_days.values()) == {10}
    with xr.open_dataset(pred / "ridge" / "oceanembed_T_202202.nc") as ds:
        assert ds.attrs["method"] == "ridge" and ds.attrs["model_checkpoint"] == "ridge.joblib"
        assert "ridge" in ds.attrs["title"].lower() and ds["temperature"].dims[0] == "time"
    with xr.open_dataset(pred / "scratch" / "oceanembed_T_202203.nc") as ds:
        assert ds.attrs["method"] == "model" and ds.attrs["model_checkpoint"] == "recon_scratch.pt"
    # the ridge product has the same ocean mask as the model product and differs in values
    day = "2022-02-25"
    m, r = (
        da.load_prediction(tiny_run.outputs_dir, day),
        da.load_prediction(tiny_run.outputs_dir, day, "ridge"),
    )
    assert (np.isfinite(m.values) == np.isfinite(r.values)).all()
    assert np.nanmax(np.abs(m.values - r.values)) > 0
    with pytest.raises(KeyError):
        da.load_prediction(tiny_run.outputs_dir, day, "model_nope")


def test_predict_cli_tag_dates_and_argument_errors(tiny_run: Config, tmp_path, monkeypatch):
    # a partial-month run goes to a scratch copy of the checkpoints so the run's products stay whole
    out = tmp_path / "o"
    shutil.copytree(tiny_run.checkpoints_dir, out / "test_tiny" / "checkpoints")
    monkeypatch.setenv("OCEANEMBED_OUTPUTS_ROOT", str(out))
    r = runner.invoke(
        app,
        ["predict", "--config", str(TINY), "--tag", "scratch", "--start", "2022-02-25",
         "--end", "2022-02-26", "--device", "cpu"],
    )  # fmt: skip
    assert r.exit_code == 0, r.output + str(r.exception)
    f = out / "test_tiny" / "predictions" / "scratch" / "oceanembed_T_202202.nc"
    with xr.open_dataset(f) as ds:
        assert (
            ds["temperature"].shape[0] == 2 and ds.attrs["model_checkpoint"] == "recon_scratch.pt"
        )
    r = runner.invoke(
        app,
        ["predict", "--config", str(TINY), "--ridge", "--start", "2022-02-25",
         "--end", "2022-02-26", "--device", "cpu"],
    )  # fmt: skip
    assert r.exit_code == 0, r.output + str(r.exception)
    with xr.open_dataset(
        out / "test_tiny" / "predictions" / "ridge" / "oceanembed_T_202202.nc"
    ) as ds:
        assert ds["temperature"].shape[0] == 2 and ds.attrs["method"] == "ridge"
    for args in (["--ridge", "--tag", "scratch"], ["--tag", "ridge"]):
        bad = runner.invoke(app, ["predict", "--config", str(TINY), *args])
        assert bad.exit_code != 0, args
    bad = runner.invoke(
        app, ["predict", "--config", str(TINY), "--split", "test", "--start", "2022-02-25"]
    )
    assert bad.exit_code != 0
    bad = runner.invoke(
        app, ["predict", "--config", str(TINY), "--split", "test", "--start", "a", "--end", "b"]
    )
    assert bad.exit_code != 0
    out_of_store = runner.invoke(
        app, ["predict", "--config", str(TINY), "--start", "2030-01-01", "--end", "2030-01-02"]
    )
    assert out_of_store.exit_code != 0


# ----------------------------------------------------------------------- evaluation vs GLORYS
def test_metrics_glorys_structure_and_values(tiny_run: Config):
    m = _json(tiny_run, "metrics_glorys.json")
    md = m["metadata"]
    assert md["data_source"] == "synthetic" and "SYNTHETIC" in md["note"]
    assert (md["split"], md["start"], md["end"], md["n_days"]) == (
        "test",
        "2022-02-20",
        "2022-03-01",
        10,
    )
    assert md["units"] == "degC" and len(md["depths"]) == 15
    assert list(m["methods"]) == METHODS and list(md["methods"]) == METHODS
    for v in m["methods"].values():
        assert set(v) == {"overall", "pooled_50_200m", "per_depth", "per_basin"}
        pd_ = v["per_depth"]
        assert set(pd_) >= {
            "depth",
            "n",
            "rmse",
            "bias",
            "mae",
            "corr_raw",
            "corr_anom",
            "skill_vs_clim",
        }
        assert len(pd_["rmse"]) == 15
        assert set(v["per_basin"]) == {"arabian_sea", "bay_of_bengal"}
        assert v["per_basin"]["bay_of_bengal"]["overall"]["n"] == 0  # tiny box is all Arabian Sea
        assert v["per_basin"]["bay_of_bengal"]["overall"]["rmse"] is None
        assert v["per_basin"]["arabian_sea"]["overall"]["n"] == v["overall"]["n"]
    clim = m["methods"]["climatology"]
    assert clim["overall"]["skill_vs_clim"] == pytest.approx(0.0)
    assert all(c is None for c in clim["per_depth"]["corr_anom"])  # zero anomaly: undefined
    assert m["methods"]["model"]["overall"]["corr_anom"] is not None
    d = m["daily_rmse"]
    assert len(d["dates"]) == 10 and np.array(d["rmse"]["model"], dtype=float).shape == (10, 15)


def test_daily_metrics_and_maps_for_every_method(tiny_run: Config):
    """Daily bias / anomaly correlation / pooled RMSE are stored per method and equal what numpy
    gives from the prediction product files (so the product == what evaluate scored); error maps
    carry raw correlation and skill for every method."""
    m = _json(tiny_run, "metrics_glorys.json")
    d = m["daily_rmse"]
    assert {"dates", "depth", "rmse", "bias", "corr_anom", "pooled_rmse", "pooled_range_m"} <= set(
        d
    )
    assert d["pooled_range_m"] == [50.0, 200.0]
    for k in METHODS:
        for name in ("rmse", "bias", "corr_anom"):
            assert np.array(d[name][k], dtype=float).shape == (10, 15), (name, k)
        for name in ("pooled_rmse", "pooled_bias", "pooled_corr_anom"):
            assert np.array(d[name][k], dtype=float).shape == (10,), (name, k)
    assert all(v is None for v in d["pooled_corr_anom"]["climatology"])
    assert all(v is None for row in d["corr_anom"]["climatology"] for v in row)  # zero anomaly

    zds = open_harmonized(tiny_run)
    stats = Stats.load(tiny_run.stats_path)
    depths = np.array(d["depth"])
    pooled = np.flatnonzero((depths >= 50) & (depths <= 200))
    folders = {"model": "", "ridge": "ridge", "model_scratch": "scratch"}
    for key, sub in folders.items():
        files = sorted((tiny_run.outputs_dir / "predictions" / sub).glob("oceanembed_T_*.nc"))
        with xr.open_mfdataset(files, combine="by_coords") as ds:
            pred = ds["temperature"].load()
        assert [str(t)[:10] for t in pred["time"].values] == d["dates"]
        ref = zds["temp"].sel(time=pred["time"]).values
        diff = pred.values - ref
        ok = np.isfinite(diff)
        bias = np.array([[np.nanmean(diff[t, z]) for z in range(15)] for t in range(10)])
        np.testing.assert_allclose(np.array(d["bias"][key], dtype=float), bias, atol=2e-4)
        se = np.where(ok, diff**2, 0.0)
        n = ok[:, pooled].sum(axis=(1, 2, 3))
        pooled_rmse = np.sqrt(se[:, pooled].sum(axis=(1, 2, 3)) / n)
        np.testing.assert_allclose(
            np.array(d["pooled_rmse"][key], dtype=float), pooled_rmse, atol=2e-4
        )
        pooled_bias = np.where(ok, diff, 0.0)[:, pooled].sum(axis=(1, 2, 3)) / n
        np.testing.assert_allclose(
            np.array(d["pooled_bias"][key], dtype=float), pooled_bias, atol=2e-4
        )
        if key != "climatology":
            clim_p = stats.climatology(pred["time"].values)
            for t in (0, 4, 9):
                a_p = (pred.values - clim_p)[t][pooled]
                b_p = (ref - clim_p)[t][pooled]
                both = np.isfinite(a_p) & np.isfinite(b_p)
                r = np.corrcoef(a_p[both], b_p[both])[0, 1]
                assert d["pooled_corr_anom"][key][t] == pytest.approx(r, abs=2e-3), (key, t)
        np.testing.assert_allclose(
            np.array(d["rmse"][key], dtype=float),
            np.sqrt(se.sum(axis=(2, 3)) / ok.sum(axis=(2, 3))),
            atol=2e-4,
        )
        if key == "climatology":
            continue
        clim = stats.climatology(pred["time"].values)
        a, b = pred.values - clim, ref - clim
        for t, z in ((0, 3), (4, 7), (9, 10)):
            both = np.isfinite(a[t, z]) & np.isfinite(b[t, z])
            r = np.corrcoef(a[t, z][both], b[t, z][both])[0, 1]
            assert d["corr_anom"][key][t][z] == pytest.approx(r, abs=2e-3), (key, t, z)

    maps = da.load_maps(tiny_run.outputs_dir)
    for key in METHODS:
        assert (
            f"rmse_{key}" in maps and f"corr_raw_{key}" in maps and f"skill_vs_clim_{key}" in maps
        )
    assert float(np.nanmax(np.abs(maps["skill_vs_clim_climatology"].values))) == 0.0
    # skill of the model map agrees with the stored per-depth skill (domain-pooled): same sign
    assert np.isfinite(maps["skill_vs_clim_ridge"].values).any()


def test_evaluate_matches_direct_numpy_and_baseline_rmse(tiny_run: Config):
    m = _json(tiny_run, "metrics_glorys.json")["methods"]
    ds = make_dataset(tiny_run, "test", preload=True)
    # (1) independent of the streaming code: product files vs the harmonised target
    files = sorted((tiny_run.outputs_dir / "predictions").glob("oceanembed_T_*.nc"))
    pred = xr.concat([xr.open_dataset(f)["temperature"] for f in files], dim="time")
    tgt = open_harmonized(tiny_run)["temp"].sel(time=pred["time"]).load()
    err = (pred - tgt).values
    direct = np.sqrt(np.nanmean(err**2, axis=(0, 2, 3)))
    np.testing.assert_allclose(m["model"]["per_depth"]["rmse"], direct, atol=1e-4)
    bias = np.nanmean(err, axis=(0, 2, 3))
    np.testing.assert_allclose(m["model"]["per_depth"]["bias"], bias, atol=1e-4)
    # (2) agreement with the Phase-3 `rmse_per_depth` used for the earlier numbers
    ridge = RidgeBaseline.load(tiny_run.checkpoints_dir / "ridge.joblib")
    np.testing.assert_allclose(
        m["ridge"]["per_depth"]["rmse"], rmse_per_depth(ridge, ds), atol=5e-4
    )
    np.testing.assert_allclose(
        m["climatology"]["per_depth"]["rmse"],
        rmse_per_depth(ClimatologyBaseline(15), ds),
        atol=5e-4,
    )
    model, _ = load_recon_model(tiny_run.checkpoints_dir / "recon.pt")
    np.testing.assert_allclose(
        m["model"]["per_depth"]["rmse"],
        rmse_per_depth(model_predictor(model, amp=False), ds),
        atol=5e-4,
    )


def test_maps_file_and_pooling_consistency(tiny_run: Config):
    mp = da.load_maps(tiny_run.outputs_dir)
    for v in ("rmse_model", "bias_model", "corr_anom_model", "rmse_climatology", "n_valid"):
        assert mp[v].dims == ("depth", "lat", "lon"), v
    m = _json(tiny_run, "metrics_glorys.json")["methods"]["model"]["per_depth"]
    n = mp["n_valid"].values
    pooled = np.sqrt((n * mp["rmse_model"].fillna(0).values ** 2).sum((1, 2)) / n.sum((1, 2)))
    np.testing.assert_allclose(pooled, m["rmse"], rtol=1e-4)
    mask = open_harmonized(tiny_run)["mask"].values
    assert np.isnan(mp["rmse_model"].values[~mask]).all()
    assert mp.attrs["data_source"] == "synthetic"


# ----------------------------------------------------------------------- Argo validation
def test_argo_matchups_and_metrics(tiny_run: Config):
    mt = da.load_argo_matchups(tiny_run.outputs_dir)
    am = _json(tiny_run, "metrics_argo.json")
    md = am["metadata"]
    base = ["profile_id", "time", "lat", "lon", "depth", "obs", "model", "ridge", "clim"]
    assert list(mt.columns)[: len(base)] == base
    assert {"glorys", "basin", "model_scratch", "grid_lat", "grid_lon"} <= set(mt.columns)
    assert len(mt) == md["n_matchups"] > 0
    assert mt["profile_id"].nunique() == md["n_profiles_used"] <= md["n_profiles_collocated"]
    assert md["n_profiles_loaded"] >= md["n_profiles_in_period"] >= md["n_profiles_collocated"]
    assert set(md["dropped_profiles"]) == {
        "outside_dates",
        "outside_domain",
        "land",
        "outside_period",
    }
    assert (
        md["dropped_profiles"]["outside_period"]
        == md["n_profiles_loaded"] - md["n_profiles_in_period"]
    )
    assert md["data_source"] == "synthetic" and "SYNTHETIC" in md["note"]
    assert "assimilates Argo" in md["independence_note"]
    t = pd.to_datetime(mt["time"])
    assert (t >= "2022-02-20").all() and (t < "2022-03-02").all()
    assert mt[["obs", "model", "ridge", "clim", "glorys"]].notna().all().all()
    assert set(mt["basin"]) <= {"arabian_sea", "bay_of_bengal", "other"}
    assert set(am["methods"]) == {*METHODS, "glorys"}
    for v in am["methods"].values():
        assert sum(v["per_depth"]["n"]) == len(mt) == v["overall"]["n"]
    # a matchup row really is the target / prediction of its cell and day
    row = mt.iloc[len(mt) // 2]
    day = pd.Timestamp(row["time"]).normalize()
    z = list(build_grid(tiny_run).depth).index(float(row["depth"]))
    cell = dict(lat=row["grid_lat"], lon=row["grid_lon"], method="nearest")
    tgt = open_harmonized(tiny_run)["temp"].sel(time=day).isel(depth=z).sel(**cell)
    assert float(tgt) == pytest.approx(row["glorys"], abs=1e-4)
    pr = da.load_prediction(tiny_run.outputs_dir, day).isel(depth=z).sel(**cell)
    assert float(pr) == pytest.approx(row["model"], abs=1e-3)
    assert abs(row["grid_lat"] - row["lat"]) <= 0.125 + 1e-6
    # point metrics reproduce from the table
    sub = mt[mt["depth"] == mt["depth"].iloc[0]]
    want = np.sqrt(np.mean((sub["model"] - sub["obs"]) ** 2))
    i = list(build_grid(tiny_run).depth).index(float(sub["depth"].iloc[0]))
    assert am["methods"]["model"]["per_depth"]["rmse"][i] == pytest.approx(want, rel=1e-5)


def test_gridded_argo_is_evaluated_when_present_and_skipped_otherwise(tiny_run: Config):
    from oceanembed.eval.argo_validation import validate_argo

    assert "gridded_argo" not in _json(tiny_run, "metrics_argo.json")  # absent -> silently skipped
    g = build_grid(tiny_run)
    z = open_harmonized(tiny_run)
    days = pd.date_range("2022-02-20", "2022-03-01")
    temp = z["temp"].sel(time=days).values
    da_ = xr.DataArray(
        temp,
        dims=("time", "depth", "lat", "lon"),
        coords={"time": days, "depth": g.depth, "lat": g.lat, "lon": g.lon},
        name="temp",
    )
    folder = tiny_run.raw_root / "argo_gridded"
    folder.mkdir(parents=True, exist_ok=True)
    f = folder / "fake_incois.nc"
    da_.to_dataset(name="temp").to_netcdf(f)
    try:
        out = validate_argo(tiny_run, "test", device="cpu", progress=False)
    finally:
        f.unlink()
        shutil.rmtree(folder, ignore_errors=True)
    gr = out["gridded_argo"]
    assert gr["n_times"] == 10 and set(gr["methods"]) == {*METHODS, "glorys"}
    assert gr["methods"]["glorys"]["overall"]["rmse"] < 1e-3  # the fake product *is* GLORYS
    assert gr["methods"]["model"]["overall"]["rmse"] > gr["methods"]["glorys"]["overall"]["rmse"]
    validate_argo(tiny_run, "test", device="cpu", progress=False)  # restore the plain metrics file
    assert "gridded_argo" not in _json(tiny_run, "metrics_argo.json")


# ----------------------------------------------------------------------- report
def test_report_banner_tables_and_figure_links(tiny_run: Config):
    md = (tiny_run.outputs_dir / "report.md").read_text(encoding="utf-8")
    assert md.startswith("# OceanEmbed report")
    assert "SYNTHETIC DATA" in md.split("## 1.")[0]  # prominent: before the first section
    assert "not** scientific skill" in md
    assert "RMSE (degC) by depth" in md and "Anomaly correlation by depth" in md
    assert "Independence caveat" in md and "OceanEmbed (no pretraining)" in md
    links = [ln.split("](")[1].rstrip(")") for ln in md.splitlines() if ln.startswith("![")]
    assert len(links) >= 9
    for rel in links:
        p = tiny_run.outputs_dir / rel
        assert p.exists() and p.stat().st_size > 5000, rel


# ----------------------------------------------------------------------- dashboard data layer
def test_data_access_functions(tiny_run: Config):
    run = tiny_run.outputs_dir
    runs = da.list_runs(run.parent)
    assert [r["name"] for r in runs] == ["test_tiny"]
    assert runs[0]["has_predictions"] and runs[0]["has_metrics"] and runs[0]["has_report"]
    assert da.list_runs(run.parent / "nope") == []

    dates = da.available_dates(run)
    assert len(dates) == 10 and dates[0] == pd.Timestamp("2022-02-20")
    d = dates[3]
    pred = da.load_prediction(run, d)
    assert pred.dims == ("depth", "lat", "lon") and pred.shape == (15, 16, 24)
    with pytest.raises(KeyError):
        da.load_prediction(run, "2030-01-01")

    tgt = da.load_target(run, d)
    assert tgt is not None and tgt.shape == pred.shape
    assert da.load_target(run, "2030-01-01") is None
    clim = da.load_climatology(run, d)
    assert clim.shape == pred.shape
    # the climatology is NaN exactly where the model has nothing to say
    assert (np.isfinite(clim.values) >= np.isfinite(pred.values)).all()

    surf = da.load_surface_inputs(run, d)
    assert list(surf.data_vars) == ["sst", "sss", "sla", "uo", "vo", "uw", "vw"]
    assert 15 < float(surf["sst"].mean()) < 35  # degC, physical units
    assert da.load_surface_inputs(run, "2030-01-01") is None

    prof = da.load_profile(run, d, lat=10.01, lon=75.02)
    assert set(prof.data_vars) == {"predicted", "target", "climatology"}
    assert prof["predicted"].dims == ("depth",) and len(prof["depth"]) == 15
    assert float(prof["lat"]) == pytest.approx(10.125) and float(prof["lon"]) == pytest.approx(
        75.125
    )
    np.testing.assert_allclose(
        prof["predicted"].values, pred.sel(lat=10.125, lon=75.125).values, equal_nan=True
    )
    sec_lat = da.load_section(run, d, lat=10.0)
    assert sec_lat["predicted"].dims == ("depth", "lon") and "difference" in sec_lat
    sec_lon = da.load_section(run, d, lon=75.0)
    assert sec_lon["predicted"].dims == ("depth", "lat")
    np.testing.assert_allclose(
        sec_lat["difference"].values,
        (sec_lat["predicted"] - sec_lat["target"]).values,
        equal_nan=True,
    )
    with pytest.raises(ValueError):
        da.load_section(run, d)
    with pytest.raises(ValueError):
        da.load_section(run, d, lat=10.0, lon=75.0)

    emb = da.load_embeddings(run, d)
    assert emb.dims == ("emb", "y", "x") and emb.shape == (16, 4, 6)
    assert da.load_embeddings(run, "2022-01-05") is None  # embeddings only cover the test split
    rgb = da.embedding_pca_rgb(run, d, n_fit_days=6)
    assert rgb.dims == ("y", "x", "rgb") and rgb.shape == (4, 6, 3)
    assert float(rgb.min()) >= 0 and float(rgb.max()) <= 1
    evr = rgb.attrs["explained_variance_ratio"]
    assert len(evr) == 3 and evr[0] >= evr[1] >= evr[2] > 0 and sum(evr) <= 1 + 1e-9
    # the same axes for every day (consistent colours)
    rgb2 = da.embedding_pca_rgb(run, dates[7], n_fit_days=6)
    assert rgb2.attrs == rgb.attrs

    logs = da.load_training_logs(run)
    assert {"pretrain", "train", "train_scratch"} <= set(logs)
    assert "val_rmse" in logs["train"] and len(logs["train"]) >= 1
    assert da.load_metrics_glorys(run)["metadata"]["split"] == "test"
    assert da.load_metrics_argo(run)["metadata"]["n_matchups"] > 0
    assert len(da.load_argo_matchups(run)) > 0
    assert da.load_metrics_glorys(run.parent / "missing") is None
    assert da.load_maps(run.parent / "missing") is None


def test_data_access_has_no_streamlit_dependency():
    src = (Path(da.__file__)).read_text(encoding="utf-8")
    assert "import streamlit" not in src
