"""Research stages R4 (ARMOR3D benchmark) and R5 (physical metrics): derived-quantity definitions on
hand-computed profiles, stratification logic, and the R4 / R5 paths on the tiny run with a small
fake ARMOR3D file (no network)."""

from __future__ import annotations

import json
import os
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from typer.testing import CliRunner

from oceanembed.cli import app
from oceanembed.config import Config, load_config
from oceanembed.data.harmonize import open_harmonized
from oceanembed.data.providers.base import MissingCredentialsError
from oceanembed.research import armor3d
from oceanembed.research import physical as ph
from oceanembed.research.r4 import (
    armor_at_matchups,
    r4_dir,
    regrid_all,
    regridded_days,
    regridded_file,
    run_r4,
)
from oceanembed.research.r5 import (
    class_names,
    day_class_masks,
    r5_dir,
    run_r5,
    strata_edges,
)
from oceanembed.research.r5_report import make_r5_report

TINY = Path(__file__).resolve().parents[1] / "configs" / "test_tiny.yaml"
Z = np.array([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000], dtype=float)


def _profile(**levels) -> np.ndarray:
    """A 15-level profile filled with NaN except the given ``z<depth>=value`` levels."""
    t = np.full(len(Z), np.nan)
    for k, v in levels.items():
        t[list(Z).index(float(k[1:]))] = v
    return t


# ----------------------------------------------------------------------------------------
# isotherm depth
# ----------------------------------------------------------------------------------------
def test_isotherm_depth_interpolates_linearly_between_levels():
    t = np.linspace(28, 5, 15)  # 28 at 0 m ... 5 at 1000 m, linear in the level index
    d20 = ph.isotherm_depth(t, Z, 20.0)
    k = np.flatnonzero(t < 20)[0]  # first level colder than 20 degC
    w = (t[k - 1] - 20.0) / (t[k - 1] - t[k])
    assert d20 == pytest.approx(Z[k - 1] + w * (Z[k] - Z[k - 1]))
    # hand value for a simple profile: 22 degC at 30 m, 19 degC at 50 m -> 20 degC at 30 + 20 * 2/3
    t = np.array([28, 27, 26, 24, 22, 19, 17, 15, 14, 13, 11, 8, 6, 5, 4], dtype=float)
    assert ph.isotherm_depth(t, Z, 20.0) == pytest.approx(30 + 20 * 2 / 3)
    assert ph.isotherm_depth(t, Z, 23.0) == pytest.approx(20 + 10 * 1 / 2)


def test_isotherm_exactly_on_a_level_and_the_equal_level_rule():
    t = np.array([28, 24, 22, 20, 18, 17, 15, 12, 10, 9, 8, 7, 6, 5, 4], dtype=float)
    assert ph.isotherm_depth(t, Z, 20.0) == pytest.approx(20.0)  # T == level at 20 m, colder below
    # T equal to the level counts as warm enough: 20, 20, 18 -> the crossing is below the second 20
    t = np.array([28, 24, 22, 20, 20, 18, 15, 12, 10, 9, 8, 7, 6, 5, 4], dtype=float)
    assert ph.isotherm_depth(t, Z, 20.0) == pytest.approx(30.0)


def test_isotherm_not_bracketed_gives_nan():
    warm = np.full(len(Z), 25.0)  # never reaches the level: not bracketed
    assert np.isnan(ph.isotherm_depth(warm, Z, 20.0))
    cold = np.full(len(Z), 15.0)  # outcrops: colder than the level everywhere
    assert np.isnan(ph.isotherm_depth(cold, Z, 20.0))
    surface_cold = np.array([15, 18, 21, 22, 20, 19, 15, 12, 10, 9, 8, 7, 6, 5, 4], dtype=float)
    assert np.isnan(ph.isotherm_depth(surface_cold, Z, 20.0))  # first level below the level
    # water column ends (NaN below 100 m) while still warmer than the level: not bracketed
    shallow = _profile(z0=28, z5=27.5, z10=27, z20=26, z30=25, z50=24, z75=23, z100=22)
    assert np.isnan(ph.isotherm_depth(shallow, Z, 20.0))
    # ... but a crossing inside the valid part is found even though deeper levels are missing
    shelf = _profile(z0=28, z5=27, z10=26, z20=24, z30=22, z50=18)
    assert ph.isotherm_depth(shelf, Z, 20.0) == pytest.approx(30 + 20 * 2 / 4)
    assert np.isnan(ph.isotherm_depth(np.full(len(Z), np.nan), Z, 20.0))


def test_isotherm_multiple_crossings_use_the_shallowest_downward_one():
    # warm, cold dip, warm again, cold again: crosses 20 downward at 5-10 m, upward at 10-20 m,
    # downward at 20-30 m. The shallowest downward crossing (5-10 m) is the answer.
    t = np.array([25, 22, 19, 22, 18, 16, 14, 12, 10, 9, 8, 7, 6, 5, 4], dtype=float)
    assert ph.isotherm_depth(t, Z, 20.0) == pytest.approx(5 + 5 * 2 / 3)


def test_isotherm_vectorised_matches_single_profiles():
    rng = np.random.default_rng(0)
    base = np.linspace(29, 4, 15)
    field = base[:, None, None] + rng.normal(0, 2.5, size=(15, 4, 5))
    field[:, 0, 0] = np.nan
    field[8:, 1, 1] = np.nan  # shallow column
    got = ph.isotherm_depth(field, Z, 23.0)
    assert got.shape == (4, 5)
    for i in range(4):
        for j in range(5):
            assert np.allclose(
                got[i, j], ph.isotherm_depth(field[:, i, j], Z, 23.0), equal_nan=True
            )
    assert np.isnan(got[0, 0])


# ----------------------------------------------------------------------------------------
# heat content and layer means
# ----------------------------------------------------------------------------------------
def test_heat_content_against_an_analytic_profile():
    t = 20.0 - 0.05 * Z  # linear in depth: the trapezoid rule is exact
    integral = 20.0 * 300 - 0.025 * 300**2  # degC m
    assert ph.layer_integral(t, Z, 300.0) == pytest.approx(integral)
    assert ph.heat_content(t, Z) == pytest.approx(ph.RHO * ph.CP * integral)
    assert ph.layer_mean(t, Z, 300.0) == pytest.approx(integral / 300)
    # J m-2 and the equivalent mean temperature are the same number up to rho * cp * 300
    assert ph.heat_content(t, Z) / (ph.RHO * ph.CP * 300) == pytest.approx(
        ph.layer_mean(t, Z, 300.0)
    )
    # uniform layer: mean temperature equals the temperature, for both layers
    uni = np.full(len(Z), 12.5)
    assert ph.layer_mean(uni, Z, 30.0) == pytest.approx(12.5)
    assert ph.layer_mean(uni, Z, 300.0) == pytest.approx(12.5)
    assert ph.heat_content(uni, Z) == pytest.approx(1025 * 3990 * 12.5 * 300)


def test_heat_content_masks_cells_that_do_not_reach_the_layer_bottom():
    t = 20.0 - 0.05 * Z
    shallow = t.copy()
    shallow[Z > 150] = np.nan  # reaches 150 m only
    assert np.isnan(ph.heat_content(shallow, Z))
    assert np.isfinite(ph.layer_mean(shallow, Z, 30.0))  # but the 0-30 m layer is fine
    holed = t.copy()
    holed[3] = np.nan  # a missing level inside the layer
    assert np.isnan(ph.layer_integral(holed, Z, 300.0))
    surf = t.copy()
    surf[0] = np.nan
    assert np.isnan(ph.layer_mean(surf, Z, 30.0))
    with pytest.raises(ValueError):
        ph.layer_integral(t, Z, 250.0)  # 250 m is not a standard level
    with pytest.raises(ValueError):
        ph.layer_integral(t[:5], Z, 300.0)  # depth axis does not match
    both = ph.derived_fields(t[:, None, None] * np.ones((1, 2, 3)), Z)
    assert set(both) == set(ph.QUANTITIES) and both["hc300"].shape == (2, 3)


# ----------------------------------------------------------------------------------------
# strata
# ----------------------------------------------------------------------------------------
def test_terciles_and_seasons():
    v = np.arange(300, dtype=float)
    edges = ph.tercile_edges(np.r_[v, np.nan])
    cls = ph.tercile_class(np.r_[v, np.nan], edges)
    assert [int((cls == k).sum()) for k in (0, 1, 2)] == [100, 100, 100] and cls[-1] == -1
    assert edges[0] == pytest.approx(99.667, abs=0.01) and edges[1] == pytest.approx(
        199.33, abs=0.01
    )
    assert np.isnan(ph.tercile_edges([np.nan, np.nan])[0])
    assert [ph.season_of_month(m) for m in (12, 1, 2, 3, 5, 6, 9, 10, 11)] == [
        "winter_monsoon", "winter_monsoon", "winter_monsoon", "pre_monsoon", "pre_monsoon",
        "summer_monsoon", "summer_monsoon", "post_monsoon", "post_monsoon",
    ]  # fmt: skip
    assert sorted(m for ms in ph.SEASONS.values() for m in ms) == list(range(1, 13))


def test_day_class_masks_partition_the_strata():
    h, w = 6, 8
    rng = np.random.default_rng(1)
    sla = rng.normal(0, 0.2, size=(5, h, w)).astype(np.float32)
    sss = rng.normal(33, 1.0, size=(5, h, w)).astype(np.float32)
    sla[:, 0, 0] = np.nan
    ocean = np.ones((h, w), bool)
    allm = np.ones((h, w), bool)
    west = np.zeros((h, w), bool)
    west[:, :4] = True
    bob = np.zeros((h, w), bool)
    bob[:, 4:] = True
    edges = strata_edges(sla, sss, ocean, bob)
    # |SLA| edges come from every ocean cell-day, salinity edges only from the "bay" cells
    assert edges["sla_abs"] == ph.tercile_edges(np.abs(sla))
    assert edges["sss_bob"] == ph.tercile_edges(sss[:, :, 4:])
    masks = day_class_masks(sla[0], sss[0], [allm, west, bob], edges)
    names = class_names()
    assert len(masks) == len(names) == 3 + 9 + 3
    pos = {n: i for i, n in enumerate(names)}
    sla_all = [masks[pos[f"sla:all:{t}"]] for t in ("low", "mid", "high")]
    assert not (sla_all[0] & sla_all[1]).any() and not (sla_all[1] & sla_all[2]).any()
    union = sla_all[0] | sla_all[1] | sla_all[2]
    assert not union[0, 0] and union.sum() == h * w - 1  # the NaN cell belongs to no tercile
    for t in ("low", "mid", "high"):  # a basin tercile is the basin part of the domain tercile
        a = masks[pos[f"sla:bay_of_bengal:{t}"]]
        assert np.array_equal(a, masks[pos[f"sla:all:{t}"]] & bob)
        assert not masks[pos[f"sss:bay_of_bengal:{t}"]][:, :4].any()  # only bay cells
    sss_union = np.logical_or.reduce(
        [masks[pos[f"sss:bay_of_bengal:{t}"]] for t in ("low", "mid", "high")]
    )
    assert np.array_equal(sss_union, bob)


# ----------------------------------------------------------------------------------------
# download plumbing (no network)
# ----------------------------------------------------------------------------------------
def test_subset_request_and_download_guards(tiny_cfg: Config, monkeypatch):
    kw = armor3d.subset_kwargs(tiny_cfg, date(2022, 2, 1), date(2022, 2, 28))
    assert kw["dataset_id"] == armor3d.DATASET_ID and kw["variables"] == ["to"]
    assert (kw["minimum_depth"], kw["maximum_depth"]) == (0.0, 1100.0)
    assert (kw["minimum_longitude"], kw["maximum_longitude"]) == (71.5, 78.5)  # grid + 0.5 halo
    assert kw["start_datetime"].startswith("2022-02-01") and kw["end_datetime"].startswith(
        "2022-02-28"
    )
    with pytest.raises(ValueError):  # the MY dataset ends 2024-12-31
        armor3d.download_armor(tiny_cfg, date(2024, 12, 1), date(2025, 1, 5))
    monkeypatch.setattr("oceanembed.research.armor3d.credentials_available", lambda: False)
    with pytest.raises(MissingCredentialsError):
        armor3d.download_armor(tiny_cfg, date(2022, 2, 1), date(2022, 2, 28))


# ----------------------------------------------------------------------------------------
# R4 / R5 on the tiny run
# ----------------------------------------------------------------------------------------
def fake_value(z, lat, lon, doy):
    """The fake ARMOR3D temperature: linear in depth, lat, lon and day, so the vertical linear
    interpolation and the 2 x 2 block mean reproduce it exactly at the canonical points."""
    return 26.0 - 0.012 * z + 0.2 * (lat - 10.0) + 0.1 * (lon - 75.0) + 0.05 * doy


def write_fake_armor(cfg: Config) -> list[Path]:
    """Raw monthly files in the layout of the real download (``latitude`` / ``longitude`` names,
    int depths, 1/8 degree grid, float64) for Feb and Mar 2022, with NaN south of 8.5 N."""
    lat = 7.5625 + 0.125 * np.arange(40)
    lon = 71.5625 + 0.125 * np.arange(56)
    depth = np.array(
        [
            0,
            5,
            10,
            20,
            30,
            50,
            75,
            100,
            125,
            150,
            200,
            250,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
            1100,
        ]
    )
    paths = []
    for first, last in (
        (date(2022, 2, 15), date(2022, 2, 28)),
        (date(2022, 3, 1), date(2022, 3, 3)),
    ):
        t = pd.date_range(first, last)
        doy = t.dayofyear.to_numpy(float)
        arr = fake_value(
            depth[None, :, None, None], lat[None, None, :, None], lon[None, None, None, :],
            doy[:, None, None, None],
        )  # fmt: skip
        arr = np.where(lat[None, None, :, None] < 8.5, np.nan, arr)
        ds = xr.Dataset(
            {"to": (("time", "depth", "latitude", "longitude"), arr)},
            coords={"time": t, "depth": depth.astype("int16"), "latitude": lat.astype("float32"),
                    "longitude": lon.astype("float32")},
        )  # fmt: skip
        path = armor3d.armor_file(cfg, first.replace(day=1))
        path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(path)
        paths.append(path)
    return paths


@pytest.fixture(scope="module")
def r45_env(_tiny_run_roots, tmp_path_factory):
    """A copy of the finished tiny run (own outputs root) with the run's data root."""
    data, out, _ = _tiny_run_roots
    root = tmp_path_factory.mktemp("oe_r45_out")
    shutil.copytree(out / "test_tiny", root / "test_tiny")
    keys = ("OCEANEMBED_DATA_ROOT", "OCEANEMBED_OUTPUTS_ROOT")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ["OCEANEMBED_DATA_ROOT"] = str(data)
    os.environ["OCEANEMBED_OUTPUTS_ROOT"] = str(root)
    try:
        yield load_config(TINY)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_r4_regrid_matches_the_analytic_field_and_keeps_the_product_time_steps(r45_env: Config):
    cfg = r45_env
    write_fake_armor(cfg)
    assert (
        armor3d.download_armor(cfg, date(2022, 2, 20), date(2022, 3, 1)) == []
    )  # nothing to fetch
    written = regrid_all(cfg, "2022-02-20", "2022-03-01")
    assert len(written) == 2 and regrid_all(cfg, "2022-02-20", "2022-03-01") == []  # resumable
    days = regridded_days(cfg, "2022-02-20", "2022-03-01")
    # only the product's own days inside the test split, and the 3-day March file stops at 3 Mar
    assert list(days) == list(pd.date_range("2022-02-20", "2022-03-01"))
    with xr.open_dataset(regridded_file(cfg, date(2022, 2, 1))) as ds:
        assert ds["temp"].dims == ("time", "depth", "lat", "lon") and ds.sizes["depth"] == 15
        z, la, lo = np.meshgrid(
            ds["depth"].values, ds["lat"].values, ds["lon"].values, indexing="ij"
        )
        t0 = pd.Timestamp(ds["time"].values[3])
        want = fake_value(z, la, lo, t0.dayofyear)
        want = np.where(la < 8.5, np.nan, want)
        assert np.allclose(ds["temp"].values[3], want, atol=1e-4, equal_nan=True)
        assert (la < 8.5).any() and np.isnan(ds["temp"].values[:, :, la[0, :, 0] < 8.5]).all()


def test_r4_scores_argo_and_grid_on_identical_samples(r45_env: Config):
    cfg = r45_env
    write_fake_armor(cfg)
    summary = run_r4(cfg, download=False, n_boot=60, progress=False)
    out = r4_dir(cfg)
    assert (out / "summary.json").exists() and (out / "summary.md").exists()
    assert list((out / "figures").glob("*.png"))
    # Argo: the armor3d column is the analytic field in the matchup's cell, day and depth
    m = pd.read_parquet(cfg.outputs_dir / "metrics" / "argo_matchups.parquet")
    arm = armor_at_matchups(cfg, m, "2022-02-20", "2022-03-01")
    day = pd.DatetimeIndex(m["time"]).normalize()
    want = fake_value(m["depth"].to_numpy(float), m["grid_lat"].to_numpy(float),
                      m["grid_lon"].to_numpy(float), day.dayofyear.to_numpy(float))  # fmt: skip
    want = np.where(m["grid_lat"].to_numpy(float) < 8.5, np.nan, want)
    assert np.allclose(arm, want, atol=1e-3, equal_nan=True)
    info = summary["argo"]["counts"]
    assert info["matchups_total"] == len(m) and info["matchups_with_armor3d"] == int(
        np.isfinite(want).sum()
    )
    # pooled 50-200 m RMSE of every product, recomputed from the table on the same rows
    keep = m[np.isfinite(want)].assign(armor3d=arm[np.isfinite(want)])
    sel = keep[(keep["depth"] >= 50) & (keep["depth"] <= 200)]
    blk = summary["argo"]["regions"]["all"]
    cols = {"armor3d": "armor3d", "model": "model", "scratch": "model_scratch", "ridge": "ridge",
            "climatology": "clim", "glorys": "glorys"}  # fmt: skip
    for p, col in cols.items():
        direct = float(np.sqrt(np.mean((sel[col] - sel["obs"]) ** 2)))
        assert blk["products"][p]["rmse"]["pooled_50_200m"]["point"] == pytest.approx(
            direct, rel=1e-4
        )
    assert blk["n"]["pooled_50_200m"] == len(sel)
    # grid: armor3d vs GLORYS over the common sample, recomputed with plain numpy
    zds = open_harmonized(cfg)
    mask = zds["mask"].values
    se, n = 0.0, 0
    for d in pd.date_range("2022-02-20", "2022-03-01"):
        z, la, lo = np.meshgrid(
            zds["depth"].values, zds["lat"].values, zds["lon"].values, indexing="ij"
        )
        a = np.where(la < 8.5, np.nan, fake_value(z, la, lo, d.dayofyear))
        g = zds["temp"].sel(time=d).values
        ok = mask & np.isfinite(a) & np.isfinite(g)
        se += float(((a - g)[ok] ** 2).sum())
        n += int(ok.sum())
    got = summary["grid"]["glorys"]["regions"]["all"]
    assert got["products"]["armor3d"]["rmse"]["overall"]["point"] == pytest.approx(
        np.sqrt(se / n), rel=1e-4
    )
    assert got["n"]["overall"] == n
    assert "armor3d" not in summary["grid"]["armor3d"]["products"]
    # the same days / depths feed every product: sample sizes agree across products of a block
    assert set(summary["grid"]["glorys"]["products"]) == {
        "climatology",
        "ridge",
        "scratch",
        "model",
        "armor3d",
    }


def test_r5_streams_and_reproduces_the_evaluate_scores(r45_env: Config):
    cfg = r45_env
    before = {p: p.stat().st_mtime_ns for p in (cfg.outputs_dir / "predictions").rglob("*.nc")}
    path = run_r5(cfg, progress=False)
    assert path == r5_dir(cfg) / "sums.npz"
    summary = make_r5_report(cfg, n_boot=60)
    out = r5_dir(cfg)
    assert (out / "summary.json").exists() and (out / "summary.md").exists()
    assert len(list((out / "figures").glob("*.png"))) == 5
    assert before == {
        p: p.stat().st_mtime_ns for p in (cfg.outputs_dir / "predictions").rglob("*.nc")
    }
    # products: the MLP checkpoint of R1 does not exist in the tiny run, so it is skipped
    assert list(summary["settings"]["products"]) == ["climatology", "ridge", "scratch", "model"]
    # the pooled 50-200 m RMSE of region:all equals `evaluate`'s pooled_50_200m RMSE per method
    ev = json.loads((cfg.outputs_dir / "metrics" / "metrics_glorys.json").read_text())["methods"]
    reg = summary["seasons"]["all_year"]["regions"]["all"]["products"]
    for mine, theirs in {"model": "model", "scratch": "model_scratch", "ridge": "ridge",
                         "climatology": "climatology"}.items():  # fmt: skip
        assert reg[mine]["rmse"]["point"] == pytest.approx(
            ev[theirs]["pooled_50_200m"]["rmse"], rel=1e-4
        )
    # seasons partition the days; terciles partition the samples of a basin
    days = {k: v["n_days"] for k, v in summary["seasons"].items()}
    assert sum(v for k, v in days.items() if k != "all_year") == days["all_year"]
    for region in ("all", "arabian_sea"):
        n_t = sum(
            summary["sla"][region]["terciles"][t]["n_samples"] for t in ("low", "mid", "high")
        )
        assert 0 < n_t <= summary["seasons"]["all_year"]["regions"][region]["n_samples"]
    # derived quantities exist, are finite on a common sample, and the climatology has no skill
    d20 = summary["derived"]["d20"]["regions"]["all"]
    assert d20["n_cell_days"] > 0
    assert d20["products"]["climatology"]["skill_vs_clim"]["point"] == pytest.approx(0.0, abs=1e-12)
    assert d20["products"]["model"]["rmse"]["point"] is not None
    # the sample is common: n is identical for every product (rmse for all, corr not for clim)
    s = np.load(out / "sums.npz")
    n_idx = list(s["fields"]).index("n")
    n_by_product = s["dq_sums"][0, 0, :, 0, n_idx].sum(axis=-1)
    assert np.allclose(n_by_product, n_by_product[0])


def test_cli_commands_run_on_the_tiny_run(r45_env: Config):
    runner = CliRunner()
    write_fake_armor(r45_env)
    r5 = runner.invoke(app, ["research", "r5", "--config", str(TINY), "--n-boot", "40"])
    assert r5.exit_code == 0, r5.output + str(r5.exception)
    r4 = runner.invoke(
        app, ["research", "r4", "--config", str(TINY), "--no-download", "--n-boot", "40"]
    )
    assert r4.exit_code == 0, r4.output + str(r4.exception)
    assert "summary.json" in r4.output
