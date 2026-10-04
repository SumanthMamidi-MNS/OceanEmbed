"""Live nowcast: availability with different delays, incremental update and resume, window pruning
(live folders only), the revision policy, provenance, input-shift statistics, verification scoring
and the API. The network is replaced by slices of the synthetic raw products."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from fastapi.testclient import TestClient

from oceanembed.api import create_app
from oceanembed.api import schemas as S
from oceanembed.config import Config, load_config
from oceanembed.data.stats import Stats
from oceanembed.live import nrt, shift, update, verify
from oceanembed.live.state import LockedError, UpdateLock, live_paths, load_days, load_state
from oceanembed.models.recon import ReconModel

SST_ID, SLA_ID = "SST_NRT_TEST", "SLA_NRT_TEST"
UTC = timezone.utc  # noqa: UP017 - datetime.UTC is the same object


def _yaml(weights: Path) -> str:
    return f"""
run_name: live_tiny
paths: {{data_root: data, outputs_root: outputs, raw_dir: raw_nrt}}
grid: {{lat_min: 8.0, lat_max: 12.0, lon_min: 72.0, lon_max: 78.0, resolution: 0.25}}
time: {{start: 2022-01-01, end: 2022-03-01}}
split:
  train: [2022-01-01, 2022-02-09]
  val: [2022-02-10, 2022-02-19]
  test: [2022-02-20, 2022-03-01]
provider: real
products:
  sst: {{provider: cmems, datasets: [{{id: {SST_ID}}}], variables: {{sst: analysed_sst}}}}
  sla: {{provider: cmems, datasets: [{{id: {SLA_ID}}}], variables: {{sla: sla}}}}
download: {{halo_deg: 0.5, retries: 0}}
model: {{emb_dim: 16, dim: 32, depth: 2, heads: 2, stem_channels: 16, main_init: scratch,
  input_groups: [sst, sla]}}
live:
  weights: {weights.as_posix()}
  window_days: 10
  revision_days: 3
  request_days: 5
  verification: {{enabled: false}}
"""


class FakeServer:
    """Replaces the two network calls: serves slices of the synthetic raw products and records
    what was asked."""

    def __init__(self, synth_root: Path, last: dict[str, date]):
        self.synth, self.last = synth_root, last
        self.requests: list[tuple[str, str, str]] = []
        self.perturb: dict[tuple[str, date], float] = {}  # (variable, day) -> offset added
        self.fail_after: int | None = None

    def describe(self, dataset_id: str) -> dict:
        return {"version": "v1", "first": date(2022, 1, 1), "last": self.last[dataset_id]}

    def subset(self, **kw) -> None:
        if self.fail_after is not None and len(self.requests) >= self.fail_after:
            raise ConnectionError("network down")
        ds_id, var = kw["dataset_id"], kw["variables"][0]
        t0, t1 = pd.Timestamp(kw["start_datetime"]), pd.Timestamp(kw["end_datetime"])
        self.requests.append((ds_id, str(t0.date()), str(t1.date())))
        product = "sst" if ds_id == SST_ID else "sla"
        parts = []
        for m in pd.period_range(t0, t1, freq="M"):
            f = self.synth / product / f"{product}_{m.year}{m.month:02d}.nc"
            with xr.open_dataset(f) as d:
                parts.append(d.load())
        raw = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
        raw = raw.sel(time=slice(t0, t1))
        raw[var] = raw[var].astype("float64")
        for (v, day), off in self.perturb.items():
            if v == var:
                sel = raw.time.dt.floor("D") == np.datetime64(day)
                raw[var] = raw[var].where(~sel, raw[var] + off)
        out = Path(kw["output_directory"]) / kw["output_filename"]
        raw.to_netcdf(out)

    def n(self, ds_id: str) -> int:
        return sum(1 for r in self.requests if r[0] == ds_id)


@pytest.fixture(scope="module")
def live_world(tiny_pipeline: Config, tmp_path_factory):
    """Released-weights folder of a tiny, randomly initialised model (inputs sst + sla) with the
    statistics and ocean mask of the tiny synthetic store."""
    tmp = tmp_path_factory.mktemp("live_weights")
    weights = tmp / "weights"
    weights.mkdir()
    yml = tmp / "live.yaml"
    yml.write_text(_yaml(weights), encoding="utf-8")
    cfg = load_config(yml)
    torch.manual_seed(0)
    model = ReconModel(cfg.model)
    torch.save(
        {"model": model.state_dict(), "config": cfg.model_dump(mode="json"), "epoch": 1,
         "pretrained": False, "tag": None, "metrics": {"val_rmse": 1.0}},
        weights / "recon.pt",
    )  # fmt: skip
    stats = Stats.load(tiny_pipeline.stats_path)
    with xr.open_zarr(tiny_pipeline.zarr_path, consolidated=False) as store:
        mask = store["mask"].values.astype(bool)
        depth = np.asarray(store["depth"].values)
    ds = stats.to_dataset()
    ds["ocean_mask"] = (("depth", "lat", "lon"), mask)
    ds.assign_coords(depth=depth).to_netcdf(weights / "stats.nc")
    return {"yaml": yml, "synth": tiny_pipeline.raw_root, "mask": mask}


@pytest.fixture
def live(live_world, tmp_path, monkeypatch):
    """Fresh data / outputs roots, the fake server and the tiny live config."""
    monkeypatch.setenv("OCEANEMBED_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("OCEANEMBED_OUTPUTS_ROOT", str(tmp_path / "outputs"))
    cfg = load_config(live_world["yaml"])
    server = FakeServer(live_world["synth"], {SST_ID: date(2022, 2, 20), SLA_ID: date(2022, 2, 20)})
    monkeypatch.setattr(nrt, "describe_dataset", server.describe)
    monkeypatch.setattr(nrt, "subset_to_file", server.subset)
    monkeypatch.setattr(nrt, "check_credentials", lambda: None)
    return cfg, server


def now_for(day: date, hours: int = 12) -> datetime:
    return datetime(day.year, day.month, day.day, hours, tzinfo=UTC)


def run(cfg, server, today: date, **kw):
    return update.run_update(cfg, device="cpu", now=now_for(today), progress=False, **kw)


# ------------------------------------------------------------------------------ pure helpers
def test_runs_of_days_split_gaps_and_the_request_length():
    d = date(2022, 1, 1)
    days = [d + timedelta(days=i) for i in (0, 1, 2, 4, 5, 6, 7, 8, 9)]
    assert nrt.runs_of_days(days, 31) == [
        (d, d + timedelta(days=2)),
        (d + timedelta(days=4), d + timedelta(days=9)),
    ]
    assert nrt.runs_of_days(days, 2)[0] == (d, d + timedelta(days=1))
    assert nrt.runs_of_days([], 5) == []


def test_window_never_shrinks_and_ends_at_the_newest_common_day():
    assert update.window_for(date(2022, 2, 10), None, 10) == (date(2022, 2, 1), date(2022, 2, 10))
    # a product falling back by a day does not move the window end backwards
    assert update.window_for(date(2022, 2, 9), date(2022, 2, 10), 10)[1] == date(2022, 2, 10)


def test_revision_set_is_the_newest_reconstructed_days():
    have = [date(2022, 2, i) for i in range(1, 11)]
    assert update.revision_set(have, date(2022, 2, 10), 3) == have[-3:]
    assert update.revision_set(have, date(2022, 2, 8), 3) == have[5:8]
    assert update.revision_set(have, date(2022, 2, 10), 0) == []


def test_revision_size_and_summary():
    depths = np.array([0.0, 50.0, 100.0, 500.0])
    a = np.zeros((4, 3, 3), np.float32)
    b = a.copy()
    b[1] += 0.5
    b[3, 0, 0] = 2.0
    in_a = {"sst": np.zeros((3, 3), np.float32), "sla": np.zeros((3, 3), np.float32)}
    in_b = {"sst": np.full((3, 3), 0.2, np.float32), "sla": np.zeros((3, 3), np.float32)}
    s = update.revision_size(in_b, in_a, b, a, depths)
    assert s["rev_sst_rmse"] == pytest.approx(0.2, abs=1e-6)
    assert s["rev_sla_rmse"] == 0.0
    assert s["rev_recon_rmse_50_200"] == pytest.approx(np.sqrt(0.25 * 1 / 2), abs=1e-6)
    assert s["rev_recon_maxabs"] == pytest.approx(2.0)
    rows = [
        {"date": "2022-02-01", "age_days": 2, "sst_rmse": 0.2, "changed_since_first": True,
         "recon_rmse_50_200": 0.3},
        {"date": "2022-02-01", "age_days": 3, "sst_rmse": 0.2, "changed_since_first": True,
         "recon_rmse_50_200": 0.4},
        {"date": "2022-02-02", "age_days": 2, "sst_rmse": 0.0, "changed_since_first": False,
         "recon_rmse_50_200": 0.0},
    ]  # fmt: skip
    st = update.summarise_revisions(rows)
    assert st["n_days_checked"] == 2 and st["n_days_revised"] == 1
    by = {r["age_days"]: r for r in st["by_age"]}
    assert by[2]["n_checks"] == 2 and by[2]["n_changed_since_first"] == 1
    assert by[3]["recon_rmse_50_200_max"] == pytest.approx(0.4)


def test_the_digest_changes_with_the_inputs():
    f = {"sst": np.ones((2, 2), np.float32), "sla": np.zeros((2, 2), np.float32)}
    g = {"sst": np.ones((2, 2), np.float32) * 1.001, "sla": np.zeros((2, 2), np.float32)}
    assert update.inputs_digest(f) == update.inputs_digest(dict(f))
    assert update.inputs_digest(f) != update.inputs_digest(g)


def test_a_model_input_without_a_near_real_time_product_is_refused(live):
    cfg, _ = live
    cfg2 = cfg.model_copy(deep=True)
    cfg2.model.input_groups = ["sst", "sla", "winds"]
    with pytest.raises(nrt.LiveError, match="winds"):
        nrt.required_products(cfg2)


def test_the_lock_refuses_a_second_update_and_takes_over_a_stale_one(tmp_path):
    lock = tmp_path / ".update.lock"
    with UpdateLock(lock):
        with pytest.raises(LockedError):
            UpdateLock(lock).__enter__()
    assert not lock.exists()
    lock.write_text("pid 1")
    old = lock.stat().st_mtime - 5 * 3600
    import os

    os.utime(lock, (old, old))
    with UpdateLock(lock):
        pass


# ------------------------------------------------------------------------------ the update
def test_cold_start_with_different_delays_and_provenance(live):
    cfg, server = live
    server.last = {SST_ID: date(2022, 2, 20), SLA_ID: date(2022, 2, 18)}  # sla two days later
    s = run(cfg, server, date(2022, 2, 21))
    assert s["last_day"] == "2022-02-18"  # the newest day both inputs have
    assert s["window"] == {
        "start": "2022-02-09",
        "end": "2022-02-18",
        "n_days": 10,
        "window_days": 10,
    }
    assert s["pending"] == ["2022-02-19", "2022-02-20"]  # sst has them, sla does not yet
    assert s["inputs"]["sst"]["age_days"] == 3 and s["inputs"]["sla"]["age_days"] == 3
    assert s["inputs"]["sst"]["delay_days_catalogue"] == 1
    assert s["inputs"]["sla"]["delay_days_catalogue"] == 3
    assert s["n_new_days"] == 10 and s["n_reconstructed"] == 10
    paths = live_paths(cfg)
    days = load_days(cfg)
    assert len(days) == 10 and days.index[0] == pd.Timestamp("2022-02-09")
    assert set(days["sst_dataset"]) == {SST_ID} and set(days["sla_dataset"]) == {SLA_ID}
    assert (
        days["sst_age_days"].astype(int)
        == (date(2022, 2, 21) - days["date"].dt.date).map(lambda x: x.days)
    ).all()
    assert not days["revised"].any() and (days["n_checks"] == 0).all()
    # run contract
    meta = json.loads((paths.run / "run_meta.json").read_text())
    assert meta["data_source"] == "real" and meta["live"]["nrt"] is True
    files = sorted(p.name for p in paths.predictions.glob("*.nc"))
    assert files == ["oceanembed_T_202202.nc"]
    with xr.open_dataset(paths.predictions / files[0]) as ds:
        assert ds.sizes["time"] == 10 and ds.attrs["live_nrt"] == "true"
        assert np.isfinite(ds["temperature"].values).any()
    with xr.open_zarr(paths.store, consolidated=False) as store:
        assert store.sizes["time"] == 10
        assert np.isfinite(store["sst"].values).any() and np.isfinite(store["sla"].values).any()
        assert not np.isfinite(store["temp"].values).any()  # no target for the live days
    # the status reads only local files
    st = update.status(cfg, now=now_for(date(2022, 2, 21)))
    assert st["last_day"] == "2022-02-18" and st["pending"] == ["2022-02-19", "2022-02-20"]
    assert st["inputs"]["sst"]["age_days"] == 3


def test_incremental_update_downloads_only_new_days_and_prunes_live_folders_only(live, tmp_path):
    cfg, server = live
    server.last = {SST_ID: date(2022, 2, 14), SLA_ID: date(2022, 2, 14)}
    run(cfg, server, date(2022, 2, 15))
    decoy_raw = cfg.data_root / "raw" / "sst"
    decoy_raw.mkdir(parents=True)
    (decoy_raw / "sst_20220101.nc").write_bytes(b"keep")  # a reprocessed raw file
    other_run = cfg.outputs_root / "final"
    other_run.mkdir(parents=True)
    (other_run / "keep.txt").write_text("keep")
    n_before = len(server.requests)
    server.last = {SST_ID: date(2022, 2, 17), SLA_ID: date(2022, 2, 17)}
    s = run(cfg, server, date(2022, 2, 18))
    assert s["n_new_days"] == 3 and s["window"]["end"] == "2022-02-17"
    # new days (one request per product) + the revision look at the newest 3 reconstructed days
    new_days = [r for r in server.requests[n_before:] if r[1] == "2022-02-15"]
    assert len(new_days) == 2 and all(r[2] == "2022-02-17" for r in new_days)
    paths = live_paths(cfg)
    days = load_days(cfg)
    assert days.index[0] == pd.Timestamp("2022-02-08") and len(days) == 10
    assert not (paths.raw / "sst" / "sst_20220207.nc").exists()  # left the window
    assert not (paths.first / "first_20220207.npz").exists()
    assert (paths.raw / "sst" / "sst_20220217.nc").exists()
    assert (decoy_raw / "sst_20220101.nc").read_bytes() == b"keep"
    assert (other_run / "keep.txt").exists()
    assert s["pruned"]["raw_files"] >= 2
    with xr.open_dataset(paths.predictions / "oceanembed_T_202202.nc") as ds:
        assert ds.sizes["time"] == 10 and str(ds.time.values[0])[:10] == "2022-02-08"
    # nothing new: no new day, the newest days are looked at again and found unchanged
    again = run(cfg, server, date(2022, 2, 18))
    assert again["n_new_days"] == 0 and again["n_reconstructed"] == 0
    assert again["n_revision_checks"] == 3 and again["n_revised_days"] == 0


def test_revision_is_detected_reconstructed_and_measured(live):
    cfg, server = live
    server.last = {SST_ID: date(2022, 2, 14), SLA_ID: date(2022, 2, 14)}
    run(cfg, server, date(2022, 2, 15))
    paths = live_paths(cfg)
    with xr.open_dataset(paths.predictions / "oceanembed_T_202202.nc") as d:
        before = d.load()
    server.perturb[("analysed_sst", date(2022, 2, 13))] = 0.8  # the product is corrected later
    s = run(cfg, server, date(2022, 2, 16))
    assert s["n_revised_days"] == 1 and s["n_reconstructed"] == 1
    days = load_days(cfg)
    r = days.loc["2022-02-13"]
    assert bool(r["revised"]) and r["n_revisions"] == 1
    assert r["rev_sst_rmse"] > 0.1 and r["rev_recon_rmse_50_200"] > 0 and r["rev_sla_rmse"] == 0
    assert not bool(days.loc["2022-02-12", "revised"]) and days.loc["2022-02-12", "n_checks"] == 1
    with xr.open_dataset(paths.predictions / "oceanembed_T_202202.nc") as d:
        after = d.load()
    t13 = np.datetime64("2022-02-13")
    assert not np.allclose(
        before["temperature"].sel(time=t13).values, after["temperature"].sel(time=t13).values,
        equal_nan=True,
    )  # fmt: skip
    t12 = np.datetime64("2022-02-12")
    np.testing.assert_array_equal(
        before["temperature"].sel(time=t12).values, after["temperature"].sel(time=t12).values
    )
    stats = load_state(cfg)["revision_stats"]
    assert stats["n_days_revised"] == 1 and stats["n_days_checked"] >= 3
    # the first-published version is kept: the correction is measured against it
    snap = update.load_snapshot(cfg, date(2022, 2, 13))
    assert snap is not None and snap[1].shape == (15, 16, 24)


def test_an_interrupted_update_is_continued_by_the_next_one(live):
    cfg, server = live
    server.last = {SST_ID: date(2022, 2, 14), SLA_ID: date(2022, 2, 14)}
    server.fail_after = 1  # the first sst request (5 of the 10 days) succeeds, the next one fails
    with pytest.raises(ConnectionError):
        run(cfg, server, date(2022, 2, 15))
    assert not live_paths(cfg).days.exists()
    assert not live_paths(cfg).lock.exists()  # the lock is released on failure
    server.fail_after = None
    s = run(cfg, server, date(2022, 2, 15))
    assert s["n_reconstructed"] == 10
    sst = [r for r in server.requests if r[0] == SST_ID]
    assert len(sst) == 2 and sst[0][1] != sst[1][1]  # the five finished days were not fetched again
    assert len(load_days(cfg)) == 10


def test_no_common_day_is_an_error_not_an_empty_run(live):
    cfg, server = live
    server.describe = lambda i: (
        {"version": "v1", "first": date(2022, 1, 1), "last": date(2022, 2, 10)}
        if i == SST_ID
        else {"version": "v1", "first": date(2022, 2, 12), "last": date(2022, 2, 14)}
    )  # the sea level product has nothing for the days sst covers
    monkeypatch_describe = nrt.describe_dataset
    try:
        nrt.describe_dataset = server.describe
        with pytest.raises(nrt.LiveError, match="no day"):
            run(cfg, server, date(2022, 2, 15))
    finally:
        nrt.describe_dataset = monkeypatch_describe


# ------------------------------------------------------------------------------ verification
def _profiles(days, n_per_day=3):
    rows = []
    pid = 0
    rng = np.random.default_rng(0)
    for d in days:
        for _ in range(n_per_day):
            pid += 1
            lat, lon = rng.uniform(9, 11), rng.uniform(74, 76)
            for z in (0.0, 10.0, 30.0, 60.0, 100.0, 150.0, 200.0, 300.0, 500.0):
                rows.append(
                    {"profile_id": f"p{pid}", "platform": "1", "cycle": 1,
                     "time": pd.Timestamp(d) + pd.Timedelta(hours=6), "latitude": lat,
                     "longitude": lon, "pres": z, "depth": z, "temp": 28.0 - 0.04 * z,
                     "temp_qc": 1}
                )  # fmt: skip
    return pd.DataFrame(rows)


def test_running_verification_scores_model_and_climatology(live, monkeypatch):
    cfg, server = live
    cfg = cfg.model_copy(deep=True)
    server.last = {SST_ID: date(2022, 2, 14), SLA_ID: date(2022, 2, 14)}
    monkeypatch.setattr(
        verify, "fetch_argo_window",
        lambda cfg, a, b: _profiles(pd.date_range(a, b, freq="D")),
    )  # fmt: skip
    monkeypatch.setattr(verify, "analysis_size_mb", lambda *a, **k: 9e9)  # too big: skipped cleanly
    s = run(cfg, server, date(2022, 2, 15), verify=True)
    v = s["verification"]
    assert v["argo"]["n_matchups"] > 0 and v["argo"]["n_profiles"] == 30
    assert v["analysis"]["available"] is False and "limit" in v["analysis"]["reason"]
    last = v["argo"]["latest"]
    assert last["date"] == "2022-02-14"
    for band in ("0_30", "50_200", "300_1000"):
        b = last["bands"][band]["rolling"]
        assert b["n"] > 0 and b["model_rmse"] is not None and b["clim_rmse"] is not None
    out = live_paths(cfg).verification
    assert (out / "verification.json").exists() and (out / "daily.parquet").exists()
    # a second update keeps the rolling record and adds to the arrival-by-age record
    run(cfg, server, date(2022, 2, 15), verify=True)
    daily = pd.read_parquet(out / "daily.parquet")
    assert not daily.duplicated(["ref", "date", "band"]).any()
    # a failing verification does not undo the update
    monkeypatch.setattr(verify, "fetch_argo_window", lambda *a: (_ for _ in ()).throw(OSError("x")))
    s = run(cfg, server, date(2022, 2, 15), verify=True)
    assert "error" in s["verification"] and s["n_reconstructed"] == 0


def test_daily_rows_and_rolling_series_on_constructed_matchups():
    t = pd.DataFrame(
        {"date": pd.to_datetime(["2022-02-01"] * 4 + ["2022-02-02"] * 2),
         "profile_id": ["a", "a", "b", "b", "c", "c"],
         "depth": [10.0, 100.0, 10.0, 100.0, 10.0, 100.0],
         "obs": [20.0] * 6, "model": [21, 21, 19, 22, 20, 20], "clim": [22, 22, 22, 22, 23, 23]}
    )  # fmt: skip
    rows = verify.daily_rows(t, "argo")
    r = rows[(rows["band"] == "0_30") & (rows["date"] == "2022-02-01")].iloc[0]
    assert r["n"] == 2 and r["sum_e2_model"] == 1 + 1 and r["sum_e_model"] == 0
    assert r["sum_e2_clim"] == 8
    m = verify.metrics_from_rows(rows[rows["band"] == "50_200"])
    assert m["n"] == 3 and m["model_rmse"] == pytest.approx(np.sqrt((1 + 4 + 0) / 3))
    assert m["clim_bias"] == pytest.approx((2 + 2 + 3) / 3)
    ser = verify.rolling_series(rows, "argo", 30)
    assert [x["date"] for x in ser] == ["2022-02-01", "2022-02-02"]
    assert ser[1]["bands"]["50_200"]["rolling"]["n"] == 3
    assert ser[1]["bands"]["50_200"]["day"]["n"] == 1
    assert verify.metrics_from_rows(rows.iloc[0:0])["n"] == 0
    short = verify.rolling_series(rows, "argo", 1)  # a one-day window sees only its own day
    assert short[1]["bands"]["50_200"]["rolling"]["n"] == 1


# ------------------------------------------------------------------------------ input shift
def test_input_shift_statistics_on_constructed_fields():
    rng = np.random.default_rng(1)
    nd, h, w, ndays = 15, 4, 6, 40
    depth = np.array([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000.0])
    mask = np.ones((nd, h, w), bool)
    mask[:, :, 0] = False
    regions = {"all": np.ones((h, w), bool), "arabian_sea": np.zeros((h, w), bool),
               "bay_of_bengal": np.zeros((h, w), bool)}  # fmt: skip
    regions["arabian_sea"][:, :3] = True
    regions["bay_of_bengal"][:, 3:] = True
    dd = np.zeros((ndays, 3, nd, 3))
    tt = np.zeros((ndays, 3, nd, 5))
    inp = np.zeros((ndays, 3, 2, 4))
    for t in range(ndays):
        target = rng.normal(size=(nd, h, w))
        rep = target + rng.normal(scale=1.0, size=target.shape)
        nr = rep + 0.3 + rng.normal(scale=0.1, size=target.shape)  # NRT: +0.3 bias, little noise
        dd[t], tt[t] = shift.recon_sums(rep, nr, target, mask, regions)
        a = rng.normal(size=(h, w))
        inp[t, :, 0] = shift.input_sums(a, a + 0.1, mask[0], regions)
        inp[t, :, 1] = shift.input_sums(a, a, mask[0], regions)
    days = np.array([str(date(2022, 1, 1) + timedelta(days=i)) for i in range(ndays)])
    sums = {"dd": dd, "tt": tt, "inp": inp, "depth": depth, "have_target": np.ones(ndays, bool),
            "days": days}  # fmt: skip
    out = shift.summarise_shift(sums, n_boot=200, seed=0, block_length=3)
    a = out["inputs"]["sst"]["all"]
    assert a["bias"] == pytest.approx(0.1) and a["rmse"] == pytest.approx(0.1)
    assert out["inputs"]["sla"]["all"]["rmse"] == pytest.approx(0.0)
    assert a["n"] == ndays * (h * (w - 1))  # land column excluded
    rd = out["recon_difference"]["all"]["pooled_50_200m"]
    assert rd["bias"] == pytest.approx(0.3, abs=0.02) and rd["rmse"] > 0.3
    c = out["vs_glorys"]["all"]["pooled_50_200m"]
    assert c["rmse_nrt"]["point"] > c["rmse_reprocessed"]["point"]
    assert c["rmse_change"]["point"] == pytest.approx(
        c["rmse_nrt"]["point"] - c["rmse_reprocessed"]["point"]
    )
    assert c["rmse_change"]["excludes_zero"] is True and c["rmse_change"]["ci_lo"] > 0
    assert out["headline"]["measurably_worse"] is True
    assert "measurably worse" in out["headline_text"]
    # identical inputs: no difference, and the interval contains 0
    dd2, tt2 = dd.copy(), tt.copy()
    tt2[..., 3:] = tt2[..., 1:3]
    same = shift.summarise_shift({**sums, "dd": dd2, "tt": tt2}, 200, 0, 3)
    ch = same["vs_glorys"]["all"]["pooled_50_200m"]["rmse_change"]
    assert ch["point"] == 0 and not ch["excludes_zero"]
    assert same["headline"]["measurably_worse"] is False


def test_overlap_days_are_clamped_to_every_product(live):
    cfg, _ = live
    cfg = cfg.model_copy(deep=True)
    cfg.live.input_shift.start, cfg.live.input_shift.end = date(2022, 1, 1), date(2022, 3, 1)
    days = shift.overlap_days(
        cfg, {"a": (date(2022, 1, 10), date(2022, 3, 5)), "b": (date(2021, 1, 1), date(2022, 2, 1))}
    )
    assert days[0] == date(2022, 1, 10) and days[-1] == date(2022, 2, 1) and len(days) == 23
    assert shift.overlap_days(cfg, {"a": (date(2023, 1, 1), date(2023, 2, 1))}) == []


# ------------------------------------------------------------------------------ the API
def test_api_live_run_flag_and_payload_and_404_for_other_runs(live, monkeypatch, tmp_path):
    cfg, server = live
    server.last = {SST_ID: date(2022, 2, 14), SLA_ID: date(2022, 2, 13)}
    run(cfg, server, date(2022, 2, 15))
    shift_dir = live_paths(cfg).shift
    shift_dir.mkdir(parents=True)
    (shift_dir / "summary.json").write_text(
        json.dumps({"headline": {"n_days": 5}, "headline_text": "x", "updated": "now"})
    )
    other = cfg.outputs_root / "plain"  # a run without a live block
    other.mkdir()
    meta = json.loads((live_paths(cfg).run / "run_meta.json").read_text())
    meta.pop("live")
    meta["run_name"] = "plain"
    (other / "run_meta.json").write_text(json.dumps(meta))
    client = TestClient(
        create_app(cfg.outputs_root, web_dist=tmp_path / "no_dist"), raise_server_exceptions=False
    )
    runs = {r["name"]: r for r in client.get("/api/runs").json()["runs"]}
    assert runs["live_tiny"]["live"] is True and runs["live_tiny"]["live_last_day"] == "2022-02-13"
    assert runs["live_tiny"]["data_source"] == "real"
    assert runs["plain"]["live"] is False
    r = client.get("/api/runs/live_tiny/live")
    assert r.status_code == 200, r.text
    S.LiveResponse.model_validate(r.json())
    body = r.json()
    assert r.headers["cache-control"] == "no-store"
    assert body["last_day"] == "2022-02-13" and body["window"]["n_days"] == 10
    ages = {i["product"]: i for i in body["inputs"]}
    assert ages["sst"]["latest_data_date"] == "2022-02-13" and ages["sla"]["age_days"] == 2
    assert body["pending"] == ["2022-02-14"]
    assert ages["sst"]["available"] is True  # both inputs have data for the newest day
    assert ages["sla"]["available"] is True
    wd = body["window_days"]
    assert len(wd) == 10 and all(
        d["complete"] and d["inputs"] == {"sst": True, "sla": True} for d in wd
    )
    assert (
        runs["live_tiny"]["artefacts"]["live"] is True
        and runs["plain"]["artefacts"]["live"] is False
    )
    assert len(body["days"]) == 10 and "inputs_digest" not in body["days"][0]
    assert body["days"][0]["sst_dataset"] == SST_ID
    assert body["input_shift"]["headline"] == {"n_days": 5} and body["verification"] is None
    assert "policy" in body["revision"]
    bad = client.get("/api/runs/plain/live")
    assert bad.status_code == 404 and "not a live run" in bad.json()["detail"]
    assert client.get("/api/runs/nope/live").status_code == 404
    # the ordinary field endpoints work on the live run
    day = "2022-02-13"
    f = client.get(f"/api/runs/live_tiny/fields?date={day}&depth=50")
    assert f.status_code == 200, f.text
    t = client.get(f"/api/runs/live_tiny/fields?date={day}&depth=50&kind=target")
    assert t.status_code == 404  # no target for live days
    assert client.get("/api/runs/live_tiny/dates").status_code == 200


def test_live_cli_commands(live, monkeypatch):
    from typer.testing import CliRunner

    from oceanembed.cli import app

    cfg, server = live
    server.last = {SST_ID: date(2022, 2, 14), SLA_ID: date(2022, 2, 14)}
    runner = CliRunner()
    cfgfile = str(Path(__file__).resolve().parents[1] / "configs" / "live.yaml")
    res = runner.invoke(app, ["live", "status", "--config", cfgfile])
    assert res.exit_code == 0 and "not been updated" in res.output
    bad = runner.invoke(
        app, ["live", "status", "--config", str(Path(cfgfile).parent / "test_tiny.yaml")]
    )
    assert bad.exit_code == 2 and "no `live` section" in bad.output
