"""Real providers are tested with mocked network calls (no credentials / downloads needed)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from typer.testing import CliRunner

from oceanembed.cli import app
from oceanembed.config import Config, DatasetSource, load_config
from oceanembed.data.providers import argo as argo_mod
from oceanembed.data.providers import cmems as cmems_mod
from oceanembed.data.providers import podaac as podaac_mod
from oceanembed.data.providers.base import (
    ARGO_COLUMNS,
    MissingCredentialsError,
    make_provider,
    months,
    source_for_range,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def poc_cfg(tmp_path, monkeypatch) -> Config:
    monkeypatch.setenv("OCEANEMBED_DATA_ROOT", str(tmp_path / "data"))
    return load_config(ROOT / "configs" / "poc.yaml")


@pytest.fixture
def two_sss_cfg(poc_cfg) -> Config:
    """poc config whose SSS product switches from the MY to the NRT dataset on 2024-01-01."""
    poc_cfg.products["sss"].datasets = [
        DatasetSource(id="cmems_obs-mob_glo_phy-sss_my_multi_P1D", end=date(2023, 12, 31)),
        DatasetSource(id="cmems_obs-mob_glo_phy-sss_nrt_multi_P1D", start=date(2024, 1, 1)),
    ]
    return poc_cfg


# --------------------------------------------------------------------------- helpers
def test_months_clips_to_requested_range():
    ms = list(months(date(2023, 11, 20), date(2024, 1, 10)))
    assert ms == [
        (date(2023, 11, 20), date(2023, 11, 30)),
        (date(2023, 12, 1), date(2023, 12, 31)),
        (date(2024, 1, 1), date(2024, 1, 10)),
    ]


def test_source_for_range_switches_datasets_at_month_boundary(two_sss_cfg):
    sss = two_sss_cfg.products["sss"]
    assert source_for_range(sss, date(2023, 12, 1), date(2023, 12, 31)).id.endswith("my_multi_P1D")
    assert source_for_range(sss, date(2024, 1, 1), date(2024, 1, 31)).id.endswith("nrt_multi_P1D")
    with pytest.raises(ValueError, match="month boundaries"):
        source_for_range(sss, date(2023, 12, 15), date(2024, 1, 15))


def test_make_provider_dispatch(poc_cfg):
    assert isinstance(make_provider(poc_cfg, "sst"), cmems_mod.CmemsProvider)
    assert isinstance(make_provider(poc_cfg, "winds"), podaac_mod.PodaacProvider)
    assert isinstance(make_provider(poc_cfg, "argo"), argo_mod.ArgoProvider)


# --------------------------------------------------------------------------- CMEMS
class FakeSubset:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, **kw):
        self.calls.append(kw)
        out = Path(kw["output_directory"]) / kw["output_filename"]
        xr.Dataset({"x": ("t", [1.0])}).to_netcdf(out)
        return None


def test_cmems_request_parameters(poc_cfg, monkeypatch):
    import copernicusmarine

    fake = FakeSubset()
    monkeypatch.setattr(copernicusmarine, "subset", fake)
    monkeypatch.setattr(cmems_mod, "credentials_available", lambda: True)
    prov = cmems_mod.CmemsProvider(poc_cfg)
    paths = prov.fetch("temp", date(2025, 2, 1), date(2025, 3, 31))
    assert [p.name for p in paths] == ["temp_202502.nc", "temp_202503.nc"]
    assert all(p.exists() for p in paths)
    assert not list(paths[0].parent.glob("*.part.nc"))
    kw = fake.calls[0]
    assert kw["dataset_id"] == "cmems_mod_glo_phy_my_0.083deg_P1D-m"
    assert kw["variables"] == ["thetao"]
    # domain 45-105E, 5-30N plus the 0.5 degree halo
    assert (kw["minimum_longitude"], kw["maximum_longitude"]) == (44.5, 105.5)
    assert (kw["minimum_latitude"], kw["maximum_latitude"]) == (4.5, 30.5)
    assert (kw["minimum_depth"], kw["maximum_depth"]) == (0.0, 1100.0)
    assert kw["start_datetime"] == "2025-02-01T00:00:00"
    assert kw["end_datetime"] == "2025-02-28T23:59:59"
    assert kw["output_filename"].startswith("temp_202502")
    assert fake.calls[1]["end_datetime"] == "2025-03-31T23:59:59"


def test_cmems_skips_existing_files(poc_cfg, monkeypatch):
    import copernicusmarine

    fake = FakeSubset()
    monkeypatch.setattr(copernicusmarine, "subset", fake)
    monkeypatch.setattr(cmems_mod, "credentials_available", lambda: True)
    prov = cmems_mod.CmemsProvider(poc_cfg)
    assert len(prov.fetch("sla", date(2024, 5, 1), date(2024, 6, 30))) == 2
    assert prov.fetch("sla", date(2024, 5, 1), date(2024, 6, 30)) == []
    assert len(fake.calls) == 2
    assert "minimum_depth" not in fake.calls[0]


def test_cmems_sss_uses_my_then_nrt_dataset(two_sss_cfg, monkeypatch):
    poc_cfg = two_sss_cfg
    import copernicusmarine

    fake = FakeSubset()
    monkeypatch.setattr(copernicusmarine, "subset", fake)
    monkeypatch.setattr(cmems_mod, "credentials_available", lambda: True)
    cmems_mod.CmemsProvider(poc_cfg).fetch("sss", date(2023, 12, 1), date(2024, 1, 31))
    assert fake.calls[0]["dataset_id"].endswith("sss_my_multi_P1D")
    assert fake.calls[1]["dataset_id"].endswith("sss_nrt_multi_P1D")
    assert fake.calls[0]["variables"] == ["sos"]


def test_cmems_missing_credentials_fails_before_any_call(poc_cfg, monkeypatch):
    import copernicusmarine

    fake = FakeSubset()
    monkeypatch.setattr(copernicusmarine, "subset", fake)
    monkeypatch.setattr(cmems_mod, "credentials_available", lambda: False)
    with pytest.raises(MissingCredentialsError, match="copernicusmarine login"):
        cmems_mod.CmemsProvider(poc_cfg).fetch("sst", date(2024, 1, 1), date(2024, 1, 31))
    assert fake.calls == []


def test_cmems_writes_ledger_line_without_secrets(poc_cfg, monkeypatch):
    import copernicusmarine

    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_USERNAME", "secret-user")
    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_PASSWORD", "secret-pass")
    monkeypatch.setattr(copernicusmarine, "subset", FakeSubset())
    cmems_mod.CmemsProvider(poc_cfg).fetch("sla", date(2024, 5, 1), date(2024, 5, 31))
    text = (poc_cfg.raw_root / "_download_log.jsonl").read_text()
    rec = json.loads(text.strip())
    assert rec["product"] == "sla" and rec["route"] == "cmems_subset" and rec["period"] == "2024-05"
    assert rec["bytes_written"] == rec["bytes_transferred"] > 0
    assert rec["dataset"].startswith("cmems_obs-sl_glo_phy")
    assert "secret" not in text and "http" not in text


def test_cmems_credentials_detection(monkeypatch, tmp_path):
    monkeypatch.setattr(cmems_mod, "CREDENTIALS_FILE", tmp_path / "nope")
    monkeypatch.delenv("COPERNICUSMARINE_SERVICE_USERNAME", raising=False)
    monkeypatch.delenv("COPERNICUSMARINE_SERVICE_PASSWORD", raising=False)
    assert cmems_mod.credentials_available() is False
    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_USERNAME", "u")
    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_PASSWORD", "p")
    assert cmems_mod.credentials_available() is True


# --------------------------------------------------------------------------- PO.DAAC
def test_podaac_missing_credentials(poc_cfg, monkeypatch):
    import earthaccess

    calls = []
    monkeypatch.setattr(earthaccess, "login", lambda **kw: calls.append("login"))
    monkeypatch.setattr(earthaccess, "search_data", lambda **kw: calls.append("search"))
    monkeypatch.setattr(podaac_mod, "credentials_strategy", lambda: None)
    with pytest.raises(MissingCredentialsError, match="Earthdata"):
        podaac_mod.PodaacProvider(poc_cfg).fetch("winds", date(2024, 1, 1), date(2024, 1, 31))
    assert calls == []


def test_podaac_credentials_detection(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
    assert podaac_mod.credentials_strategy() is None
    (tmp_path / "_netrc").write_text("machine urs.earthdata.nasa.gov login a password b\n")
    assert podaac_mod.credentials_strategy() == "netrc"
    monkeypatch.setenv("EARTHDATA_USERNAME", "a")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "b")
    assert podaac_mod.credentials_strategy() == "environment"


# --------------------------------------------------------------------------- Argo
def _argo_raw(n_prof=2, n_lev=4, t0="2023-03-10"):
    rows = []
    for p in range(n_prof):
        for k in range(n_lev):
            rows.append(
                {
                    "PLATFORM_NUMBER": 2900000 + p,
                    "CYCLE_NUMBER": 5 + p,
                    "DIRECTION": "A",
                    "PRES": 5.0 + 100.0 * k,
                    "PRES_QC": 1,
                    "TEMP": 28.0 - 3.0 * k,
                    "TEMP_QC": 1,
                    "LATITUDE": 12.0 + p,
                    "LONGITUDE": 85.0,
                    "TIME": pd.Timestamp(t0) + pd.Timedelta(days=p * 40),
                }
            )
    return pd.DataFrame(rows)


def test_tidy_profiles_qc_and_schema():
    raw = _argo_raw()
    raw.loc[1, "TEMP_QC"] = 4  # bad
    raw.loc[2, "TEMP_QC"] = 2  # probably good: kept
    raw.loc[3, "TEMP"] = np.nan
    out = argo_mod.tidy_profiles(raw, [1, 2], 1100.0)
    assert list(out.columns) == ARGO_COLUMNS
    assert len(out) == 8 - 2
    assert set(out.temp_qc) == {1, 2}
    assert out.profile_id.iloc[0] == "2900000_005"
    # depth ~ pressure (slightly smaller at depth)
    assert (out.depth < out.pres).all()
    deep = argo_mod.tidy_profiles(raw, [1, 2], 150.0)
    assert deep.depth.max() <= 150.0


def test_fetch_argo_box_tiles_and_filters_dates(monkeypatch):
    calls = []

    def fake(source, lon0, lon1, lat0, lat1, pmax, date0, date1):
        calls.append((source, lon0, lon1, lat0, lat1, pmax, date0, date1))
        return _argo_raw(1, 3) if lon0 == 80 and lat0 == 10 else pd.DataFrame()

    monkeypatch.setattr(argo_mod, "_fetch_region", fake)
    df = argo_mod.fetch_argo_box(
        "erddap", (80, 90), (10, 20), date(2023, 3, 1), date(2023, 3, 31), box_deg=5
    )
    assert len(calls) == 4  # 2 x 2 tiles
    assert calls[0] == ("erddap", 80, 85, 10, 15, 1150.0, "2023-03-01", "2023-04-01")
    assert len(df) == 3 and df.profile_id.nunique() == 1
    assert df.time.between("2023-03-01", "2023-04-01").all()


def test_argo_provider_writes_monthly_parquet(tiny_cfg, monkeypatch):
    monkeypatch.setattr(
        argo_mod,
        "_fetch_region",
        lambda src, lo0, lo1, la0, la1, pm, d0, d1: _argo_raw(2, 5, t0=d0),
    )
    cfg = tiny_cfg.model_copy(update={"provider": "real"})
    paths = argo_mod.ArgoProvider(cfg).fetch("argo", date(2022, 1, 1), date(2022, 2, 28))
    assert [p.name for p in paths] == ["argo_202201.parquet", "argo_202202.parquet"]
    df = pd.read_parquet(paths[0])
    assert list(df.columns) == ARGO_COLUMNS
    assert argo_mod.ArgoProvider(cfg).fetch("argo", date(2022, 1, 1), date(2022, 2, 28)) == []
    assert len(df) == 5  # one profile falls inside the month, the other is filtered out
    assert len(argo_mod.load_argo(cfg)) == 10
    ledger = (cfg.raw_root / "_download_log.jsonl").read_text().splitlines()
    lines = [json.loads(x) for x in ledger]
    assert [x["period"] for x in lines] == ["2022-01", "2022-02"]
    assert all(x["product"] == "argo" and x["bytes_written"] > 0 for x in lines)
    assert all(x["bytes_transferred"] is None for x in lines)  # not measurable via argopy


def test_fetch_region_treats_empty_box_as_no_profiles(monkeypatch):
    """ERDDAP answers an empty box with HTTP 404, which argopy raises as FileNotFoundError."""
    import argopy

    class Empty:
        def __init__(self, **kw):
            pass

        def region(self, box):
            raise FileNotFoundError("Your query produced no matching results")

    monkeypatch.setattr(argopy, "DataFetcher", Empty)
    df = argo_mod._fetch_region(
        "erddap", 75.0, 90.0, 20.0, 30.0, 1150.0, "2024-01-01", "2024-02-01"
    )
    assert len(df) == 0
    assert len(argo_mod.tidy_profiles(df, [1, 2], 1100.0)) == 0


def test_netcdf3_fallback_reopens_in_memory_data_with_scipy(monkeypatch):
    """netCDF4 rejects some small valid ERDDAP responses; the fallback re-opens them with scipy."""
    raw = bytes(xr.Dataset({"temp": ("row", [8.784, 8.101])}).to_netcdf(engine="scipy"))
    real_open = xr.open_dataset
    engines = []

    def picky(target, *args, **kwargs):
        engines.append(kwargs.get("engine"))
        if kwargs.get("engine") is None:
            raise PermissionError("[Errno 1] Operation not permitted: '<xarray-in-memory-read>'")
        return real_open(target, *args, **kwargs)

    monkeypatch.setattr(xr, "open_dataset", picky)
    with argo_mod._netcdf3_fallback():
        ds = xr.open_dataset(raw)
        assert ds["temp"].values.tolist() == [8.784, 8.101]
        with pytest.raises(PermissionError):  # files on disk are not retried
            xr.open_dataset("some_file.nc")
    assert engines == [None, "scipy", None]
    assert xr.open_dataset is picky  # restored on exit


def test_fetch_region_retries_transient_errors(monkeypatch):
    """A truncated ERDDAP response raises e.g. PermissionError from netCDF4: retry, then raise."""
    import argopy

    calls = {"n": 0}

    class Flaky:
        fail_times = 2

        def __init__(self, **kw):
            pass

        def region(self, box):
            calls["n"] += 1
            if calls["n"] <= self.fail_times:
                raise PermissionError(
                    "[Errno 1] Operation not permitted: '<xarray-in-memory-read>'"
                )
            return self

        def to_dataframe(self):
            return _argo_raw(1, 3)

    monkeypatch.setattr(argopy, "DataFetcher", Flaky)
    monkeypatch.setattr(argo_mod, "FETCH_BACKOFF_S", 0.0)
    args = ("erddap", 75.0, 90.0, 5.0, 20.0, 1150.0, "2018-11-01", "2018-12-01")
    assert len(argo_mod._fetch_region(*args)) == 3
    assert calls["n"] == 3

    calls["n"] = 0
    Flaky.fail_times = 99
    with pytest.raises(PermissionError):
        argo_mod._fetch_region(*args)
    assert calls["n"] == argo_mod.FETCH_RETRIES + 1


def test_incois_gridded_absent_is_skipped(tiny_cfg):
    assert argo_mod.load_incois_gridded(tiny_cfg) is None


def test_incois_gridded_regridded_when_present(tiny_cfg):
    folder = tiny_cfg.raw_root / "argo_gridded"
    folder.mkdir(parents=True)
    lat = np.arange(7.5, 12.6, 1.0)
    lon = np.arange(71.5, 78.6, 1.0)
    depth = np.array([0.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 1500.0])
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    arr = (20.0 + 0.1 * la)[None, None] - 0.01 * depth[None, :, None, None] + 0 * lo
    ds = xr.Dataset(
        {"TEMP": (("time", "depth", "latitude", "longitude"), arr.astype("float32"))},
        coords={
            "time": pd.to_datetime(["2022-01-15"]),
            "depth": depth,
            "latitude": lat,
            "longitude": lon,
        },
    )
    ds.to_netcdf(folder / "incois_argo_gridded_2022.nc")
    out = argo_mod.load_incois_gridded(tiny_cfg)
    assert out is not None and out.dims == ("time", "depth", "lat", "lon")
    assert out.shape == (1, 15, 16, 24)
    iy = np.argmin(abs(out.lat.values - 10.125))
    expected = 20.0 + 0.1 * 10.125 - 0.01 * 100.0
    assert out.sel(depth=100.0).values[0, iy, 5] == pytest.approx(expected, abs=0.01)


# --------------------------------------------------------------------------- CLI
def test_cli_help_lists_phase1_commands():
    res = CliRunner().invoke(app, ["--help"])
    assert res.exit_code == 0
    for cmd in ("synth", "download", "harmonize", "stats"):
        assert cmd in res.output


def test_cli_download_reports_missing_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("OCEANEMBED_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(cmems_mod, "credentials_available", lambda: False)
    res = CliRunner().invoke(
        app, ["download", "-c", str(ROOT / "configs" / "poc.yaml"), "-v", "sst"]
    )
    assert res.exit_code == 2
    assert "copernicusmarine login" in res.output


def test_cli_synth_rejects_real_provider_config():
    res = CliRunner().invoke(app, ["synth", "-c", str(ROOT / "configs" / "poc.yaml")])
    assert res.exit_code != 0


def test_cli_download_rejects_unknown_product():
    res = CliRunner().invoke(
        app, ["download", "-c", str(ROOT / "configs" / "poc.yaml"), "-v", "nonsense"]
    )
    assert res.exit_code != 0
