"""PO.DAAC provider: OPeNDAP constraint construction, resumable pieces, fallback, ledger.

The network is replaced by ``FakeDap``, a tiny DAP4 server that slices a global in-memory field
according to the constraint expression it receives -- so the tests check that the requested
index ranges really select the intended values, with no credentials and no downloads.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from oceanembed.config import Config, load_config
from oceanembed.data.providers import podaac as pm
from oceanembed.data.regrid import standardize

ROOT = Path(__file__).resolve().parents[1]
NC = "application/x-netcdf;ver=4"
XML = "application/vnd.opendap.dap4.dataset-metadata+xml"

OSCAR_DMR = """<?xml version="1.0" encoding="ISO-8859-1"?>
<Dataset xmlns="http://xml.opendap.org/ns/DAP/4.0#" name="g.nc">
  <Dimension name="longitude" size="{nx}"/><Dimension name="time" size="1"/>
  <Dimension name="latitude" size="{ny}"/>
  <Float64 name="lon"><Dim name="/longitude"/>
    <Attribute name="units" type="String"><Value>degrees_east</Value></Attribute></Float64>
  <Float64 name="time"><Dim name="/time"/>
    <Attribute name="units" type="String"><Value>days since 1990-1-1</Value></Attribute></Float64>
  <Float64 name="lat"><Dim name="/latitude"/>
    <Attribute name="units" type="String"><Value>degrees_north</Value></Attribute></Float64>
  <Float64 name="u"><Dim name="/time"/><Dim name="/longitude"/><Dim name="/latitude"/></Float64>
  <Float64 name="v"><Dim name="/time"/><Dim name="/longitude"/><Dim name="/latitude"/></Float64>
</Dataset>"""

CCMP_DMR = """<?xml version="1.0" encoding="ISO-8859-1"?>
<Dataset xmlns="http://xml.opendap.org/ns/DAP/4.0#" name="g.nc">
  <Dimension name="time" size="4"/><Dimension name="latitude" size="{ny}"/>
  <Dimension name="longitude" size="{nx}"/>
  <Float32 name="latitude"><Dim name="/latitude"/>
    <Attribute name="units" type="String"><Value>degrees_north</Value></Attribute></Float32>
  <Float32 name="longitude"><Dim name="/longitude"/>
    <Attribute name="units" type="String"><Value>degrees_east</Value></Attribute></Float32>
  <Float64 name="time"><Dim name="/time"/>
    <Attribute name="units" type="String"><Value>hours since 1970-1-1</Value>
    </Attribute></Float64>
  <Float32 name="uwnd"><Dim name="/time"/><Dim name="/latitude"/><Dim name="/longitude"/></Float32>
  <Float32 name="vwnd"><Dim name="/time"/><Dim name="/latitude"/><Dim name="/longitude"/></Float32>
</Dataset>"""


# ------------------------------------------------------------------ pure helpers
def test_index_runs_ascending_0_360_grid():
    lon = np.arange(0.0, 360.0, 0.25)  # OSCAR-like
    assert pm.index_runs(lon, 44.5, 105.5, periodic=True) == [(178, 422)]
    lat = np.arange(-89.75, 90.0, 0.25)
    assert pm.index_runs(lat, 4.5, 30.5) == [(377, 481)]


def test_index_runs_half_cell_centres():
    lon = np.arange(0.125, 360.0, 0.25)  # CCMP-like: centres, window edges between nodes
    ((i0, i1),) = pm.index_runs(lon, 44.5, 105.5, periodic=True)
    assert lon[i0] == pytest.approx(44.625) and lon[i1] == pytest.approx(105.375)


def test_index_runs_descending_coordinate():
    lat = np.arange(89.875, -90.0, -0.25)
    ((i0, i1),) = pm.index_runs(lat, 4.5, 30.5)
    assert i0 < i1 and lat[i0] == pytest.approx(30.375) and lat[i1] == pytest.approx(4.625)


def test_index_runs_lon_conventions_and_seam():
    lon180 = np.arange(-180.0, 180.0, 1.0)
    ((i0, i1),) = pm.index_runs(lon180, 44.5, 105.5, periodic=True)  # 0-360 window, -180..180 data
    assert lon180[i0] == 45.0 and lon180[i1] == 105.0
    lon = np.arange(0.0, 360.0, 1.0)
    # a window of -10..10 on a 0-360 grid straddles the seam -> two runs, low indices first
    assert pm.index_runs(lon, -10.0, 10.0, periodic=True) == [(0, 10), (350, 359)]
    # a full circle selects everything
    assert pm.index_runs(lon, 0.0, 360.0, periodic=True) == [(0, 359)]
    with pytest.raises(ValueError):
        pm.index_runs(lon, 400.0, 410.0)


def test_build_constraint_oscar_layout_time_lon_lat():
    layout = pm.parse_dmr(OSCAR_DMR.format(nx=1440, ny=719))
    assert layout.coord_var == {"longitude": "lon", "time": "time", "latitude": "lat"}
    assert layout.axis_dim == {"lon": "longitude", "lat": "latitude", "time": "time"}
    ce = pm.build_constraint(layout, ["u", "v"], {"longitude": (178, 422), "latitude": (377, 481)})
    assert ce == (
        "/time;/lon[178:422];/lat[377:481];/u[0:0][178:422][377:481];/v[0:0][178:422][377:481]"
    )


def test_build_constraint_ccmp_layout_time_lat_lon_and_missing_variable():
    layout = pm.parse_dmr(CCMP_DMR.format(nx=1440, ny=720))
    ce = pm.build_constraint(layout, ["uwnd"], {"longitude": (178, 421), "latitude": (378, 481)})
    assert ce == "/time;/latitude[378:481];/longitude[178:421];/uwnd[0:3][378:481][178:421]"
    with pytest.raises(pm.SubsettingRefused, match="not in the granule"):
        pm.build_constraint(layout, ["nope"], {})


def test_standardize_promotes_coordinate_variables_and_fixes_julian_time():
    import cftime

    ds = xr.Dataset(
        {"u": (("time", "longitude", "latitude"), np.zeros((1, 3, 2)))},
        coords={"time": [cftime.DatetimeJulian(2024, 1, 5)]},
    )
    ds["lon"] = ("longitude", [359.0, 1.0, 0.0], {"units": "degrees_east"})
    ds["lat"] = ("latitude", [10.0, 5.0], {"units": "degrees_north"})
    out = standardize(ds)
    assert list(out["lon"].values) == [0.0, 1.0, 359.0] and list(out["lat"].values) == [5.0, 10.0]
    assert out["time"].dtype == np.dtype("datetime64[ns]")
    assert str(out["time"].values[0])[:10] == "2024-01-05"


# ------------------------------------------------------------------ fake server
class FakeDap:
    """DAP4 server over global fields; one granule per day."""

    def __init__(self, kind: str, tmp: Path, first=date(2024, 1, 1), days=3):
        self.kind, self.tmp = kind, tmp
        self.days = [first + timedelta(d) for d in range(days)]
        if kind == "currents":
            self.lon = np.arange(0.0, 360.0, 1.0)
            self.lat = np.arange(-60.0, 61.0, 1.0)
            self.dmr = OSCAR_DMR.format(nx=len(self.lon), ny=len(self.lat))
            self.vars, self.nt = ("u", "v"), 1
        else:
            self.lon = np.arange(0.5, 360.0, 1.0)
            self.lat = np.arange(60.5, -61.0, -1.0)  # descending, as in the real CCMP
            self.dmr = CCMP_DMR.format(nx=len(self.lon), ny=len(self.lat))
            self.vars, self.nt = ("uwnd", "vwnd"), 4
        self.requests: list[str] = []  # data constraint expressions
        self.fail_data = 0  # next N data requests raise TransientError
        self.fail_days: set[str] = set()  # YYYYMMDD whose data requests always fail
        self.status = 200  # forced status for every request when != 200

    def field(self, day: date, k: int) -> np.ndarray:
        n = (day - date(2024, 1, 1)).days + 1
        shape = (
            (self.nt, len(self.lon), len(self.lat))
            if self.kind == "currents"
            else (self.nt, len(self.lat), len(self.lon))
        )
        rng = np.random.default_rng(n * 10 + k)
        return rng.normal(size=shape).astype("float32")

    def globals_(self, day: date) -> dict:
        t = pd.date_range(day, periods=self.nt, freq="6h" if self.nt > 1 else "D")
        lon_n, lat_n = ("lon", "lat") if self.kind == "currents" else ("longitude", "latitude")
        d = {
            lon_n: (("longitude",), self.lon),
            lat_n: (("latitude",), self.lat),
            "time": (("time",), t.values),
        }
        order = (
            ("time", "longitude", "latitude")
            if self.kind == "currents"
            else ("time", "latitude", "longitude")
        )
        for k, v in enumerate(self.vars):
            d[v] = (order, self.field(day, k))
        return d

    def get(self, url: str, params: dict | None = None) -> pm.Reply:
        if self.status != 200:
            return pm.Reply(self.status, "text/html", b"<html/>", 100)
        if url.endswith(".dmr"):
            return pm.Reply(200, XML, self.dmr.encode(), len(self.dmr))
        stamp = re.search(r"_(\d{8})", url).group(1)
        day = pd.Timestamp(stamp).date()
        ce = params["dap4.ce"]
        parts = ce.split(";")
        is_data = any(p.lstrip("/").split("[")[0] in self.vars for p in parts)
        if is_data:
            if stamp in self.fail_days or self.fail_data > 0:
                self.fail_data = max(0, self.fail_data - 1)
                raise pm.TransientError("ConnectionError")
            self.requests.append(ce)
        full = self.globals_(day)
        out = {}
        for part in parts:
            name = part.split("[")[0].lstrip("/")
            dims, arr = full[name]
            ranges = re.findall(r"\[(\d+):(\d+)\]", part)
            idx = tuple(slice(int(a), int(b) + 1) for a, b in ranges) or (slice(None),) * len(dims)
            out[name] = (dims, arr[idx])
        body_path = self.tmp / f"resp_{uuid.uuid4().hex}.nc"
        xr.Dataset(out).to_netcdf(body_path)
        body = body_path.read_bytes()
        return pm.Reply(200, NC, body, len(body) // 3)  # pretend 3x compression on the wire


def _granule(kind: str, day: date) -> dict:
    ur = (
        f"oscar_currents_final_{day:%Y%m%d}"
        if kind == "currents"
        else f"CCMP_Wind_Analysis_{day:%Y%m%d}_V03.1_L4"
    )
    coll = "C2098858642-POCLOUD" if kind == "currents" else "C2916514952-POCLOUD"
    return {
        "umm": {
            "GranuleUR": ur,
            "TemporalExtent": {"RangeDateTime": {"BeginningDateTime": f"{day}T00:00:00.000Z"}},
            "RelatedUrls": [
                {"Type": "GET DATA", "URL": f"https://archive.example.org/{ur}.nc"},
                {
                    "Type": "USE SERVICE API",
                    "URL": f"https://opendap.earthdata.nasa.gov/collections/{coll}/granules/{ur}",
                },
            ],
        },
        "meta": {"collection-concept-id": coll},
    }


class FakeEarthaccess:
    """search_data / download with global NetCDF granules (laid out like the real files)."""

    def __init__(self, dap: FakeDap):
        self.dap = dap
        self.downloads: list[Path] = []
        self.fail_download = False

    def login(self, strategy="all", **kw):
        return type("Auth", (), {"authenticated": True})()

    def search_data(self, **kw):
        return [_granule(self.dap.kind, d) for d in self.dap.days]

    def download(self, granules, local_path=None, **kw):
        if self.fail_download:
            raise ConnectionError("down")
        out = []
        for g in granules:
            begin = g["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
            day = date.fromisoformat(begin[:10])
            p = Path(local_path) / f"{g['umm']['GranuleUR']}.nc"
            xr.Dataset(self.dap.globals_(day)).to_netcdf(p)
            self.downloads.append(p)
            out.append(p)
        return out


@pytest.fixture
def cfg(tmp_path, monkeypatch) -> Config:
    monkeypatch.setenv("OCEANEMBED_DATA_ROOT", str(tmp_path / "data"))
    c = load_config(ROOT / "configs" / "poc.yaml")
    c.download.retry_backoff_s = 0.0
    c.download.workers = 2
    return c


def _setup(monkeypatch, tmp_path, kind="currents", days=3):
    import earthaccess

    srv = tmp_path / "srv"
    srv.mkdir(exist_ok=True)
    dap = FakeDap(kind, srv, days=days)
    ea = FakeEarthaccess(dap)
    for n in ("login", "search_data", "download"):
        monkeypatch.setattr(earthaccess, n, getattr(ea, n))
    monkeypatch.setattr(pm, "credentials_strategy", lambda: "netrc")
    monkeypatch.setattr(
        pm.PodaacProvider, "_get", lambda self, url, params=None: dap.get(url, params)
    )
    sleeps: list[float] = []
    monkeypatch.setattr(pm.time, "sleep", sleeps.append)
    return dap, ea, sleeps


def _ledger(cfg) -> list[dict]:
    return [json.loads(x) for x in (cfg.raw_root / "_download_log.jsonl").read_text().splitlines()]


# ------------------------------------------------------------------ OPeNDAP route
@pytest.mark.parametrize("kind,names", [("currents", ("u", "v")), ("winds", ("uwnd", "vwnd"))])
def test_opendap_month_matches_global_field_cropped(cfg, monkeypatch, tmp_path, kind, names):
    dap, ea, _ = _setup(monkeypatch, tmp_path, kind)
    (path,) = pm.PodaacProvider(cfg).fetch(kind, date(2024, 1, 1), date(2024, 1, 31))
    assert path.name == f"{kind}_202401.nc"
    with xr.open_dataset(path) as ds:
        ds = ds.load()
    assert set(ds.data_vars) == set(names)  # configured variables only
    assert ds.sizes["time"] == 3 * dap.nt
    assert ds.lon.min() >= 44.5 and ds.lon.max() <= 105.5  # domain + halo, 0-360
    assert ds.lat.min() >= 4.5 and ds.lat.max() <= 30.5
    assert (np.diff(ds.lat.values) > 0).all()  # ascending even for the descending CCMP latitude
    # the values are exactly the global field cut at the same coordinates (day 2, first variable)
    day2 = dap.days[1]
    g = dap.globals_(day2)
    lon_n, lat_n = ("lon", "lat") if kind == "currents" else ("longitude", "latitude")
    dims, arr = g[names[0]]
    ref = xr.DataArray(
        arr,
        dims=dims,
        coords={"longitude": g[lon_n][1], "latitude": g[lat_n][1]},
    ).transpose("time", "latitude", "longitude")
    ref = ref.sel(latitude=ds.lat.values, longitude=ds.lon.values).values
    mine = ds[names[0]].isel(time=slice(dap.nt, 2 * dap.nt)).values
    assert np.allclose(mine, ref)
    assert not (cfg.raw_root / "_parts").exists()  # parts removed after the merge
    assert not list(cfg.raw_root.rglob("*.tmp"))
    assert len(dap.requests) == 3 and all("[0:" in ce for ce in dap.requests)  # one per day
    assert ea.downloads == []  # no global granule needed


def test_opendap_ledger_lines_and_no_secrets(cfg, monkeypatch, tmp_path):
    monkeypatch.setenv("EARTHDATA_USERNAME", "secret-user")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "secret-pass")
    _setup(monkeypatch, tmp_path)
    pm.PodaacProvider(cfg).fetch("currents", date(2024, 1, 1), date(2024, 1, 31))
    raw = (cfg.raw_root / "_download_log.jsonl").read_text()
    recs = _ledger(cfg)
    days = [r for r in recs if r["period"] != "layout"]
    assert sorted(r["period"] for r in days) == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert sum(r["period"] == "layout" for r in recs) == 1  # coordinates fetched once, not per day
    for r in recs:
        assert set(r) == {
            "ts", "product", "dataset", "period", "route", "bytes_written",
            "bytes_transferred", "transfer_basis", "seconds",
        }  # fmt: skip
        assert r["product"] == "currents" and r["dataset"] == "OSCAR_L4_OC_FINAL_V2.0"
        assert r["route"] == "opendap" and r["transfer_basis"] == "wire"
    assert all(r["bytes_written"] > 0 and r["bytes_transferred"] > 0 for r in days)
    assert "secret" not in raw and "http" not in raw and "token" not in raw.lower()


def test_retries_with_exponential_backoff(cfg, monkeypatch, tmp_path):
    dap, _, sleeps = _setup(monkeypatch, tmp_path, days=1)
    cfg.download.retry_backoff_s = 2.0
    cfg.download.workers = 1
    dap.fail_data = 2
    pm.PodaacProvider(cfg).fetch("currents", date(2024, 1, 1), date(2024, 1, 1))
    assert sleeps == [2.0, 4.0]
    assert [r["route"] for r in _ledger(cfg) if r["period"] != "layout"] == ["opendap"]


# ------------------------------------------------------------------ resume / atomicity
def test_resume_skips_finished_parts_and_incomplete_month_is_not_merged(cfg, monkeypatch, tmp_path):
    dap, ea, _ = _setup(monkeypatch, tmp_path)
    cfg.download.workers = 1
    cfg.download.retries = 0
    dap.fail_days = {"20240103"}  # the 3rd day fails on OPeNDAP ...
    ea.fail_download = True  # ... and on the granule fallback -> the month stays incomplete
    with pytest.raises(RuntimeError, match="re-run the download to resume"):
        pm.PodaacProvider(cfg).fetch("currents", date(2024, 1, 1), date(2024, 1, 31))
    month = cfg.raw_root / "currents" / "currents_202401.nc"
    parts = cfg.raw_root / "_parts" / "currents_202401"
    assert not month.exists()  # an incomplete month is never merged
    assert sorted(p.name for p in parts.glob("*.nc")) == ["20240101.nc", "20240102.nc"]
    assert not list(parts.glob("*.tmp"))  # atomic writes leave no temporary files

    # second invocation with a healthy server: only the missing day is requested
    dap.fail_days = set()
    ea.fail_download = False
    dap.requests.clear()
    (path,) = pm.PodaacProvider(cfg).fetch("currents", date(2024, 1, 1), date(2024, 1, 31))
    assert path == month and month.exists() and not parts.exists()
    assert len(dap.requests) == 1  # days 1 and 2 were not downloaded again
    with xr.open_dataset(month) as ds:
        assert ds.sizes["time"] == 3
    # a third invocation does nothing at all
    assert pm.PodaacProvider(cfg).fetch("currents", date(2024, 1, 1), date(2024, 1, 31)) == []


def test_part_write_is_atomic(tmp_path, monkeypatch):
    target = tmp_path / "20240101.nc"
    ds = xr.Dataset({"u": ("x", np.arange(3.0))})

    def boom(self, path, **kw):
        Path(path).write_bytes(b"half a file")
        raise OSError("disk full")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", boom)
    with pytest.raises(OSError):
        pm._write_nc(ds, target)
    assert not target.exists()  # the final name only ever appears complete


# ------------------------------------------------------------------ fallback
def test_falls_back_to_granule_route_when_opendap_refuses(cfg, monkeypatch, tmp_path, caplog):
    dap, ea, _ = _setup(monkeypatch, tmp_path, days=6)
    dap.status = 403  # e.g. the Earthdata application is not authorised
    cfg.download.workers = 1
    with caplog.at_level("WARNING"):
        (path,) = pm.PodaacProvider(cfg).fetch("currents", date(2024, 1, 1), date(2024, 1, 31))
    with xr.open_dataset(path) as ds:
        assert ds.sizes["time"] == 6 and ds.lon.min() >= 44.5 and ds.lon.max() <= 105.5
        assert (np.diff(ds.lat.values) > 0).all() and (np.diff(ds.lon.values) > 0).all()
    assert len(ea.downloads) == 6 and not any(p.exists() for p in ea.downloads)  # deleted
    recs = _ledger(cfg)
    assert {r["route"] for r in recs} == {"granule_fallback"}
    assert all(r["transfer_basis"] == "file_size" and r["bytes_transferred"] > 1000 for r in recs)
    msgs = " ".join(m.getMessage() for m in caplog.records)
    assert "falling back to the global granule download" in msgs and "HTTP 403" in msgs


def test_repeated_transient_failures_switch_opendap_off(cfg, monkeypatch, tmp_path):
    dap, ea, _ = _setup(monkeypatch, tmp_path, days=6)
    cfg.download.workers = 1
    cfg.download.retries = 0
    dap.fail_data = 10**6  # every data request fails
    pm.PodaacProvider(cfg).fetch("currents", date(2024, 1, 1), date(2024, 1, 31))
    assert len(dap.requests) == 0 and len(ea.downloads) == 6
    # after three failures the remaining days did not even try OPeNDAP
    assert dap.fail_data == 10**6 - 3


def test_granule_mode_never_uses_opendap(cfg, monkeypatch, tmp_path):
    dap, ea, _ = _setup(monkeypatch, tmp_path, kind="winds", days=2)
    cfg.download.podaac_mode = "granule"
    cfg.download.delete_global_granules = False
    seen: list[str] = []
    monkeypatch.setattr(pm.PodaacProvider, "_get", lambda self, url, params=None: seen.append(url))
    (path,) = pm.PodaacProvider(cfg).fetch("winds", date(2024, 1, 1), date(2024, 1, 2))
    assert seen == [] and len(ea.downloads) == 2 and all(p.exists() for p in ea.downloads)
    with xr.open_dataset(path) as ds:
        assert ds.sizes["time"] == 8 and set(ds.data_vars) == {"uwnd", "vwnd"}
    assert {r["route"] for r in _ledger(cfg)} == {"granule"}


def test_search_kwargs_and_missing_granules(cfg, monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, days=0)
    prov = pm.PodaacProvider(cfg)
    assert prov.search_kwargs("currents", date(2024, 1, 1), date(2024, 1, 31)) == {
        "short_name": "OSCAR_L4_OC_FINAL_V2.0",
        "temporal": ("2024-01-01", "2024-01-31"),
        "bounding_box": (44.5, 4.5, 105.5, 30.5),
    }
    with pytest.raises(RuntimeError, match="no currents granules found"):
        prov.fetch("currents", date(2024, 1, 1), date(2024, 1, 31))


def test_opendap_url_falls_back_to_concept_ids():
    g = _granule("currents", date(2024, 1, 5))
    g["umm"]["RelatedUrls"] = []
    assert pm.opendap_url(g) == (
        "https://opendap.earthdata.nasa.gov/collections/C2098858642-POCLOUD/granules/"
        "oscar_currents_final_20240105"
    )
    assert pm.granule_day(g) == date(2024, 1, 5)
