"""Argo profiles (via ``argopy``, no credentials) and an optional INCOIS gridded-ARGO loader.

Raw output: ``<raw>/argo/argo_<YYYYMM>.parquet`` -- one row per (profile, pressure level) with the
schema in ``ARGO_COLUMNS`` (identical to the synthetic provider's Argo-like profiles):

``profile_id, platform, cycle, time, latitude, longitude, pres [dbar], depth [m], temp [degC],
temp_qc`` -- only good-QC temperature (flags 1 and 2 by default) is kept.

Windows note: aiohttp (used by argopy) can fail with ``unable to get local issuer certificate``
with the Windows certificate store; we point ``SSL_CERT_FILE`` at certifi's CA bundle when the
user has not set it.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from oceanembed.config import Config
from oceanembed.data.providers.base import ARGO_COLUMNS, log_transfer, months, raw_dir, raw_file
from oceanembed.data.providers.synthetic_world import pressure_to_depth
from oceanembed.data.regrid import interp_to_depths, regrid_horizontal, standardize

log = logging.getLogger(__name__)

# Transient ERDDAP failures (timeouts, proxy errors, truncated responses) are retried per box.
FETCH_RETRIES = 4
FETCH_BACKOFF_S = 5.0


def _ensure_ca_bundle() -> None:
    if "SSL_CERT_FILE" not in os.environ:
        try:
            import certifi

            os.environ["SSL_CERT_FILE"] = certifi.where()
        except ImportError:  # pragma: no cover
            pass


@contextlib.contextmanager
def _netcdf3_fallback():
    """While active, in-memory datasets that netCDF4 refuses are re-opened with scipy.

    ERDDAP returns classic NetCDF-3; netCDF4's in-memory reader rejects some small but valid
    responses (seen: a 2-row file -> ``PermissionError: '<xarray-in-memory-read>'``), which
    argopy cannot recover from. scipy reads the same bytes fine.
    """
    original = xr.open_dataset

    def open_dataset(target, *args, **kwargs):
        in_memory = isinstance(target, bytes | bytearray | io.BytesIO)
        if not in_memory or "engine" in kwargs:
            return original(target, *args, **kwargs)
        try:
            return original(target, *args, **kwargs)
        except Exception:  # noqa: BLE001 - whatever netCDF4 raised, try the pure reader
            if isinstance(target, io.BytesIO):
                target.seek(0)
            return original(target, *args, engine="scipy", **kwargs)

    xr.open_dataset = open_dataset
    try:
        yield
    finally:
        xr.open_dataset = original


def _fetch_region(
    source: str, lon0: float, lon1: float, lat0: float, lat1: float, pmax: float,
    date0: str, date1: str,
) -> pd.DataFrame:  # fmt: skip
    """Raw argopy dataframe for one box / period (separated out so tests can mock it)."""
    _ensure_ca_bundle()
    from argopy import DataFetcher
    from argopy.errors import DataNotFound, ErddapHTTPNotFound, NoData

    box = [lon0, lon1, lat0, lat1, 0.0, pmax, date0, date1]
    for attempt in range(FETCH_RETRIES + 1):
        fetcher = DataFetcher(src=source, ds="phy", mode="standard", progress=False, cache=False)
        try:
            with _netcdf3_fallback():
                return fetcher.region(box).to_dataframe()
        except (FileNotFoundError, DataNotFound, ErddapHTTPNotFound, NoData):
            # ERDDAP answers an empty box / period with HTTP 404 ("no matching results"); argopy
            # surfaces that as FileNotFoundError. Not an error: there simply were no profiles.
            log.debug("no Argo profiles in %s", box)
            return pd.DataFrame()
        except Exception as e:  # noqa: BLE001 - ERDDAP hiccups surface as many exception types
            # e.g. a truncated / non-NetCDF response raises PermissionError from netCDF4.
            if attempt == FETCH_RETRIES:
                raise
            wait = FETCH_BACKOFF_S * 2**attempt
            log.warning(
                "Argo fetch failed for %s (%s: %s); retry %d/%d in %.0f s",
                box, type(e).__name__, e, attempt + 1, FETCH_RETRIES, wait,
            )  # fmt: skip
            time.sleep(wait)
    return pd.DataFrame()  # pragma: no cover - loop always returns or raises


def tidy_profiles(df: pd.DataFrame, qc_flags: list[int], max_depth_m: float) -> pd.DataFrame:
    """argopy dataframe -> tidy table with good-QC temperature only."""
    if df is None or len(df) == 0:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in ARGO_COLUMNS})
    d = df.reset_index(drop=True).copy()
    keep = d["TEMP"].notna() & d["PRES"].notna()
    if "TEMP_QC" in d:
        keep &= pd.to_numeric(d["TEMP_QC"], errors="coerce").isin(qc_flags)
    if "PRES_QC" in d:
        keep &= pd.to_numeric(d["PRES_QC"], errors="coerce").isin(qc_flags)
    d = d[keep]
    plat = d["PLATFORM_NUMBER"].astype(str)
    cyc = d["CYCLE_NUMBER"].astype(int)
    direction = d["DIRECTION"].astype(str) if "DIRECTION" in d else pd.Series("A", index=d.index)
    out = pd.DataFrame(
        {
            "profile_id": plat
            + "_"
            + cyc.astype(str).str.zfill(3)
            + np.where(direction == "D", "D", ""),
            "platform": plat,
            "cycle": cyc.astype("int32"),
            "time": pd.to_datetime(d["TIME"]).astype("datetime64[ns]"),
            "latitude": d["LATITUDE"].astype("float64"),
            "longitude": d["LONGITUDE"].astype("float64"),
            "pres": d["PRES"].astype("float32"),
            "depth": pressure_to_depth(d["PRES"].to_numpy(), d["LATITUDE"].to_numpy()).astype(
                "float32"
            ),
            "temp": d["TEMP"].astype("float32"),
            "temp_qc": pd.to_numeric(d["TEMP_QC"], errors="coerce").fillna(1).astype("int8")
            if "TEMP_QC" in d
            else np.int8(1),
        }
    )
    out = out[out["depth"] <= max_depth_m]
    return out.drop_duplicates(["profile_id", "pres"]).reset_index(drop=True)[ARGO_COLUMNS]


def fetch_argo_box(
    source: str,
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
    first: date,
    last: date,
    qc_flags: list[int] | None = None,
    max_depth_m: float = 1100.0,
    box_deg: float = 15.0,
) -> pd.DataFrame:
    """Tidy good-QC Argo temperature profiles for a box and period (tiled into ``box_deg``)."""
    qc_flags = qc_flags or [1, 2]
    date0 = pd.Timestamp(first).strftime("%Y-%m-%d")
    date1 = (pd.Timestamp(last) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    lons = np.arange(lon_range[0], lon_range[1], box_deg).tolist() + [lon_range[1]]
    lats = np.arange(lat_range[0], lat_range[1], box_deg).tolist() + [lat_range[1]]
    frames = []
    for i in range(len(lons) - 1):
        for j in range(len(lats) - 1):
            lo1, la1 = min(lons[i + 1], lon_range[1]), min(lats[j + 1], lat_range[1])
            if lo1 <= lons[i] or la1 <= lats[j]:
                continue
            raw = _fetch_region(
                source, lons[i], lo1, lats[j], la1, max_depth_m + 50.0, date0, date1
            )
            frames.append(tidy_profiles(raw, qc_flags, max_depth_m))
    frames = [f for f in frames if len(f)]
    if not frames:
        return tidy_profiles(pd.DataFrame(), qc_flags, max_depth_m)
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["profile_id", "pres"])
    t = df["time"]
    return df[(t >= pd.Timestamp(date0)) & (t < pd.Timestamp(date1))].reset_index(drop=True)


class ArgoProvider:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def fetch(self, variable: str, start: date, end: date) -> list[Path]:
        if variable != "argo":
            raise ValueError("ArgoProvider only serves the 'argo' product")
        g, a = self.cfg.grid, self.cfg.argo
        written: list[Path] = []
        for first, last in tqdm(list(months(start, end)), desc="argo", leave=False):
            path = raw_file(self.cfg, "argo", first)
            if path.exists() and not self.cfg.download.overwrite:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            df = fetch_argo_box(
                a.source, (g.lon_min, g.lon_max), (g.lat_min, g.lat_max), first, last,
                a.qc_flags, a.max_depth_m, a.box_deg,
            )  # fmt: skip
            df.to_parquet(path, index=False)
            log_transfer(
                self.cfg, product="argo", dataset=f"argopy:{a.source}", period=f"{first:%Y-%m}",
                route="argopy", bytes_written=path.stat().st_size, bytes_transferred=None,
                transfer_basis="unknown", seconds=time.time() - t0,
            )  # fmt: skip
            written.append(path)
        return written


def load_argo(cfg: Config) -> pd.DataFrame:
    """All raw Argo monthly tables concatenated (empty frame if none present)."""
    files = sorted(raw_dir(cfg, "argo").glob("argo_*.parquet"))
    if not files:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in ARGO_COLUMNS})
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


# ----------------------------------------------------------------------------------------
# optional INCOIS gridded ARGO
# ----------------------------------------------------------------------------------------
_TEMP_NAMES = ("temp", "temperature", "TEMP", "thetao", "pottmp", "ptemp")


def load_incois_gridded(cfg: Config) -> xr.DataArray | None:
    """Regrid a user-supplied INCOIS gridded-ARGO NetCDF (``<raw>/argo_gridded/*.nc``) onto the
    canonical grid and standard depths. Returns ``(time, depth, lat, lon)`` or ``None`` when no
    file has been dropped there (the validation step then simply skips it)."""
    from oceanembed.grid import build_grid

    folder = raw_dir(cfg, "argo_gridded")
    files = sorted(folder.glob("*.nc")) if folder.exists() else []
    if not files:
        log.info("no INCOIS gridded-ARGO file in %s; skipping", folder)
        return None
    grid = build_grid(cfg)
    parts = []
    for f in files:
        with xr.open_dataset(f) as ds:
            name = next((n for n in _TEMP_NAMES if n in ds.data_vars), None)
            if name is None:
                raise KeyError(f"{f.name}: no temperature variable among {_TEMP_NAMES}")
            da = standardize(ds[name], {"depth": "depth"} if "depth" in ds[name].dims else None)
            if "depth" not in da.dims:
                raise ValueError(f"{f.name}: gridded ARGO must have a depth/lev dimension")
            da = interp_to_depths(da.load(), grid.depth) if "depth" in da.dims else da
            parts.append(regrid_horizontal(da, grid.lat, grid.lon, grid.resolution, "auto", 0.5))
    out = xr.concat(parts, dim="time").sortby("time") if "time" in parts[0].dims else parts[0]
    out.name = "temp"
    return out
