"""ARMOR3D (observation-based 3-D temperature) for research stage R4: download and regridding.

Product ``MULTIOBS_GLO_PHY_TSUV_3D_MYNRT_015_012`` (Copernicus Marine), temperature ``to``,
0.125 deg, 50 levels, daily. Catalogue facts (``copernicusmarine describe``, checked 2026-10-03):

* ``cmems_obs-mob_glo_phy_my_0.125deg_P1D-m`` -- reprocessed (MY), version 202511,
  1993-01-01 .. 2024-12-31: covers the whole 2024 test year, so one dataset serves every day and
  no near-real-time switch happens inside the test period;
* ``cmems_obs-mob_glo_phy_nrt_0.125deg_P1D-m`` -- near real time, 2024-09-08 onwards (not used).

The download follows the other CMEMS products: a server-side subset per calendar month
(domain + halo, 0-1100 m), written to ``<raw>/armor3d/armor3d_<YYYYMM>.nc``, resumable (existing
months are skipped) and recorded in the transfer ledger. Regridding reuses
:func:`oceanembed.data.harmonize.process_variable` (the GLORYS rules): vertical linear
interpolation to the 15 standard depths, then a 2 x 2 block mean onto the 0.25 deg grid. The time
axis is the product's own daily axis; nothing is interpolated in time.
"""

from __future__ import annotations

import logging
import time as _time
from datetime import date, datetime, time
from pathlib import Path

import numpy as np
import xarray as xr

from oceanembed.config import Config
from oceanembed.data.harmonize import process_variable
from oceanembed.data.providers.base import (
    MissingCredentialsError,
    log_transfer,
    months,
    raw_dir,
    region_with_halo,
)
from oceanembed.data.providers.cmems import credentials_available
from oceanembed.data.regrid import standardize
from oceanembed.grid import Grid, daily_axis

log = logging.getLogger(__name__)

PRODUCT = "armor3d"
PRODUCT_ID = "MULTIOBS_GLO_PHY_TSUV_3D_MYNRT_015_012"
DATASET_ID = "cmems_obs-mob_glo_phy_my_0.125deg_P1D-m"  # reprocessed, daily, to 2024-12-31
DATASET_END = date(2024, 12, 31)
VARIABLE = "to"
MAX_DEPTH_M = 1100.0


def armor_dir(cfg: Config) -> Path:
    return raw_dir(cfg, PRODUCT)


def armor_file(cfg: Config, month_start: date) -> Path:
    return armor_dir(cfg) / f"{PRODUCT}_{month_start:%Y%m}.nc"


def subset_kwargs(cfg: Config, first: date, last: date) -> dict:
    """Arguments of ``copernicusmarine.subset`` for one month (domain plus the usual halo)."""
    lon0, lon1, lat0, lat1 = region_with_halo(cfg)
    return {
        "dataset_id": DATASET_ID,
        "variables": [VARIABLE],
        "minimum_longitude": lon0,
        "maximum_longitude": lon1,
        "minimum_latitude": lat0,
        "maximum_latitude": lat1,
        "minimum_depth": 0.0,
        "maximum_depth": MAX_DEPTH_M,
        "start_datetime": datetime.combine(first, time(0, 0, 0)).isoformat(),
        "end_datetime": datetime.combine(last, time(23, 59, 59)).isoformat(),
        "coordinates_selection_method": "inside",
        "netcdf_compression_level": 1,
        "disable_progress_bar": True,
    }


def download_armor(cfg: Config, start: date, end: date) -> list[Path]:
    """Fetch the months overlapping ``[start, end]``; existing files are kept. Returns the paths
    written by this call."""
    if end > DATASET_END:
        raise ValueError(f"the reprocessed ARMOR3D dataset ends {DATASET_END}; asked for {end}")
    todo = [
        (first, last)
        for first, last in months(start, end)
        if cfg.download.overwrite or not armor_file(cfg, first).exists()
    ]
    if todo and not credentials_available():
        raise MissingCredentialsError(
            "No Copernicus Marine credentials found (see docs/usage.md); R4 needs them once."
        )
    written: list[Path] = []
    if todo:
        import copernicusmarine  # late import: tests patch ``subset``
    for first, last in todo:
        path = armor_file(cfg, first)
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".part.nc")
        kw = subset_kwargs(cfg, first, last)
        log.info("cmems %s %s -> %s", PRODUCT, kw["dataset_id"], path.name)
        t0 = _time.time()
        copernicusmarine.subset(
            output_directory=str(path.parent), output_filename=part.name, overwrite=True, **kw
        )
        part.replace(path)
        size = path.stat().st_size
        log_transfer(
            cfg, product=PRODUCT, dataset=kw["dataset_id"], period=f"{first:%Y-%m}",
            route="cmems_subset", bytes_written=size, bytes_transferred=size,
            transfer_basis="file_size", seconds=_time.time() - t0,
        )  # fmt: skip
        written.append(path)
    return written


def raw_size_bytes(cfg: Config) -> int:
    return sum(p.stat().st_size for p in armor_dir(cfg).glob(f"{PRODUCT}_*.nc"))


# ----------------------------------------------------------------------------------------
# regridding
# ----------------------------------------------------------------------------------------
def open_month(cfg: Config, month_start: date) -> xr.Dataset | None:
    """Raw monthly file with canonical dimension names and ascending axes (``None`` if absent)."""
    path = armor_file(cfg, month_start)
    if not path.exists():
        return None
    ds = standardize(xr.open_dataset(path))
    if "depth" in ds.dims:
        ds = ds.sortby("depth")
    return ds


def regrid_days(
    raw: xr.DataArray, grid: Grid, days: np.ndarray, min_valid: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """One raw ``(time, depth, lat, lon)`` temperature array -> ``(n_days, 15, H, W)`` on the
    canonical grid, plus the matching ``datetime64[D]`` days (the product's own time steps)."""
    present = np.unique(raw["time"].values.astype("datetime64[D]"))
    present = present[np.isin(present, np.asarray(days).astype("datetime64[D]"))]
    arr = process_variable(raw, "temp", grid, present.astype("datetime64[ns]"), min_valid)
    return arr, present


def regrid_month(
    cfg: Config, grid: Grid, first: date, last: date, chunk_days: int = 4
) -> tuple[np.ndarray, np.ndarray] | None:
    """Regrid the downloaded file of one month: ``(fields (n, 15, H, W), days datetime64[D])`` for
    the days of ``[first, last]`` the file holds (``None`` when the file is absent). A few days at
    a time, so the 1/8 deg 3-D field never sits in memory for a whole month."""
    ds = open_month(cfg, first)
    if ds is None:
        return None
    want = daily_axis(first, last).values
    try:
        temp = ds[VARIABLE]
        parts, kept = [], []
        for s in range(0, temp.sizes["time"], chunk_days):
            sub = temp.isel(time=slice(s, s + chunk_days)).load()
            fields, days = regrid_days(sub, grid, want, cfg.harmonize.min_valid_fraction)
            if len(days):
                parts.append(fields)
                kept.append(days)
    finally:
        ds.close()
    if not parts:
        return None
    return np.concatenate(parts), np.concatenate(kept)
