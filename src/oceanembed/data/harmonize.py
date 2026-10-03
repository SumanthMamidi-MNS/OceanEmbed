"""Raw native-grid monthly files -> harmonised Zarr store on the canonical grid.

Layout of ``<data_root>/processed/<run>.zarr`` (see docs/architecture.md):

* ``sst sss sla uo vo uw vw`` : ``(time, lat, lon)`` float32, NaN over land / gaps
* ``temp``                    : ``(time, depth, lat, lon)`` float32, NaN below sea floor / land
* ``mask``                    : ``(depth, lat, lon)`` bool, ocean and above the sea floor
* one day per chunk along ``time``

Processing is month by month and, inside a month, in small time chunks, so memory stays bounded
even for the 1/12 deg 3-D temperature field.
"""

from __future__ import annotations

import logging
import shutil
from datetime import date
from pathlib import Path

import dask.array as da_
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from zarr.codecs import BloscCodec

from oceanembed.config import SURFACE_VARS, Config
from oceanembed.data.providers.base import months, raw_file
from oceanembed.data.regrid import (
    crop,
    interp_to_depths,
    regrid_horizontal,
    standardize,
    to_celsius,
    to_daily,
)
from oceanembed.grid import Grid, build_grid, daily_axis

log = logging.getLogger(__name__)

VAR_ATTRS: dict[str, dict[str, str]] = {
    "sst": {"long_name": "sea surface temperature", "standard_name": "sea_surface_temperature",
            "units": "degC"},
    "sss": {"long_name": "sea surface salinity", "standard_name": "sea_surface_salinity",
            "units": "1e-3"},
    "sla": {"long_name": "sea level anomaly",
            "standard_name": "sea_surface_height_above_sea_level", "units": "m"},
    "uo": {"long_name": "eastward surface current", "standard_name": "eastward_sea_water_velocity",
           "units": "m s-1"},
    "vo": {"long_name": "northward surface current",
           "standard_name": "northward_sea_water_velocity", "units": "m s-1"},
    "uw": {"long_name": "eastward 10 m wind", "standard_name": "eastward_wind", "units": "m s-1"},
    "vw": {"long_name": "northward 10 m wind", "standard_name": "northward_wind", "units": "m s-1"},
    "temp": {"long_name": "sea water potential temperature",
             "standard_name": "sea_water_potential_temperature", "units": "degC"},
}  # fmt: skip

# product -> canonical variables it provides (taken from the config at run time)
SURFACE_PRODUCTS = ["sst", "sss", "sla", "currents", "winds"]
_COMPRESSOR = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")


def _template(cfg: Config, grid: Grid, time: pd.DatetimeIndex) -> xr.Dataset:
    h, w = grid.shape
    nt, nd = len(time), grid.n_depth
    coords = {"time": time.values, "depth": grid.depth, "lat": grid.lat, "lon": grid.lon}
    data = {}
    for v in SURFACE_VARS:
        arr = da_.full((nt, h, w), np.nan, chunks=(1, h, w), dtype="float32")
        data[v] = (("time", "lat", "lon"), arr, VAR_ATTRS[v])
    arr = da_.full((nt, nd, h, w), np.nan, chunks=(1, nd, h, w), dtype="float32")
    data["temp"] = (("time", "depth", "lat", "lon"), arr, VAR_ATTRS["temp"])
    ds = xr.Dataset(data, coords=coords)
    ds["lat"].attrs.update(units="degrees_north", standard_name="latitude", axis="Y")
    ds["lon"].attrs.update(units="degrees_east", standard_name="longitude", axis="X")
    ds["depth"].attrs.update(units="m", standard_name="depth", positive="down", axis="Z")
    ds.attrs.update(
        title=f"OceanEmbed harmonised dataset: {cfg.run_name}",
        Conventions="CF-1.8",
        grid_resolution_deg=cfg.grid.resolution,
        provider=cfg.provider,
        history="created by oceanembed harmonize",
    )
    return ds


def _open_month(cfg: Config, product: str, first: date) -> xr.Dataset:
    path = raw_file(cfg, product, first)
    if not path.exists():
        cmd = "synth" if cfg.provider == "synthetic" else "download"
        raise FileNotFoundError(
            f"missing raw file {path}. Run `oceanembed {cmd} --config <yaml>` first."
        )
    ds = xr.open_dataset(path)
    ds = standardize(ds, cfg.products[product].coords)
    g = build_grid(cfg)
    pad = max(cfg.download.halo_deg, 0.0) + g.resolution
    return crop(
        ds,
        (g.lat_edges[0] - pad, g.lat_edges[1] + pad),
        (g.lon_edges[0] - pad, g.lon_edges[1] + pad),
    )


def process_variable(
    raw: xr.DataArray, canon: str, grid: Grid, days: np.ndarray, min_valid: float = 0.5
) -> np.ndarray:
    """One raw variable (standardised dims, any time sampling) -> canonical daily array
    ``(len(days), [depth,] lat, lon)`` float32 (NaN where missing)."""
    if "depth" in raw.dims and canon != "temp":
        # surface products such as the CMEMS SSS carry a singleton depth axis (0 m)
        if raw.sizes["depth"] != 1:
            raise ValueError(f"{canon}: surface field has {raw.sizes['depth']} depth levels")
        raw = raw.isel(depth=0, drop=True)
    order = [d for d in ("time", "depth", "lat", "lon") if d in raw.dims]
    raw = raw.transpose(*order)
    if canon in ("sst", "temp"):
        raw = to_celsius(raw)
    daily = to_daily(raw)
    if "depth" in daily.dims:
        daily = interp_to_depths(daily, grid.depth)
    out = regrid_horizontal(daily, grid.lat, grid.lon, grid.resolution, "auto", min_valid)
    # place onto the requested day axis, NaN for days absent from the raw data
    full_shape = (len(days), *out.shape[1:])
    full = np.full(full_shape, np.nan, dtype=np.float32)
    pos = pd.DatetimeIndex(days).get_indexer(pd.DatetimeIndex(out["time"].values))
    ok = pos >= 0
    full[pos[ok]] = out.values[ok]
    return full


def harmonize(cfg: Config, progress: bool = True, surface_only: bool = False) -> Path:
    """Build the harmonised store. ``surface_only`` skips the GLORYS temperature: the store then
    holds the five surface products only (``temp`` stays NaN, the ocean ``mask`` is all False) --
    enough to predict with released weights, which carry their own mask and statistics."""
    grid = build_grid(cfg)
    time = daily_axis(cfg.time.start, cfg.time.end)
    store = cfg.zarr_path
    store.parent.mkdir(parents=True, exist_ok=True)
    if store.exists():
        shutil.rmtree(store)

    template = _template(cfg, grid, time)
    enc = {v: {"compressors": [_COMPRESSOR]} for v in template.data_vars}
    template.to_zarr(store, mode="w", compute=False, encoding=enc, consolidated=False)

    h, w = grid.shape
    valid_count = np.zeros((grid.n_depth, h, w), dtype=np.int32)
    missing_days: dict[str, int] = {}
    chunk = cfg.harmonize.time_chunk
    month_list = list(months(cfg.time.start, cfg.time.end))
    bar = tqdm(month_list, desc="harmonize", disable=not progress)
    for first, last in bar:
        mdays = daily_axis(first, last)
        i0 = int(time.get_loc(mdays[0]))
        opened: dict[str, xr.Dataset] = {
            p: _open_month(cfg, p, first)
            for p in (SURFACE_PRODUCTS if surface_only else [*SURFACE_PRODUCTS, "temp"])
        }
        try:
            for s in range(0, len(mdays), chunk):
                days = mdays[s : s + chunk]
                t0, t1 = days[0], days[-1] + np.timedelta64(1, "D")
                out: dict[str, xr.DataArray] = {}
                for product, ds in opened.items():
                    sub = ds.sel(time=slice(t0, t1 - np.timedelta64(1, "ns")))
                    for canon, rawname in cfg.products[product].variables.items():
                        if rawname not in sub:
                            raise KeyError(
                                f"variable {rawname!r} not found in raw {product} data "
                                f"(have {list(sub.data_vars)})"
                            )
                        arr = process_variable(
                            sub[rawname],
                            canon,
                            grid,
                            days.values,
                            cfg.harmonize.min_valid_fraction,
                        )
                        nmiss = int(np.isnan(arr.reshape(len(days), -1)).all(axis=1).sum())
                        if nmiss:
                            missing_days[canon] = missing_days.get(canon, 0) + nmiss
                        dims = (
                            ("time", "depth", "lat", "lon")
                            if canon == "temp"
                            else ("time", "lat", "lon")
                        )
                        out[canon] = xr.DataArray(arr, dims=dims)
                        if canon == "temp":
                            valid_count += np.isfinite(arr).sum(axis=0).astype(np.int32)
                chunk_ds = xr.Dataset(out, coords={"time": days.values})
                chunk_ds.to_zarr(
                    store, region={"time": slice(i0 + s, i0 + s + len(days))}, consolidated=False
                )
        finally:
            for ds in opened.values():
                ds.close()

    mask = valid_count >= max(1, int(np.ceil(0.5 * len(time))))
    if surface_only:
        mask = np.zeros_like(mask)  # unknown without the target: released weights bring theirs
    mds = xr.Dataset(
        {
            "mask": (
                ("depth", "lat", "lon"),
                mask,
                {"long_name": "ocean and above sea floor", "flag_values": "0 1"},
            )
        },
        coords={"depth": grid.depth, "lat": grid.lat, "lon": grid.lon},
    )
    mds.to_zarr(store, mode="a", consolidated=False)
    # optional user-supplied INCOIS gridded ARGO, regridded for the validation step
    from oceanembed.data.providers.argo import load_incois_gridded

    gridded = load_incois_gridded(cfg)
    if gridded is not None:
        gridded.to_dataset(name="temp").to_netcdf(
            cfg.processed_root / f"{cfg.run_name}_argo_gridded.nc"
        )
    for canon, n in missing_days.items():
        log.warning("%s: %d day(s) had no raw data and were left NaN", canon, n)
    return store


def open_harmonized(cfg: Config) -> xr.Dataset:
    return xr.open_zarr(cfg.zarr_path, consolidated=False)


__all__ = ["VAR_ATTRS", "harmonize", "open_harmonized", "process_variable"]
