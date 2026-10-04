"""Raw near-real-time files -> the canonical grid, and the harmonised store of the live window.

The processing is exactly that of the batch pipeline
(:func:`oceanembed.data.harmonize.process_variable`:
block mean or bilinear regridding, daily mean, kelvin to degC, vertical interpolation), applied to
the files of the days asked for. The live store ``data/processed/live.zarr`` has the layout of every
harmonised store (seven surface variables, ``temp``, ``mask``), so the ordinary dataset and the API
read it; the surface variables the model does not use stay NaN and ``temp`` stays NaN (there is no
target for recent days).
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Sequence
from pathlib import Path

import dask.array as da_
import numpy as np
import pandas as pd
import xarray as xr
from zarr.codecs import BloscCodec

from oceanembed.config import SURFACE_VARS, Config
from oceanembed.data.harmonize import VAR_ATTRS, process_variable
from oceanembed.data.regrid import crop, standardize
from oceanembed.grid import Grid, build_grid

log = logging.getLogger(__name__)

_COMPRESSOR = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")


def harmonise_files(
    cfg: Config,
    product: str,
    files: Sequence[Path],
    days: pd.DatetimeIndex,
    variables: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """Harmonise the raw ``files`` of ``product`` onto the canonical grid for ``days``.

    Returns ``{canonical variable: (len(days), [depth,] lat, lon) float32}``; days that no file
    covers are NaN. Each file is processed on its own in chunks of ``harmonize.time_chunk`` days,
    so a large 3-D file never has to be loaded whole."""
    grid = build_grid(cfg)
    variables = variables or cfg.products[product].variables
    pad = max(cfg.download.halo_deg, 0.0) + grid.resolution
    lat_r = (grid.lat_edges[0] - pad, grid.lat_edges[1] + pad)
    lon_r = (grid.lon_edges[0] - pad, grid.lon_edges[1] + pad)
    out: dict[str, np.ndarray] = {}
    chunk = cfg.harmonize.time_chunk
    for path in files:
        with xr.open_dataset(path) as raw:
            ds = crop(standardize(raw, cfg.products[product].coords), lat_r, lon_r)
            times = pd.DatetimeIndex(ds["time"].values)
            for s in range(0, len(times), chunk):
                sub = ds.isel(time=slice(s, s + chunk))
                cdays = pd.DatetimeIndex(sorted(set(times[s : s + chunk].normalize())))
                pos = days.get_indexer(cdays)
                keep = pos >= 0
                if not keep.any():
                    continue
                for canon, rawname in variables.items():
                    if rawname not in sub:
                        raise KeyError(
                            f"variable {rawname!r} not found in {path.name} "
                            f"(have {list(sub.data_vars)})"
                        )
                    arr = process_variable(
                        sub[rawname], canon, grid, cdays.values, cfg.harmonize.min_valid_fraction
                    )
                    if canon not in out:
                        out[canon] = np.full((len(days), *arr.shape[1:]), np.nan, np.float32)
                    have = np.isfinite(arr.reshape(len(cdays), -1)).any(axis=1) & keep
                    out[canon][pos[have]] = arr[have]
    for canon in variables:
        if canon not in out:
            n_depth = grid.n_depth if canon == "temp" else None
            shape = (len(days), *((n_depth,) if n_depth else ()), *grid.shape)
            out[canon] = np.full(shape, np.nan, dtype=np.float32)
    return out


def store_template(
    cfg: Config, grid: Grid, days: pd.DatetimeIndex, mask: np.ndarray, title: str
) -> xr.Dataset:
    h, w = grid.shape
    nt, nd = len(days), grid.n_depth
    data: dict = {}
    for v in SURFACE_VARS:
        data[v] = (
            ("time", "lat", "lon"),
            da_.full((nt, h, w), np.nan, chunks=(1, h, w), dtype="float32"),
            VAR_ATTRS[v],
        )
    data["temp"] = (
        ("time", "depth", "lat", "lon"),
        da_.full((nt, nd, h, w), np.nan, chunks=(1, nd, h, w), dtype="float32"),
        VAR_ATTRS["temp"],
    )
    data["mask"] = (
        ("depth", "lat", "lon"),
        np.asarray(mask, dtype=bool),
        {"long_name": "ocean and above sea floor", "flag_values": "0 1"},
    )
    ds = xr.Dataset(
        data,
        coords={"time": days.values, "depth": grid.depth, "lat": grid.lat, "lon": grid.lon},
    )
    ds["lat"].attrs.update(units="degrees_north", standard_name="latitude", axis="Y")
    ds["lon"].attrs.update(units="degrees_east", standard_name="longitude", axis="X")
    ds["depth"].attrs.update(units="m", standard_name="depth", positive="down", axis="Z")
    ds.attrs.update(
        title=title,
        Conventions="CF-1.8",
        grid_resolution_deg=cfg.grid.resolution,
        provider=cfg.provider,
        history="created by oceanembed live",
    )
    return ds


def write_store(
    path: Path,
    cfg: Config,
    days: pd.DatetimeIndex,
    surface: dict[str, np.ndarray],
    mask: np.ndarray,
    temp: np.ndarray | None = None,
    title: str | None = None,
) -> Path:
    """Write a harmonised store with the given surface fields (``(time, lat, lon)`` per canonical
    variable; the others NaN) and optional ``temp``. The store is built next to ``path`` and put
    in place at the end, so a reader never sees a half-written store."""
    grid = build_grid(cfg)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tpl = store_template(cfg, grid, days, mask, title or f"OceanEmbed live store: {cfg.run_name}")
    for v, arr in surface.items():
        tpl[v] = (("time", "lat", "lon"), np.asarray(arr, dtype=np.float32), VAR_ATTRS[v])
    if temp is not None:
        tpl["temp"] = (
            ("time", "depth", "lat", "lon"),
            np.asarray(temp, dtype=np.float32),
            VAR_ATTRS["temp"],
        )
    enc = {v: {"compressors": [_COMPRESSOR]} for v in tpl.data_vars if v != "mask"}
    path.parent.mkdir(parents=True, exist_ok=True)
    tpl.to_zarr(tmp, mode="w", encoding=enc, consolidated=False)
    if path.exists():
        shutil.rmtree(path)
    tmp.replace(path)
    return path
