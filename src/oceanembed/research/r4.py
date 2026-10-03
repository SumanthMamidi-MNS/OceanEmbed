"""Research stage R4 (external benchmark): ARMOR3D against OceanEmbed, the baselines and GLORYS.

Pipeline (each step resumable, nothing under the main run is modified):

1. **download** the ARMOR3D temperature for the test period (:mod:`oceanembed.research.armor3d`);
2. **regrid** every downloaded month to the canonical 0.25 degree grid and 15 depths on the
   product's own daily time steps (no temporal interpolation) into
   ``data/processed/<run>_armor3d/armor3d_<YYYYMM>.nc``;
3. **score**, on exactly the same samples:

   * against Argo: the stored matchups of ``validate-argo`` (``metrics/argo_matchups.parquet``)
     get an ``armor3d`` column from the regridded field in the same cell and on the same day; a
     matchup is kept only where ARMOR3D has a value, so ARMOR3D, OceanEmbed, no-pretraining,
     ridge, climatology and GLORYS are all scored on the identical profile levels;
   * on the grid: per day and depth, sums of every product against GLORYS and, in addition,
     against ARMOR3D, on the cells where the static mask, GLORYS and ARMOR3D are all defined.

Outputs: ``outputs/<run>/research/r4/grid_sums.npz`` (streaming pass), ``summary.json``,
``summary.md`` and ``figures/`` (see :mod:`oceanembed.research.r4_report`).
"""

from __future__ import annotations

import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from oceanembed.config import Config
from oceanembed.data.dataset import make_surface_dataset
from oceanembed.data.harmonize import open_harmonized
from oceanembed.data.providers.base import months
from oceanembed.data_access import load_prediction
from oceanembed.eval.argo_validation import load_argo_period
from oceanembed.eval.evaluate import read_target, split_bounds
from oceanembed.eval.metrics import SUM_FIELDS, sums_from_arrays
from oceanembed.grid import build_grid
from oceanembed.research import armor3d

REGIONS = ("all", "arabian_sea", "bay_of_bengal")
# product key -> (label, column in the Argo matchup table, key in data_access.load_prediction)
PRODUCTS = {
    "climatology": ("Climatology", "clim", None),
    "ridge": ("Ridge regression", "ridge", "ridge"),
    "scratch": ("OceanEmbed (no pretraining)", "model_scratch", "model_scratch"),
    "model": ("OceanEmbed (pretrained)", "model", "model"),
    "armor3d": ("ARMOR3D", "armor3d", None),
    "glorys": ("GLORYS", "glorys", None),
}
LABELS = {k: v[0] for k, v in PRODUCTS.items()}
REFS = ("glorys", "armor3d")
GRID_SUMS = "grid_sums.npz"
EXAMPLE_DAY = "2024-07-15"
EXAMPLE_DEPTH = 100.0


def r4_dir(cfg: Config) -> Path:
    return cfg.outputs_dir / "research" / "r4"


def regrid_dir(cfg: Config) -> Path:
    return cfg.processed_root / f"{cfg.run_name}_armor3d"


def regridded_file(cfg: Config, month_start: date) -> Path:
    return regrid_dir(cfg) / f"armor3d_{month_start:%Y%m}.nc"


def _say(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------------------------------------------------------------------
# step 2: regrid
# ----------------------------------------------------------------------------------------
def regrid_all(cfg: Config, start: str, end: str, force: bool = False) -> list[Path]:
    """Regrid the downloaded months of ``[start, end]``; existing outputs are kept unless
    ``force``. Returns the files written. Months without a raw file are skipped (and reported by
    :func:`regridded_days`)."""
    grid = build_grid(cfg)
    written = []
    for first, last in months(pd.Timestamp(start).date(), pd.Timestamp(end).date()):
        out = regridded_file(cfg, first)
        if out.exists() and not force:
            continue
        res = armor3d.regrid_month(cfg, grid, first, last)
        if res is None:
            continue
        fields, days = res
        out.parent.mkdir(parents=True, exist_ok=True)
        ds = xr.Dataset(
            {"temp": (("time", "depth", "lat", "lon"), fields, {"units": "degC"})},
            coords={
                "time": days.astype("datetime64[ns]"),
                "depth": grid.depth,
                "lat": grid.lat,
                "lon": grid.lon,
            },
            attrs={
                "source": f"{armor3d.PRODUCT_ID} / {armor3d.DATASET_ID}, "
                f"variable {armor3d.VARIABLE}",
                "regridding": "vertical linear interpolation to the standard depths, then 2x2 "
                "block mean (harmonize.process_variable); product's own daily steps",
            },
        )
        tmp = out.with_suffix(".tmp.nc")
        ds.to_netcdf(tmp, encoding={"temp": {"zlib": True, "complevel": 3}})
        os.replace(tmp, out)
        written.append(out)
    return written


def regridded_days(cfg: Config, start: str, end: str) -> pd.DatetimeIndex:
    """Days of ``[start, end]`` that have a regridded ARMOR3D field."""
    days = []
    for first, _ in months(pd.Timestamp(start).date(), pd.Timestamp(end).date()):
        f = regridded_file(cfg, first)
        if f.exists():
            with xr.open_dataset(f) as ds:
                days.append(pd.DatetimeIndex(ds["time"].values))
    if not days:
        return pd.DatetimeIndex([])
    idx = days[0].append(days[1:]).sort_values()
    want = pd.date_range(start, end, freq="D")
    return idx[idx.isin(want)]


# ----------------------------------------------------------------------------------------
# step 3a: Argo
# ----------------------------------------------------------------------------------------
def armor_at_matchups(cfg: Config, matchups: pd.DataFrame, start: str, end: str) -> np.ndarray:
    """ARMOR3D (regridded) at each matchup row: same cell, same UTC day, same standard depth.
    NaN where ARMOR3D is undefined or the day has no field."""
    grid = build_grid(cfg)
    depths = grid.depth
    out = np.full(len(matchups), np.nan, dtype=np.float32)
    day = pd.DatetimeIndex(matchups["time"]).normalize()
    j = np.round((matchups["grid_lat"].to_numpy(float) - grid.lat[0]) / grid.resolution).astype(int)
    k = np.round((matchups["grid_lon"].to_numpy(float) - grid.lon[0]) / grid.resolution).astype(int)
    d = np.searchsorted(depths, matchups["depth"].to_numpy(float))
    for first, _ in months(pd.Timestamp(start).date(), pd.Timestamp(end).date()):
        f = regridded_file(cfg, first)
        sel = np.flatnonzero((day.year == first.year) & (day.month == first.month))
        if not f.exists() or len(sel) == 0:
            continue
        with xr.open_dataset(f) as ds:
            times = pd.DatetimeIndex(ds["time"].values)
            pos = times.get_indexer(day[sel])
            ok = pos >= 0
            arr = ds["temp"].values
        rows = sel[ok]
        out[rows] = arr[pos[ok], d[rows], j[rows], k[rows]]
    return out


def argo_table(cfg: Config, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    """The stored Argo matchups with an ``armor3d`` column, restricted to the rows where ARMOR3D is
    defined (the common sample), plus a dict of counts."""
    path = cfg.outputs_dir / "metrics" / "argo_matchups.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; R4 scores ARMOR3D on the validate-argo matchups"
        )
    m = pd.read_parquet(path)
    m["armor3d"] = armor_at_matchups(cfg, m, start, end)
    keep = np.isfinite(m["armor3d"].to_numpy())
    info = {
        "matchups_total": int(len(m)),
        "profiles_total": int(m["profile_id"].nunique()),
        "matchups_with_armor3d": int(keep.sum()),
        "profiles_with_armor3d": int(m.loc[keep, "profile_id"].nunique()),
        "matchups_dropped_armor3d_undefined": int((~keep).sum()),
    }
    return m[keep].reset_index(drop=True), info


def float_density(cfg: Config, prof: pd.DataFrame, start: str, end: str) -> np.ndarray:
    """For each profile of ``prof`` (``profile_id, time, lat, lon``): the number of *other* Argo
    profiles within +-3 days and +-2 degrees in latitude and longitude, from every profile of the
    raw Argo files of the period. A crude local data density: how many floats ARMOR3D's in-situ
    analysis had around the matchup."""
    raw = load_argo_period(cfg, pd.Timestamp(start) - pd.Timedelta(days=3), end)
    allp = raw.drop_duplicates("profile_id")[["profile_id", "time", "latitude", "longitude"]]
    t = pd.to_datetime(allp["time"]).to_numpy("datetime64[s]").astype(float) / 86400.0
    la, lo = allp["latitude"].to_numpy(float), allp["longitude"].to_numpy(float)
    pid = allp["profile_id"].to_numpy()
    pt = pd.to_datetime(prof["time"]).to_numpy("datetime64[s]").astype(float) / 86400.0
    out = np.zeros(len(prof))
    for i, (ti, lai, loi, pi) in enumerate(
        zip(
            pt,
            prof["lat"].to_numpy(float),
            prof["lon"].to_numpy(float),
            prof["profile_id"],
            strict=True,
        )
    ):
        near = (np.abs(t - ti) <= 3.0) & (np.abs(la - lai) <= 2.0) & (np.abs(lo - loi) <= 2.0)
        out[i] = near.sum() - int(np.any(near & (pid == pi)))
    return out


def per_day_sums(
    x: np.ndarray, y: np.ndarray, t_idx: np.ndarray, d_idx: np.ndarray, shape
) -> np.ndarray:
    """``(F, T, D)`` additive sums of matchup rows (prediction ``x``, reference ``y``)."""
    n_t, n_d = shape
    flat = t_idx * n_d + d_idx
    e = x - y
    cols = [
        np.ones_like(x), x, y, x * x, y * y, x * y, np.abs(e), e * e,
    ]  # fmt: skip
    return np.stack(
        [np.bincount(flat, weights=c, minlength=n_t * n_d).reshape(n_t, n_d) for c in cols]
    )


def argo_daily(
    table: pd.DataFrame, days: pd.DatetimeIndex, depths: np.ndarray, products: list[str]
) -> dict:
    """Per-day, per-depth sums of every product vs the Argo observation for the regions
    (``all``, ``arabian_sea``, ``bay_of_bengal``): ``{'raw'|'anom': (R, P, F, T, D)}``."""
    t_idx = days.get_indexer(pd.DatetimeIndex(table["time"]).normalize())
    d_idx = np.searchsorted(depths, table["depth"].to_numpy(float))
    ok = t_idx >= 0
    obs = table["obs"].to_numpy(np.float64)
    clim = table["clim"].to_numpy(np.float64)
    out = {
        k: np.zeros((len(REGIONS), len(products), len(SUM_FIELDS), len(days), len(depths)))
        for k in ("raw", "anom")
    }
    for r, region in enumerate(REGIONS):
        sel = ok if region == "all" else ok & (table["basin"].to_numpy() == region)
        for p, name in enumerate(products):
            col = PRODUCTS[name][1]
            x = table[col].to_numpy(np.float64)[sel]
            args = (t_idx[sel], d_idx[sel], (len(days), len(depths)))
            out["raw"][r, p] = per_day_sums(x, obs[sel], *args)
            out["anom"][r, p] = per_day_sums(x - clim[sel], obs[sel] - clim[sel], *args)
    return out


# ----------------------------------------------------------------------------------------
# step 3b: grid
# ----------------------------------------------------------------------------------------
def stream_grid(cfg: Config, start: str, end: str, progress: bool = True) -> Path:
    """Stream the days that have an ARMOR3D field and write ``grid_sums.npz``: per day and depth the
    sums of every product against GLORYS and against ARMOR3D on the common sample (static mask,
    GLORYS and ARMOR3D defined), for the whole domain and each basin; plus one example day."""
    out = r4_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    ds = make_surface_dataset(cfg, start, end)
    zds = open_harmonized(cfg)
    grid = build_grid(cfg)
    dates = ds.dates()
    depths = np.asarray(ds.depth, dtype=np.float64)
    armor_days = regridded_days(cfg, start, end)
    have = [i for i, d in enumerate(dates) if d in armor_days]
    if not have:
        raise FileNotFoundError("no regridded ARMOR3D day overlaps the test split")
    basins = grid.basin_masks()
    region_masks = [np.ones(grid.shape, bool), basins["arabian_sea"], basins["bay_of_bengal"]]
    products = list(PRODUCTS)
    shape = (len(REGIONS), len(products), len(SUM_FIELDS), len(dates), len(depths))
    sums = {ref: {k: np.zeros(shape) for k in ("raw", "anom")} for ref in REFS}
    valid_days = np.zeros(len(dates), dtype=bool)
    example = {}
    bar = progress and sys.stderr.isatty()
    t0 = time.time()
    done = 0
    for first, _ in months(pd.Timestamp(start).date(), pd.Timestamp(end).date()):
        f = regridded_file(cfg, first)
        if not f.exists():
            continue
        with xr.open_dataset(f) as ads:
            atimes = pd.DatetimeIndex(ads["time"].values)
            afields = ads["temp"].values
        for ai, day in enumerate(tqdm(atimes, desc=f"r4 {first:%Y-%m}", disable=not bar)):
            if day not in dates:
                continue
            t = int(dates.get_loc(day))
            glorys = read_target(zds, ds, t, t + 1)[0]
            clim = ds.stats.climatology(dates[t : t + 1].values)[0]
            fields = {"glorys": glorys, "armor3d": afields[ai], "climatology": clim}
            for name, (_, _, src) in PRODUCTS.items():
                if src is not None:
                    fields[name] = load_prediction(cfg.outputs_dir, day, src).values
            fields = {k: np.where(ds.mask, v, np.nan).astype(np.float32) for k, v in fields.items()}
            valid = ds.mask & np.isfinite(fields["glorys"]) & np.isfinite(fields["armor3d"])
            for ref in REFS:
                y = fields[ref]
                for p, name in enumerate(products):
                    if name == ref:
                        continue
                    x = fields[name]
                    for r, rm in enumerate(region_masks):
                        v = valid & rm[None]
                        for kind, (xx, yy) in (("raw", (x, y)), ("anom", (x - clim, y - clim))):
                            part = sums_from_arrays(xx, yy, v, axis=(1, 2))
                            for i, name_f in enumerate(SUM_FIELDS):
                                sums[ref][kind][r, p, i, t] = part[name_f]
            valid_days[t] = True
            if str(day.date()) == EXAMPLE_DAY:
                z = int(np.argmin(np.abs(depths - EXAMPLE_DEPTH)))
                example = {name: f[z] for name, f in fields.items()} | {"depth": depths[z]}
            done += 1
    _say(f"r4 grid: {done} days in {time.time() - t0:.0f}s")
    tmp = out / "grid_sums.tmp.npz"
    arrays = {f"sums_{ref}_{kind}": sums[ref][kind] for ref in REFS for kind in ("raw", "anom")}
    arrays |= {f"example_{k}": np.asarray(v) for k, v in example.items()}
    np.savez_compressed(
        tmp,
        dates=np.array([str(d.date()) for d in dates]),
        valid_days=valid_days,
        depths=depths,
        products=np.array(products),
        regions=np.array(REGIONS),
        refs=np.array(REFS),
        fields=np.array(SUM_FIELDS),
        lat=grid.lat,
        lon=grid.lon,
        example_date=np.array(EXAMPLE_DAY),
        **arrays,
    )
    os.replace(tmp, out / GRID_SUMS)
    return out / GRID_SUMS


def load_grid_sums(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------------------
def run_r4(
    cfg: Config,
    download: bool = True,
    recompute: bool = False,
    n_boot: int = 2000,
    block_length: int | None = None,
    seed: int = 0,
    progress: bool = True,
) -> dict:
    """Download (unless ``download`` is False), regrid, stream, score and write the report. Every
    step skips what is already on disk; ``recompute`` redoes the regridding and the streaming pass
    (the downloaded files are kept)."""
    from oceanembed.research.r4_report import make_r4_report

    start, end = split_bounds(cfg, "test")
    if download:
        written = armor3d.download_armor(cfg, pd.Timestamp(start).date(), pd.Timestamp(end).date())
        _say(
            f"download: {len(written)} month(s) fetched, "
            f"{armor3d.raw_size_bytes(cfg) / 1e6:.0f} MB on disk"
        )
    regridded = regrid_all(cfg, start, end, force=recompute)
    _say(f"regrid: {len(regridded)} month(s) written")
    if recompute or not (r4_dir(cfg) / GRID_SUMS).exists():
        stream_grid(cfg, start, end, progress)
    return make_r4_report(cfg, n_boot=n_boot, block_length=block_length, seed=seed)
