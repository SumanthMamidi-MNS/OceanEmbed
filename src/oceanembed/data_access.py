"""Read-only data access for the OceanEmbed dashboard (no UI code, no Streamlit import).

Every function takes the *run directory* (``outputs/<run_name>``, a ``str`` or ``Path``) and plain
hashable arguments (dates may be ``str``, ``datetime``, ``np.datetime64`` or ``pd.Timestamp``), so
each one can be wrapped directly in ``st.cache_data``. Paths are never guessed: the harmonised
Zarr and statistics locations come from the run's ``run_meta.json`` (written by the pipeline).

Conventions: temperatures are degC ``xarray.DataArray``\\s with dims ``(depth, lat, lon)``; land /
below-sea-floor cells are NaN. Functions that read optional products return ``None`` when the file
(or the date inside it) does not exist; functions that need a prediction raise ``KeyError`` for a
date outside :func:`available_dates`.
"""

from __future__ import annotations

import json
import re
import threading
from functools import lru_cache
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

from oceanembed.config import SURFACE_VARS, Config
from oceanembed.data.stats import Stats, harmonic_design
from oceanembed.grid import build_grid
from oceanembed.infer.predict import PRODUCT_FILE, RIDGE_PRODUCT_DIR
from oceanembed.runmeta import RUN_META_FILE, read_run_meta

try:  # the lock xarray holds around netCDF4 / HDF5 calls: the C libraries are not thread-safe
    from xarray.backends.locks import HDF5_LOCK, NETCDFC_LOCK, combine_locks

    NETCDF4_LOCK = combine_locks([NETCDFC_LOCK, HDF5_LOCK])
except ImportError:  # pragma: no cover
    NETCDF4_LOCK = threading.RLock()

__all__ = [
    "available_dates",
    "climatology_info",
    "embedding_dates",
    "embedding_pca_mean",
    "embedding_pca_rgb",
    "list_runs",
    "load_argo_matchups",
    "load_climatology",
    "load_climatology_point",
    "load_embeddings",
    "load_maps",
    "load_metrics_argo",
    "load_metrics_glorys",
    "load_prediction",
    "load_profile",
    "load_run_meta",
    "load_section",
    "load_surface_inputs",
    "load_target",
    "load_training_logs",
    "prediction_file_dates",
    "prediction_methods",
    "split_day_counts",
]

MAX_PCA_COMPONENTS = 8
_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,30}$")


def _run(run) -> Path:
    return Path(run)


def _ts(date) -> pd.Timestamp:
    return pd.Timestamp(date).normalize()


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ----------------------------------------------------------------------------------------
# runs and metadata
# ----------------------------------------------------------------------------------------
def list_runs(outputs_root) -> list[dict]:
    """Runs under ``outputs_root`` (sub-folders holding a ``run_meta.json``), sorted by name.

    Each entry: ``name``, ``path``, ``data_source``, ``updated`` and availability flags
    ``has_predictions`` / ``has_metrics`` / ``has_argo`` / ``has_embeddings`` / ``has_report``.
    """
    root = Path(outputs_root)
    out = []
    if not root.exists():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / RUN_META_FILE).exists()):
        meta = read_run_meta(d)
        out.append(
            {
                "name": meta.get("run_name", d.name),
                "path": str(d),
                "data_source": meta.get("data_source", "unknown"),
                "updated": meta.get("updated"),
                "has_predictions": any((d / "predictions").glob("oceanembed_T_*.nc")),
                "has_metrics": (d / "metrics" / "metrics_glorys.json").exists(),
                "has_argo": (d / "metrics" / "metrics_argo.json").exists(),
                "has_embeddings": (d / "embeddings" / "embeddings.zarr").exists(),
                "has_report": (d / "report.md").exists(),
            }
        )
    return out


def load_run_meta(run) -> dict:
    """``run_meta.json``: config snapshot, ``data_source``, split dates, grid, data paths."""
    return read_run_meta(_run(run))


@lru_cache(maxsize=8)
def _config_cached(meta_path: str, mtime: float) -> Config:
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    return Config.model_validate(meta["config"])


def _config(run) -> Config:
    p = _run(run) / RUN_META_FILE
    return _config_cached(str(p), p.stat().st_mtime)


def _paths(run) -> dict[str, Path]:
    """Data paths from ``run_meta.json``. The pipeline stores them as the config gave them (relative
    to the project root by default), so relative ones are anchored at the project root, which is
    recovered from the run directory itself (``<root>/<outputs_root>/<run>``) rather than from the
    current working directory."""
    meta = load_run_meta(run)
    paths = meta["paths"]
    out = Path(paths.get("outputs_dir", ""))
    root = None
    if not out.is_absolute() and out.parts:
        run_dir = _run(run).resolve()
        if len(run_dir.parents) >= len(out.parts):
            root = run_dir.parents[len(out.parts) - 1]
    resolved = {}
    for k, v in paths.items():
        p = Path(v)
        resolved[k] = root / p if root is not None and not p.is_absolute() else p
    return resolved


# ----------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------
def load_metrics_glorys(run) -> dict | None:
    """``metrics/metrics_glorys.json`` (per method: overall / per_depth / per_basin; daily RMSE)."""
    return _json(_run(run) / "metrics" / "metrics_glorys.json")


def load_metrics_argo(run) -> dict | None:
    """``metrics/metrics_argo.json`` (same structure, vs Argo; plus counts and notes)."""
    return _json(_run(run) / "metrics" / "metrics_argo.json")


def load_argo_matchups(run) -> pd.DataFrame | None:
    """``metrics/argo_matchups.parquet``: one row per (profile, depth)."""
    p = _run(run) / "metrics" / "argo_matchups.parquet"
    return pd.read_parquet(p) if p.exists() else None


def load_maps(run) -> xr.Dataset | None:
    """``metrics/maps_glorys.nc`` loaded into memory: ``rmse_<method>``, ``bias_<method>``,
    ``corr_anom_<method>`` ... on ``(depth, lat, lon)``."""
    p = _run(run) / "metrics" / "maps_glorys.nc"
    if not p.exists():
        return None
    with xr.open_dataset(p) as ds:
        return ds.load()


def load_training_logs(run) -> dict[str, pd.DataFrame]:
    """JSON-lines training logs as DataFrames keyed by file stem (``pretrain``, ``train``,
    ``train_scratch`` ...)."""
    out = {}
    for p in sorted((_run(run) / "logs").glob("*.jsonl")):
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]
        out[p.stem] = pd.DataFrame(rows)
    return out


# ----------------------------------------------------------------------------------------
# predictions and reference fields
# ----------------------------------------------------------------------------------------
def prediction_methods(run) -> dict[str, Path]:
    """Methods that have a NetCDF day-field product, as ``{method key: folder}``.

    ``model`` is ``predictions/`` itself, ``ridge`` is ``predictions/ridge/`` and every other
    sub-folder ``predictions/<tag>/`` is the ablation ``model_<tag>`` (the same keys the metrics
    use). Only folders that hold at least one product file are listed; ``model`` comes first,
    then ``ridge``, then the ablations by name.
    """
    base = _run(run) / "predictions"
    prefix = PRODUCT_FILE.split("{")[0]
    out: dict[str, Path] = {}
    if any(_readable(f) for f in base.glob(f"{prefix}*.nc")):
        out["model"] = base
    if base.is_dir():
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            if not any(_readable(f) for f in d.glob(f"{prefix}*.nc")):
                continue
            if d.name == RIDGE_PRODUCT_DIR:
                out["ridge"] = d
            elif _TAG_RE.match(d.name):
                out[f"model_{d.name}"] = d
    return dict(sorted(out.items(), key=lambda kv: (kv[0] != "model", kv[0] != "ridge", kv[0])))


def _prediction_files(run, method: str = "model") -> dict[str, Path]:
    folder = prediction_methods(run).get(method)
    if folder is None:
        return {}
    prefix = PRODUCT_FILE.split("{")[0]
    files = sorted(f for f in folder.glob(f"{prefix}*.nc") if _readable(f))
    return {f.stem.removeprefix(prefix): f for f in files}


def _readable(f: Path) -> bool:
    """False for a product file that cannot be opened yet (``predict`` writes in place, so a
    month that is still being produced can be a truncated HDF5 file for a while)."""
    try:
        st = f.stat()
        _file_dates(str(f), st.st_mtime_ns, st.st_size)
    except (OSError, KeyError, ValueError, RuntimeError):
        return False
    return True


@lru_cache(maxsize=64)
def _file_dates(path: str, mtime_ns: int, size: int) -> pd.DatetimeIndex:
    # netCDF4 directly (a few ms): the product's time axis is int days since an epoch, and
    # xarray's open_dataset costs ~0.4 s per file, which adds up over 12 monthly files x 3 methods
    with NETCDF4_LOCK, netCDF4.Dataset(path) as nc:
        var = nc.variables["time"]
        units = str(getattr(var, "units", ""))
        raw = np.asarray(var[:])
    if units.startswith("days since "):
        origin = pd.Timestamp(units.removeprefix("days since ").strip())
        return pd.DatetimeIndex(origin + pd.to_timedelta(raw.astype(np.int64), unit="D"))
    with xr.open_dataset(path) as ds:  # any other encoding: let xarray decode it
        return pd.DatetimeIndex(ds["time"].values)


def prediction_file_dates(run, method: str = "model") -> dict[str, pd.DatetimeIndex]:
    """Days held by each monthly product file, keyed by ``YYYYMM`` (cached per mtime / size)."""
    out = {}
    for ym, f in _prediction_files(run, method).items():
        st = f.stat()
        out[ym] = _file_dates(str(f), st.st_mtime_ns, st.st_size)
    return out


def available_dates(run, method: str = "model") -> pd.DatetimeIndex:
    """Sorted days for which a prediction exists (read from the monthly NetCDF files)."""
    times = list(prediction_file_dates(run, method).values())
    if not times:
        return pd.DatetimeIndex([])
    return times[0].append(times[1:]).sort_values()


def load_prediction(run, date, method: str = "model") -> xr.DataArray:
    """Predicted temperature ``(depth, lat, lon)`` degC for one day (``KeyError`` if absent).

    ``method`` selects the product: ``model`` (default), ``ridge`` or an ablation ``model_<tag>``.
    One day is read straight through ``netCDF4`` (one compressed chunk, ~10 ms); going through
    xarray's lazy ``open_dataset().sel()`` costs ~0.3 s per call on the same file.
    """
    d = _ts(date)
    f = _prediction_files(run, method).get(d.strftime("%Y%m"))
    if f is None:
        raise KeyError(f"no '{method}' prediction file for {d:%Y-%m} in {run}")
    st = f.stat()
    days = _file_dates(str(f), st.st_mtime_ns, st.st_size)
    pos = days.get_indexer([d])[0]
    if pos < 0:
        raise KeyError(f"{d.date()} not in {f.name}")
    with NETCDF4_LOCK, netCDF4.Dataset(f) as nc:
        var = nc.variables["temperature"]
        arr = np.ma.filled(var[pos], np.nan).astype(np.float32)
        coords = {k: np.asarray(nc.variables[k][:]) for k in ("depth", "lat", "lon")}
        attrs = {k: var.getncattr(k) for k in var.ncattrs() if not k.startswith("_")}
    return xr.DataArray(
        arr,
        dims=("depth", "lat", "lon"),
        coords={**coords, "time": d.to_datetime64()},
        name="temperature",
        attrs=attrs,
    )


@lru_cache(maxsize=8)
def _open_zarr_cached(path: str, stamp: float) -> xr.Dataset:
    return xr.open_zarr(path, consolidated=False)


def _open_store(run) -> xr.Dataset | None:
    """The harmonised Zarr of a run (handle cached per ``run_meta.json`` mtime)."""
    z = _paths(run)["zarr"]
    if not z.exists():
        return None
    return _open_zarr_cached(str(z), (_run(run) / RUN_META_FILE).stat().st_mtime)


def load_target(run, date) -> xr.DataArray | None:
    """Harmonised GLORYS target ``(depth, lat, lon)``; ``None`` if the store, the date or the data
    (all NaN) is unavailable."""
    ds = _open_store(run)
    d = _ts(date)
    if ds is None or d not in pd.DatetimeIndex(ds["time"].values):
        return None
    arr = ds["temp"].sel(time=d).load()
    return arr if bool(np.isfinite(arr.values).any()) else None


@lru_cache(maxsize=4)
def _stats_cached(path: str, mtime: float) -> Stats:
    return Stats.load(Path(path))


def load_climatology(run, date) -> xr.DataArray:
    """Harmonic climatology ``(depth, lat, lon)`` for a day (NaN where it was not fitted, i.e.
    land / below the sea floor)."""
    p = _paths(run)["stats"]
    stats = _stats_cached(str(p), p.stat().st_mtime)
    g = build_grid(_config(run))
    clim = stats.climatology(np.array([_ts(date).to_datetime64()]))[0]
    return xr.DataArray(
        clim,
        dims=("depth", "lat", "lon"),
        coords=g.coords(),
        name="climatology",
        attrs={"units": "degC", "long_name": "harmonic climatology (train split)"},
    )


def load_climatology_point(run, dates, i: int, j: int) -> np.ndarray:
    """Harmonic climatology ``(n_dates, depth)`` float32 at grid cell ``(i, j)`` (NaN where it was
    not fitted). Cheap: five coefficients per depth, no full-grid evaluation."""
    p = _paths(run)["stats"]
    stats = _stats_cached(str(p), p.stat().st_mtime)
    x = harmonic_design(np.asarray(pd.DatetimeIndex(dates).values)).astype(np.float32)
    return (x @ stats.clim_coef[:, :, i, j]).astype(np.float32)


def climatology_info(run) -> dict | None:
    """Attributes of the statistics file (``train_start``, ``train_end``, ``n_train_days``,
    ``n_harmonic_terms``: 1 = mean only, 3 = + annual, 5 = + semi-annual); ``None`` if missing."""
    try:
        p = _paths(run)["stats"]
        if not p.exists():
            return None
        with NETCDF4_LOCK, netCDF4.Dataset(p) as nc:
            return {k: nc.getncattr(k) for k in nc.ncattrs() if not str(k).startswith("_")}
    except (OSError, KeyError, ValueError):
        return None


def split_day_counts(run) -> dict[str, int]:
    """Days per split (``train`` / ``val`` / ``test``): days present in the harmonised store inside
    each split range, else the calendar days of the range."""
    meta = load_run_meta(run)
    store = None
    try:
        store = _open_store(run)
    except (OSError, KeyError, ValueError):
        pass
    times = pd.DatetimeIndex(store["time"].values) if store is not None else None
    out = {}
    for name, span in (meta.get("split") or {}).items():
        lo, hi = pd.Timestamp(span["start"]), pd.Timestamp(span["end"])
        if times is not None:
            out[name] = int(((times >= lo) & (times <= hi)).sum())
        else:
            out[name] = int((hi - lo).days + 1)
    return out


def load_surface_inputs(run, date) -> xr.Dataset | None:
    """The 7 surface input fields ``(lat, lon)`` in physical units (sst, sss, sla, uo, vo, uw, vw);
    ``None`` if the store or the date is unavailable."""
    ds = _open_store(run)
    d = _ts(date)
    if ds is None or d not in pd.DatetimeIndex(ds["time"].values):
        return None
    return ds[SURFACE_VARS].sel(time=d).load()


def _nearest(arr: xr.DataArray | xr.Dataset, **sel) -> xr.DataArray | xr.Dataset:
    return arr.sel(method="nearest", **sel)


def _triplet(run, date) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """prediction, target (NaN-filled when unavailable), climatology on one common grid."""
    pred = load_prediction(run, date)
    tgt = load_target(run, date)
    if tgt is None:
        tgt = xr.full_like(pred, np.nan)
    clim = load_climatology(run, date)
    pred = pred.drop_vars("time", errors="ignore")
    coords = {k: pred[k].values for k in ("depth", "lat", "lon")}  # identical grid everywhere
    tgt = tgt.drop_vars("time", errors="ignore").assign_coords(coords)
    return pred, tgt, clim.assign_coords(coords).where(np.isfinite(pred))


def load_profile(run, date, lat: float, lon: float) -> xr.Dataset:
    """Vertical profiles at the grid cell nearest to ``(lat, lon)``: a Dataset over ``depth``
    with ``predicted``, ``target`` (NaN if unavailable) and ``climatology``; the cell's
    coordinates are the scalar coords ``lat`` / ``lon``."""
    pred, tgt, clim = _triplet(run, date)
    ds = xr.Dataset({"predicted": pred, "target": tgt, "climatology": clim})
    ds = _nearest(ds, lat=lat, lon=lon)
    ds.attrs.update(date=str(_ts(date).date()), units="degC")
    return ds


def load_section(run, date, lat: float | None = None, lon: float | None = None) -> xr.Dataset:
    """Vertical section along a latitude (``lat=``, dims ``depth, lon``) or a longitude
    (``lon=``, dims ``depth, lat``) at the nearest grid line: ``predicted``, ``target``,
    ``climatology`` and ``difference`` (predicted - target, NaN if no target)."""
    if (lat is None) == (lon is None):
        raise ValueError("give exactly one of lat= or lon=")
    pred, tgt, clim = _triplet(run, date)
    ds = xr.Dataset(
        {"predicted": pred, "target": tgt, "climatology": clim, "difference": pred - tgt}
    )
    ds = _nearest(ds, lat=lat) if lat is not None else _nearest(ds, lon=lon)
    ds.attrs.update(date=str(_ts(date).date()), units="degC")
    return ds


# ----------------------------------------------------------------------------------------
# embeddings
# ----------------------------------------------------------------------------------------
def _embedding_path(run) -> Path:
    return _run(run) / "embeddings" / "embeddings.zarr"


def load_embeddings(run, date) -> xr.DataArray | None:
    """Embedding map ``(emb, y, x)`` for one day (cells of 4x4 grid pixels, with ``lat(y)`` /
    ``lon(x)`` coordinates); ``None`` if there is no embeddings store or the date is not in it."""
    p = _embedding_path(run)
    if not p.exists():
        return None
    ds = _open_zarr_cached(str(p), (_run(run) / RUN_META_FILE).stat().st_mtime)
    d = _ts(date)
    if d not in pd.DatetimeIndex(ds["time"].values):
        return None
    return ds["embedding"].sel(time=d).load()


def embedding_dates(run) -> pd.DatetimeIndex:
    """Days that have an embedding map (the test split); empty if there is no embeddings store."""
    p = _embedding_path(run)
    if not p.exists():
        return pd.DatetimeIndex([])
    ds = _open_zarr_cached(str(p), (_run(run) / RUN_META_FILE).stat().st_mtime)
    return pd.DatetimeIndex(ds["time"].values)


@lru_cache(maxsize=4)
def _fit_pca(path: str, mtime: float, n_fit_days: int, seed: int):
    ds = xr.open_zarr(path, consolidated=False)
    n_time = ds.sizes["time"]
    rng = np.random.default_rng(seed)
    days = np.sort(rng.choice(n_time, size=min(n_fit_days, n_time), replace=False))
    emb = ds["embedding"].isel(time=days).values  # (n, E, Y, X)
    n_emb = emb.shape[1]
    x = emb.transpose(0, 2, 3, 1).reshape(-1, n_emb).astype(np.float64)
    mean = x.mean(axis=0)
    _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
    k = min(MAX_PCA_COMPONENTS, vt.shape[0])  # the first three do not depend on k
    comps = vt[:k]
    flip = np.sign(comps[np.arange(k), np.abs(comps).argmax(axis=1)])
    comps = comps * np.where(flip == 0, 1.0, flip)[:, None]
    proj = (x - mean) @ comps.T
    lo, hi = np.percentile(proj, [2, 98], axis=0)
    evr = (s[:k] ** 2) / np.sum(s**2)
    return mean, comps, lo, np.maximum(hi, lo + 1e-9), evr


def embedding_pca_mean(run, n_fit_days: int = 24, seed: int = 0) -> np.ndarray | None:
    """Mean embedding vector ``(emb,)`` of the PCA day sample (``None`` without embeddings)."""
    p = _embedding_path(run)
    if not p.exists():
        return None
    return _fit_pca(str(p), p.stat().st_mtime, n_fit_days, seed)[0]


def embedding_pca_rgb(
    run, date, n_fit_days: int = 24, seed: int = 0, n_components: int = 3
) -> xr.DataArray | None:
    """PCA of the embedding map as an image ``(y, x, rgb)`` in [0, 1] (the dimension is called
    ``rgb`` for any ``n_components`` <= 8; 3 gives R, G, B = PC1..PC3).

    The PCA is fitted once per run on a random sample of ``n_fit_days`` days (cached), so every
    date is projected on the same axes and colours are comparable between days; each component is
    scaled to its 2nd-98th percentile over the sample. ``attrs['explained_variance_ratio']`` holds
    the variance fractions of the returned components. ``None`` if the day has no embedding.
    """
    emb = load_embeddings(run, date)
    if emb is None:
        return None
    p = _embedding_path(run)
    mean, comps, lo, hi, evr = _fit_pca(str(p), p.stat().st_mtime, n_fit_days, seed)
    n = max(1, min(int(n_components), len(comps)))
    comps, lo, hi, evr = comps[:n], lo[:n], hi[:n], evr[:n]
    x = emb.transpose("y", "x", "emb").values.astype(np.float64)
    rgb = np.clip(((x - mean) @ comps.T - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return xr.DataArray(
        rgb,
        dims=("y", "x", "rgb"),
        coords={
            "lat": emb["lat"],
            "lon": emb["lon"],
            "rgb": [f"pc{i + 1}" for i in range(n)],
        },
        name="embedding_pca_rgb",
        attrs={"explained_variance_ratio": [float(v) for v in evr], "n_fit_days": n_fit_days},
    )
