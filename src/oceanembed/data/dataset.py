"""Torch ``Dataset`` over the harmonised Zarr for one split.

Per day it returns

* ``x``    ``(12, H, W)`` float32 -- 7 standardised surface fields (NaN -> 0), ocean mask,
  sin/cos day-of-year, normalised lat, normalised lon (channel order in ``INPUT_CHANNELS``)
* ``y``    ``(15, H, W)`` float32 -- standardised temperature anomaly
  ``(temp - climatology) / anomaly_std[depth]`` (0 where masked)
* ``mask`` ``(15, H, W)`` bool -- ocean and above the sea floor
* ``sv``   ``(7, H, W)`` bool -- the surface fields that were really observed (not NaN-filled)
* ``t``    int64 -- days since 1970-01-01 (collate-friendly date) ; ``index`` -- position in split

The dataset only stores paths and small numpy arrays, so it pickles cleanly for ``spawn``
workers (the Zarr store is opened lazily inside each process). ``num_workers=0`` always works.

Reading one Zarr chunk per day through xarray is slow, so :meth:`OceanDataset.preload` reads the
whole split into RAM once (surface fields, targets and masks as float32 / bool; roughly 2 GB for
two years of the full 100x240 grid). After that ``__getitem__`` only assembles arrays.

For long periods that do not fit in RAM (11 training years are about 11 GB as float32),
:meth:`OceanDataset.use_cache` keeps the same four arrays in ``.npy`` files on disk (float16 or
float32), built once from the store and read through ``np.memmap`` by day. The files live under
``<data_root>/processed/cache/``, carry a ``meta.json`` written last (a partial build is never used)
and are rebuilt whenever they are missing or no longer match the store, the statistics or the split.
``make_dataset`` chooses between the two with ``train.cache`` (default ``ram``: nothing changes for
existing configs).
"""

from __future__ import annotations

import copy
import gc
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

from oceanembed.config import INPUT_GROUPS, SURFACE_VARS, Config
from oceanembed.data.stats import Stats, split_indices

INPUT_CHANNELS = [*SURFACE_VARS, "ocean_mask", "doy_sin", "doy_cos", "lat_norm", "lon_norm"]
N_INPUT_CHANNELS = len(INPUT_CHANNELS)  # 12
N_SURFACE = len(SURFACE_VARS)  # 7
CACHE_ARRAYS = ("surf", "sv", "y", "valid")
CACHE_VERSION = 1


class OceanDataset(Dataset):
    def __init__(self, zarr_path: str | Path, stats: Stats, start, end):
        self.zarr_path = str(zarr_path)
        self.stats = stats
        self._ds: xr.Dataset | None = None
        ds = self._open()
        self.time = ds["time"].values
        self.indices = split_indices(self.time, start, end)
        if len(self.indices) == 0:
            raise ValueError(f"no days between {start} and {end} in {self.zarr_path}")
        lat, lon = ds["lat"].values, ds["lon"].values
        self.lat, self.lon = lat, lon
        self.mask = ds["mask"].values.astype(bool)  # (D, H, W)
        self.depth = ds["depth"].values
        h, w = len(lat), len(lon)
        self.shape = (h, w)
        lat_plane = np.broadcast_to((2 * (lat - lat[0]) / (lat[-1] - lat[0]) - 1)[:, None], (h, w))
        lon_plane = np.broadcast_to((2 * (lon - lon[0]) / (lon[-1] - lon[0]) - 1)[None, :], (h, w))
        self._lat_plane = lat_plane.astype(np.float32)
        self._lon_plane = lon_plane.astype(np.float32)
        self._mean = stats.input_mean.astype(np.float32)[:, None, None]
        self._std = stats.input_std.astype(np.float32)[:, None, None]
        self._anom_std = stats.anom_std.astype(np.float32)[:, None, None]
        # day-of-year phase for every split item
        dates = pd.DatetimeIndex(self.time[self.indices])
        n = np.where(dates.is_leap_year, 366.0, 365.0)
        phi = 2 * np.pi * (dates.dayofyear.values - 1) / n
        self._sin = np.sin(phi).astype(np.float32)
        self._cos = np.cos(phi).astype(np.float32)
        self._cache: dict[str, np.ndarray] | None = None
        self._memmap = False  # the cache arrays are read-only memory maps (see use_cache)
        self._zero_surface: tuple[int, ...] = ()  # surface channels of dropped input groups
        self._ds = None  # do not keep an open handle (pickling / workers)

    # --- input groups ----------------------------------------------------------------
    def set_input_groups(self, groups) -> OceanDataset:
        """Keep only the surface inputs of ``groups``: every other surface channel is zeroed (the
        training mean, in standardised units) in the inputs ``x`` / ``gather`` return. The cached
        arrays are never changed, so one cache serves every input set. Returns ``self``."""
        keep = {v for g in groups for v in INPUT_GROUPS[g]}
        self._zero_surface = tuple(i for i, v in enumerate(SURFACE_VARS) if v not in keep)
        return self

    def _zero_surf(self, surf: np.ndarray, axis: int) -> np.ndarray:
        """``surf`` with the dropped channels (on ``axis``) set to 0 -- a copy, never in place."""
        if not self._zero_surface:
            return surf
        surf = np.array(surf, dtype=np.float32, copy=True)
        index = [slice(None)] * surf.ndim
        index[axis] = list(self._zero_surface)
        surf[tuple(index)] = 0.0
        return surf

    # --- lazy zarr handle -------------------------------------------------------------
    def _open(self) -> xr.Dataset:
        if self._ds is None:
            self._ds = xr.open_zarr(self.zarr_path, consolidated=False)
        return self._ds

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_ds"] = None
        return state

    # --- helpers -----------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.indices)

    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.time[self.indices])

    @property
    def preloaded(self) -> bool:
        """True once the arrays are cached (preload or use_cache)."""
        return self._cache is not None

    def climatology(self, index: int | slice | np.ndarray) -> np.ndarray:
        """Climatological temperature ``(D, H, W)`` for split item(s) (leading axis if many)."""
        dates = self.time[self.indices[index]]
        out = self.stats.climatology(np.atleast_1d(dates))
        return out[0] if np.ndim(dates) == 0 else out

    def _standardise_surface(self, raw: np.ndarray):
        """Raw surface ``(n,7,H,W)`` -> standardised ``surf`` (NaN -> 0) and observed flags."""
        surf = (raw.astype(np.float32) - self._mean) / self._std
        ok = np.isfinite(surf) & self.mask[0][None, None]
        surf = np.where(ok, surf, 0.0).astype(np.float32)
        return surf, ok

    def _standardise(self, raw: np.ndarray, temp: np.ndarray, times: np.ndarray):
        """Raw surface ``(n,7,H,W)`` + temperature ``(n,D,H,W)`` -> standardised arrays."""
        surf, ok = self._standardise_surface(raw)
        clim = self.stats.climatology(times)
        y = (temp.astype(np.float32) - clim) / self._anom_std
        valid = self.mask[None] & np.isfinite(y)
        y = np.where(valid, y, 0.0).astype(np.float32)
        return surf, ok, y, valid

    def preload(self, chunk_days: int = 32) -> OceanDataset:
        """Read the whole split into RAM (idempotent). Returns ``self``."""
        if self._cache is not None:
            return self
        ds = self._open()
        n = len(self.indices)
        lo, hi = int(self.indices[0]), int(self.indices[-1]) + 1
        if hi - lo != n:
            raise ValueError("split days are not contiguous in the store")
        h, w = self.shape
        d = len(self.depth)
        surf = np.empty((n, N_SURFACE, h, w), np.float32)
        sv = np.empty((n, N_SURFACE, h, w), bool)
        y = np.empty((n, d, h, w), np.float32)
        valid = np.empty((n, d, h, w), bool)
        for a in range(0, n, chunk_days):
            b = min(a + chunk_days, n)
            sl = slice(lo + a, lo + b)
            raw = np.stack([ds[v].isel(time=sl).values for v in SURFACE_VARS], axis=1)
            temp = ds["temp"].isel(time=sl).values
            surf[a:b], sv[a:b], y[a:b], valid[a:b] = self._standardise(raw, temp, self.time[sl])
        self._cache = {"surf": surf, "sv": sv, "y": y, "valid": valid}
        self._ds = None
        return self

    # --- on-disk cache ---------------------------------------------------------------
    def _cache_meta(self, dtype: str) -> dict:
        """What a cache must match to be reused (everything its content depends on)."""
        return {
            "version": CACHE_VERSION,
            "dtype": dtype,
            "n_days": int(len(self.indices)),
            "first_day": str(self.time[self.indices[0]].astype("datetime64[D]")),
            "last_day": str(self.time[self.indices[-1]].astype("datetime64[D]")),
            "grid_shape": [len(self.depth), *self.shape],
            "stats": self.stats.fingerprint(),
            "store": self.zarr_path,
            "store_days": int(len(self.time)),
        }

    def cache_ready(self, cache_dir: Path, dtype: str = "float16") -> bool:
        """True when ``cache_dir`` holds a complete cache that matches this dataset."""
        f = Path(cache_dir) / "meta.json"
        if not f.exists():
            return False
        try:
            meta = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        want = self._cache_meta(dtype)
        files = all((Path(cache_dir) / f"{k}.npy").exists() for k in CACHE_ARRAYS)
        return (
            files
            and meta.get("complete") is True
            and all(meta.get(k) == v for k, v in want.items())
        )

    def build_cache(self, cache_dir: Path, dtype: str = "float16", chunk_days: int = 32) -> dict:
        """(Re)build the on-disk cache in ``cache_dir`` from the store and return its metadata.

        The files are written to ``<cache_dir>.tmp`` and the folder is renamed only after
        ``meta.json`` (with the round-off the chosen dtype introduced) is complete."""
        if dtype not in ("float16", "float32"):
            raise ValueError("cache dtype must be float16 or float32")
        cache_dir = Path(cache_dir)
        tmp = cache_dir.with_name(cache_dir.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        ds = self._open()
        n = len(self.indices)
        lo, hi = int(self.indices[0]), int(self.indices[-1]) + 1
        if hi - lo != n:
            raise ValueError("split days are not contiguous in the store")
        h, w = self.shape
        d = len(self.depth)
        fdt = np.dtype(dtype)
        shapes = {
            "surf": ((n, N_SURFACE, h, w), fdt),
            "sv": ((n, N_SURFACE, h, w), np.dtype(bool)),
            "y": ((n, d, h, w), fdt),
            "valid": ((n, d, h, w), np.dtype(bool)),
        }
        mm = {
            k: np.lib.format.open_memmap(tmp / f"{k}.npy", mode="w+", dtype=dt, shape=shape)
            for k, (shape, dt) in shapes.items()
        }
        sd = self._anom_std[None]  # (1, D, 1, 1)
        err = {"surf_max": 0.0, "surf_sq": 0.0, "surf_n": 0, "y_max": 0.0, "y_sq": 0.0, "y_n": 0}
        for a in range(0, n, chunk_days):
            b = min(a + chunk_days, n)
            sl = slice(lo + a, lo + b)
            raw = np.stack([ds[v].isel(time=sl).values for v in SURFACE_VARS], axis=1)
            temp = ds["temp"].isel(time=sl).values
            surf, sv, y, valid = self._standardise(raw, temp, self.time[sl])
            mm["surf"][a:b], mm["sv"][a:b] = surf, sv
            mm["y"][a:b], mm["valid"][a:b] = y, valid
            if fdt != np.float32:  # round-off of the stored values
                es = np.abs(mm["surf"][a:b].astype(np.float32) - surf)  # standardised units
                ey = np.abs(mm["y"][a:b].astype(np.float32) - y) * sd  # degC
                ey = np.where(valid, ey, 0.0)
                err["surf_max"] = max(err["surf_max"], float(es.max()))
                err["surf_sq"] += float((es.astype(np.float64) ** 2).sum())
                err["surf_n"] += es.size
                err["y_max"] = max(err["y_max"], float(ey.max()))
                err["y_sq"] += float((ey.astype(np.float64) ** 2).sum())
                err["y_n"] += int(valid.sum())
        for m in mm.values():
            m.flush()
        del m, mm  # release every handle: a directory with open files cannot be renamed on Windows
        gc.collect()
        roundoff = None
        if fdt != np.float32:
            roundoff = {
                "surface_max_abs_standardised": err["surf_max"],
                "surface_rms_standardised": (err["surf_sq"] / max(err["surf_n"], 1)) ** 0.5,
                "temperature_max_abs_degC": err["y_max"],
                "temperature_rms_degC": (err["y_sq"] / max(err["y_n"], 1)) ** 0.5,
            }
        meta = {**self._cache_meta(dtype), "complete": True, "roundoff": roundoff}
        (tmp / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        shutil.rmtree(cache_dir, ignore_errors=True)
        os.replace(tmp, cache_dir)
        return meta

    def use_cache(
        self, cache_dir: Path, dtype: str = "float16", chunk_days: int = 32
    ) -> OceanDataset:
        """Like :meth:`preload`, but the four arrays are read-only memory maps of ``.npy`` files in
        ``cache_dir`` (built first when missing or stale), so RAM no longer limits the period.
        ``float16`` halves the size; its round-off is recorded in ``meta.json``. Idempotent."""
        if self._cache is not None:
            return self
        cache_dir = Path(cache_dir)
        if not self.cache_ready(cache_dir, dtype):
            self.build_cache(cache_dir, dtype, chunk_days)
        self._cache = {k: np.load(cache_dir / f"{k}.npy", mmap_mode="r") for k in CACHE_ARRAYS}
        self._memmap = True
        self._ds = None
        return self

    def arrays(self) -> dict[str, np.ndarray]:
        """The cached ``surf (n,7,H,W)``, ``sv``, ``y (n,15,H,W)``, ``valid`` arrays (in RAM after
        :meth:`preload`, read-only memory maps after :meth:`use_cache`)."""
        if self._cache is None:
            raise RuntimeError("call preload() or use_cache() first")
        return self._cache

    @property
    def memory_mapped(self) -> bool:
        return self._memmap

    def tail(self, n_days: int) -> OceanDataset:
        """A view of the last ``n_days`` days of this cached dataset (shares the arrays)."""
        if self._cache is None:
            raise RuntimeError("call preload() or use_cache() first")
        n_days = min(int(n_days), len(self))
        out = copy.copy(self)
        out._ds = None
        out.indices = self.indices[-n_days:]
        out._sin, out._cos = self._sin[-n_days:], self._cos[-n_days:]
        out._cache = {k: v[-n_days:] for k, v in self._cache.items()}
        return out

    def gather(self, t: np.ndarray, p: np.ndarray, names: tuple[str, ...] = CACHE_ARRAYS):
        """Values at (split day ``t``, flat pixel ``p``) pairs from the cached arrays ``names``
        (default ``surf (n,7)``, ``sv (n,7)``, ``y (n,D)``, ``valid (n,D)``), as a tuple. A RAM
        cache is indexed directly; a memory map is read day by day (each day's block once) instead
        of one scattered read per sample."""
        out = self._gather(t, p, names)
        if self._zero_surface and "surf" in names:
            i = names.index("surf")
            out = (*out[:i], self._zero_surf(out[i], axis=1), *out[i + 1 :])
        return out

    def _gather(self, t: np.ndarray, p: np.ndarray, names: tuple[str, ...]):
        a = self.arrays()
        n_days, (h, w) = len(self), self.shape
        if not self._memmap:
            return tuple(a[k].reshape(n_days, a[k].shape[1], h * w)[t, :, p] for k in names)
        out = {
            k: np.empty(
                (len(t), a[k].shape[1]), np.float32 if a[k].dtype.kind == "f" else a[k].dtype
            )
            for k in names
        }
        order = np.argsort(t, kind="stable")
        ts = t[order]
        for grp in np.split(np.arange(len(ts)), np.flatnonzero(np.diff(ts)) + 1):
            if len(grp) == 0:
                continue
            day, sel = int(ts[grp[0]]), order[grp]
            for k in names:
                block = np.asarray(a[k][day]).reshape(a[k].shape[1], h * w)
                out[k][sel] = block[:, p[sel]].T
        return tuple(out[k] for k in names)

    def _day(self, i: int):
        """Standardised ``surf (7,H,W)``, ``sv``, ``y``, ``valid`` for split item ``i``."""
        if self._cache is not None:
            c = self._cache
            if self._memmap:  # copy one day out of the file (float16 -> float32)
                return (
                    np.array(c["surf"][i], dtype=np.float32),
                    np.array(c["sv"][i]),
                    np.array(c["y"][i], dtype=np.float32),
                    np.array(c["valid"][i]),
                )
            return c["surf"][i], c["sv"][i], c["y"][i], c["valid"][i]
        ds = self._open()
        ti = int(self.indices[i])
        raw = np.stack([ds[v].isel(time=ti).values for v in SURFACE_VARS])[None]
        temp = ds["temp"].isel(time=ti).values[None]
        surf, sv, y, valid = self._standardise(raw, temp, self.time[ti : ti + 1])
        return surf[0], sv[0], y[0], valid[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        if i < 0:
            i += len(self)
        h, w = self.shape
        surf, sv, y, valid = self._day(i)
        planes = np.empty((5, h, w), np.float32)
        planes[0] = self.mask[0]
        planes[1] = self._sin[i]
        planes[2] = self._cos[i]
        planes[3] = self._lat_plane
        planes[4] = self._lon_plane
        x = np.concatenate([self._zero_surf(surf, axis=0), planes])
        ti = int(self.indices[i])
        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(np.ascontiguousarray(y)),
            "mask": torch.from_numpy(np.ascontiguousarray(valid)),
            "sv": torch.from_numpy(np.ascontiguousarray(sv)),
            "t": torch.tensor(int(self.time[ti].astype("datetime64[D]").astype(np.int64))),
            "index": torch.tensor(i),
        }

    # --- de-normalisation --------------------------------------------------------------
    def denormalize(self, y, t) -> np.ndarray:
        """Standardised anomaly -> temperature in degC.

        ``y`` is ``(B, D, H, W)`` (or ``(D, H, W)``), ``t`` the matching ``t`` value(s) from the
        batch (days since 1970-01-01). Returns a numpy array shaped like ``y``.
        """
        y = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
        tt = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
        single = y.ndim == 3
        if single:
            y = y[None]
        dates = np.atleast_1d(tt).astype("datetime64[D]").astype("datetime64[ns]")
        clim = self.stats.climatology(dates)
        out = clim + y * self._anom_std[None]
        return out[0] if single else out


class SurfaceOnlyDataset(OceanDataset):
    """Model inputs only -- the ``temp`` target is never read.

    Used for prediction / evaluation on any day present in the harmonised store, including days
    where the target product is missing (the store then holds NaN). :meth:`batch` reads a
    contiguous block of days at once; ``__getitem__`` returns ``x``, ``sv``, ``t`` and ``index``.
    """

    def batch(self, first: int, stop: int) -> dict[str, torch.Tensor]:
        """Split items ``first .. stop-1`` as a batch (``x``, ``sv``, ``t``, ``index``)."""
        idx = np.arange(first, stop)
        ti = self.indices[idx]
        if len(ti) and ti[-1] - ti[0] + 1 != len(ti):
            raise ValueError("split days are not contiguous in the store")
        ds = self._open()
        sl = slice(int(ti[0]), int(ti[-1]) + 1)
        raw = np.stack([ds[v].isel(time=sl).values for v in SURFACE_VARS], axis=1)
        surf, ok = self._standardise_surface(raw)
        surf = self._zero_surf(surf, axis=1)
        n, (h, w) = len(idx), self.shape
        planes = np.empty((n, 5, h, w), np.float32)
        planes[:, 0] = self.mask[0]
        planes[:, 1] = self._sin[idx][:, None, None]
        planes[:, 2] = self._cos[idx][:, None, None]
        planes[:, 3] = self._lat_plane
        planes[:, 4] = self._lon_plane
        t = self.time[ti].astype("datetime64[D]").astype(np.int64)
        return {
            "x": torch.from_numpy(np.concatenate([surf, planes], axis=1)),
            "sv": torch.from_numpy(ok),
            "t": torch.from_numpy(t),
            "index": torch.from_numpy(idx),
        }

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        if i < 0:
            i += len(self)
        out = self.batch(i, i + 1)
        return {k: v[0] for k, v in out.items()}

    def preload(self, chunk_days: int = 32) -> OceanDataset:  # pragma: no cover
        raise NotImplementedError("SurfaceOnlyDataset reads inputs on demand")


def _stats_for(cfg: Config, stats: Stats | None = None) -> Stats:
    """The run's statistics (or the given ``stats``); a run that shares another run's store must
    have been fitted on the same train period."""
    if stats is None:
        stats = Stats.load(cfg.stats_path)
    if cfg.shares_store:
        want = (str(cfg.split.train.start), str(cfg.split.train.end))
        if (stats.train_start, stats.train_end) != want:
            raise ValueError(
                f"the statistics of store '{cfg.store_name}' were fitted on "
                f"{stats.train_start} .. {stats.train_end}, but this config trains on "
                f"{want[0]} .. {want[1]}"
            )
    return stats


def make_surface_dataset(
    cfg: Config, start, end, *, stats: Stats | None = None, mask: np.ndarray | None = None
) -> SurfaceOnlyDataset:
    """Inputs-only dataset for an arbitrary date range inside the harmonised store, with the
    config's input groups applied. ``stats`` / ``mask`` replace the run's statistics file and the
    store's ocean mask (prediction from released weights)."""
    ds = SurfaceOnlyDataset(cfg.zarr_path, _stats_for(cfg, stats), start, end)
    if mask is not None:
        ds.mask = np.asarray(mask, dtype=bool)
    return ds.set_input_groups(cfg.model.input_groups)


def make_dataset(cfg: Config, split: str, preload: bool = False) -> OceanDataset:
    """Dataset for ``'train' | 'val' | 'test'`` using the config's date ranges and saved stats."""
    r = cfg.split.get(split)
    ds = OceanDataset(cfg.zarr_path, _stats_for(cfg), r.start, r.end)
    ds.set_input_groups(cfg.model.input_groups)
    if not preload:
        return ds
    if cfg.train.cache == "memmap":
        return ds.use_cache(cache_dir(cfg, split), cfg.train.cache_dtype)
    return ds.preload()


def cache_dir(cfg: Config, split: str) -> Path:
    """Folder of a split's on-disk cache: ``<data_root>/processed/cache/<store>/<split>_...``
    (the store is the run itself unless ``paths.store`` points at another run's data)."""
    r = cfg.split.get(split)
    name = f"{split}_{r.start:%Y%m%d}_{r.end:%Y%m%d}_{cfg.train.cache_dtype}"
    return cfg.processed_root / "cache" / cfg.store_name / name
