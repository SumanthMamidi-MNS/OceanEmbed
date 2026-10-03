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
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

from oceanembed.config import SURFACE_VARS, Config
from oceanembed.data.stats import Stats, split_indices

INPUT_CHANNELS = [*SURFACE_VARS, "ocean_mask", "doy_sin", "doy_cos", "lat_norm", "lon_norm"]
N_INPUT_CHANNELS = len(INPUT_CHANNELS)  # 12
N_SURFACE = len(SURFACE_VARS)  # 7


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
        self._ds = None  # do not keep an open handle (pickling / workers)

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

    def arrays(self) -> dict[str, np.ndarray]:
        """The preloaded ``surf (n,7,H,W)``, ``sv``, ``y (n,15,H,W)``, ``valid`` arrays."""
        if self._cache is None:
            raise RuntimeError("call preload() first")
        return self._cache

    def _day(self, i: int):
        """Standardised ``surf (7,H,W)``, ``sv``, ``y``, ``valid`` for split item ``i``."""
        if self._cache is not None:
            c = self._cache
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
        x = np.concatenate([surf, planes])
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


def make_surface_dataset(cfg: Config, start, end) -> SurfaceOnlyDataset:
    """Inputs-only dataset for an arbitrary date range inside the harmonised store."""
    return SurfaceOnlyDataset(cfg.zarr_path, Stats.load(cfg.stats_path), start, end)


def make_dataset(cfg: Config, split: str, preload: bool = False) -> OceanDataset:
    """Dataset for ``'train' | 'val' | 'test'`` using the config's date ranges and saved stats."""
    stats = Stats.load(cfg.stats_path)
    r = cfg.split.get(split)
    ds = OceanDataset(cfg.zarr_path, stats, r.start, r.end)
    return ds.preload() if preload else ds
