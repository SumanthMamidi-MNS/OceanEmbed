"""Train-split statistics: input normalisation, harmonic climatology, anomaly std.

Everything is computed on the **train** date range only and in a streaming fashion (a few days
at a time; the harmonic fit accumulates normal equations), so the full cube is never loaded.

Climatology model, evaluable for any date (leap years included)::

    T_clim(x, z, date) = c0 + c1 cos(phi) + c2 sin(phi) + c3 cos(2 phi) + c4 sin(2 phi)
    phi = 2 pi * (date - Jan 1 of its year) / (days in that year)

When the train period is short the number of harmonics is reduced (< 180 days: mean only,
< 548 days: mean + annual) because the seasonal terms are not identifiable; unused coefficients
are stored as zero.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from oceanembed.config import SURFACE_VARS, Config
from oceanembed.data.harmonize import open_harmonized

N_HARMONICS = 5  # [1, cos, sin, cos2, sin2]
MIN_POINT_DAYS = 30


def harmonic_design(dates) -> np.ndarray:
    """Design matrix ``(n, 5)`` for the given dates (any date-like sequence)."""
    d = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
    year_start = pd.to_datetime(d.year.astype(str) + "-01-01")
    year_len = np.where(d.is_leap_year, 366.0, 365.0)
    phi = 2.0 * np.pi * ((d - year_start) / pd.Timedelta(days=1)) / year_len
    phi = np.asarray(phi, dtype=np.float64)
    return np.stack(
        [np.ones_like(phi), np.cos(phi), np.sin(phi), np.cos(2 * phi), np.sin(2 * phi)], axis=1
    )


def n_terms_for_span(n_days: int) -> int:
    if n_days < 180:
        return 1
    if n_days < 548:
        return 3
    return 5


@dataclass
class Stats:
    channels: list[str]
    input_mean: np.ndarray  # (C,)
    input_std: np.ndarray  # (C,)
    clim_coef: np.ndarray  # (5, D, H, W) float32
    anom_std: np.ndarray  # (D,)
    train_start: str
    train_end: str
    n_train_days: int
    n_terms: int

    def fingerprint(self) -> str:
        """SHA-1 of everything the normalisation and the climatology depend on (cache validity)."""
        h = hashlib.sha1()
        for a in (self.input_mean, self.input_std, self.clim_coef, self.anom_std):
            h.update(np.ascontiguousarray(a, dtype=np.float64 if a.ndim == 1 else np.float32))
        h.update(f"{self.train_start}|{self.train_end}|{self.n_terms}".encode())
        return h.hexdigest()

    def climatology(self, dates) -> np.ndarray:
        """Climatological temperature ``(n, D, H, W)`` float32 for the given dates."""
        x = harmonic_design(dates).astype(np.float32)
        return np.tensordot(x, self.clim_coef, axes=([1], [0])).astype(np.float32)

    def to_dataset(self) -> xr.Dataset:
        return xr.Dataset(
            {
                "input_mean": (("channel",), self.input_mean.astype(np.float64)),
                "input_std": (("channel",), self.input_std.astype(np.float64)),
                "clim_coef": (("harmonic", "depth", "lat", "lon"), self.clim_coef),
                "anom_std": (("depth",), self.anom_std.astype(np.float64)),
            },
            coords={"channel": np.array(self.channels, dtype="U8")},
            attrs={
                "title": "OceanEmbed train-split statistics",
                "train_start": self.train_start,
                "train_end": self.train_end,
                "n_train_days": self.n_train_days,
                "n_harmonic_terms": self.n_terms,
            },
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ds = self.to_dataset()
        enc = {"clim_coef": {"zlib": True, "complevel": 3}}
        ds.to_netcdf(path, encoding=enc)

    @classmethod
    def load(cls, path: Path) -> Stats:
        with xr.open_dataset(path) as ds:
            return cls(
                channels=[str(c) for c in ds["channel"].values],
                input_mean=ds["input_mean"].values.copy(),
                input_std=ds["input_std"].values.copy(),
                clim_coef=ds["clim_coef"].values.astype(np.float32),
                anom_std=ds["anom_std"].values.copy(),
                train_start=str(ds.attrs["train_start"]),
                train_end=str(ds.attrs["train_end"]),
                n_train_days=int(ds.attrs["n_train_days"]),
                n_terms=int(ds.attrs["n_harmonic_terms"]),
            )


def split_indices(time: np.ndarray, start, end) -> np.ndarray:
    t = pd.DatetimeIndex(time)
    return np.nonzero((t >= pd.Timestamp(start)) & (t <= pd.Timestamp(end)))[0]


def compute_stats(cfg: Config, chunk_days: int = 16, progress: bool = True) -> Stats:
    ds = open_harmonized(cfg)
    time = ds["time"].values
    idx = split_indices(time, cfg.split.train.start, cfg.split.train.end)
    if len(idx) == 0:
        raise ValueError("no days of the train split found in the harmonised store")
    mask = ds["mask"].values  # (D, H, W)
    ocean0 = mask[0]
    n_depth, h, w = mask.shape
    p = n_depth * h * w
    k = n_terms_for_span(len(idx))

    s1 = np.zeros(len(SURFACE_VARS))
    s2 = np.zeros(len(SURFACE_VARS))
    cnt = np.zeros(len(SURFACE_VARS))
    xty = np.zeros((N_HARMONICS, p))
    xtx = np.zeros((N_HARMONICS * N_HARMONICS, p))
    npts = np.zeros(p)
    flat_mask = mask.reshape(-1)

    starts = list(range(0, len(idx), chunk_days))
    for s in tqdm(starts, desc="stats pass 1", disable=not progress):
        sel = idx[s : s + chunk_days]
        sl = slice(int(sel[0]), int(sel[-1]) + 1)
        for ci, v in enumerate(SURFACE_VARS):
            a = ds[v].isel(time=sl).values.astype(np.float64)
            ok = np.isfinite(a) & ocean0[None]
            vals = a[ok]
            s1[ci] += vals.sum()
            s2[ci] += (vals**2).sum()
            cnt[ci] += vals.size
        temp = ds["temp"].isel(time=sl).values.reshape(len(sel), -1)
        valid = np.isfinite(temp) & flat_mask[None]
        y = np.where(valid, temp, 0.0).astype(np.float64)
        vf = valid.astype(np.float64)
        x = harmonic_design(time[sel])
        x[:, k:] = 0.0
        xty += x.T @ y
        outer = (x[:, :, None] * x[:, None, :]).reshape(len(sel), -1)
        xtx += outer.T @ vf
        npts += vf.sum(axis=0)

    mean = s1 / np.maximum(cnt, 1)
    std = np.sqrt(np.maximum(s2 / np.maximum(cnt, 1) - mean**2, 0.0))
    std = np.where(std > 1e-12, std, 1.0)

    # solve the (k x k) normal equations per point
    coef = np.full((N_HARMONICS, p), np.nan)
    good = npts >= MIN_POINT_DAYS
    a = xtx.reshape(N_HARMONICS, N_HARMONICS, p)[:k, :k][:, :, good].transpose(2, 0, 1)
    b = xty[:k][:, good].T[:, :, None]
    a = a + 1e-9 * np.eye(k)[None] * np.maximum(a[:, :1, :1], 1.0)
    sol = np.linalg.solve(a, b)[:, :, 0]  # (n_good, k)
    cf = np.zeros((N_HARMONICS, int(good.sum())))
    cf[:k] = sol.T
    coef[:, good] = cf
    clim_coef = coef.reshape(N_HARMONICS, n_depth, h, w).astype(np.float32)

    # pass 2: anomaly std per depth
    ss = np.zeros(n_depth)
    nn = np.zeros(n_depth)
    for s in tqdm(starts, desc="stats pass 2", disable=not progress):
        sel = idx[s : s + chunk_days]
        sl = slice(int(sel[0]), int(sel[-1]) + 1)
        temp = ds["temp"].isel(time=sl).values
        xd = harmonic_design(time[sel]).astype(np.float32)
        clim = np.tensordot(xd, clim_coef, axes=([1], [0]))
        anom = temp - clim
        ok = np.isfinite(anom) & mask[None]
        anom = np.where(ok, anom, 0.0).astype(np.float64)
        ss += (anom**2).sum(axis=(0, 2, 3))
        nn += ok.sum(axis=(0, 2, 3))
    anom_std = np.sqrt(ss / np.maximum(nn, 1))
    anom_std = np.where(anom_std > 1e-6, anom_std, 1.0)

    stats = Stats(
        channels=list(SURFACE_VARS),
        input_mean=mean,
        input_std=std,
        clim_coef=clim_coef,
        anom_std=anom_std,
        train_start=str(cfg.split.train.start),
        train_end=str(cfg.split.train.end),
        n_train_days=int(len(idx)),
        n_terms=k,
    )
    stats.save(cfg.stats_path)
    return stats
