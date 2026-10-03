"""Baselines: harmonic climatology (anomaly = 0) and pixel-wise ridge regression.

Both are *predictors* in the sense of :mod:`oceanembed.infer.predict`: they map the model input
``x (B,12,H,W)`` to a standardised anomaly ``(B,15,H,W)``.

Ridge features per pixel (11): the 7 standardised surface values, sin/cos day-of-year and the
normalised lat / lon (dataset channels 0-6 and 8-11). One ridge per depth, fitted on a random
subsample of train-split ocean points that are valid at that depth (so points below the sea floor
never enter that depth's fit); the ocean-mask channel is constant on ocean points and unused.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import Ridge

from oceanembed.config import Config
from oceanembed.data.dataset import N_SURFACE, OceanDataset

FEATURE_CHANNELS = [*range(N_SURFACE), N_SURFACE + 1, N_SURFACE + 2, N_SURFACE + 3, N_SURFACE + 4]
N_FEATURES = len(FEATURE_CHANNELS)  # 11
MIN_FIT_POINTS = 50
RIDGE_FILE = "ridge.joblib"


class ClimatologyBaseline:
    """Predicts the climatology: a zero standardised anomaly everywhere."""

    def __init__(self, n_depths: int = 15):
        self.n_depths = n_depths

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        return torch.zeros(b, self.n_depths, h, w, dtype=torch.float32, device=x.device)


class RidgeBaseline:
    def __init__(self, coef: np.ndarray, intercept: np.ndarray, meta: dict | None = None):
        self.coef = np.asarray(coef, dtype=np.float32)  # (D, F)
        self.intercept = np.asarray(intercept, dtype=np.float32)  # (D,)
        self.meta = meta or {}

    # --- inference ----------------------------------------------------------------------
    def predict_anomaly(self, x: torch.Tensor) -> torch.Tensor:
        """``(B,12,H,W)`` torch -> ``(B,15,H,W)`` standardised anomaly (same device)."""
        f = x[:, FEATURE_CHANNELS]
        w = torch.from_numpy(self.coef).to(x.device, torch.float32)
        b = torch.from_numpy(self.intercept).to(x.device, torch.float32)
        return torch.einsum("bfhw,df->bdhw", f.float(), w) + b[None, :, None, None]

    __call__ = predict_anomaly

    def predict(self, x) -> np.ndarray:
        """Numpy convenience: ``(12,H,W)`` -> ``(15,H,W)`` or ``(B,12,H,W)`` -> ``(B,15,H,W)``."""
        t = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        single = t.ndim == 3
        out = self.predict_anomaly(t[None] if single else t).numpy()
        return out[0] if single else out

    # --- persistence --------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"coef": self.coef, "intercept": self.intercept, "meta": self.meta}, path, compress=3
        )

    @classmethod
    def load(cls, path: Path) -> RidgeBaseline:
        d = joblib.load(path)
        return cls(d["coef"], d["intercept"], d.get("meta"))


def sample_points(ds: OceanDataset, max_points: int, seed: int):
    """Random (day, ocean pixel) samples from a preloaded dataset.

    Returns ``features (n, 11)``, ``target (n, D)`` and ``valid (n, D)`` for samples whose 7
    surface values were all observed.
    """
    n_days = len(ds)
    ocean = np.flatnonzero(ds.mask[0].reshape(-1))
    rng = np.random.default_rng(seed)
    n = min(max_points, n_days * len(ocean))
    t = rng.integers(0, n_days, n)
    p = ocean[rng.integers(0, len(ocean), n)]
    surf, sv, y, valid = ds.gather(t, p)  # (n, 7), (n, 7), (n, D), (n, D)
    ok = sv.all(axis=1)
    lat = ds._lat_plane.reshape(-1)[p]
    lon = ds._lon_plane.reshape(-1)[p]
    feats = np.column_stack([surf, ds._sin[t], ds._cos[t], lat, lon]).astype(np.float32)
    return feats[ok], y[ok], valid[ok]


def fit_ridge(cfg: Config, train_ds: OceanDataset) -> RidgeBaseline:
    feats, y, valid = sample_points(train_ds, cfg.baseline.ridge_max_points, cfg.baseline.seed)
    n_depth = y.shape[1]
    coef = np.zeros((n_depth, N_FEATURES), np.float32)
    intercept = np.zeros(n_depth, np.float32)
    n_fit = []
    for d in range(n_depth):
        v = valid[:, d]
        n_fit.append(int(v.sum()))
        if v.sum() < MIN_FIT_POINTS:
            continue
        r = Ridge(alpha=cfg.baseline.ridge_alpha).fit(feats[v], y[v, d])
        coef[d], intercept[d] = r.coef_, r.intercept_
    meta = {
        "alpha": cfg.baseline.ridge_alpha,
        "n_samples": int(len(feats)),
        "n_fit_per_depth": n_fit,
        "features": [
            "sst",
            "sss",
            "sla",
            "uo",
            "vo",
            "uw",
            "vw",
            "doy_sin",
            "doy_cos",
            "lat_norm",
            "lon_norm",
        ],
    }
    return RidgeBaseline(coef, intercept, meta)


@torch.no_grad()
def rmse_per_depth(predictor, ds: OceanDataset, batch_size: int = 16) -> np.ndarray:
    """RMSE in degC per depth of a predictor over a (preloaded) dataset, valid points only.

    The climatology cancels in ``pred - truth``, so the error is the anomaly error times the
    per-depth anomaly std.
    """
    a = ds.arrays()
    sd = torch.from_numpy(ds.stats.anom_std.astype(np.float32))[None, :, None, None]
    n_depth = sd.shape[1]
    se = np.zeros(n_depth)
    cnt = np.zeros(n_depth)
    for i in range(0, len(ds), batch_size):
        batch = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
        x = torch.stack([b["x"] for b in batch])
        y = torch.from_numpy(np.asarray(a["y"][i : i + len(batch)], dtype=np.float32))
        m = torch.from_numpy(np.array(a["valid"][i : i + len(batch)]))
        err = torch.where(m, (predictor(x).float() - y) * sd, 0.0).double()
        se += (err**2).sum(dim=(0, 2, 3)).numpy()
        cnt += m.sum(dim=(0, 2, 3)).numpy()
    return np.sqrt(se / np.maximum(cnt, 1))
