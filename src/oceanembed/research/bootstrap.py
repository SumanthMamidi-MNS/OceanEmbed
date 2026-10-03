"""Moving-block bootstrap over test days, computed from per-day sufficient statistics.

Everything here works on ``daily`` arrays of shape ``(F, T, D)``: the eight additive fields of
:data:`oceanembed.eval.metrics.SUM_FIELDS` (``n, sx, sy, sxx, syy, sxy, sae, se2``) summed over the
grid points of one region, for ``T`` days and ``D`` depths. A bootstrap replicate is a *multiset of
days*: it is stored as a count vector over days, so the resampled sums of any method are one matrix
product ``counts @ daily`` and all methods, depths, regions and metrics see the same replicates.
That is what makes paired differences exact: the same resampled days enter both methods.

Block design: overlapping, non-circular blocks of ``block_length`` consecutive days (moving-block
bootstrap); ``ceil(T / L)`` blocks are drawn with their start uniform on ``0 .. T - L``,
concatenated and truncated to ``T`` days.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from oceanembed.eval.metrics import SUM_FIELDS, metrics_from_sums, skill_score

N_FIELDS = len(SUM_FIELDS)
METRIC_NAMES = ("rmse", "bias", "mae", "corr_raw", "corr_anom", "skill_vs_clim")
CI_LEVEL = 0.95


# ----------------------------------------------------------------------------------------
# columns: the depths, the pooled thermocline range and everything pooled
# ----------------------------------------------------------------------------------------
def column_selector(depths, pooled_range=(50.0, 200.0)) -> tuple[list[str], np.ndarray]:
    """Column names and a ``(K, D)`` 0/1 selector: one column per depth, then the pooled range
    (``pooled_<lo>_<hi>m``) and ``overall`` (all depths)."""
    d = np.asarray(depths, dtype=float)
    names = [f"{z:g}" for z in d]
    pooled = ((d >= pooled_range[0]) & (d <= pooled_range[1])).astype(float)
    sel = np.vstack([np.eye(len(d)), pooled[None], np.ones((1, len(d)))])
    names += [f"pooled_{pooled_range[0]:g}_{pooled_range[1]:g}m", "overall"]
    return names, sel


# ----------------------------------------------------------------------------------------
# resampling
# ----------------------------------------------------------------------------------------
def block_counts(n_days: int, block_length: int, n_boot: int, seed: int) -> np.ndarray:
    """``(n_boot, n_days)`` counts: how often each day appears in each moving-block replicate."""
    length = max(1, min(int(block_length), n_days))
    n_blocks = math.ceil(n_days / length)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_days - length + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(length)).reshape(n_boot, -1)[:, :n_days]
    flat = (idx + np.arange(n_boot)[:, None] * n_days).ravel()
    return np.bincount(flat, minlength=n_boot * n_days).reshape(n_boot, n_days).astype(np.float64)


def column_sums(daily: np.ndarray, weights: np.ndarray, sel: np.ndarray) -> dict[str, np.ndarray]:
    """Weighted sums over days, then over the depths of each column: ``{field: (B, K)}``.

    ``weights`` is ``(B, T)`` (bootstrap counts) or ``(1, T)`` of ones for the point estimate."""
    per_depth = np.einsum("bt,ftd->bfd", weights, daily)  # (B, F, D)
    cols = np.einsum("bfd,kd->bfk", per_depth, sel)  # (B, F, K)
    return {f: cols[:, i, :] for i, f in enumerate(SUM_FIELDS)}


def metrics_from_daily(
    raw: np.ndarray, anom: np.ndarray, clim_raw: np.ndarray, sel: np.ndarray, weights: np.ndarray
) -> dict[str, np.ndarray]:
    """All metrics ``{name: (B, K)}`` of one method from its per-day sums.

    ``raw`` are the sums of (prediction, reference) temperatures, ``anom`` those with the
    climatology removed from both, ``clim_raw`` the raw sums of the climatology method (the
    skill reference). ``mse`` is returned as well."""
    r = metrics_from_sums(column_sums(raw, weights, sel))
    a = metrics_from_sums(column_sums(anom, weights, sel))
    c = metrics_from_sums(column_sums(clim_raw, weights, sel))
    return {
        "rmse": r["rmse"],
        "mse": r["mse"],
        "bias": r["bias"],
        "mae": r["mae"],
        "corr_raw": r["corr"],
        "corr_anom": a["corr"],
        "skill_vs_clim": skill_score(r["mse"], c["mse"]),
    }


def percentile_ci(dist: np.ndarray, level: float = CI_LEVEL) -> tuple[np.ndarray, np.ndarray]:
    """Percentile interval of a ``(B, ...)`` bootstrap distribution (NaN replicates ignored)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns (e.g. climatology corr)
        lo, hi = np.nanpercentile(dist, [50 * (1 - level), 50 * (1 + level)], axis=0)
    return lo, hi


def paired_difference(boot_a: np.ndarray, boot_b: np.ndarray, point_a, point_b) -> dict:
    """Bootstrap summary of ``A - B`` for ``(B, K)`` distributions taken on the *same* replicates.

    Returns arrays over ``K``: ``diff`` (point estimate), ``ci_lo``, ``ci_hi``, ``excludes_zero``
    and ``p_two_sided`` (twice the smaller tail mass of the replicate differences, with the
    ``+1`` correction so it is never exactly 0)."""
    if boot_a.shape != boot_b.shape:
        raise ValueError("paired distributions need the same replicates")
    d = boot_a - boot_b
    lo, hi = percentile_ci(d)
    n = d.shape[0]
    le = (d <= 0).sum(axis=0)
    ge = (d >= 0).sum(axis=0)
    p = np.minimum(1.0, 2.0 * (np.minimum(le, ge) + 1.0) / (n + 1.0))
    return {
        "diff": np.asarray(point_a) - np.asarray(point_b),
        "ci_lo": lo,
        "ci_hi": hi,
        "excludes_zero": (lo > 0) | (hi < 0),
        "p_two_sided": p,
    }


# ----------------------------------------------------------------------------------------
# decorrelation time of a daily series -> block length
# ----------------------------------------------------------------------------------------
def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation of a 1-D series for lags ``0 .. max_lag`` (lag 0 = 1)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    den = float((x * x).sum())
    max_lag = min(max_lag, len(x) - 1)
    if den <= 0:
        return np.r_[1.0, np.zeros(max_lag)]
    return np.array([float((x[: len(x) - k] * x[k:]).sum()) / den for k in range(max_lag + 1)])


def decorrelation_time(x: np.ndarray, max_lag: int | None = None) -> dict[str, float]:
    """``efolding_days``: first lag where the autocorrelation drops below 1/e (``max_lag`` if it
    never does). ``tau_int_days``: integrated autocorrelation time ``1 + 2 sum rho_k`` with
    Sokal's automatic window (smallest ``M >= 5 tau(M)``), a measure of how many days carry one
    independent piece of information."""
    n = len(x)
    max_lag = max_lag or max(2, n // 3)
    rho = autocorrelation(x, max_lag)
    below = np.flatnonzero(rho < 1 / math.e)
    efold = float(below[0]) if len(below) else float(len(rho) - 1)
    tau = 1.0 + 2.0 * np.cumsum(rho[1:])
    window = len(tau)
    for m in range(1, len(tau) + 1):
        if m >= 5.0 * tau[m - 1]:
            window = m
            break
    return {"efolding_days": efold, "tau_int_days": float(max(tau[window - 1], 1.0))}


def auto_block_length(series: list[np.ndarray]) -> int:
    """Median e-folding time (days) of the daily series, between 1 and a quarter of the period."""
    n = len(series[0])
    efolds = [decorrelation_time(s)["efolding_days"] for s in series]
    return int(max(1, min(round(float(np.median(efolds))), max(1, n // 4))))


def seed_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    """Mean / sample SD (NaN for one seed) / min / max over the leading seed axis."""
    v = np.asarray(values, dtype=float)
    sd = v.std(axis=0, ddof=1) if v.shape[0] > 1 else np.full(v.shape[1:], np.nan)
    return {"mean": v.mean(axis=0), "sd": sd, "min": v.min(axis=0), "max": v.max(axis=0)}
