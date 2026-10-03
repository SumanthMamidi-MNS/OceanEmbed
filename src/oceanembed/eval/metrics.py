"""Masked verification metrics as streaming accumulators.

:class:`MetricAccumulator` keeps, for every ``(depth, lat, lon)`` grid point, the running sums
``n, sum x, sum y, sum x^2, sum y^2, sum xy, sum |e|, sum e^2`` (``x`` = prediction, ``y`` =
reference, ``e = x - y``). Because sums are additive they can be reduced afterwards over any group
of points -- everything, one depth, a basin box at one depth, or nothing (per-point maps) -- and
days can be fed in any number of chunks, so a test period never has to sit in RAM at once.

Metrics from the sums (see :func:`metrics_from_sums`): ``rmse``, ``bias`` (pred - ref), ``mae``,
``corr`` (Pearson), plus means / standard deviations. :func:`skill_score` is
``1 - MSE_model / MSE_clim``.

Correlation naming used throughout the package:

* ``corr_raw``  -- correlation of the temperatures themselves; inflated by the seasonal cycle, the
  horizontal gradient and (when pooled over depths) the vertical gradient.
* ``corr_anom`` -- correlation of anomalies, i.e. with the climatology removed from both the
  prediction and the reference; the meaningful measure of day-to-day skill.
"""

from __future__ import annotations

import numpy as np

SUM_FIELDS = ("n", "sx", "sy", "sxx", "syy", "sxy", "sae", "se2")
MIN_CORR_N = 3  # no correlation from fewer samples
_VAR_FLOOR = 1e-12


def _zero_sums(shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    return {f: np.zeros(shape, dtype=np.float64) for f in SUM_FIELDS}


def sums_from_arrays(
    pred: np.ndarray, ref: np.ndarray, valid: np.ndarray | None = None, axis=0
) -> dict[str, np.ndarray]:
    """Sums of the eight fields over ``axis`` for samples where both values are finite (and
    ``valid``). Used by the accumulator (``axis=0`` = batch axis) and for 1-D matchup vectors."""
    x = np.asarray(pred, dtype=np.float64)
    y = np.asarray(ref, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if valid is not None:
        ok &= valid
    x = np.where(ok, x, 0.0)
    y = np.where(ok, y, 0.0)
    e = x - y
    return {
        "n": ok.sum(axis=axis).astype(np.float64),
        "sx": x.sum(axis=axis),
        "sy": y.sum(axis=axis),
        "sxx": (x * x).sum(axis=axis),
        "syy": (y * y).sum(axis=axis),
        "sxy": (x * y).sum(axis=axis),
        "sae": np.abs(e).sum(axis=axis),
        "se2": (e * e).sum(axis=axis),
    }


def metrics_from_sums(s: dict[str, np.ndarray | float]) -> dict[str, np.ndarray]:
    """Metrics (any shape) from the summed fields; NaN where there are no / too few samples."""
    n = np.asarray(s["n"], dtype=np.float64)
    has = n > 0
    nn = np.where(has, n, 1.0)
    mx, my = s["sx"] / nn, s["sy"] / nn
    vx = np.maximum(s["sxx"] / nn - mx * mx, 0.0)
    vy = np.maximum(s["syy"] / nn - my * my, 0.0)
    cov = s["sxy"] / nn - mx * my
    ok_corr = (n >= MIN_CORR_N) & (vx > _VAR_FLOOR) & (vy > _VAR_FLOOR)
    denom = np.sqrt(np.where(ok_corr, vx * vy, 1.0))
    nan = np.nan
    return {
        "n": n,
        "rmse": np.where(has, np.sqrt(s["se2"] / nn), nan),
        "mse": np.where(has, s["se2"] / nn, nan),
        "bias": np.where(has, mx - my, nan),
        "mae": np.where(has, s["sae"] / nn, nan),
        "corr": np.where(ok_corr, cov / denom, nan),
        "mean_pred": np.where(has, mx, nan),
        "mean_ref": np.where(has, my, nan),
        "std_pred": np.where(has, np.sqrt(vx), nan),
        "std_ref": np.where(has, np.sqrt(vy), nan),
    }


def skill_score(mse_model, mse_clim):
    """``1 - MSE_model / MSE_clim`` (1 = perfect, 0 = as good as climatology, < 0 = worse);
    NaN where the climatology error is zero or undefined."""
    m = np.asarray(mse_model, dtype=np.float64)
    c = np.asarray(mse_clim, dtype=np.float64)
    ok = np.isfinite(m) & np.isfinite(c) & (c > 0)
    return np.where(ok, 1.0 - m / np.where(ok, c, 1.0), np.nan)


class MetricAccumulator:
    """Streaming per-grid-point sums for ``(D, H, W)`` fields; feed ``(B, D, H, W)`` or
    ``(D, H, W)`` arrays any number of times."""

    def __init__(self, shape: tuple[int, int, int]):
        self.shape = tuple(shape)
        self.sums = _zero_sums(self.shape)

    def update(self, pred: np.ndarray, ref: np.ndarray, valid: np.ndarray | None = None) -> None:
        """Add samples. Entries where either value is NaN/inf or ``valid`` is False are skipped."""
        pred, ref = np.asarray(pred), np.asarray(ref)
        if pred.ndim == 3:
            pred, ref = pred[None], ref[None]
            valid = None if valid is None else np.asarray(valid)[None]
        if pred.shape[1:] != self.shape or ref.shape != pred.shape:
            raise ValueError(f"expected (B,{self.shape}) arrays, got {pred.shape} / {ref.shape}")
        part = sums_from_arrays(pred, ref, valid, axis=0)
        for f in SUM_FIELDS:
            self.sums[f] += part[f]

    def merge(self, other: MetricAccumulator) -> None:
        for f in SUM_FIELDS:
            self.sums[f] += other.sums[f]

    # --- reductions ------------------------------------------------------------------------
    def per_point(self) -> dict[str, np.ndarray]:
        """Sums at every grid point ``(D, H, W)`` (for maps)."""
        return self.sums

    def by_depth(self, region: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Sums per depth ``(D,)`` over all points, or over the ``(H, W)`` boolean ``region``."""
        out = {}
        for f in SUM_FIELDS:
            a = self.sums[f]
            out[f] = a.sum(axis=(1, 2)) if region is None else a[:, region].sum(axis=1)
        return out

    def total(
        self, region: np.ndarray | None = None, depth_idx: np.ndarray | list[int] | None = None
    ) -> dict[str, float]:
        """Sums pooled over a region and a subset of depth indices (default: everything)."""
        d = self.by_depth(region)
        sel = slice(None) if depth_idx is None else np.asarray(depth_idx, dtype=int)
        return {f: float(d[f][sel].sum()) for f in SUM_FIELDS}

    def metrics_by_depth(self, region: np.ndarray | None = None) -> dict[str, np.ndarray]:
        return metrics_from_sums(self.by_depth(region))

    def metrics_total(self, region=None, depth_idx=None) -> dict[str, float]:
        m = metrics_from_sums(self.total(region, depth_idx))
        return {k: float(v) for k, v in m.items()}

    def metrics_map(self) -> dict[str, np.ndarray]:
        return metrics_from_sums(self.per_point())


def point_metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Metrics of two 1-D vectors (e.g. Argo matchups); NaN pairs are ignored."""
    s = sums_from_arrays(np.asarray(pred).ravel(), np.asarray(ref).ravel(), axis=None)
    return {k: float(v) for k, v in metrics_from_sums(s).items()}
