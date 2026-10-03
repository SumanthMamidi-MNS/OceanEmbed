"""Small helpers shared by the R4 and R5 reports: bootstrap scoring of daily sums, interval cells,
and Markdown number formatting. The bootstrap itself is :mod:`oceanembed.research.bootstrap`."""

from __future__ import annotations

import numpy as np

from oceanembed.research.bootstrap import (
    block_counts,
    decorrelation_time,
    metrics_from_daily,
    percentile_ci,
)

SCORES = ("rmse", "bias", "corr_anom", "skill_vs_clim", "mse")
ONE = np.ones((1, 1))  # selector of a single "depth" column


def choose_block_length(daily_mse: list[np.ndarray], n_days: int, given: int | None = None):
    """The block length (days): ``given``, else the median e-folding time of the daily MSE series
    of the methods (the R1 rule), at least 1 and at most a quarter of the period."""
    if given:
        return int(given), None
    efolds = [decorrelation_time(s)["efolding_days"] for s in daily_mse]
    auto = int(max(1, min(round(float(np.median(efolds))), max(1, n_days // 4))))
    return auto, auto


def make_counts(n_days: int, block_length: int, n_boot: int, seed: int) -> np.ndarray:
    return block_counts(n_days, block_length, n_boot, seed)


def score_series(
    raw: np.ndarray, anom: np.ndarray, clim: np.ndarray, w_point: np.ndarray, w_boot: np.ndarray
) -> dict[str, tuple[float, np.ndarray]]:
    """Metrics of one method from its per-day sums ``(F, T)`` (``raw``: prediction vs reference,
    ``anom``: climatology removed from both, ``clim``: raw sums of the climatology method).
    Returns ``{score: (point estimate, (B,) bootstrap replicates)}``."""
    a, b, c = raw[:, :, None], anom[:, :, None], clim[:, :, None]
    pt = metrics_from_daily(a, b, c, ONE, w_point)
    bt = metrics_from_daily(a, b, c, ONE, w_boot)
    return {m: (float(pt[m][0, 0]), bt[m][:, 0]) for m in SCORES}


def interval(point: float, boot: np.ndarray) -> dict:
    """``{point, ci_lo, ci_hi}`` (95 % percentile interval; ``None`` when undefined)."""
    if not np.isfinite(point) or not np.isfinite(boot).any():
        return {"point": None, "ci_lo": None, "ci_hi": None}
    lo, hi = percentile_ci(boot[:, None])
    return {"point": float(point), "ci_lo": float(lo[0]), "ci_hi": float(hi[0])}


def paired(a: tuple[float, np.ndarray], b: tuple[float, np.ndarray]) -> dict:
    """Bootstrap summary of ``A - B`` for two ``(point, replicates)`` pairs (same replicates)."""
    d = a[1] - b[1]
    out = interval(a[0] - b[0], d)
    if out["point"] is None:
        return out | {"excludes_zero": None}
    out["excludes_zero"] = bool(out["ci_lo"] > 0 or out["ci_hi"] < 0)
    return out


def metric_cells(scores: dict[str, tuple[float, np.ndarray]]) -> dict[str, dict]:
    return {m: interval(*scores[m]) for m in SCORES if m != "mse"}


# ----------------------------------------------------------------------------------------
# formatting
# ----------------------------------------------------------------------------------------
def fin(x) -> bool:
    return x is not None and bool(np.isfinite(x))


def num(x, nd: int = 3) -> str:
    return f"{x:.{nd}f}" if fin(x) else "-"


def signed(x, nd: int = 3) -> str:
    return f"{x:+.{nd}f}" if fin(x) else "-"


def ci_text(cell: dict, nd: int = 3, scale: float = 1.0) -> str:
    """``point [lo, hi]`` from an :func:`interval` cell."""
    if not cell or not fin(cell.get("point")):
        return "-"
    p, lo, hi = (cell[k] * scale for k in ("point", "ci_lo", "ci_hi"))
    return f"{p:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out
