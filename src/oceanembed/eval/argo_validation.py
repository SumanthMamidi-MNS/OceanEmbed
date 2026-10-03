"""Validation against Argo profiles (and, if supplied, the INCOIS gridded-ARGO product).

Each profile is (1) interpolated to the standard depths with a conservative rule,
(2) collocated to the grid cell containing it and to the same UTC day, and (3) compared with the
model, the baselines, the climatology and the GLORYS target in that cell on that day.

Vertical interpolation rule (:func:`interp_profile`), applied independently per standard depth
``z`` to a profile's good-QC levels sorted by depth:

* an observation exactly at ``z`` is used as is;
* otherwise ``z`` must be *bracketed* by observations above and below, whose spacing must not
  exceed ``max_gap(z) = max(10 m, 0.2 z)`` (10 m at <= 50 m, 20 m at 100 m, 40 m at 200 m,
  100 m at 500 m, 200 m at 1000 m, i.e. never interpolate across a hole larger than a fifth of
  the depth); then the value is linearly interpolated;
* surface exception: for ``z <= 5 m`` (0 and 5 m) the shallowest observation is used when it lies
  within 10 m of ``z`` and ``z`` is not bracketed with an acceptable gap;
* there is never any extrapolation below the deepest observation or above the shallowest one
  (except the surface exception), so deep levels need a profile that reaches past them.

Argo is independent of the model's *inputs* (satellite surface fields) but GLORYS assimilates Argo
and the model is trained on GLORYS: the comparison is not independent of the training target.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from oceanembed.config import Config
from oceanembed.data.dataset import make_surface_dataset
from oceanembed.data.harmonize import open_harmonized
from oceanembed.data.providers.argo import load_incois_gridded
from oceanembed.data.providers.base import ARGO_COLUMNS, make_provider, months, raw_file
from oceanembed.eval.evaluate import (
    POOLED_RANGE,
    SYNTHETIC_NOTE,
    _block,
    load_methods,
    method_label,
    metrics_dir,
    read_target,
    split_bounds,
    write_json,
)
from oceanembed.eval.metrics import MetricAccumulator, point_metrics, skill_score
from oceanembed.grid import BASINS, Grid, build_grid
from oceanembed.infer.predict import predict_batch
from oceanembed.runmeta import data_source
from oceanembed.train.utils import get_device

log = logging.getLogger(__name__)

SURFACE_DEPTH_LIMIT_M = 5.0  # depths <= this may use the shallowest observation
SURFACE_MAX_OFFSET_M = 10.0  # ... if it is at most this far away
MIN_GAP_M = 10.0
GAP_FRACTION = 0.2

INTERPOLATION_RULE = (
    "per standard depth z: exact level, else linear interpolation between bracketing levels "
    "if their spacing <= max(10 m, 0.2*z); for z <= 5 m the shallowest level is used if within "
    "10 m; no extrapolation otherwise"
)
INDEPENDENCE_NOTE = (
    "Argo profiles are independent of the model's inputs (satellite surface fields) but GLORYS "
    "assimilates Argo and the model is trained on GLORYS, so this is not independent of the "
    "training target. GLORYS-vs-Argo is therefore a reference for the reanalysis's own "
    "consistency, not an independent error estimate."
)
PARQUET_COLUMNS = {"climatology": "clim"}  # method key -> matchup column (others: same name)


def max_gap(z: float) -> float:
    return max(MIN_GAP_M, GAP_FRACTION * float(z))


def interp_profile(depth_obs, temp_obs, targets) -> np.ndarray:
    """Interpolate one profile to ``targets`` with the rule in the module docstring (NaN = no
    acceptable value)."""
    d = np.asarray(depth_obs, dtype=np.float64)
    t = np.asarray(temp_obs, dtype=np.float64)
    ok = np.isfinite(d) & np.isfinite(t)
    d, t = d[ok], t[ok]
    order = np.argsort(d, kind="stable")
    d, t = d[order], t[order]
    out = np.full(len(targets), np.nan)
    if len(d) == 0:
        return out
    for i, z in enumerate(targets):
        k = int(np.searchsorted(d, z, side="left"))  # first level with d >= z
        if k < len(d) and d[k] == z:
            out[i] = t[k]
            continue
        if 0 < k < len(d):  # bracketed: d[k-1] < z < d[k]
            if d[k] - d[k - 1] <= max_gap(z):
                w = (z - d[k - 1]) / (d[k] - d[k - 1])
                out[i] = (1 - w) * t[k - 1] + w * t[k]
                continue
        if z <= SURFACE_DEPTH_LIMIT_M and abs(d[0] - z) <= SURFACE_MAX_OFFSET_M:
            out[i] = t[0]
    return out


def collocate(time, lat, lon, grid: Grid, days: pd.DatetimeIndex, ocean2d: np.ndarray):
    """Nearest cell (the cell containing the point) and same-day collocation.

    Returns ``(day_idx, j, k, status)`` arrays; ``status`` is ``'ok'``, ``'outside_dates'``,
    ``'outside_domain'`` or ``'land'`` (checked in that order). ``day_idx`` indexes ``days``.
    """
    t = pd.DatetimeIndex(pd.to_datetime(np.asarray(time))).normalize()
    day_idx = days.get_indexer(t)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    (la0, la1), (lo0, lo1) = grid.lat_edges, grid.lon_edges
    inside = (lat >= la0) & (lat < la1) & (lon >= lo0) & (lon < lo1)
    j = np.floor((lat - la0) / grid.resolution).astype(int)
    k = np.floor((lon - lo0) / grid.resolution).astype(int)
    j = np.where(inside, np.clip(j, 0, len(grid.lat) - 1), -1)
    k = np.where(inside, np.clip(k, 0, len(grid.lon) - 1), -1)
    land = np.zeros(len(lat), dtype=bool)
    land[inside] = ~ocean2d[j[inside], k[inside]]
    status = np.full(len(lat), "ok", dtype=object)
    status[land] = "land"
    status[~inside] = "outside_domain"
    status[day_idx < 0] = "outside_dates"
    return day_idx, j, k, status.astype(str)


def basin_of(j: np.ndarray, k: np.ndarray, grid: Grid) -> np.ndarray:
    """Basin name per cell index (``'other'`` outside both boxes)."""
    out = np.full(len(j), "other", dtype=object)
    for name, mask in grid.basin_masks().items():
        out[mask[j, k]] = name
    return out.astype(str)


# ----------------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------------
def load_argo_period(cfg: Config, start, end) -> pd.DataFrame:
    """Tidy Argo table for ``[start, end]``: cached monthly parquet files, fetching/generating
    the missing months first (synthetic: instant; real: argopy download, needs network)."""
    month_list = list(months(pd.Timestamp(start).date(), pd.Timestamp(end).date()))
    missing = [f for f, _ in month_list if not raw_file(cfg, "argo", f).exists()]
    if missing:
        log.info("fetching Argo profiles for %d missing month(s)", len(missing))
        make_provider(cfg, "argo").fetch(
            "argo", pd.Timestamp(start).date(), pd.Timestamp(end).date()
        )
    frames = [pd.read_parquet(raw_file(cfg, "argo", f)) for f, _ in month_list]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in ARGO_COLUMNS})
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(["profile_id", "pres"]).reset_index(drop=True)


def profile_table(df: pd.DataFrame, depths: np.ndarray):
    """Profile-level table (id, time, lat, lon) and the interpolated ``(P, D)`` observations."""
    if len(df) == 0:
        empty = pd.DataFrame(
            {"profile_id": [], "time": pd.to_datetime([]), "lat": [], "lon": []}
        ).astype({"lat": float, "lon": float})
        return empty, np.empty((0, len(depths)))
    df = df.sort_values(["profile_id", "depth"], kind="stable")
    pid = df["profile_id"].to_numpy()
    uniq, first = np.unique(pid, return_index=True)
    bounds = np.append(first, len(df))
    z = df["depth"].to_numpy(dtype=np.float64)
    temp = df["temp"].to_numpy(dtype=np.float64)
    obs = np.stack(
        [
            interp_profile(z[a:b], temp[a:b], depths)
            for a, b in zip(bounds[:-1], bounds[1:], strict=True)
        ]
    )
    head = df.iloc[first]
    prof = pd.DataFrame(
        {
            "profile_id": uniq,
            "time": head["time"].to_numpy(),
            "lat": head["latitude"].to_numpy(dtype=np.float64),
            "lon": head["longitude"].to_numpy(dtype=np.float64),
        }
    )
    return prof, obs


# ----------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------
def _argo_summary(sub: pd.DataFrame, col: str) -> dict:
    if len(sub) == 0:
        return {
            k: None for k in ("rmse", "bias", "mae", "corr_raw", "corr_anom", "skill_vs_clim")
        } | {"n": 0}
    m = point_metrics(sub[col].to_numpy(), sub["obs"].to_numpy())
    a = point_metrics((sub[col] - sub["clim"]).to_numpy(), (sub["obs"] - sub["clim"]).to_numpy())
    c = point_metrics(sub["clim"].to_numpy(), sub["obs"].to_numpy())
    return {
        "n": int(m["n"]),
        "rmse": m["rmse"],
        "bias": m["bias"],
        "mae": m["mae"],
        "corr_raw": m["corr"],
        "corr_anom": a["corr"],
        "skill_vs_clim": float(skill_score(m["mse"], c["mse"])),
    }


def _argo_block(df: pd.DataFrame, col: str, depths: np.ndarray) -> dict:
    per = [_argo_summary(df[df["depth"] == z], col) for z in depths]
    pooled = df[(df["depth"] >= POOLED_RANGE[0]) & (df["depth"] <= POOLED_RANGE[1])]
    keys = ("n", "rmse", "bias", "mae", "corr_raw", "corr_anom", "skill_vs_clim")
    return {
        "overall": _argo_summary(df, col),
        "pooled_50_200m": _argo_summary(pooled, col),
        "per_depth": {"depth": [float(z) for z in depths]} | {k: [p[k] for p in per] for k in keys},
    }


# ----------------------------------------------------------------------------------------
# INCOIS gridded ARGO (optional)
# ----------------------------------------------------------------------------------------
def plan_gridded_matches(gtimes: pd.DatetimeIndex, days: pd.DatetimeIndex):
    """Pair each gridded time with the evaluated days it is compared against.

    Daily products are matched to the same day. A product with a time step over ~20 days is
    treated as monthly: it is compared with the mean of the model over that calendar month, and
    only if *every* day of the month lies in ``days``. Returns ``[(gridded_time_index,
    [day_positions])]``.
    """
    gt = pd.DatetimeIndex(gtimes).normalize()
    step = (
        np.median(np.diff(gt.values).astype("timedelta64[D]").astype(float)) if len(gt) > 1 else 1
    )
    monthly = step > 20
    plan = []
    for gi, g in enumerate(gt):
        if monthly:
            month_days = pd.date_range(g.replace(day=1), g + pd.offsets.MonthEnd(0), freq="D")
            pos = days.get_indexer(month_days)
            if (pos >= 0).all():
                plan.append((gi, [int(p) for p in pos]))
        else:
            p = days.get_indexer([g])[0]
            if p >= 0:
                plan.append((gi, [int(p)]))
    return plan


def evaluate_gridded_argo(cfg, ds, zds, methods, grid, dev, progress=True) -> dict | None:
    gridded = load_incois_gridded(cfg)
    if gridded is None:
        return None
    if "time" not in gridded.dims:
        log.warning("gridded ARGO has no time dimension; skipped")
        return None
    plan = plan_gridded_matches(pd.DatetimeIndex(gridded["time"].values), ds.dates())
    if not plan:
        log.warning("gridded ARGO has no time overlapping the evaluated period")
        return {"n_times": 0}
    shape = (len(ds.depth), *ds.shape)
    keys = [*methods.predictors, "glorys"]
    raw = {k: MetricAccumulator(shape) for k in keys}
    anom = {k: MetricAccumulator(shape) for k in keys}
    dates = ds.dates()
    for gi, positions in tqdm(plan, desc="gridded argo", disable=not progress):
        sums: dict[str, np.ndarray] = {}
        clim_sum = 0.0
        for p in positions:
            batch = ds.batch(p, p + 1)
            fields = {
                k: predict_batch(pr, batch, ds, dev)[0] for k, pr in methods.predictors.items()
            }
            fields["glorys"] = np.where(ds.mask, read_target(zds, ds, p, p + 1)[0], np.nan)
            for k, f in fields.items():
                sums[k] = sums.get(k, 0.0) + f
            clim_sum = clim_sum + ds.stats.climatology(dates[p : p + 1].values)[0]
        n = len(positions)
        ref = gridded.isel(time=gi).transpose("depth", "lat", "lon").values.astype(np.float32)
        clim = clim_sum / n
        valid = ds.mask & np.isfinite(ref)
        for k in keys:
            f = sums[k] / n
            raw[k].update(f, ref, valid)
            anom[k].update(f - clim, ref - clim, valid)
    out = {
        "n_times": len(plan),
        "dates": [str(pd.Timestamp(gridded["time"].values[gi]).date()) for gi, _ in plan],
        "methods": {k: _block(raw[k], anom[k], raw["climatology"], grid.depth) for k in keys},
    }
    return out


# ----------------------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------------------
def validate_argo(
    cfg: Config, split: str = "test", device: str | None = None, progress: bool = True
) -> dict:
    dev = get_device(device)
    start, end = split_bounds(cfg, split)
    ds = make_surface_dataset(cfg, start, end)
    zds = open_harmonized(cfg)
    grid = build_grid(cfg)
    methods = load_methods(cfg, dev)
    days = ds.dates()
    depths = ds.depth
    n_depth = len(depths)

    raw = load_argo_period(cfg, start, end)
    n_loaded = int(raw["profile_id"].nunique()) if len(raw) else 0
    t = pd.to_datetime(raw["time"]) if len(raw) else pd.Series([], dtype="datetime64[ns]")
    in_period = (t >= pd.Timestamp(start)) & (t < pd.Timestamp(end) + pd.Timedelta(days=1))
    raw = raw[in_period.to_numpy()] if len(raw) else raw
    prof, obs = profile_table(raw, depths)
    n_in_period = len(prof)

    day_idx, j, k, status = collocate(
        prof["time"], prof["lat"], prof["lon"], grid, days, ds.mask[0]
    )
    dropped = {s: int((status == s).sum()) for s in ("outside_dates", "outside_domain", "land")}
    dropped["outside_period"] = n_loaded - n_in_period
    log.info("Argo: %d profiles loaded, %d in period, dropped: %s", n_loaded, n_in_period, dropped)
    ok_idx = np.flatnonzero(status == "ok")
    n_p = len(ok_idx)

    keys = [*methods.predictors, "glorys"]
    fields = {key: np.full((n_p, n_depth), np.nan, dtype=np.float32) for key in keys}
    clim_pt = np.full((n_p, n_depth), np.nan, dtype=np.float32)
    mask_pt = np.zeros((n_p, n_depth), dtype=bool)
    if n_p:
        pj, pk, pd_idx = j[ok_idx], k[ok_idx], day_idx[ok_idx]
        mask_pt[:] = ds.mask[:, pj, pk].T
        for di in tqdm(np.unique(pd_idx), desc="argo collocation", disable=not progress):
            sel = np.flatnonzero(pd_idx == di)
            batch = ds.batch(int(di), int(di) + 1)
            clim_day = ds.stats.climatology(days[int(di) : int(di) + 1].values)[0]
            clim_pt[sel] = clim_day[:, pj[sel], pk[sel]].T
            tgt = read_target(zds, ds, int(di), int(di) + 1)[0]
            fields["glorys"][sel] = tgt[:, pj[sel], pk[sel]].T
            for key, predictor in methods.predictors.items():
                temp = predict_batch(predictor, batch, ds, dev)[0]
                fields[key][sel] = temp[:, pj[sel], pk[sel]].T

    obs_ok = obs[ok_idx] if n_p else np.empty((0, n_depth))
    valid = np.isfinite(obs_ok) & mask_pt & np.isfinite(fields["glorys"]) & np.isfinite(clim_pt)
    for key in methods.predictors:
        valid &= np.isfinite(fields[key])
    n_levels_obs = int(np.isfinite(obs_ok).sum())
    pi, di_ = np.nonzero(valid)
    prof_ok = prof.iloc[ok_idx].reset_index(drop=True)
    basins = basin_of(j[ok_idx], k[ok_idx], grid) if n_p else np.array([], dtype=str)
    table = pd.DataFrame(
        {
            "profile_id": prof_ok["profile_id"].to_numpy()[pi],
            "time": prof_ok["time"].to_numpy()[pi],
            "lat": prof_ok["lat"].to_numpy()[pi],
            "lon": prof_ok["lon"].to_numpy()[pi],
            "depth": depths[di_].astype(np.float32),
            "obs": obs_ok[pi, di_].astype(np.float32),
        }
    )
    for key in methods.predictors:
        table[PARQUET_COLUMNS.get(key, key)] = fields[key][pi, di_]
    table["glorys"] = fields["glorys"][pi, di_]
    table["basin"] = basins[pi]
    table["grid_lat"] = grid.lat[j[ok_idx]][pi].astype(np.float32) if n_p else []
    table["grid_lon"] = grid.lon[k[ok_idx]][pi].astype(np.float32) if n_p else []
    # canonical column order: spec columns first
    first_cols = ["profile_id", "time", "lat", "lon", "depth", "obs", "model", "ridge", "clim"]
    first_cols = [c for c in first_cols if c in table.columns]
    rest = [c for c in table.columns if c not in first_cols and c not in ("basin", "glorys")]
    table = table[[*first_cols, *rest, "glorys", "basin"]]
    mdir = metrics_dir(cfg)
    mdir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(mdir / "argo_matchups.parquet", index=False)

    results = {}
    for key in keys:
        col = PARQUET_COLUMNS.get(key, key)
        entry = _argo_block(table, col, depths)
        entry["per_basin"] = {
            name: _argo_block(table[table["basin"] == name], col, depths) for name in BASINS
        }
        results[key] = entry

    synthetic = data_source(cfg) == "synthetic"
    info = {k: {"label": methods.labels[k], **methods.info[k]} for k in methods.predictors}
    info["glorys"] = {"label": method_label("glorys"), "kind": "reanalysis"}
    out = {
        "metadata": {
            "data_source": data_source(cfg),
            "run_name": cfg.run_name,
            "split": split,
            "start": start,
            "end": end,
            "depths": [float(d) for d in depths],
            "units": "degC",
            "bias_convention": "method minus Argo",
            "n_profiles_loaded": n_loaded,
            "n_profiles_in_period": n_in_period,
            "n_profiles_collocated": n_p,
            "n_profiles_used": int(table["profile_id"].nunique()),
            "n_matchups": int(len(table)),
            "n_levels_with_obs": n_levels_obs,
            "dropped_profiles": dropped,
            "interpolation_rule": INTERPOLATION_RULE,
            "collocation": "cell containing the profile position, same UTC day (daily fields)",
            "basins": {name: list(box) for name, box in BASINS.items()},
            "column_for_method": {k: PARQUET_COLUMNS.get(k, k) for k in keys},
            "methods": info,
            "independence_note": INDEPENDENCE_NOTE,
            "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "note": SYNTHETIC_NOTE
            + " Synthetic Argo-like profiles are samples of the same analytic world that "
            "generates the synthetic GLORYS target (plus 0.03 degC noise)."
            if synthetic
            else None,
        },
        "methods": results,
    }
    gridded = evaluate_gridded_argo(cfg, ds, zds, methods, grid, dev, progress)
    if gridded is not None:
        out["gridded_argo"] = gridded
    write_json(mdir / "metrics_argo.json", out)
    return out


__all__ = [
    "collocate",
    "interp_profile",
    "load_argo_period",
    "max_gap",
    "plan_gridded_matches",
    "validate_argo",
]
