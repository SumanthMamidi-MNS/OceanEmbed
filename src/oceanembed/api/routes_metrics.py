"""Validation: metrics vs GLORYS and vs Argo, error maps, Argo matchups and profiles."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Query, Request

from oceanembed.api import schemas as S
from oceanembed.api.deps import get_store, reply, run_dep
from oceanembed.api.encode import array_to_list
from oceanembed.api.metrics import method_infos, reshape_metrics
from oceanembed.api.routes_runs import ERRORS, _argo, _glorys
from oceanembed.api.store import (
    DATE_RE,
    ApiError,
    Run,
    Store,
    _sym_limit,
    robust_range,
)
from oceanembed.grid import BASINS

router = APIRouter()

MAP_METRICS = ("skill_vs_clim", "corr_anom", "corr_raw", "rmse", "bias")  # longest prefixes first
MAP_UNITS = {
    "rmse": "degC",
    "bias": "degC",
    "corr_anom": "1",
    "corr_raw": "1",
    "skill_vs_clim": "1",
}
PROFILE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")
_FIXED_COLUMNS = {
    "profile_id", "time", "lat", "lon", "depth", "obs", "grid_lat", "grid_lon", "glorys", "basin",
}  # fmt: skip


@router.get(
    "/runs/{run}/metrics/glorys",
    response_model=S.MetricsResponse,
    responses=ERRORS,
    summary="Metrics vs GLORYS: overall, pooled 50-200 m, per depth, per basin, daily RMSE",
    description="Metric-major reshape of `metrics_glorys.json`: `per_depth[metric][method]` is a list "
    "over `depths`. Metrics: n, rmse, bias (method minus reference), mae, corr_raw, corr_anom, "
    "skill_vs_clim. NaN is `null`.",
)
def metrics_glorys(
    request: Request, run: Run = Depends(run_dep), store: Store = Depends(get_store)
):
    raw = _glorys(store, run)
    if raw is None:
        raise ApiError(404, "no metrics vs GLORYS in this run (run `oceanembed evaluate`)")
    return reply(request, reshape_metrics(run.name, raw, "GLORYS"))


@router.get(
    "/runs/{run}/metrics/argo",
    response_model=S.MetricsResponse,
    responses=ERRORS,
    summary="Metrics vs Argo: model, ridge, climatology, GLORYS; per depth and basin",
    description="Same structure as the GLORYS metrics. `metadata` holds profile counts, the "
    "interpolation rule and the `independence_note`: GLORYS assimilates Argo and the model is "
    "trained on GLORYS, so this is not independent of the training target.",
)
def metrics_argo(request: Request, run: Run = Depends(run_dep), store: Store = Depends(get_store)):
    raw = _argo(store, run)
    if raw is None:
        raise ApiError(404, "no metrics vs Argo in this run (run `oceanembed validate-argo`)")
    return reply(request, reshape_metrics(run.name, raw, "Argo"))


# ----------------------------------------------------------------------------------------------
# error maps
# ----------------------------------------------------------------------------------------------
def _split_map_name(name: str) -> tuple[str, str] | None:
    for m in MAP_METRICS:
        if name.startswith(m + "_"):
            return m, name[len(m) + 1 :]
    return None


def _maps(store: Store, run: Run) -> dict[str, np.ndarray]:
    maps = store.map_arrays(run)
    if maps is None:
        raise ApiError(404, "no error maps in this run (metrics/maps_glorys.nc missing)")
    return maps


def _map_index(maps: dict[str, np.ndarray]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name in maps:
        parts = _split_map_name(name)
        if parts:
            out.setdefault(parts[0], []).append(parts[1])
    return out


@router.get(
    "/runs/{run}/metrics/maps/index",
    response_model=S.MapsIndex,
    responses=ERRORS,
    summary="Which metric / method combinations have an error map",
)
def maps_index(request: Request, run: Run = Depends(run_dep), store: Store = Depends(get_store)):
    maps = _maps(store, run)
    return reply(
        request,
        {
            "run": run.name,
            "metrics": _map_index(maps),
            "depths": run.depths,
            "has_n_valid": "n_valid" in maps,
        },
    )


SKILL_LIMIT_PERCENTILE = 95.0
SKILL_MIN_LIMIT = 0.5
SKILL_MAX_LIMIT = 3.0


def _range_info(arr: np.ndarray, rng: dict) -> dict[str, Any]:
    """What the returned map does to its own colour range: data extremes and how many valid
    points lie outside ``[vmin, vmax]`` (they saturate the colour scale)."""
    v = np.asarray(arr, dtype=np.float64)
    v = v[np.isfinite(v)]
    below, above = int((v < rng["vmin"]).sum()), int((v > rng["vmax"]).sum())
    return {
        "n_valid": int(v.size),
        "data_min": float(v.min()) if v.size else None,
        "data_max": float(v.max()) if v.size else None,
        "n_below": below,
        "n_above": above,
        "fraction_outside": round((below + above) / v.size, 4) if v.size else 0.0,
        "exceeds_range": bool(below or above),
        "limit_rule": None,
        "limit_capped": None,
    }


@router.get(
    "/runs/{run}/metrics/maps",
    response_model=S.MapResponse,
    responses=ERRORS,
    summary="2-D error map (per grid point, over the evaluated period) at one depth",
)
def metrics_map(
    request: Request,
    run: Run = Depends(run_dep),
    store: Store = Depends(get_store),
    metric: str = Query("rmse", description="rmse | bias | corr_anom | corr_raw | skill_vs_clim"),
    method: str = Query("model"),
    depth: float | None = Query(None, description="Metres; must equal a grid depth"),
    depth_index: int | None = Query(None),
):
    maps = _maps(store, run)
    index = _map_index(maps)
    if metric not in MAP_UNITS:
        raise ApiError(400, f"metric must be one of {', '.join(MAP_UNITS)}")
    if metric not in index:
        raise ApiError(
            404, f"this run has no '{metric}' maps; available: {', '.join(index) or 'none'}"
        )
    if method not in index[metric]:
        raise ApiError(
            404, f"no '{metric}' map for method '{method}'; available: {', '.join(index[metric])}"
        )
    k = store.depth_index(run, depth, depth_index)
    arr = maps[f"{metric}_{method}"][k]
    # one range for every method at this depth so panels are comparable
    group = [maps[f"{metric}_{m}"][k] for m in index[metric]]
    if metric == "rmse":
        lo, hi = robust_range(group, 0.0, 99.0)
        rng = {"vmin": 0.0, "vmax": hi, "diverging": False}
    elif metric == "bias":
        lim = _sym_limit(group)
        rng = {"vmin": -lim, "vmax": lim, "diverging": True}
    elif metric == "skill_vs_clim":
        # symmetric about zero (0 = as good as the climatology). Skill never exceeds 1 but is
        # unbounded below, so the limit is the 95th percentile of |skill| over every method at this
        # depth, kept between SKILL_MIN_LIMIT and SKILL_MAX_LIMIT; range_info reports what falls
        # outside it.
        raw_lim = _sym_limit(group, SKILL_LIMIT_PERCENTILE)
        lim = round(float(np.clip(raw_lim, SKILL_MIN_LIMIT, SKILL_MAX_LIMIT)), 3)
        rng = {"vmin": -lim, "vmax": lim, "diverging": True}
    else:
        lo, hi = robust_range(group, 1.0, 99.0)
        rng = {"vmin": lo, "vmax": min(hi, 1.0), "diverging": False}
    info = _range_info(arr, rng)
    if metric == "skill_vs_clim":
        info.update(
            limit_rule=f"{SKILL_LIMIT_PERCENTILE:g}th percentile of |skill| over all methods at this "
            f"depth, kept within {SKILL_MIN_LIMIT:g}..{SKILL_MAX_LIMIT:g}; skill is at most 1 and "
            "unbounded below",
            limit_capped=bool(raw_lim > SKILL_MAX_LIMIT),
        )
    labels = {m["key"]: m["label"] for m in method_infos(_glorys(store, run))}
    return reply(
        request,
        {
            "run": run.name,
            "metric": metric,
            "method": method,
            "label": labels.get(method, method),
            "depth": run.depths[k],
            "depth_index": k,
            "units": MAP_UNITS[metric],
            "shape": list(arr.shape),
            "data": arr,
            "color_range": rng,
            "range_info": info,
        },
    )


# ----------------------------------------------------------------------------------------------
# Argo matchups and profiles
# ----------------------------------------------------------------------------------------------
def _matchups(store: Store, run: Run) -> pd.DataFrame:
    df = store.argo_matchups(run)
    if df is None:
        raise ApiError(404, "no Argo matchups in this run (run `oceanembed validate-argo`)")
    return df


def _method_columns(df: pd.DataFrame, store: Store, run: Run) -> list[str]:
    """Prediction columns of the matchup table (model, ridge, clim, ablations), in file order."""
    meta = (_argo(store, run) or {}).get("metadata", {})
    mapped = [c for k, c in (meta.get("column_for_method") or {}).items() if k != "glorys"]
    cols = [c for c in mapped if c in df.columns and c != "obs"]
    extra = [c for c in df.columns if c not in _FIXED_COLUMNS and c not in cols]
    return [*cols, *extra]


def _iso(series: pd.Series) -> list[str]:
    return series.dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()


@router.get(
    "/runs/{run}/argo/matchups",
    response_model=S.MatchupsResponse,
    responses=ERRORS,
    summary="Argo-vs-prediction matchups, columnar, deterministically down-sampled",
    description="One row per (profile, depth). When there are more rows than `max_points`, rows are "
    "taken at evenly spaced positions of the stored order (same request -> same rows).",
)
def argo_matchups(
    request: Request,
    run: Run = Depends(run_dep),
    store: Store = Depends(get_store),
    max_points: int = Query(20000, ge=100, le=1_000_000, description="Row limit"),
):
    df = _matchups(store, run)
    n = len(df)
    if n > max_points:
        idx = np.unique(np.linspace(0, n - 1, max_points).round().astype(np.int64))
        df = df.iloc[idx]
    methods = _method_columns(df, store, run)
    cols: dict[str, list[Any]] = {
        "profile_id": df["profile_id"].astype(str).tolist(),
        "time": _iso(df["time"]),
        "lat": array_to_list(df["lat"].to_numpy(), 4),
        "lon": array_to_list(df["lon"].to_numpy(), 4),
        "depth": array_to_list(df["depth"].to_numpy(), 2),
        "basin": df["basin"].astype(str).tolist() if "basin" in df else [None] * len(df),
        "obs": array_to_list(df["obs"].to_numpy(), 3),
    }
    if "glorys" in df:
        cols["glorys"] = array_to_list(df["glorys"].to_numpy(), 3)
    for c in methods:
        cols[c] = array_to_list(df[c].to_numpy(), 3)
    return reply(
        request,
        {
            "run": run.name,
            "n_total": int(n),
            "n_returned": int(len(df)),
            "downsampled": bool(len(df) < n),
            "methods": methods,
            "columns": cols,
        },
    )


def _alias_climatology(d: dict[str, Any]) -> None:
    """The matchup column of the climatology is ``clim``; expose it as ``climatology`` too (the key
    every other endpoint uses), keeping ``clim`` for older clients."""
    if "clim" in d and "climatology" not in d:
        d["climatology"] = d["clim"]


def _profile_table(store: Store, run: Run) -> list[dict[str, Any]]:
    def build():
        df = _matchups(store, run)
        methods = _method_columns(df, store, run) + (["glorys"] if "glorys" in df else [])
        g = df.groupby("profile_id", sort=False)
        head = g.agg(
            time=("time", "first"),
            lat=("lat", "first"),
            lon=("lon", "first"),
            basin=("basin", "first"),
            n_levels=("depth", "size"),
        )
        err = pd.DataFrame({c: (df[c] - df["obs"]) ** 2 for c in methods})
        rmse = np.sqrt(err.groupby(df["profile_id"], sort=False).mean()).round(3).to_dict("index")
        out = []
        for pid, row in head.sort_values(["time"], kind="stable").iterrows():
            errs = {c: float(rmse[pid][c]) for c in methods}
            _alias_climatology(errs)
            out.append(
                {
                    "profile_id": str(pid),
                    "time": pd.Timestamp(row["time"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "lat": round(float(row["lat"]), 4),
                    "lon": round(float(row["lon"]), 4),
                    "basin": None if pd.isna(row["basin"]) else str(row["basin"]),
                    "n_levels": int(row["n_levels"]),
                    "rmse": errs,
                }
            )
        return out

    return store.cached(run, "argo_profiles", build, nbytes=2_000_000)


DEFAULT_PROFILE_LIMIT = 5000
MAX_PROFILE_LIMIT = 50000


def _check_day(text: str | None, what: str) -> str | None:
    if text is None:
        return None
    if not DATE_RE.match(text):
        raise ApiError(400, f"'{what}' must be a date formatted YYYY-MM-DD, got {text!r}")
    try:
        pd.Timestamp(text)
    except ValueError as e:
        raise ApiError(400, f"'{what}' is not a valid calendar date: {text!r}") from e
    return text


@router.get(
    "/runs/{run}/argo/profiles",
    response_model=S.ArgoProfilesResponse,
    responses=ERRORS,
    summary="Argo profiles with location and per-method RMSE (filter, sort, paginate)",
    description="Filters: `basin`, `start` / `end` (inclusive days). `sort` is `time` (default) or "
    "`rmse` (per-profile RMSE of `method`, default `model`; profiles without a value come last) "
    "with `order` asc / desc. At most `limit` profiles are returned (default 5000, maximum 50000); "
    "`n_profiles` counts all matches and `has_more` says whether `offset + limit` stops short of "
    "them. Equal keys keep time order, so a page is reproducible. `lat_min` / `lat_max` / "
    "`lon_min` / `lon_max` (inclusive, any subset) filter by a bounding box. `around=<profile_id>` "
    "returns the page that contains that profile under the current filter and sort (`offset` is "
    "then ignored) and reports its position as `around_index`; 404 when the profile is unknown or "
    "filtered out. Each profile's `rmse` has the climatology under `climatology` (and the older "
    "`clim`).",
)
def argo_profiles(
    request: Request,
    run: Run = Depends(run_dep),
    store: Store = Depends(get_store),
    basin: str | None = Query(None, description="arabian_sea | bay_of_bengal"),
    start: str | None = Query(None, description="first day, YYYY-MM-DD"),
    end: str | None = Query(None, description="last day, YYYY-MM-DD (inclusive)"),
    sort: str = Query("time", description="time | rmse"),
    order: str = Query("asc", description="asc | desc"),
    method: str = Query("model", description="method column used by sort=rmse"),
    limit: int = Query(DEFAULT_PROFILE_LIMIT, ge=1, le=MAX_PROFILE_LIMIT),
    offset: int = Query(0, ge=0),
    lat_min: float | None = Query(None, description="bounding box, degrees north (inclusive)"),
    lat_max: float | None = Query(None),
    lon_min: float | None = Query(None, description="bounding box, degrees east (inclusive)"),
    lon_max: float | None = Query(None),
    around: str | None = Query(
        None, description="profile_id: return the page containing it (offset is ignored)"
    ),
):
    for lo_name, lo, hi_name, hi in (
        ("lat_min", lat_min, "lat_max", lat_max),
        ("lon_min", lon_min, "lon_max", lon_max),
    ):
        if lo is not None and hi is not None and lo > hi:
            raise ApiError(400, f"{lo_name} {lo:g} is greater than {hi_name} {hi:g}")
    if basin is not None and basin not in BASINS:
        raise ApiError(400, f"basin must be one of {', '.join(BASINS)}, got {basin!r}")
    start, end = _check_day(start, "start"), _check_day(end, "end")
    if start and end and start > end:
        raise ApiError(400, f"start {start} is after end {end}")
    if sort not in ("time", "rmse"):
        raise ApiError(400, "sort must be 'time' or 'rmse'")
    if order not in ("asc", "desc"):
        raise ApiError(400, "order must be 'asc' or 'desc'")
    table = _profile_table(store, run)
    if sort == "rmse" and table and method not in table[0]["rmse"]:
        raise ApiError(400, f"method must be one of {', '.join(table[0]['rmse'])}, got {method!r}")
    rows = table
    if basin:
        rows = [r for r in rows if r["basin"] == basin]
    if start:
        rows = [r for r in rows if r["time"][:10] >= start]
    if end:
        rows = [r for r in rows if r["time"][:10] <= end]
    for key, lo, hi in (("lat", lat_min, lat_max), ("lon", lon_min, lon_max)):
        if lo is not None:
            rows = [r for r in rows if r[key] >= lo]
        if hi is not None:
            rows = [r for r in rows if r[key] <= hi]
    desc = order == "desc"
    if sort == "rmse":
        val = lambda r: r["rmse"].get(method)  # noqa: E731
        ok = [r for r in rows if val(r) is not None and np.isfinite(val(r))]
        bad = [r for r in rows if not (val(r) is not None and np.isfinite(val(r)))]
        rows = sorted(ok, key=val, reverse=desc) + bad
    elif desc:
        rows = rows[::-1]
    around_index = None
    if around is not None:
        around_index = next((i for i, r in enumerate(rows) if r["profile_id"] == around), None)
        if around_index is None:
            known = any(r["profile_id"] == around for r in table)
            raise ApiError(
                404,
                f"Argo profile '{around}' is not among the {len(rows)} profiles matching the "
                "current filters (remove a filter to find it)"
                if known
                else f"Argo profile '{around}' not found in run '{run.name}'",
            )
        offset = (around_index // limit) * limit
    page = rows[offset : offset + limit]
    return reply(
        request,
        {
            "run": run.name,
            "n_profiles": len(rows),
            "n_total": len(table),
            "n_returned": len(page),
            "offset": offset,
            "limit": limit,
            "has_more": offset + len(page) < len(rows),
            "around": around,
            "around_index": around_index,
            "profiles": page,
        },
    )


@router.get(
    "/runs/{run}/argo/profiles/{profile_id}",
    response_model=S.ArgoProfileDetail,
    responses=ERRORS,
    summary="One Argo profile: observation vs model / ridge / climatology / GLORYS by depth",
)
def argo_profile(
    request: Request,
    profile_id: str,
    run: Run = Depends(run_dep),
    store: Store = Depends(get_store),
):
    df = _matchups(store, run)
    if not PROFILE_ID_RE.match(profile_id) or not (df["profile_id"] == profile_id).any():
        raise ApiError(404, f"Argo profile '{profile_id}' not found in run '{run.name}'")
    rows = df[df["profile_id"] == profile_id].sort_values("depth")
    first = rows.iloc[0]
    methods = _method_columns(df, store, run) + (["glorys"] if "glorys" in df else [])
    series = {c: array_to_list(rows[c].to_numpy(), 3) for c in methods}
    _alias_climatology(series)
    return reply(
        request,
        {
            "run": run.name,
            "profile_id": profile_id,
            "time": pd.Timestamp(first["time"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "lat": float(first["lat"]),
            "lon": float(first["lon"]),
            "grid_lat": float(first["grid_lat"]) if "grid_lat" in rows else None,
            "grid_lon": float(first["grid_lon"]) if "grid_lon" in rows else None,
            "basin": None
            if "basin" not in rows or pd.isna(first["basin"])
            else str(first["basin"]),
            "depth": array_to_list(rows["depth"].to_numpy(), 2),
            "obs": array_to_list(rows["obs"].to_numpy(), 3),
            "series": series,
            "units": "degC",
        },
    )
