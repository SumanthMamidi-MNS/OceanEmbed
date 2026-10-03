"""Gridded data: fields, surface inputs, profiles, sections, time series, embeddings."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Query, Request

from oceanembed import data_access as da
from oceanembed.api import schemas as S
from oceanembed.api.deps import get_store, reply, reply_bytes, run_dep
from oceanembed.api.encode import array_to_list
from oceanembed.api.routes_runs import ERRORS
from oceanembed.api.store import (
    DATE_RE,
    FIELD_KINDS,
    MAIN_METHOD,
    RANGE_SAMPLE_DAYS,
    ApiError,
    Run,
    Store,
    _sym_limit,
    kind_label,
    robust_range,
)
from oceanembed.eval.evaluate import method_label

MAX_COMPONENTS = da.MAX_PCA_COMPONENTS
METHOD_DESC = (
    "Prediction product to serve: `model` (default), `ridge` or an ablation such as "
    "`model_scratch`; the ones available are `field_methods` of the run detail."
)

router = APIRouter()

SURFACE_META = {  # long name, units, diverging colour scale
    "sst": ("Sea surface temperature", "degC", False),
    "sss": ("Sea surface salinity", "PSU", False),
    "sla": ("Sea level anomaly", "m", True),
    "uo": ("Eastward surface current", "m s-1", True),
    "vo": ("Northward surface current", "m s-1", True),
    "uw": ("Eastward 10 m wind", "m s-1", True),
    "vw": ("Northward 10 m wind", "m s-1", True),
}


def _wants_binary(fmt: str | None, accept: str) -> bool:
    f = (fmt or "").lower()
    if f not in ("", "json", "f32"):
        raise ApiError(400, "format must be 'json' or 'f32'")
    if f:
        return f == "f32"
    first = accept.split(",")[0].strip().lower()
    return first.startswith("application/octet-stream")


def _range_dict(hint: dict) -> dict:
    return {"vmin": hint["vmin"], "vmax": hint["vmax"], "diverging": hint["diverging"]}


def _stats(a: np.ndarray, kind: str) -> dict:
    v = a[np.isfinite(a)].astype(np.float64)
    out = {"n_valid": int(v.size), "min": None, "max": None, "mean": None, "std": None}
    if v.size:
        out.update(min=float(v.min()), max=float(v.max()), mean=float(v.mean()), std=float(v.std()))
        if kind == "difference":
            out.update(
                rmse=float(np.sqrt((v**2).mean())),
                bias=float(v.mean()),
                mae=float(np.abs(v).mean()),
            )
    return out


@router.get(
    "/runs/{run}/fields",
    response_model=S.FieldResponse,
    responses={
        **ERRORS,
        200: {
            "description": "JSON, or with `format=f32` / `Accept: application/octet-stream` a raw "
            "little-endian float32 array (NaN preserved, C order). Shape, colour range and the "
            "other metadata are in the `X-*` response headers (see docs/api.md).",
            "content": {"application/octet-stream": {}},
        },
    },
    summary="One 2-D field (with depth) or the full 15-level volume (without) for a day",
)
def fields(
    request: Request,
    run: Run = Depends(run_dep),
    store: Store = Depends(get_store),
    date: str | None = Query(None, description="YYYY-MM-DD, a predicted day"),
    kind: str = Query("prediction", description="|".join(FIELD_KINDS)),
    depth: float | None = Query(None, description="Metres; must equal a grid depth"),
    depth_index: int | None = Query(None, description="0-based level index (alternative to depth)"),
    format: str | None = Query(None, description="json (default) or f32 (binary)"),
    method: str = Query(MAIN_METHOD, description=METHOD_DESC),
):
    if kind not in FIELD_KINDS:
        raise ApiError(400, f"kind must be one of {', '.join(FIELD_KINDS)}, got {kind!r}")
    method = store.method(run, method)
    d = store.parse_date(run, date)
    binary = _wants_binary(format, request.headers.get("accept", ""))
    level = (
        store.depth_index(run, depth, depth_index)
        if (depth is not None or depth_index is not None)
        else None
    )
    vol = store.volume(run, kind, d, method)
    arr = vol if level is None else vol[level]
    hint = store.colour_hint(run, kind, d, level, method)
    has_target = store.target(run, d) is not None
    per_depth = None
    if level is None:
        per_depth = [store.colour_hint(run, kind, d, k, method) for k in range(vol.shape[0])]

    if binary:
        headers = {
            "X-Shape": ",".join(str(n) for n in arr.shape),
            "X-Dtype": "float32",
            "X-Byte-Order": "little",
            "X-Kind": kind,
            "X-Method": method,
            "X-Date": d.strftime("%Y-%m-%d"),
            "X-Depth-Index": "" if level is None else str(level),
            "X-Color-Range": f"{hint['vmin']},{hint['vmax']}",
            "X-Color-Diverging": "1" if hint["diverging"] else "0",
            "X-Has-Target": "1" if has_target else "0",
        }
        if per_depth is not None:
            headers["X-Color-Range-Per-Depth"] = json.dumps(
                [[h["vmin"], h["vmax"]] for h in per_depth]
            )
        body = np.ascontiguousarray(arr, dtype="<f4").tobytes()
        return reply_bytes(request, body, "application/octet-stream", headers)

    content = {
        "run": run.name,
        "date": d.strftime("%Y-%m-%d"),
        "kind": kind,
        "method": method,
        "label": kind_label(kind, method, run.main_init),
        "units": "degC",
        "depth": None if level is None else run.depths[level],
        "depth_index": level,
        "shape": list(arr.shape),
        "data": arr,
        "color_range": _range_dict(hint),
        "color_range_per_depth": None if per_depth is None else [_range_dict(h) for h in per_depth],
        "stats": None if level is None else _stats(arr, kind),
        "has_target": has_target,
    }
    return reply(request, content)


@router.get(
    "/runs/{run}/ranges",
    response_model=S.RangesResponse,
    responses=ERRORS,
    summary="Period-wide colour ranges per depth (for a 'hold range' that needs no day volumes)",
    description="Per-depth colour limits over the whole prediction period, estimated once per run "
    "from `samples` evenly spaced days and cached in memory per run version (nothing is written to "
    "the run folder). `temperature` is the 1st-99th percentile of the main model's prediction "
    "united with the GLORYS target; per method, `difference` and `anomaly` are symmetric about "
    "zero (99th percentile of the absolute value). See `method_note` in the response.",
)
def ranges(
    request: Request,
    run: Run = Depends(run_dep),
    store: Store = Depends(get_store),
    samples: int = Query(
        RANGE_SAMPLE_DAYS, ge=2, le=60, description="Days sampled over the prediction period"
    ),
):
    return reply(request, store.period_ranges(run, samples))


@router.get(
    "/runs/{run}/surface",
    response_model=S.SurfaceResponse,
    responses=ERRORS,
    summary="The 7 surface input fields of a day in physical units",
)
def surface(
    request: Request,
    run: Run = Depends(run_dep),
    store: Store = Depends(get_store),
    date: str | None = Query(None, description="YYYY-MM-DD"),
):
    d = store.parse_date(run, date)
    fields_ = store.surface(run, d)
    if fields_ is None:
        raise ApiError(
            404, f"no harmonised surface inputs for {d.date()} (processed store missing)"
        )
    out = {}
    for name, arr in fields_.items():
        long_name, units, diverging = SURFACE_META.get(name, (name, "", False))
        if diverging:
            lim = _sym_limit([arr])
            rng = {"vmin": -lim, "vmax": lim, "diverging": True}
        else:
            lo, hi = robust_range([arr])
            rng = {"vmin": lo, "vmax": hi, "diverging": False}
        out[name] = {"long_name": long_name, "units": units, "data": arr, "color_range": rng}
    shape = list(next(iter(fields_.values())).shape)
    return reply(
        request, {"run": run.name, "date": d.strftime("%Y-%m-%d"), "shape": shape, "fields": out}
    )


@router.get(
    "/runs/{run}/profile",
    response_model=S.ProfileResponse,
    responses=ERRORS,
    summary="Vertical profiles (prediction, target, climatology) at the nearest cell",
    description="`method` picks the prediction (model, ridge or an ablation); target and "
    "climatology do not depend on it. The ridge profile of Argo locations is also in "
    "`/argo/profiles/{id}`.",
)
def profile(
    request: Request,
    run: Run = Depends(run_dep),
    store: Store = Depends(get_store),
    date: str | None = Query(None),
    lat: float = Query(..., description="degrees north, inside the grid"),
    lon: float = Query(..., description="degrees east, inside the grid"),
    method: str = Query(MAIN_METHOD, description=METHOD_DESC),
):
    method = store.method(run, method)
    d = store.parse_date(run, date)
    i, j = store.cell(run, lat, lon)
    pred = store.prediction(run, d, method)[:, i, j]
    tgt_vol = store.target(run, d)
    clim_vol = store.climatology(run, d)
    nan = np.full_like(pred, np.nan)
    tgt = nan if tgt_vol is None else tgt_vol[:, i, j]
    clim = nan if clim_vol is None else np.where(np.isfinite(pred), clim_vol[:, i, j], np.nan)
    return reply(
        request,
        {
            "run": run.name,
            "date": d.strftime("%Y-%m-%d"),
            "method": method,
            "method_label": method_label(method, run.main_init),
            "requested": {"lat": lat, "lon": lon},
            "lat": float(run.grid.lat[i]),
            "lon": float(run.grid.lon[j]),
            "lat_index": i,
            "lon_index": j,
            "is_ocean": bool(np.isfinite(pred).any()),
            "depths": run.depths,
            "prediction": pred,
            "target": tgt,
            "climatology": clim,
            "has_target": tgt_vol is not None,
            "units": "degC",
        },
    )


@router.get(
    "/runs/{run}/section",
    response_model=S.SectionResponse,
    responses=ERRORS,
    summary="Depth x distance section along a latitude or a longitude",
)
def section(
    request: Request,
    run: Run = Depends(run_dep),
    store: Store = Depends(get_store),
    date: str | None = Query(None),
    lat: float | None = Query(None, description="zonal section along this latitude"),
    lon: float | None = Query(None, description="meridional section along this longitude"),
    method: str = Query(MAIN_METHOD, description=METHOD_DESC),
):
    if (lat is None) == (lon is None):
        raise ApiError(
            400, "give exactly one of 'lat' (zonal section) or 'lon' (meridional section)"
        )
    method = store.method(run, method)
    d = store.parse_date(run, date)
    g = run.grid
    pred = store.prediction(run, d, method)
    tgt_vol = store.target(run, d)
    clim_vol = store.climatology(run, d)
    if lat is not None:
        i, _ = store.cell(run, lat, float(g.lon[0]))
        sel = lambda a: a[:, i, :]  # noqa: E731
        along, fixed, fixed_value, requested, distance = "lon", "lat", float(g.lat[i]), lat, g.lon
    else:
        _, j = store.cell(run, float(g.lat[0]), lon)
        sel = lambda a: a[:, :, j]  # noqa: E731
        along, fixed, fixed_value, requested, distance = "lat", "lon", float(g.lon[j]), lon, g.lat
    p = sel(pred)
    t = np.full_like(p, np.nan) if tgt_vol is None else sel(tgt_vol)
    c = (
        np.full_like(p, np.nan)
        if clim_vol is None
        else np.where(np.isfinite(p), sel(clim_vol), np.nan)
    )
    diff = p - t
    lo, hi = robust_range([p, t])
    lim = _sym_limit([diff])
    return reply(
        request,
        {
            "run": run.name,
            "date": d.strftime("%Y-%m-%d"),
            "method": method,
            "method_label": method_label(method, run.main_init),
            "along": along,
            "fixed": fixed,
            "fixed_value": fixed_value,
            "requested": requested,
            "distance": array_to_list(distance, 6),
            "depths": run.depths,
            "prediction": p,
            "target": t,
            "climatology": c,
            "difference": diff,
            "color_range": {"vmin": lo, "vmax": hi, "diverging": False},
            "difference_range": {"vmin": -lim, "vmax": lim, "diverging": True},
            "has_target": tgt_vol is not None,
            "units": "degC",
        },
    )


@router.get(
    "/runs/{run}/timeseries",
    response_model=S.TimeseriesResponse,
    responses=ERRORS,
    summary="Time x depth (Hovmoeller) of prediction, target and climatology at a point, plus "
    "per-day error",
)
def timeseries(
    request: Request,
    run: Run = Depends(run_dep),
    store: Store = Depends(get_store),
    lat: float = Query(...),
    lon: float = Query(...),
    method: str = Query(MAIN_METHOD, description=METHOD_DESC),
):
    method = store.method(run, method)
    i, j = store.cell(run, lat, lon)
    if len(store.dates(run)) == 0:
        raise ApiError(404, "this run has no predictions")
    s = store.point_series(run, i, j, method)
    pred, tgt = s["prediction"], s["target"]
    clim = store.point_climatology(run, i, j, s["dates"])
    diff = pred - tgt
    ok = np.isfinite(diff)
    n = ok.sum(axis=1)
    ss = np.where(ok, diff, 0.0).astype(np.float64)
    rmse = np.where(n > 0, np.sqrt((ss**2).sum(axis=1) / np.maximum(n, 1)), np.nan)
    bias = np.where(n > 0, ss.sum(axis=1) / np.maximum(n, 1), np.nan)
    lo, hi = robust_range([pred, tgt])
    return reply(
        request,
        {
            "run": run.name,
            "method": method,
            "method_label": method_label(method, run.main_init),
            "requested": {"lat": lat, "lon": lon},
            "lat": float(run.grid.lat[i]),
            "lon": float(run.grid.lon[j]),
            "is_ocean": bool(np.isfinite(pred).any()),
            "dates": [d.strftime("%Y-%m-%d") for d in s["dates"]],
            "depths": run.depths,
            "prediction": pred,
            "target": tgt,
            "rmse_by_day": rmse,
            "bias_by_day": bias,
            "climatology": clim,
            "color_range": {"vmin": lo, "vmax": hi, "diverging": False},
            "units": "degC",
        },
    )


# ----------------------------------------------------------------------------------------------
# embeddings
# ----------------------------------------------------------------------------------------------
def _need_embeddings(run: Run) -> pd.DatetimeIndex:
    days = da.embedding_dates(run.path)
    if len(days) == 0:
        raise ApiError(404, "this run has no embeddings (run `oceanembed embed`)")
    return days


@router.get(
    "/runs/{run}/embeddings/dates",
    response_model=S.EmbeddingDatesResponse,
    responses=ERRORS,
    summary="Days that have an embedding map (the test split)",
)
def embedding_dates(request: Request, run: Run = Depends(run_dep)):
    days = _need_embeddings(run)
    return reply(request, {"run": run.name, "dates": [d.strftime("%Y-%m-%d") for d in days]})


def _coarse_ocean(lat, lon, ocean: np.ndarray, lat_e, lon_e) -> np.ndarray:
    iy = np.abs(np.asarray(lat, float)[:, None] - lat_e[None, :]).argmin(axis=1)
    ix = np.abs(np.asarray(lon, float)[:, None] - lon_e[None, :]).argmin(axis=1)
    out = np.zeros((lat_e.size, lon_e.size), dtype=bool)
    np.logical_or.at(out, (iy[:, None], ix[None, :]), ocean)
    return out


def _embedding_day(run: Run, days: pd.DatetimeIndex, date: str | None) -> pd.Timestamp:
    if date is None or not DATE_RE.match(date):
        raise ApiError(400, f"'date' must be a date formatted YYYY-MM-DD, got {date!r}")
    try:
        d = pd.Timestamp(date)
    except ValueError as e:
        raise ApiError(400, f"'date' is not a valid calendar date: {date!r}") from e
    if d < days[0] or d > days[-1]:
        raise ApiError(
            400, f"date {date} is outside the embedding range {days[0].date()} .. {days[-1].date()}"
        )
    if d not in days:
        raise ApiError(404, f"no embedding for {date} (a gap inside the embedding range)")
    return d


def _embedding_ocean(store: Store, run: Run, lat_e: np.ndarray, lon_e: np.ndarray) -> np.ndarray:
    all_days = store.dates(run)
    if len(all_days) == 0:
        return np.ones((lat_e.size, lon_e.size), dtype=bool)
    surf = np.isfinite(store.prediction(run, all_days[0])[0])
    return _coarse_ocean(run.grid.lat, run.grid.lon, surf, lat_e, lon_e)


def _cell_size(lat_e: np.ndarray, lon_e: np.ndarray) -> list[float]:
    return [
        float(np.diff(lat_e).mean()) if lat_e.size > 1 else float("nan"),
        float(np.diff(lon_e).mean()) if lon_e.size > 1 else float("nan"),
    ]


@router.get(
    "/runs/{run}/embeddings",
    response_model=S.EmbeddingResponse,
    responses=ERRORS,
    summary="PCA components of the embedding map as an (n, y, x) array scaled 0..1",
    description="Components 1-3 (the default) are the R, G, B channels. The PCA is fitted once per "
    "run on a sample of days, so colours are comparable between days; `n_components` (up to 8) "
    "returns more components, each scaled 0..1 on its own, with their explained variance.",
)
def embeddings(
    request: Request,
    run: Run = Depends(run_dep),
    store: Store = Depends(get_store),
    date: str | None = Query(None, description="YYYY-MM-DD, one of /embeddings/dates"),
    n_components: int = Query(3, ge=1, le=MAX_COMPONENTS, description="PCA components, 1..8"),
):
    days = _need_embeddings(run)
    d = _embedding_day(run, days, date)
    rgb = da.embedding_pca_rgb(run.path, d, n_components=n_components)
    if rgb is None:
        raise ApiError(404, f"no embedding for {date}")
    if n_components > rgb.sizes["rgb"]:
        raise ApiError(400, f"n_components must be at most {rgb.sizes['rgb']} (embedding size)")
    lat_e, lon_e = np.asarray(rgb["lat"].values, float), np.asarray(rgb["lon"].values, float)
    ocean = _embedding_ocean(store, run, lat_e, lon_e)
    data = rgb.transpose("rgb", "y", "x").values
    return reply(
        request,
        {
            "run": run.name,
            "date": d.strftime("%Y-%m-%d"),
            "shape": list(data.shape),
            "data": data,
            "explained_variance_ratio": rgb.attrs["explained_variance_ratio"],
            "lat": array_to_list(lat_e, 6),
            "lon": array_to_list(lon_e, 6),
            "cell_size": _cell_size(lat_e, lon_e),
            "ocean": ocean.astype(int),
        },
    )


@router.get(
    "/runs/{run}/embeddings/similar",
    response_model=S.EmbeddingSimilarityResponse,
    responses=ERRORS,
    summary="Cosine similarity of one embedding cell to every cell, over all embedding features",
    description="The reference is the embedding cell nearest to (`lat`, `lon`). The similarity uses "
    "the full embedding vector (`emb_dim` features), not the PCA components. With `center=true` "
    "the run's mean embedding (the PCA sample mean) is subtracted first, which removes the "
    "component common to all cells and spreads the values over -1..1. Land cells (`ocean` = 0) are "
    "`null`; the range covers the other ocean cells.",
)
def embeddings_similar(
    request: Request,
    run: Run = Depends(run_dep),
    store: Store = Depends(get_store),
    date: str | None = Query(None, description="YYYY-MM-DD, one of /embeddings/dates"),
    lat: float = Query(..., description="degrees north, inside the grid"),
    lon: float = Query(..., description="degrees east, inside the grid"),
    center: bool = Query(False, description="subtract the run's mean embedding first"),
):
    days = _need_embeddings(run)
    d = _embedding_day(run, days, date)
    store.cell(run, lat, lon)  # 400 when the point is outside the grid
    emb = da.load_embeddings(run.path, d)  # (emb, y, x)
    if emb is None:
        raise ApiError(404, f"no embedding for {date}")
    lat_e, lon_e = np.asarray(emb["lat"].values, float), np.asarray(emb["lon"].values, float)
    iy, ix = int(np.abs(lat_e - lat).argmin()), int(np.abs(lon_e - lon).argmin())
    x = emb.transpose("y", "x", "emb").values.astype(np.float64)
    if center:
        mean = da.embedding_pca_mean(run.path)
        if mean is not None:
            x = x - mean
    norm = np.linalg.norm(x, axis=-1)
    ref = x[iy, ix]
    denom = norm * np.linalg.norm(ref)
    sim = np.where(denom > 0, (x @ ref) / np.where(denom > 0, denom, 1.0), np.nan)
    sim = np.clip(sim, -1.0, 1.0)
    ocean = _embedding_ocean(store, run, lat_e, lon_e)
    sim = np.where(ocean, sim, np.nan)  # land cells carry no meaningful embedding: null
    others = ocean.copy()
    others[iy, ix] = False
    vals = sim[others & np.isfinite(sim)]
    rng = {
        "min": float(vals.min()) if vals.size else None,
        "max": float(vals.max()) if vals.size else None,
    }
    return reply(
        request,
        {
            "run": run.name,
            "date": d.strftime("%Y-%m-%d"),
            "requested": {"lat": lat, "lon": lon},
            "lat": float(lat_e[iy]),
            "lon": float(lon_e[ix]),
            "y_index": iy,
            "x_index": ix,
            "emb_dim": int(x.shape[-1]),
            "centered": bool(center),
            "reference_is_ocean": bool(ocean[iy, ix]),
            "shape": list(sim.shape),
            "data": np.round(sim, 4),
            "similarity_range": rng,
            "lat_values": array_to_list(lat_e, 6),
            "lon_values": array_to_list(lon_e, 6),
            "cell_size": _cell_size(lat_e, lon_e),
            "ocean": ocean.astype(int),
        },
    )
