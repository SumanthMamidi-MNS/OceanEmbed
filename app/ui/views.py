"""Page-level builders: turn a run's metrics / fields into cards, tables and figures.

Pure functions (no Streamlit, no file IO) so the pages stay thin and the tests can build every
main figure from a finished run. Low-level drawing lives in ``figures``; tokens in ``theme``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from app.ui import figures as F
from app.ui import theme as T

POOLED = "pooled_50_200m"
MAP_QUANTITIES = ("rmse", "bias", "corr_anom", "corr_raw", "skill_vs_clim")


# ----------------------------------------------------------------------------------------
# methods
# ----------------------------------------------------------------------------------------
def method_labels(metrics: dict | None) -> dict[str, str]:
    info = (metrics or {}).get("metadata", {}).get("methods", {})
    return {k: str(v.get("label", k)) for k, v in info.items()}


def styles_for(metrics: dict | None, extra: tuple[str, ...] = ()) -> dict[str, T.MethodStyle]:
    """Fixed style of every method of a metrics file (drawing order, back to front)."""
    keys = [*(metrics or {}).get("methods", {}), *extra]
    return T.method_styles(keys, method_labels(metrics))


def pooled_label(metrics: dict | None) -> str:
    rng = (metrics or {}).get("metadata", {}).get("pooled_depth_range_m") or [50, 200]
    return f"{rng[0]:g}–{rng[1]:g} m"


# ----------------------------------------------------------------------------------------
# overview
# ----------------------------------------------------------------------------------------
def pipeline_steps(meta: dict) -> list[tuple[str, str, str]]:
    """(stage, title, description) of the three stages, with this run's actual sizes."""
    grid = meta.get("grid", {})
    model = meta.get("config", {}).get("model", {})
    n_lat, n_lon, res = grid.get("n_lat"), grid.get("n_lon"), grid.get("resolution")
    depths = grid.get("depths") or []
    size = f"{n_lat} × {n_lon} cells at {res:g}°" if n_lat and n_lon and res else "the model grid"
    emb = model.get("emb_dim")
    latent = (
        f"{emb} features on a {n_lat // 4} × {n_lon // 4} map"
        if emb and n_lat and n_lon
        else "a compact latent map"
    )
    levels = (
        f"{len(depths)} standard depths, {depths[0]:g}–{depths[-1]:g} m"
        if depths
        else "the standard depths"
    )
    return [
        (
            "Input",
            "Surface satellite fields",
            "SST, salinity, sea level anomaly, currents (U, V) and winds (U, V): 7 daily "
            f"fields on {size}.",
        ),
        (
            "Embedding",
            "Satellite embedding engine",
            "A CNN stem and Transformer encoder, pretrained by masked reconstruction, compress "
            f"the surface state into {latent}.",
        ),
        (
            "Output",
            "Subsurface temperature",
            f"A decoder reconstructs temperature at {levels}, daily, on the same grid. No "
            "subsurface data is used at prediction time.",
        ),
    ]


def headline_cards(metrics: dict, argo: dict | None) -> list[T.Card]:
    """Model vs its baselines over the pooled depth band, plus the Argo coverage."""
    pooled = {k: v.get(POOLED, {}) for k, v in metrics.get("methods", {}).items()}
    model = pooled.get("model", {})
    ridge = pooled.get("ridge", {})
    clim = pooled.get("climatology", {})
    band = pooled_label(metrics)
    gain = ""
    if T.is_number(model.get("rmse")) and T.is_number(clim.get("rmse")) and clim["rmse"] > 0:
        pct = 100 * (1 - model["rmse"] / clim["rmse"])
        gain = f"<br><b>{T.fmt(pct, 0)} %</b> lower than climatology"
    out = [
        T.Card(
            f"RMSE, {band}",
            T.fmt(model.get("rmse")),
            "°C",
            f"Ridge <b>{T.fmt(ridge.get('rmse'))}</b> · climatology "
            f"<b>{T.fmt(clim.get('rmse'))}</b> °C{gain}",
            lead=True,
        ),
        T.Card(
            f"Skill vs climatology, {band}",
            T.fmt(model.get("skill_vs_clim")),
            "",
            "1 − MSE / MSE of climatology (0 = no skill)<br>"
            f"Ridge <b>{T.fmt(ridge.get('skill_vs_clim'))}</b>",
        ),
        T.Card(
            f"Anomaly correlation, {band}",
            T.fmt(model.get("corr_anom")),
            "",
            f"Ridge <b>{T.fmt(ridge.get('corr_anom'))}</b><br>Raw correlation "
            f"<b>{T.fmt(model.get('corr_raw'))}</b> is inflated by seasonal and vertical gradients",
        ),
    ]
    if argo:
        meta = argo.get("metadata", {})
        vs = argo.get("methods", {}).get("model", {}).get("overall", {})
        out.append(
            T.Card(
                "Argo matchups",
                T.fmt_int(meta.get("n_matchups")),
                "",
                f"<b>{T.fmt_int(meta.get('n_profiles_used'))}</b> profiles<br>OceanEmbed RMSE vs "
                f"Argo <b>{T.fmt(vs.get('rmse'))}</b> °C, all depths",
            )
        )
    else:
        out.append(T.Card("Argo matchups", T.MISSING, "", "No Argo validation for this run yet"))
    return out


def run_summary(meta: dict, metrics: dict | None) -> list[tuple[str, str]]:
    grid = meta.get("grid", {})
    split = meta.get("split", {})

    def span(d: dict | None) -> str:
        return f"{d.get('start', '?')} → {d.get('end', '?')}" if d else T.MISSING

    rows = [
        ("Data source", str(meta.get("data_source", "unknown"))),
        ("Period", span(meta.get("time"))),
        ("Train", span(split.get("train"))),
        ("Validation", span(split.get("val"))),
        ("Test", span(split.get("test"))),
    ]
    if grid:
        lat, lon = grid.get("lat", ["?", "?"]), grid.get("lon", ["?", "?"])
        depths = grid.get("depths") or []
        rows.append(
            (
                "Grid",
                f"{grid.get('n_lat')} × {grid.get('n_lon')} cells at {grid.get('resolution')}°, "
                f"{lat[0]}–{lat[1]}°N, {lon[0]}–{lon[1]}°E",
            )
        )
        rows.append(
            ("Depths", f"{len(depths)} levels: " + ", ".join(f"{d:g}" for d in depths) + " m")
        )
    if metrics:
        m = metrics.get("metadata", {})
        rows.append(
            ("Evaluated", f"{m.get('split', '?')} split, {m.get('n_days', '?')} days vs GLORYS")
        )
    if meta.get("updated"):
        rows.append(("Updated", str(meta["updated"]).replace("T", " ").replace("Z", " UTC")))
    return rows


# ----------------------------------------------------------------------------------------
# metrics against depth
# ----------------------------------------------------------------------------------------
def metric_profile(
    blocks: dict[str, dict],
    styles: dict[str, T.MethodStyle],
    metric: str = "rmse",
    *,
    reference: str = "GLORYS",
    height: int = 440,
    showlegend: bool = True,
    x_range: tuple[float, float] | None = None,
    y_title: str | None = "depth (m)",
) -> go.Figure:
    """One metric against depth for every method. ``blocks`` maps method -> metric block (with a
    column-oriented ``per_depth``)."""
    spec = T.METRICS[metric]
    present = [k for k in styles if k in blocks and metric in blocks[k].get("per_depth", {})]
    depths = blocks[present[0]]["per_depth"]["depth"] if present else [0.0, 1.0]
    series = [F.Series(styles[k], blocks[k]["per_depth"][metric]) for k in present]
    title = spec.axis if not spec.unit else f"{spec.label} vs {reference} ({spec.unit})"
    fig = F.depth_profiles(
        depths, series, x_title=title, unit=spec.unit, decimals=spec.decimals + 1,
        zero_line=spec.zero_line, height=height, showlegend=showlegend, x_range=x_range,
        y_title=y_title,
    )  # fmt: skip
    if metric in ("rmse", "mae") and x_range is None:
        fig.update_xaxes(rangemode="tozero")
    return fig


def metric_range(blocks_list: list[dict[str, dict]], metric: str) -> tuple[float, float] | None:
    """Common x range of one metric over several panels (e.g. the two basins)."""
    vals = [
        T.to_array(b[k]["per_depth"][metric])
        for b in blocks_list
        for k in b
        if metric in b[k].get("per_depth", {})
    ]
    v = np.concatenate(vals) if vals else np.array([])
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    lo, hi = float(v.min()), float(v.max())
    if metric in ("rmse", "mae"):
        lo = 0.0
    pad = 0.05 * (hi - lo or 1.0)
    return lo - (0 if lo == 0 else pad), hi + pad


def summary_table(
    blocks: dict[str, dict], styles: dict[str, T.MethodStyle], block: str = POOLED
) -> pd.DataFrame:
    """One row per method for the ``overall`` or pooled block (the table twin of the charts)."""
    rows = []
    for k in T.legend_order(styles):
        b = blocks.get(k, {}).get(block)
        if not b:
            continue
        rows.append(
            {
                "Method": styles[k].label,
                "RMSE (°C)": b.get("rmse"),
                "Bias (°C)": b.get("bias"),
                "MAE (°C)": b.get("mae"),
                "Anomaly corr.": b.get("corr_anom"),
                "Raw corr.": b.get("corr_raw"),
                "Skill": b.get("skill_vs_clim"),
                "N": b.get("n"),
            }
        )
    df = pd.DataFrame(rows)
    numeric = [c for c in df.columns if c != "Method"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")  # None -> NaN
    return df


def per_depth_table(
    blocks: dict[str, dict], styles: dict[str, T.MethodStyle], metric: str
) -> pd.DataFrame:
    """Depth x method table of one metric."""
    cols: dict[str, np.ndarray] = {}
    depths = None
    for k in T.legend_order(styles):
        pd_block = blocks.get(k, {}).get("per_depth", {})
        if metric in pd_block:
            depths = pd_block["depth"]
            cols[styles[k].label] = T.to_array(pd_block[metric])
    if depths is None:
        return pd.DataFrame()
    return pd.DataFrame({"Depth (m)": [f"{d:g}" for d in depths], **cols})


def basin_blocks(metrics: dict, basin: str) -> dict[str, dict]:
    """method -> metric blocks of one basin."""
    return {
        k: v["per_basin"][basin]
        for k, v in metrics.get("methods", {}).items()
        if basin in v.get("per_basin", {})
    }


def basins_of(metrics: dict) -> list[str]:
    out: list[str] = []
    for v in metrics.get("methods", {}).values():
        out.extend(b for b in v.get("per_basin", {}) if b not in out)
    return out


def daily_rmse(
    metrics: dict, styles: dict[str, T.MethodStyle], depth: float | None = None
) -> go.Figure | None:
    """Daily domain-mean RMSE of every method at one depth (``None``: RMS over the pooled band)."""
    daily = metrics.get("daily_rmse") or {}
    if not daily.get("dates") or not daily.get("rmse"):
        return None
    depths = np.asarray(daily["depth"], dtype=float)
    if depth is None:
        lo, hi = metrics.get("metadata", {}).get("pooled_depth_range_m") or [50, 200]
        sel = (depths >= lo) & (depths <= hi)
        title = f"RMSE vs GLORYS, {pooled_label(metrics)} (°C)"
    else:
        sel = np.isclose(depths, depth)
        title = f"RMSE vs GLORYS at {T.fmt_depth(depth)} (°C)"
    series = []
    for k, s in styles.items():
        if k not in daily["rmse"]:
            continue
        arr = np.array(
            [[np.nan if v is None else v for v in row] for row in daily["rmse"][k]], dtype=float
        )
        with np.errstate(invalid="ignore"):
            series.append(F.Series(s, np.sqrt(np.nanmean(arr[:, sel] ** 2, axis=1))))
    return F.time_series(daily["dates"], series, y_title=title)


# ----------------------------------------------------------------------------------------
# maps of one day
# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DaySlices:
    """Prediction, its reference and their difference at one depth, with shared colour limits."""

    predicted: xr.DataArray
    reference: xr.DataArray | None
    difference: xr.DataArray | None
    reference_name: str
    difference_name: str
    has_target: bool
    vmin: float
    vmax: float
    limit: float


def day_slices(
    pred: xr.DataArray, target: xr.DataArray | None, clim: xr.DataArray | None, depth: float
) -> DaySlices:
    """Slices at ``depth``. Without a target the reference falls back to the climatology (and the
    difference becomes the predicted anomaly); without either, only the prediction is shown."""
    p = pred.sel(depth=depth, method="nearest")
    ref, ref_name, diff_name = None, "", ""
    if target is not None:
        ref = target.sel(depth=depth, method="nearest")
        ref_name, diff_name = "GLORYS target", "Predicted − target"
    elif clim is not None:
        ref = clim.sel(depth=depth, method="nearest")
        ref_name, diff_name = "Climatology", "Predicted − climatology (anomaly)"
    diff = None
    if ref is not None:
        ref = ref.assign_coords(lat=p["lat"].values, lon=p["lon"].values)
        diff = p - ref
    vmin, vmax = F.shared_range(p.values, None if ref is None else ref.values)
    limit = F.symmetric_limit(diff.values) if diff is not None else 1.0
    return DaySlices(p, ref, diff, ref_name, diff_name, target is not None, vmin, vmax, limit)


def slice_stats(s: DaySlices) -> dict[str, float]:
    """RMSE, bias, largest error and pattern correlation over the ocean cells of one slice."""
    if s.reference is None:
        return {}
    a, b = np.asarray(s.predicted.values, float), np.asarray(s.reference.values, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return {}
    d = a[ok] - b[ok]
    corr = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 and d.std() > 0 else np.nan
    return {
        "rmse": float(np.sqrt(np.mean(d**2))),
        "bias": float(d.mean()),
        "max_abs": float(np.abs(d).max()),
        "corr": corr,
        "n": int(ok.sum()),
    }


def slice_cards(s: DaySlices) -> list[T.Card]:
    st = slice_stats(s)
    if not st:
        return []
    ref = "target" if s.has_target else "clim."
    return [
        T.Card("RMSE" if s.has_target else "RMS anomaly", T.fmt(st["rmse"]), "°C"),
        T.Card("Bias" if s.has_target else "Mean anomaly", T.fmt(st["bias"], signed=True), "°C"),
        T.Card("Max |diff|", T.fmt(st["max_abs"]), "°C"),
        T.Card(f"Corr. with {ref}", T.fmt(st["corr"], 2)),
    ]


def surface_map(
    surface: xr.Dataset, name: str, *, width: int = F.WIDTH_QUARTER, compact: bool = True
):
    """Map of one surface input field with its own colormap (diverging for signed fields)."""
    spec = T.surface_field(name, dict(surface[name].attrs))
    return F.field_map(
        surface[name], colorscale=spec.scale, unit=spec.unit, quantity=spec.label,
        diverging=spec.diverging, width=width, compact=compact, decimals=3,
    )  # fmt: skip


# ----------------------------------------------------------------------------------------
# profile at a point
# ----------------------------------------------------------------------------------------
def default_point(pred: xr.DataArray) -> tuple[float, float]:
    """A sensible first point: the full-depth ocean cell closest to the centre of the domain."""
    lat, lon = np.asarray(pred["lat"].values, float), np.asarray(pred["lon"].values, float)
    deep = np.isfinite(pred.isel(depth=-1).values)
    if not deep.any():
        deep = np.isfinite(pred.isel(depth=0).values)
    if not deep.any():
        return float(lat[lat.size // 2]), float(lon[lon.size // 2])
    dist = (lat[:, None] - lat.mean()) ** 2 + (lon[None, :] - lon.mean()) ** 2
    j, i = np.unravel_index(np.where(deep, dist, np.inf).argmin(), deep.shape)
    return float(lat[j]), float(lon[i])


def has_target(ds: xr.Dataset) -> bool:
    """Whether a profile / section Dataset carries a real target (not the NaN placeholder)."""
    return bool(np.isfinite(ds["target"].values).any())


def profile_is_ocean(prof: xr.Dataset) -> bool:
    return bool(np.isfinite(prof["predicted"].values).any())


def point_cards(prof: xr.Dataset, depth: float) -> list[T.Card]:
    """The selected cell in numbers: values at the chosen depth and the whole-profile error."""
    at = prof.sel(depth=depth, method="nearest")
    pred, tgt, clim = (float(at[v]) for v in ("predicted", "target", "climatology"))
    where = T.fmt_depth(float(at["depth"]))
    valid = prof["depth"].values[np.isfinite(prof["predicted"].values)]
    out = [T.Card(f"Predicted, {where}", T.fmt(pred), "°C")]
    if np.isfinite(tgt):
        diff = (prof["predicted"] - prof["target"]).values
        out.append(T.Card(f"Target, {where}", T.fmt(tgt), "°C"))
        out.append(T.Card("Profile RMSE", T.fmt(float(np.sqrt(np.nanmean(diff**2)))), "°C"))
    else:
        out.append(T.Card(f"Climatology, {where}", T.fmt(clim), "°C"))
    if valid.size:
        out.append(T.Card("Deepest level", f"{valid.max():g}", "m"))
    return out


def point_profile(prof: xr.Dataset, *, height: int = 470) -> go.Figure:
    """Predicted vs target vs climatology temperature at one grid cell."""
    styles = T.method_styles(["model", "glorys", "climatology"])
    series = [
        F.Series(styles["climatology"], prof["climatology"].values, "Climatology"),
        F.Series(styles["glorys"], prof["target"].values, "GLORYS target"),
        F.Series(styles["model"], prof["predicted"].values, "OceanEmbed prediction"),
    ]
    return F.depth_profiles(prof["depth"].values, series, x_title="temperature (°C)", height=height)


def profile_table(prof: xr.Dataset) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Depth (m)": [f"{d:g}" for d in prof["depth"].values],
            "Predicted (°C)": prof["predicted"].values,
            "Target (°C)": prof["target"].values,
            "Climatology (°C)": prof["climatology"].values,
        }
    )
    df["Predicted − target (°C)"] = df["Predicted (°C)"] - df["Target (°C)"]
    return df


# ----------------------------------------------------------------------------------------
# verification maps
# ----------------------------------------------------------------------------------------
def map_inventory(maps: xr.Dataset) -> dict[str, list[str]]:
    """quantity -> methods available in the verification maps (``<quantity>_<method>``)."""
    out: dict[str, list[str]] = {}
    for q in MAP_QUANTITIES:
        methods = [
            str(v).removeprefix(f"{q}_") for v in maps.data_vars if str(v).startswith(q + "_")
        ]
        if methods:
            out[q] = methods
    return out


def error_map(
    maps: xr.Dataset,
    quantity: str,
    method: str,
    depth: float,
    *,
    limits: tuple[float, float] | None = None,
    boxes: dict | None = None,
    width: int = F.WIDTH_HALF,
) -> go.Figure:
    """Map of one verification quantity for one method at one depth. ``limits`` (from
    ``error_limits``) lets two methods share a colour range."""
    spec = T.METRICS[quantity]
    field = maps[f"{quantity}_{method}"].sel(depth=depth, method="nearest")
    lo, hi = limits or error_limits(maps, quantity, [method], depth)
    diverging = quantity in ("bias", "skill_vs_clim")
    return F.field_map(
        field,
        colorscale=T.DIVERGING_SCALE if diverging else T.MAGNITUDE_SCALE,
        zmin=lo, zmax=hi, unit=spec.unit, quantity=spec.label, diverging=diverging,
        width=width, boxes=boxes, decimals=3,
    )  # fmt: skip


def error_limits(
    maps: xr.Dataset, quantity: str, methods: list[str], depth: float
) -> tuple[float, float]:
    """Colour limits shared by ``methods``: zero-based for errors, symmetric for signed ones."""
    fields = [
        maps[f"{quantity}_{m}"].sel(depth=depth, method="nearest").values
        for m in methods
        if f"{quantity}_{m}" in maps
    ]
    if quantity == "bias":
        lim = F.symmetric_limit(*fields)
        return -lim, lim
    if quantity == "skill_vs_clim":
        return -1.0, 1.0
    lo, hi = F.shared_range(*fields, pct=1.0)
    if quantity in ("rmse", "mae"):
        return 0.0, hi
    return min(lo, 0.0) if lo < 0 else lo, min(hi, 1.0)


# ----------------------------------------------------------------------------------------
# Argo
# ----------------------------------------------------------------------------------------
def argo_columns(argo: dict | None, matchups: pd.DataFrame) -> dict[str, str]:
    """method key -> column of the matchup table (the climatology column is called ``clim``)."""
    cols = dict((argo or {}).get("metadata", {}).get("column_for_method") or {})
    if not cols:
        cols = {m: ("clim" if m == "climatology" else m) for m in (argo or {}).get("methods", {})}
    return {m: c for m, c in cols.items() if c in matchups.columns}


def argo_cards(argo: dict) -> list[T.Card]:
    meta = argo.get("metadata", {})
    dropped = meta.get("dropped_profiles") or {}
    n_dropped = sum(int(v) for v in dropped.values() if T.is_number(v))
    detail = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in dropped.items() if v) or "none"
    model = argo.get("methods", {}).get("model", {}).get("overall", {})
    glorys = argo.get("methods", {}).get("glorys", {}).get("overall", {})
    return [
        T.Card(
            "Profiles used",
            T.fmt_int(meta.get("n_profiles_used")),
            "",
            f"of <b>{T.fmt_int(meta.get('n_profiles_loaded'))}</b> loaded",
        ),
        T.Card("Matchups", T.fmt_int(meta.get("n_matchups")), "", "(profile, depth) pairs"),
        T.Card("Profiles dropped", T.fmt_int(n_dropped), "", detail),
        T.Card(
            "OceanEmbed RMSE vs Argo",
            T.fmt(model.get("rmse")),
            "°C",
            f"GLORYS itself <b>{T.fmt(glorys.get('rmse'))}</b> °C (assimilates Argo)",
            lead=True,
        ),
    ]


def argo_scatter(matchups: pd.DataFrame, column: str, label: str, max_points: int = 20000):
    """Observed vs predicted for one method (a fixed random subsample above ``max_points``)."""
    df = matchups[["obs", column, "depth"]].dropna()
    if len(df) > max_points:
        df = df.sample(max_points, random_state=0)
    return F.obs_scatter(df["obs"], df[column], df["depth"], label=label)


def argo_profiles(matchups: pd.DataFrame) -> pd.DataFrame:
    """One row per profile: position, day and number of matched levels."""
    g = matchups.groupby("profile_id", sort=False)
    out = g.agg(lat=("lat", "first"), lon=("lon", "first"), time=("time", "first"))
    out["levels"] = g.size()
    return out.reset_index()


def argo_locations(
    matchups: pd.DataFrame,
    ocean: xr.DataArray | None,
    boxes: dict | None = None,
    width: int = F.WIDTH_HALF,
) -> go.Figure:
    prof = argo_profiles(matchups)
    hover = [
        f"{pd.Timestamp(t):%Y-%m-%d} · {n} levels"
        for t, n in zip(prof["time"], prof["levels"], strict=True)
    ]
    return F.points_map(ocean, prof["lat"], prof["lon"], hover, boxes=boxes, width=width)


# ----------------------------------------------------------------------------------------
# embeddings
# ----------------------------------------------------------------------------------------
def coarse_ocean(ocean: xr.DataArray | None, rgb: xr.DataArray) -> np.ndarray | None:
    """Ocean mask on the embedding grid: a cell is ocean if any model cell nearest to it is."""
    if ocean is None:
        return None
    lat_e, lon_e = np.asarray(rgb["lat"].values, float), np.asarray(rgb["lon"].values, float)
    iy = np.abs(np.asarray(ocean["lat"].values, float)[:, None] - lat_e[None, :]).argmin(axis=1)
    ix = np.abs(np.asarray(ocean["lon"].values, float)[:, None] - lon_e[None, :]).argmin(axis=1)
    out = np.zeros((lat_e.size, lon_e.size), dtype=bool)
    np.logical_or.at(out, (iy[:, None], ix[None, :]), np.asarray(ocean.values, dtype=bool))
    return out


def embedding_figure(rgb: xr.DataArray, ocean: xr.DataArray | None, *, width: int = F.WIDTH_HALF):
    return F.rgb_map(rgb, coarse_ocean(ocean, rgb), width=width)


def variance_text(rgb: xr.DataArray) -> str:
    evr = rgb.attrs.get("explained_variance_ratio") or []
    names = ("red", "green", "blue")
    parts = [f"PC{i + 1} ({names[i]}) {100 * v:.0f} %" for i, v in enumerate(evr[:3])]
    total = f", {100 * sum(evr[:3]):.0f} % of the variance in total" if evr else ""
    return " · ".join(parts) + total
