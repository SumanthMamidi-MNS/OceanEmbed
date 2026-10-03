"""Reshape the metrics JSON files into frontend-friendly structures (pure functions on dicts)."""

from __future__ import annotations

from typing import Any

import numpy as np

METRIC_NAMES = ["n", "rmse", "bias", "mae", "corr_raw", "corr_anom", "skill_vs_clim"]
HEADLINE_DEPTHS = (0.0, 50.0, 100.0, 200.0, 500.0)


def _num(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else x


def method_infos(*raws: dict | None) -> list[dict[str, Any]]:
    """Methods described by the ``metadata.methods`` blocks of the given metrics files (union, in
    first-seen order) with flags telling in which file each one is scored."""
    glorys, argo = (raws + (None, None))[:2]
    infos: dict[str, dict[str, Any]] = {}
    for raw in (glorys, argo):
        for key, m in ((raw or {}).get("metadata", {}).get("methods") or {}).items():
            infos.setdefault(key, {"key": key, **m})
    out = []
    for key, m in infos.items():
        out.append(
            {
                "key": key,
                "label": str(m.get("label", key)),
                "kind": str(m.get("kind", "model")),
                "tag": m.get("tag"),
                "pretrained": m.get("pretrained"),
                "epoch": m.get("epoch"),
                "val_rmse": _num(m.get("val_rmse")),
                "description": m.get("description"),
                "in_glorys_metrics": key in (glorys or {}).get("methods", {}),
                "in_argo_metrics": key in (argo or {}).get("methods", {}),
            }
        )
    return out


def _pooled_key(method: dict) -> str | None:
    return next((k for k in method if k.startswith("pooled")), None)


def _blocks(methods: dict[str, dict]) -> dict[str, Any]:
    """overall / pooled / per_depth (metric -> method -> list) of a ``{method: block}`` mapping."""
    overall = {k: m.get("overall") for k, m in methods.items() if m.get("overall")}
    pooled = {}
    for k, m in methods.items():
        pk = _pooled_key(m)
        if pk and m.get(pk):
            pooled[k] = m[pk]
    depths: list[float] = []
    per_depth: dict[str, dict[str, list]] = {}
    for k, m in methods.items():
        pd_block = m.get("per_depth")
        if not pd_block:
            continue
        depths = depths or pd_block.get("depth", [])
        for metric, values in pd_block.items():
            if metric != "depth":
                per_depth.setdefault(metric, {})[k] = values
    return {"overall": overall, "pooled": pooled, "per_depth": per_depth, "depths": depths}


def _reshape_year(block: dict) -> dict[str, Any]:
    """One ``per_year`` entry (``{n_days | n_profiles, methods: {method: block}}``): the counts as
    they are, the methods metric-major like the whole period."""
    methods = block.get("methods", {})
    out = {k: v for k, v in block.items() if k != "methods"}
    top = _blocks(methods)
    basins: dict[str, dict[str, Any]] = {}
    for k, m in methods.items():
        for basin, b in (m.get("per_basin") or {}).items():
            basins.setdefault(basin, {})[k] = b
    per_basin = {b: _blocks(ms) for b, ms in basins.items()}
    for b in per_basin.values():
        b.pop("depths", None)
    return {
        **out,
        "overall": top["overall"],
        "pooled": top["pooled"],
        "per_depth": top["per_depth"],
        "per_basin": per_basin,
    }


def reshape_metrics(run: str, raw: dict, reference: str) -> dict[str, Any]:
    """Method-major file -> metric-major structure. Nothing from the source file is dropped: unknown
    top-level keys (e.g. ``gridded_argo``) are passed through in ``extra``."""
    methods = raw.get("methods", {})
    top = _blocks(methods)
    basins: dict[str, dict[str, Any]] = {}
    for k, m in methods.items():
        for basin, block in (m.get("per_basin") or {}).items():
            basins.setdefault(basin, {})[k] = block
    per_basin = {b: _blocks(ms) for b, ms in basins.items()}
    for b in per_basin.values():
        b.pop("depths", None)
    meta = raw.get("metadata", {})
    rng = meta.get("pooled_depth_range_m") or [50.0, 200.0]
    daily = raw.get("daily_rmse")
    daily_out = (
        {
            "dates": daily.get("dates", []),
            "depths": daily.get("depth", []),
            "methods": daily.get("rmse", {}),
        }
        if daily
        else None
    )
    # richer daily block (bias, spatial anomaly correlation, pooled RMSE); runs evaluated before
    # these were stored simply lack the keys
    daily_full = (
        {
            "dates": daily.get("dates", []),
            "depths": daily.get("depth", []),
            "pooled_range_m": daily.get("pooled_range_m") or [float(rng[0]), float(rng[1])],
            "rmse": daily.get("rmse", {}),
            **{
                k: daily[k]
                for k in ("bias", "corr_anom", "pooled_rmse", "pooled_bias", "pooled_corr_anom")
                if k in daily
            },
        }
        if daily
        else None
    )
    per_year = (
        {y: _reshape_year(b) for y, b in raw["per_year"].items()} if raw.get("per_year") else None
    )
    extra = {
        k: v for k, v in raw.items() if k not in ("metadata", "methods", "daily_rmse", "per_year")
    }
    return {
        "run": run,
        "reference": reference,
        "metadata": meta,
        "methods": method_infos(raw),
        "metric_names": METRIC_NAMES,
        "overall": top["overall"],
        "pooled": top["pooled"],
        "pooled_range_m": [float(rng[0]), float(rng[1])],
        "depths": top["depths"] or meta.get("depths", []),
        "per_depth": top["per_depth"],
        "per_basin": per_basin,
        "daily_rmse": daily_out,
        "daily": daily_full,
        "per_year": per_year,
        "extra": extra,
    }


def _gain(model: float | None, ref: float | None) -> float | None:
    if model is None or ref is None or not ref or not np.isfinite(model) or not np.isfinite(ref):
        return None
    return round(100.0 * (1.0 - model / ref), 2)


def experiment_rows(
    glorys: dict | None, argo: dict | None, depths: tuple[float, ...] = HEADLINE_DEPTHS
) -> tuple[list[dict[str, Any]], list[float]]:
    """One row per method / ablation with the headline numbers of both validations."""
    infos = {m["key"]: m for m in method_infos(glorys, argo)}
    g_methods = (glorys or {}).get("methods", {})
    a_methods = (argo or {}).get("methods", {})
    keys = list(dict.fromkeys([*g_methods, *a_methods]))
    all_depths = (glorys or {}).get("metadata", {}).get("depths") or (argo or {}).get(
        "metadata", {}
    ).get("depths", [])
    chosen = [d for d in depths if d in all_depths]

    def pooled_rmse(k):
        m = g_methods.get(k, {})
        pk = _pooled_key(m)
        return _num((m.get(pk) or {}).get("rmse")) if pk else None

    clim, ridge = pooled_rmse("climatology"), pooled_rmse("ridge")
    rows = []
    for k in keys:
        info = infos.get(k, {"key": k, "label": k, "kind": "model"})
        g = g_methods.get(k, {})
        pk = _pooled_key(g)
        per_depth = g.get("per_depth") or {}
        depth_blocks = None
        if per_depth.get("depth"):
            depth_blocks = {}
            for d in chosen:
                i = per_depth["depth"].index(d)
                depth_blocks[f"{d:g}"] = {
                    m: _num(per_depth[m][i]) for m in METRIC_NAMES if m in per_depth
                }
        rows.append(
            {
                "method": k,
                "label": info["label"],
                "kind": info["kind"],
                "pretrained": info.get("pretrained"),
                "epoch": info.get("epoch"),
                "val_rmse": info.get("val_rmse"),
                "glorys_overall": g.get("overall"),
                "glorys_pooled": g.get(pk) if pk else None,
                "glorys_depths": depth_blocks,
                "argo_overall": a_methods.get(k, {}).get("overall"),
                "pooled_rmse_gain_vs_climatology_pct": _gain(pooled_rmse(k), clim),
                "pooled_rmse_gain_vs_ridge_pct": _gain(pooled_rmse(k), ridge),
            }
        )
    return rows, chosen
