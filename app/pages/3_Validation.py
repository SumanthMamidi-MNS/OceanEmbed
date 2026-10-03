"""Validation: skill against GLORYS by depth and basin, error maps, daily series and the
comparison with Argo profiles."""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit only puts app/ on sys.path; the UI package is imported as `app.ui` from the repo
# root (setup_page drops the duplicate entries that reruns would otherwise pile up).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
from app.ui import components as C
from app.ui import data as D
from app.ui import figures as F
from app.ui import theme as T
from app.ui import views as V

SECTIONS = ["By depth", "Basins", "Error maps", "Daily series", "Argo"]
DEPTH_PANELS = [["rmse", "bias", "skill_vs_clim"], ["corr_anom", "corr_raw"]]


def _table(df, precision: int = 3) -> None:
    st.dataframe(
        df.style.format(precision=precision, na_rep=T.MISSING, thousands=","), hide_index=True
    )


def pick_methods(styles: dict[str, T.MethodStyle]) -> dict[str, T.MethodStyle]:
    """Sidebar method filter; an empty selection means all methods."""
    order = T.legend_order(styles)
    with st.sidebar:
        st.markdown('<div class="oe-side-label">Methods</div>', unsafe_allow_html=True)
        chosen = st.multiselect(
            "Methods", order, default=order, format_func=lambda k: styles[k].label,
            key="validation_methods", label_visibility="collapsed",
            placeholder="All methods",
        )  # fmt: skip
    chosen = chosen or order
    return {k: s for k, s in styles.items() if k in chosen}


def render_by_depth(metrics: dict, styles: dict[str, T.MethodStyle]) -> None:
    blocks = metrics["methods"]
    C.legend([styles[k] for k in T.legend_order(styles)])
    for row in DEPTH_PANELS:
        cols = st.columns(3, gap="medium")
        for col, metric in zip(cols, row, strict=False):
            spec = T.METRICS[metric]
            with col:
                C.panel_title(spec.label, spec.note)
                fig = V.metric_profile(blocks, styles, metric, height=400, showlegend=False)
                C.plot(fig, f"validation_depth_{metric}")
        if len(row) < len(cols):
            with cols[-1]:
                C.note(
                    "<b>Two correlations, on purpose.</b> Temperature is dominated by the "
                    "seasonal cycle and the vertical gradient, which any method reproduces, so "
                    "the raw correlation is close to 1 even for climatology. The anomaly "
                    "correlation removes the climatology from both sides and is the number to "
                    "trust. Climatology has no anomaly correlation by construction."
                )
    with st.expander("Tables: metrics by method and by depth"):
        st.caption("All depths, whole domain")
        _table(V.summary_table(blocks, styles, "overall"))
        st.caption(f"Pooled {V.pooled_label(metrics)}")
        _table(V.summary_table(blocks, styles))
        metric = st.selectbox(
            "Per-depth table", list(T.METRICS), format_func=lambda m: T.METRICS[m].label,
            key="validation_table_metric",
        )  # fmt: skip
        _table(V.per_depth_table(blocks, styles, metric))


def render_basins(metrics: dict, styles: dict[str, T.MethodStyle]) -> None:
    basins = V.basins_of(metrics)
    if not basins:
        C.empty_state("No basin breakdown", "This metrics file has no per-basin blocks.")
        return
    metric = st.segmented_control(
        "Metric", ["rmse", "bias", "corr_anom", "skill_vs_clim"], default="rmse",
        format_func=lambda m: T.METRICS[m].label, key="validation_basin_metric", required=True,
    )  # fmt: skip
    blocks = {b: V.basin_blocks(metrics, b) for b in basins}
    x_range = V.metric_range(
        [{k: v for k, v in blocks[b].items() if k in styles} for b in basins], metric
    )
    boxes = metrics.get("metadata", {}).get("basins", {})
    C.legend([styles[k] for k in T.legend_order(styles)])
    cols = st.columns(len(basins), gap="medium")
    for col, basin in zip(cols, basins, strict=True):
        with col:
            box = boxes.get(basin)
            caption = (
                f"{box[0]:g}–{box[1]:g}°N, {box[2]:g}–{box[3]:g}°E · same axis in both panels"
                if box
                else "Same axis in both panels"
            )
            C.panel_title(f"{T.basin_label(basin)}: {T.METRICS[metric].label}", caption)
            fig = V.metric_profile(
                blocks[basin], styles, metric, height=420, showlegend=False, x_range=x_range
            )
            C.plot(fig, f"validation_basin_{basin}")
    with st.expander("Tables: metrics by basin", expanded=True):
        for basin in basins:
            st.caption(f"{T.basin_label(basin)}, pooled {V.pooled_label(metrics)}")
            _table(V.summary_table(blocks[basin], styles))


def render_maps(run: D.Run, metrics: dict, styles: dict[str, T.MethodStyle]) -> None:
    maps = D.error_maps(run.path, run.stamp)
    if maps is None:
        C.empty_state(
            "No verification maps",
            "metrics/maps_glorys.nc is written by the evaluate step.",
            C.command_for(run, "evaluate"),
        )
        return
    inventory = V.map_inventory(maps)
    if not inventory:
        C.empty_state("No verification maps", "The maps file holds no known quantities.")
        return
    all_styles = V.styles_for(metrics)
    label = lambda k: all_styles[k].label if k in all_styles else k  # noqa: E731
    left, mid, right = st.columns(3, gap="medium")
    quantity = left.selectbox(
        "Quantity", list(inventory), format_func=lambda q: T.METRICS[q].label,
        key="validation_map_quantity",
    )  # fmt: skip
    order = [k for k in T.legend_order(all_styles) if k in inventory[quantity]]
    order += [k for k in inventory[quantity] if k not in order]
    method = mid.selectbox("Method", order, format_func=label, key="validation_map_method")
    others = [None] + [k for k in order if k != method]
    default = others.index("climatology") if "climatology" in others else min(1, len(others) - 1)
    other = right.selectbox(
        "Compare with", others, index=default, key=f"validation_map_other_{quantity}_{method}",
        format_func=lambda k: "Nothing" if k is None else label(k),
    )  # fmt: skip
    depth = C.depth_control([float(d) for d in maps["depth"].values])
    shown = [method] + ([other] if other else [])
    limits = V.error_limits(maps, quantity, shown, depth)
    boxes = metrics.get("metadata", {}).get("basins", {})
    spec = T.METRICS[quantity]
    cols = st.columns(len(shown), gap="medium")
    for col, m in zip(cols, shown, strict=True):
        with col:
            C.panel_title(
                f"{spec.label}: {label(m)}",
                f"At {T.fmt_depth(depth)}, over the test period. "
                + ("Shared colour range. " if len(shown) > 1 else "")
                + "Dotted boxes are the basins.",
            )
            fig = V.error_map(
                maps, quantity, m, depth, limits=limits, boxes=boxes,
                width=F.WIDTH_HALF if len(shown) > 1 else F.WIDTH_FULL,
            )  # fmt: skip
            C.plot(fig, f"validation_map_{m}")
    C.note(f"<b>{spec.label}.</b> {spec.note}")


def render_daily(metrics: dict, styles: dict[str, T.MethodStyle]) -> None:
    daily = metrics.get("daily_rmse") or {}
    if not daily.get("dates"):
        C.empty_state("No daily series", "This metrics file has no daily RMSE block.")
        return
    band = V.pooled_label(metrics)
    scope = st.segmented_control(
        "Depth", ["pooled", "level"], default="pooled", key="validation_daily_scope",
        format_func={"pooled": f"Pooled {band}", "level": "One depth level"}.get, required=True,
    )  # fmt: skip
    depth = None
    if scope == "level":
        depth = C.depth_control([float(d) for d in daily["depth"]])
    C.panel_title(
        "Daily RMSE against GLORYS, whole domain",
        (f"RMS over the {band} levels" if depth is None else f"At {T.fmt_depth(depth)}")
        + ". Hover to compare methods on one day.",
    )
    C.plot(V.daily_rmse(metrics, styles, depth), "validation_daily")


def render_argo(run: D.Run, styles_glorys: dict[str, T.MethodStyle]) -> None:
    argo = D.metrics_argo(run.path, run.stamp)
    if not argo or not argo.get("methods"):
        C.empty_state(
            "No Argo validation for this run",
            "Argo profiles are collocated with the predictions by the validate-argo step.",
            C.command_for(run, "validate-argo"),
        )
        return
    meta = argo.get("metadata", {})
    styles = V.styles_for(argo)
    keep = set(styles_glorys) | {"glorys"}
    styles = {k: s for k, s in styles.items() if k in keep}
    C.cards(V.argo_cards(argo))
    C.note(
        "<b>GLORYS assimilates Argo.</b> "
        + C.esc(
            meta.get("independence_note")
            or "Argo is independent of the model's inputs but not of its training target, "
            "because the GLORYS reanalysis assimilates Argo profiles."
        )
    )
    matchups = D.argo_matchups(run.path, run.stamp)
    C.legend([styles[k] for k in T.legend_order(styles)])
    cols = st.columns(3, gap="medium")
    for col, metric in zip(cols, ["rmse", "bias", "corr_anom"], strict=True):
        with col:
            spec = T.METRICS[metric]
            C.panel_title(f"{spec.label} vs Argo", spec.note.replace("reference", "Argo"))
            fig = V.metric_profile(
                argo["methods"], styles, metric, reference="Argo", height=400, showlegend=False
            )
            C.plot(fig, f"validation_argo_{metric}")
    if matchups is None or matchups.empty:
        C.empty_state(
            "No matchup table",
            "metrics/argo_matchups.parquet is written by the validate-argo step.",
            C.command_for(run, "validate-argo"),
        )
    else:
        columns = V.argo_columns(argo, matchups)
        left, right = st.columns(2, gap="medium")
        with left:
            options = [k for k in T.legend_order(styles) if k in columns]
            method = st.selectbox(
                "Method on the vertical axis", options,
                format_func=lambda k: styles[k].label, key="validation_argo_method",
            )  # fmt: skip
            C.panel_title(
                f"Argo observation vs {styles[method].label}",
                f"{len(matchups):,} (profile, depth) matchups, coloured by depth. Dashed: 1:1.",
            )
            fig = V.argo_scatter(matchups, columns[method], styles[method].label)
            C.plot(fig, "validation_argo_scatter")
        with right:
            C.panel_title(
                "Argo profile locations",
                f"{matchups['profile_id'].nunique():,} profiles used, "
                f"{meta.get('start', '?')} → {meta.get('end', '?')}",
            )
            fig = V.argo_locations(
                matchups, D.ocean_mask(run.path, run.stamp), meta.get("basins", {})
            )
            C.plot(fig, "validation_argo_map")
            C.key_values(
                [
                    ("Collocation", str(meta.get("collocation", T.MISSING))),
                    ("Vertical rule", str(meta.get("interpolation_rule", T.MISSING))),
                    ("Bias sign", str(meta.get("bias_convention", T.MISSING))),
                ]
            )
    with st.expander("Table: metrics vs Argo"):
        st.caption("All depths")
        _table(V.summary_table(argo["methods"], styles, "overall"))
        st.caption(f"Pooled {V.pooled_label(argo)}")
        _table(V.summary_table(argo["methods"], styles))
    gridded = argo.get("gridded_argo") or {}
    if gridded.get("methods"):
        st.divider()
        C.panel_title(
            "Gridded Argo product (INCOIS)",
            f"RMSE against the gridded product over {gridded.get('n_times', '?')} time steps.",
        )
        C.plot(
            V.metric_profile(gridded["methods"], styles, "rmse", reference="gridded Argo"),
            "validation_argo_gridded",
        )


def render() -> None:
    run = C.select_run()
    if run is None:
        return
    metrics = D.metrics_glorys(run.path, run.stamp)
    has_metrics = bool(metrics and metrics.get("methods"))
    meta = (metrics or {}).get("metadata", {})
    chips = ""
    if has_metrics:
        chips = C.chip("Split", str(meta.get("split", "?"))) + C.chip(
            "Period", f"{meta.get('start', '?')} → {meta.get('end', '?')}"
        ) + C.chip("Days", str(meta.get("n_days", "?")))  # fmt: skip
    C.page_header(
        "Validation",
        "How close the reconstruction is to the GLORYS reanalysis and to Argo profiles, always "
        "next to the ridge and climatology baselines.",
        run,
        chips=chips,
    )
    section = st.segmented_control(
        "View", SECTIONS, default=SECTIONS[0], key="validation_section", required=True,
        label_visibility="collapsed",
    )  # fmt: skip
    styles = V.styles_for(metrics) if has_metrics else {}
    if styles and section != "Error maps":  # the maps have their own method selectors
        styles = pick_methods(styles)
    if section == "Argo":
        render_argo(run, styles or V.styles_for(D.metrics_argo(run.path, run.stamp)))
        return
    if not has_metrics:
        C.empty_state(
            "No evaluation metrics yet",
            "metrics/metrics_glorys.json is written by the evaluate step.",
            C.command_for(run, "evaluate"),
        )
        return
    if section == "By depth":
        render_by_depth(metrics, styles)
    elif section == "Basins":
        render_basins(metrics, styles)
    elif section == "Error maps":
        render_maps(run, metrics, styles)
    else:
        render_daily(metrics, styles)


C.setup_page("Validation")
C.run_page(render)
