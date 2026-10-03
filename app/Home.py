"""OceanEmbed dashboard, Overview page. Run with ``streamlit run app/Home.py``."""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit only puts app/ on sys.path; the UI package is imported as `app.ui` from the repo
# root (setup_page drops the duplicate entries that reruns would otherwise pile up).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from app.ui import components as C
from app.ui import data as D
from app.ui import theme as T
from app.ui import views as V


def render_pipeline(run: D.Run) -> None:
    steps = V.pipeline_steps(run.meta)
    parts = []
    for i, (stage, title, text) in enumerate(steps, start=1):
        arrow = '<span aria-hidden="true">→</span>' if i < len(steps) else ""
        parts.append(
            f'<div class="oe-step"><div class="n">{i} · {C.esc(stage)}{arrow}</div>'
            f'<div class="t">{C.esc(title)}</div><div class="d">{C.esc(text)}</div></div>'
        )
    st.markdown(f'<div class="oe-flow">{"".join(parts)}</div>', unsafe_allow_html=True)


def availability_chips(run: D.Run) -> list[str]:
    flags = [
        ("has_predictions", "Predictions"),
        ("has_metrics", "GLORYS metrics"),
        ("has_argo", "Argo validation"),
        ("has_embeddings", "Embeddings"),
        ("has_report", "Report"),
    ]
    return [
        C.chip(label + ("" if run.info.get(k) else ": missing"), off=not run.info.get(k))
        for k, label in flags
    ]


def render() -> None:
    run = C.select_run()
    if run is None:
        return
    C.page_header(
        "Overview",
        "OceanEmbed reconstructs subsurface ocean temperature over the North Indian Ocean from "
        "surface satellite observations alone.",
        run,
    )
    render_pipeline(run)

    metrics = D.metrics_glorys(run.path, run.stamp)
    argo = D.metrics_argo(run.path, run.stamp)
    has_metrics = bool(metrics and metrics.get("methods"))
    if has_metrics:
        C.panel_title(
            "Headline results",
            f"Test-period skill against the GLORYS reanalysis, pooled over "
            f"{V.pooled_label(metrics)}: the thermocline, where the surface says least about "
            "the interior.",
        )
        C.cards(V.headline_cards(metrics, argo))
    else:
        C.empty_state(
            "No evaluation metrics yet",
            "Headline metrics and the per-depth error profile appear once the run is evaluated.",
            C.command_for(run, "evaluate"),
        )

    left, right = (
        st.columns([3, 2], gap="large") if has_metrics else (st.container(), st.container())
    )
    with left:
        if has_metrics:
            styles = V.styles_for(metrics)
            C.panel_title(
                "RMSE by depth, all methods",
                "Lower is better. Depth increases downward on a square-root axis.",
            )
            C.plot(V.metric_profile(metrics["methods"], styles, "rmse", height=470), "home_rmse")
            with st.expander("Table: pooled metrics by method"):
                st.dataframe(
                    V.summary_table(metrics["methods"], styles).style.format(
                        precision=3, na_rep=T.MISSING, thousands=","
                    ),
                    hide_index=True,
                )
    with right:
        C.panel_title("Run summary", f"Run “{run.name}”")
        C.key_values(V.run_summary(run.meta, metrics))
        C.chips(availability_chips(run))
        C.note(
            "<b>How to read this.</b> Climatology is the no-skill reference and ridge regression "
            "the simple statistical baseline. The anomaly correlation removes the seasonal cycle "
            "from both sides, so it is always shown next to the raw correlation."
        )


C.setup_page("Overview")
C.run_page(render)
