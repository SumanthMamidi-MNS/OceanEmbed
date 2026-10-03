"""Embeddings: a PCA view of the satellite embedding map next to the surface fields it encodes."""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit only puts app/ on sys.path; the UI package is imported as `app.ui` from the repo
# root (setup_page drops the duplicate entries that reruns would otherwise pile up).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import streamlit as st
from app.ui import components as C
from app.ui import data as D
from app.ui import figures as F
from app.ui import theme as T
from app.ui import views as V

SUBTITLE = "What the satellite embedding engine extracts from the surface fields, day by day."


def explain(run: D.Run, rgb) -> None:
    emb = run.meta.get("config", {}).get("model", {}).get("emb_dim")
    lat = np.asarray(rgb["lat"].values, dtype=float)
    cell = f"{abs(lat[1] - lat[0]):g}°" if lat.size > 1 else "coarse"
    size = f"{emb} numbers" if emb else "a vector of numbers"
    C.note(
        "<b>What is the embedding?</b> For every day the encoder turns the seven surface maps "
        f"into a coarser map ({cell} cells) that holds {size} per cell: a learned summary of the "
        "local surface state and its surroundings, and the only thing the temperature decoder "
        "builds on. That many numbers cannot be drawn, so a principal component analysis keeps "
        "the three directions that vary most and paints them as red, green and blue. "
        "<b>Cells with similar colours are in a similar state as far as the model is "
        "concerned.</b> The colours have no physical unit; the axes are fitted once per run, so "
        "colours are comparable between days."
    )


def render() -> None:
    run = C.select_run()
    if run is None:
        return
    days = D.dates(run.path, run.stamp)
    if not run.info.get("has_embeddings"):
        C.page_header("Embeddings", SUBTITLE, run)
        C.empty_state(
            "No embeddings exported for this run",
            "Embedding maps for the test period are written to embeddings/embeddings.zarr by the "
            "embed step.",
            C.command_for(run, "embed"),
        )
        return
    if not days:
        C.page_header("Embeddings", SUBTITLE, run)
        C.no_predictions(run)
        return
    day = D.day_key(C.day_control(days))
    C.page_header("Embeddings", SUBTITLE, run, chips=C.chip("Day", day))
    rgb = D.embedding_rgb(run.path, run.stamp, day)
    if rgb is None:
        C.empty_state(
            "No embedding for this day",
            "Embeddings are exported for the test period only. Pick a day inside it, or export "
            "them again.",
            C.command_for(run, "embed"),
        )
        return
    explain(run, rgb)
    surface = D.surface_inputs(run.path, run.stamp, day)
    left, right = st.columns(2, gap="medium")
    with left:
        C.panel_title("Embedding map, first three principal components", V.variance_text(rgb))
        C.plot(V.embedding_figure(rgb, D.ocean_mask(run.path, run.stamp)), "embeddings_rgb")
    with right:
        if surface is None:
            C.panel_title("Surface field")
            C.surface_grid(run, None, "embeddings_surface")
        else:
            names = list(surface.data_vars)
            name = st.session_state.get("embeddings_field") or names[0]
            name = name if name in names else names[0]
            spec = T.surface_field(name, dict(surface[name].attrs))
            C.panel_title(f"Surface input: {spec.label}", "Compare its patterns with the colours")
            fig = V.surface_map(surface, name, width=F.WIDTH_HALF, compact=False)
            C.plot(fig, "embeddings_field_map")
            st.segmented_control(
                "Surface field shown above", names, default=names[0], key="embeddings_field",
                required=True, format_func=str.upper,
            )  # fmt: skip
    if surface is not None:
        st.divider()
        C.panel_title("All surface inputs for this day", "The seven fields the encoder sees")
        C.surface_grid(run, surface, "embeddings_surface")


C.setup_page("Embeddings")
C.run_page(render)
