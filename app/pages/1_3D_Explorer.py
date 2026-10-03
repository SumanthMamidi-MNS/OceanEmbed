"""3-D Explorer: prediction, target and difference maps at a chosen day and depth, plus the
surface inputs the prediction was made from."""

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


def render_maps(s: V.DaySlices) -> None:
    left, right = st.columns(2, gap="medium")
    with left:
        C.panel_title("OceanEmbed prediction", "From surface fields only")
        C.plot(F.field_map(s.predicted, zmin=s.vmin, zmax=s.vmax), "explorer_pred")
    with right:
        if s.reference is not None:
            C.panel_title(s.reference_name, "Same colour range as the prediction")
            fig = F.field_map(s.reference, zmin=s.vmin, zmax=s.vmax, quantity="temperature")
            C.plot(fig, "explorer_ref")
        else:
            C.panel_title("Reference")
            C.empty_state(
                "No reference field for this day",
                "Neither the GLORYS target nor the climatology statistics could be read, so only "
                "the prediction is shown.",
            )
    if s.difference is None:
        return
    left, right = st.columns(2, gap="medium")
    with left:
        C.panel_title(s.difference_name, "Diverging scale centred on zero, symmetric limits")
        fig = F.field_map(
            s.difference, colorscale=T.DIVERGING_SCALE, zmax=s.limit, diverging=True,
            quantity="difference",
        )  # fmt: skip
        C.plot(fig, "explorer_diff")
    with right:
        C.panel_title(
            "This day and depth" if s.has_target else "Prediction vs climatology",
            "Over all ocean cells of the map",
        )
        C.cards(V.slice_cards(s), tight=True)
        label = "predicted − target" if s.has_target else "predicted − climatology"
        C.plot(F.difference_histogram(s.difference.values, label=label), "explorer_hist")


def render_surface(run: D.Run, day: str) -> None:
    C.panel_title(
        "Surface inputs for this day",
        "The seven satellite-observable fields the model sees. Signed fields use a diverging "
        "scale centred on zero.",
    )
    C.surface_grid(run, D.surface_inputs(run.path, run.stamp, day), "explorer_surface")


def render() -> None:
    run = C.select_run()
    if run is None:
        return
    days = D.dates(run.path, run.stamp)
    if not days:
        C.page_header("3-D Explorer", "Maps of the reconstructed temperature field.", run)
        C.no_predictions(run)
        return
    day = C.day_control(days)
    key = D.day_key(day)
    pred = D.prediction(run.path, run.stamp, key)
    depths = [float(d) for d in pred["depth"].values] if pred is not None else run.depths
    depth = C.depth_control(depths)
    C.page_header(
        "3-D Explorer",
        "Reconstructed temperature at one day and depth, next to the GLORYS target and their "
        "difference. Step with the sidebar buttons or the arrow keys.",
        run,
        chips=C.chip("Day", key) + C.chip("Depth", T.fmt_depth(depth)),
    )
    if pred is None:
        C.empty_state("No prediction for this day", "Pick another day in the sidebar.")
        return
    target = D.target(run.path, run.stamp, key)
    clim = None if target is not None else D.climatology(run.path, run.stamp, key)
    if target is None:
        C.note(
            "<b>No GLORYS target for this day.</b> This is a prediction-only date, so the "
            "prediction is compared with the climatology instead."
        )
    render_maps(V.day_slices(pred, target, clim, depth))
    st.divider()
    render_surface(run, key)


C.setup_page("3-D Explorer")
C.run_page(render)
