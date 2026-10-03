"""Profiles & Sections: the vertical temperature profile at a chosen point and a zonal or
meridional section through it."""

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

LOCATOR_KEY = "profiles_locator"
ORIENTATIONS = {
    "zonal": "Zonal, along a latitude",
    "meridional": "Meridional, along a longitude",
}


def render_point(run: D.Run, day: str, pred, depth: float, lat: float, lon: float, zonal: bool):
    prof = D.profile(run.path, run.stamp, day, lat, lon)
    cell = (float(prof["lat"]), float(prof["lon"])) if prof is not None else (lat, lon)
    left, right = st.columns([3, 2], gap="medium")
    with left:
        C.panel_title(
            f"Pick a point: prediction at {T.fmt_depth(depth)}",
            "Click an ocean cell, or type coordinates in the sidebar. The dashed line is the "
            "section below.",
        )
        mark = F.MapMark(
            lat=cell[0], lon=cell[1],
            line_lat=cell[0] if zonal else None, line_lon=None if zonal else cell[1],
        )  # fmt: skip
        fig = F.field_map(
            pred.sel(depth=depth, method="nearest"), mark=mark, clickable=True, width=F.WIDTH_WIDE
        )
        if prof is not None and V.profile_is_ocean(prof):
            C.cards(V.point_cards(prof, depth), tight=True)
        C.clickable_plot(fig, LOCATOR_KEY)
    with right:
        C.panel_title(
            f"Vertical profile at {T.fmt_lat(cell[0])}, {T.fmt_lon(cell[1])}",
            "Nearest grid cell. Depth increases downward.",
        )
        if prof is None:
            C.empty_state(
                "Profile unavailable",
                "The climatology statistics or the harmonised store of this run could not be "
                "read, so the profile cannot be assembled.",
                C.command_for(run, "stats"),
            )
        elif not V.profile_is_ocean(prof):
            C.empty_state(
                "This cell is land",
                "There is no temperature here. Click an ocean cell on the map or change the "
                "coordinates in the sidebar.",
            )
        else:
            C.plot(V.point_profile(prof), "profiles_profile")
            with st.expander("Table: profile values"):
                st.dataframe(
                    V.profile_table(prof).style.format(precision=2, na_rep=T.MISSING),
                    hide_index=True,
                )
    return cell


def render_section(run: D.Run, day: str, cell: tuple[float, float], zonal: bool) -> None:
    along = f"latitude {T.fmt_lat(cell[0])}" if zonal else f"longitude {T.fmt_lon(cell[1])}"
    title = f"{'Zonal' if zonal else 'Meridional'} section along {along}"
    section = D.section(
        run.path, run.stamp, day,
        lat=cell[0] if zonal else None, lon=None if zonal else cell[1],
    )  # fmt: skip
    if section is None:
        C.panel_title(title)
        C.empty_state(
            "Section unavailable",
            "The climatology statistics or the harmonised store of this run could not be read.",
            C.command_for(run, "stats"),
        )
        return
    reference = "target" if V.has_target(section) else "climatology"
    C.panel_title(
        title,
        f"Prediction and {reference} share one colour range; the difference is centred on zero. "
        "The dashed line marks the selected point, grey is land or the sea floor.",
    )
    if not V.profile_is_ocean(section):
        C.empty_state("This line is entirely over land", "Move the point to an ocean cell.")
        return
    C.plot(F.section_figure(section, mark=cell[1] if zonal else cell[0]), "profiles_section")


def render() -> None:
    run = C.select_run()
    if run is None:
        return
    days = D.dates(run.path, run.stamp)
    if not days:
        C.page_header("Profiles & Sections", "Vertical structure of the reconstruction.", run)
        C.no_predictions(run)
        return
    day = D.day_key(C.day_control(days, shortcuts=False))
    pred = D.prediction(run.path, run.stamp, day)
    if pred is None:
        C.page_header("Profiles & Sections", "Vertical structure of the reconstruction.", run)
        C.empty_state("No prediction for this day", "Pick another day in the sidebar.")
        return
    lat, lon = C.point_control(pred["lat"].values, pred["lon"].values, V.default_point(pred))
    with st.sidebar:
        st.markdown('<div class="oe-side-label">Section</div>', unsafe_allow_html=True)
        key = C.remember("oe_orientation", "zonal", list(ORIENTATIONS))
        st.radio(
            "Section", list(ORIENTATIONS), key=key, format_func=ORIENTATIONS.get,
            on_change=C.sync, args=("oe_orientation",), label_visibility="collapsed",
        )  # fmt: skip
    zonal = st.session_state["oe_orientation"] == "zonal"
    depth = C.depth_control([float(d) for d in pred["depth"].values], shortcuts=False)
    C.page_header(
        "Profiles & Sections",
        "The vertical temperature profile at one point and a section through it, against the "
        "GLORYS target and the climatology.",
        run,
        chips=C.chip("Day", day),
    )
    if D.target(run.path, run.stamp, day) is None:
        C.note(
            "<b>No GLORYS target for this day.</b> The profile shows the prediction and the "
            "climatology; the section compares the prediction with the climatology."
        )
    cell = render_point(run, day, pred, depth, lat, lon, zonal)
    st.divider()
    render_section(run, day, cell, zonal)


C.setup_page("Profiles & Sections")
C.run_page(render)
