"""Dashboard UI: figure builders on a finished tiny run, design tokens, and a Streamlit
``AppTest`` smoke run of every page (including the empty states)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest
from app import data_access as da
from app.ui import figures as F
from app.ui import theme as T
from app.ui import views as V
from streamlit.testing.v1 import AppTest

from oceanembed.config import Config

APP = Path(__file__).resolve().parents[1] / "app"
HOME = str(APP / "Home.py")
PAGES = {
    "explorer": "pages/1_3D_Explorer.py",
    "profiles": "pages/2_Profiles_and_Sections.py",
    "validation": "pages/3_Validation.py",
    "embeddings": "pages/4_Embeddings.py",
}


def _heatmaps(fig: go.Figure) -> list[go.Heatmap]:
    return [t for t in fig.data if isinstance(t, go.Heatmap)]


# ----------------------------------------------------------------------- tokens
def test_method_styles_are_fixed_and_never_colour_only():
    keys = ["climatology", "model", "ridge", "model_scratch", "glorys", "model_other"]
    styles = T.method_styles(keys)
    assert list(styles)[-1] == "model"  # the model is drawn last (on top)
    assert T.legend_order(styles)[0] == "model"
    # every method differs from every other in colour AND in dash-or-marker
    assert len({s.color for s in styles.values()}) == len(keys)
    assert len({(s.dash, s.symbol) for s in styles.values()}) == len(keys)
    # a method keeps its style when other methods are filtered out
    assert T.method_styles(["ridge", "model"])["ridge"] == styles["ridge"]
    assert T.method_styles(["model_scratch"])["model_scratch"] == styles["model_scratch"]
    assert styles["model_scratch"].label == "OceanEmbed, no pretraining"


def test_formatting_helpers():
    assert T.fmt(-0.1234) == "−0.12" and T.fmt(None) == T.MISSING and T.fmt(float("nan")) == "–"
    assert T.fmt(1.5, 1, "°C", signed=True) == "+1.5 °C"
    assert T.fmt_int(12686) == "12,686" and T.fmt_depth(100.0) == "100 m"
    assert np.isnan(T.to_array([1.0, None])[1])


def test_colour_ranges():
    a = np.array([[1.0, np.nan], [3.0, 5.0]])
    lo, hi = F.shared_range(a, a + 10, pct=0.0)
    assert (lo, hi) == (1.0, 15.0)
    assert F.shared_range(np.full((2, 2), np.nan)) == (0.0, 1.0)
    assert F.symmetric_limit(np.array([-4.0, 1.0, np.nan]), pct=100) == 4.0


# ----------------------------------------------------------------------- figure builders
def test_overview_and_validation_builders(tiny_run: Config):
    run = tiny_run.outputs_dir
    metrics, argo = da.load_metrics_glorys(run), da.load_metrics_argo(run)
    styles = V.styles_for(metrics)
    assert set(styles) == set(metrics["methods"])

    cards = V.headline_cards(metrics, argo)
    assert len(cards) == 4 and cards[0].lead and cards[0].unit == "°C"
    assert V.headline_cards(metrics, None)[-1].value == T.MISSING
    assert [s[0] for s in V.pipeline_steps(da.load_run_meta(run))] == [
        "Input",
        "Embedding",
        "Output",
    ]
    assert dict(V.run_summary(da.load_run_meta(run), metrics))["Data source"] == "synthetic"

    fig = V.metric_profile(metrics["methods"], styles, "rmse")
    assert len(fig.data) == len(styles)
    y0, y1 = fig.layout.yaxis.range
    assert y0 > y1  # depth increases downward
    assert "°C" in fig.layout.xaxis.title.text and fig.layout.yaxis.title.text == "depth (m)"
    # climatology has no anomaly correlation: its all-NaN line is dropped, not drawn
    corr = V.metric_profile(metrics["methods"], styles, "corr_anom")
    assert "Climatology" not in [t.name for t in corr.data]
    # each line carries colour + dash + marker
    for trace in fig.data:
        assert trace.line.dash and trace.marker.symbol and trace.mode == "lines+markers"

    table = V.summary_table(metrics["methods"], styles)
    assert list(table["Method"])[0] == "OceanEmbed" and len(table) == len(styles)
    assert np.isnan(table.set_index("Method").loc["Climatology", "Anomaly corr."])
    assert len(V.per_depth_table(metrics["methods"], styles, "bias")) == 15

    basins = V.basins_of(metrics)
    assert basins == ["arabian_sea", "bay_of_bengal"]
    blocks = [V.basin_blocks(metrics, b) for b in basins]
    lo, hi = V.metric_range(blocks, "rmse")
    assert lo == 0.0 and hi > 0
    assert V.metric_profile(blocks[0], styles, "rmse", x_range=(lo, hi)).layout.xaxis.range == (
        lo,
        hi,
    )

    daily = V.daily_rmse(metrics, styles)
    assert len(daily.data) == len(styles) and len(daily.data[0].x) == metrics["metadata"]["n_days"]
    assert V.daily_rmse(metrics, styles, depth=100.0) is not None
    assert V.daily_rmse({"methods": {}}, styles) is None

    maps = da.load_maps(run)
    inventory = V.map_inventory(maps)
    assert {"rmse", "bias"} <= set(inventory) and "model" in inventory["rmse"]
    limits = V.error_limits(maps, "rmse", ["model", "climatology"], 100.0)
    assert limits[0] == 0.0
    emap = V.error_map(maps, "rmse", "model", 100.0, limits=limits)
    assert (_heatmaps(emap)[0].zmin, _heatmaps(emap)[0].zmax) == limits
    bias = _heatmaps(V.error_map(maps, "bias", "model", 100.0))[0]
    assert bias.zmin == -bias.zmax  # diverging, centred on zero

    matchups = da.load_argo_matchups(run)
    columns = V.argo_columns(argo, matchups)
    assert columns["climatology"] == "clim" and "glorys" in columns
    assert len(V.argo_cards(argo)) == 4
    scatter = V.argo_scatter(matchups, columns["model"], "OceanEmbed")
    assert scatter.layout.xaxis.range == scatter.layout.yaxis.range  # square, with the 1:1 line
    locations = V.argo_locations(matchups, None, argo["metadata"]["basins"])
    assert len(locations.data[-1].x) == matchups["profile_id"].nunique()
    assert V.metric_profile(argo["methods"], V.styles_for(argo), "rmse", reference="Argo").data


def test_map_profile_section_and_embedding_builders(tiny_run: Config):
    run = tiny_run.outputs_dir
    day = da.available_dates(run)[3]
    pred = da.load_prediction(run, day).drop_vars("time", errors="ignore")
    target = da.load_target(run, day).drop_vars("time", errors="ignore")
    clim = da.load_climatology(run, day)
    depth = 100.0

    s = V.day_slices(pred, target, clim, depth)
    assert s.has_target and s.reference_name == "GLORYS target" and s.vmin < s.vmax
    pmap, tmap = (
        F.field_map(s.predicted, zmin=s.vmin, zmax=s.vmax),
        F.field_map(s.reference, zmin=s.vmin, zmax=s.vmax),
    )
    hp, ht = _heatmaps(pmap)[0], _heatmaps(tmap)[0]
    assert (hp.zmin, hp.zmax) == (ht.zmin, ht.zmax) == (s.vmin, s.vmax)  # shared colour range
    assert "°C" in hp.colorbar.title.text
    # land stays NaN (transparent over the flat land background), never a data colour
    assert np.isnan(np.asarray(hp.z, dtype=float)).sum() == int(np.isnan(s.predicted.values).sum())
    assert pmap.layout.plot_bgcolor == T.LAND and hp.hoverongaps is False
    dmap = _heatmaps(
        F.field_map(s.difference, colorscale=T.DIVERGING_SCALE, zmax=s.limit, diverging=True)
    )[0]
    assert dmap.zmin == -s.limit and dmap.zmax == s.limit
    stats = V.slice_stats(s)
    assert stats["rmse"] >= abs(stats["bias"]) and stats["n"] > 0
    assert len(V.slice_cards(s)) == 4
    assert F.difference_histogram(s.difference.values).data[0].type == "bar"

    # prediction-only date: falls back to the climatology, then to the prediction alone
    fallback = V.day_slices(pred, None, clim, depth)
    assert not fallback.has_target and fallback.reference_name == "Climatology"
    alone = V.day_slices(pred, None, None, depth)
    assert alone.reference is None and alone.difference is None and V.slice_cards(alone) == []

    surface = da.load_surface_inputs(run, day)
    for name in surface.data_vars:
        heat = _heatmaps(V.surface_map(surface, name))[0]
        if T.surface_field(name).diverging:
            assert heat.zmin == -heat.zmax

    lat, lon = V.default_point(pred)
    prof = da.load_profile(run, day, lat, lon)
    assert V.profile_is_ocean(prof) and V.has_target(prof)
    pfig = V.point_profile(prof)
    assert [t.name for t in pfig.data] == [
        "Climatology",
        "GLORYS target",
        "OceanEmbed prediction",
    ]
    assert pfig.layout.yaxis.range[0] > pfig.layout.yaxis.range[1]
    assert len(V.profile_table(prof)) == 15 and V.point_cards(prof, depth)
    clickable = F.field_map(
        s.predicted, clickable=True, mark=F.MapMark(lat=lat, lon=lon, line_lat=lat)
    )
    assert clickable.layout.clickmode == "event+select"
    assert len(clickable.data[1].x) == int(np.isfinite(s.predicted.values).sum())

    for kw in ({"lat": lat}, {"lon": lon}):
        section = da.load_section(run, day, **kw)
        sfig = F.section_figure(section, mark=lon if "lat" in kw else lat)
        heats = _heatmaps(sfig)
        assert len(heats) == 3
        assert (heats[0].zmin, heats[0].zmax) == (heats[1].zmin, heats[1].zmax)
        assert heats[2].zmin == -heats[2].zmax
        assert sfig.layout.yaxis.range[0] > sfig.layout.yaxis.range[1]
    no_target = section.assign(target=section["target"] * np.nan)
    titles = [a.text for a in F.section_figure(no_target).layout.annotations]
    assert "Climatology" in titles and "Predicted − climatology" in titles

    rgb = da.embedding_pca_rgb(run, day, n_fit_days=6)
    ocean = np.isfinite(pred.isel(depth=0))
    coarse = V.coarse_ocean(ocean, rgb)
    assert coarse.shape == rgb.shape[:2] and coarse.any()
    efig = V.embedding_figure(rgb, ocean)
    assert efig.data[0].type == "image" and np.asarray(efig.data[0].z).shape == rgb.shape
    assert "PC1 (red)" in V.variance_text(rgb)


# ----------------------------------------------------------------------- Streamlit pages
EMPTY = 'class="oe-empty"'  # an empty state was rendered


def _texts(at: AppTest) -> str:
    return "\n".join(str(m.value) for m in at.markdown)


def _run_page(page: str | None) -> AppTest:
    at = AppTest.from_file(HOME, default_timeout=120)
    at.run()
    if page is not None:
        at.switch_page(page)
        at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_home_page_shows_banner_and_headlines(tiny_run: Config):
    at = _run_page(None)
    text = _texts(at)
    assert "SYNTHETIC DATA" in text and "oe-strip" in text  # banner + persistent strip
    assert "Headline results" in text and "Run summary" in text
    assert at.sidebar.selectbox[0].value == "test_tiny"
    assert len(at.get("plotly_chart")) == 1


@pytest.mark.parametrize("name", list(PAGES))
def test_pages_render_without_errors(tiny_run: Config, name: str):
    at = _run_page(PAGES[name])
    text = _texts(at)
    assert "SYNTHETIC DATA" in text and EMPTY not in text
    assert len(at.get("plotly_chart")) >= 2


def test_validation_views_and_run_selection_persist(tiny_run: Config):
    at = _run_page(PAGES["validation"])
    for view in ["Basins", "Error maps", "Daily series", "Argo"]:
        at.segmented_control(key="validation_section").set_value(view).run()
        assert not at.exception, (view, [e.value for e in at.exception])
        assert at.get("plotly_chart"), view
    assert "GLORYS assimilates Argo" in _texts(at)
    # the day chosen on one page is kept on the next one
    at.switch_page(PAGES["explorer"]).run()
    at.sidebar.button(key="oe_day_next").click().run()
    day = at.session_state["oe_day"]
    at.switch_page(PAGES["embeddings"]).run()
    assert not at.exception and at.session_state["oe_day"] == day
    assert day != str(da.available_dates(tiny_run.outputs_dir)[0].date())


def test_empty_states(tiny_run: Config, tmp_path, monkeypatch):
    source = tiny_run.outputs_dir  # resolved before the outputs root is redirected
    # no runs at all
    monkeypatch.setenv("OCEANEMBED_OUTPUTS_ROOT", str(tmp_path / "nothing"))
    for page in [None, *PAGES.values()]:
        at = _run_page(page)
        assert "No finished run found" in _texts(at)
        assert "run-all" in at.code[0].value

    # a run with predictions only: no metrics, no Argo, no embeddings
    root = tmp_path / "outputs"
    run = root / tiny_run.run_name
    shutil.copytree(source / "predictions", run / "predictions")
    shutil.copy(source / "run_meta.json", run / "run_meta.json")
    monkeypatch.setenv("OCEANEMBED_OUTPUTS_ROOT", str(root))
    commands = {None: "evaluate", PAGES["validation"]: "evaluate", PAGES["embeddings"]: "embed"}
    for page, step in commands.items():
        at = _run_page(page)
        assert EMPTY in _texts(at)
        assert any(f"oceanembed {step} " in c.value for c in at.code), (page, step)
    at = _run_page(PAGES["validation"])
    at.segmented_control(key="validation_section").set_value("Argo").run()
    assert not at.exception and any("validate-argo" in c.value for c in at.code)
    # the map pages still work from the predictions alone
    assert len(_run_page(PAGES["explorer"]).get("plotly_chart")) >= 2


def test_relative_run_meta_paths_resolve_from_any_dir(tiny_run: Config, tmp_path, monkeypatch):
    """`run_meta.json` stores paths as the config gave them (relative to the project root); the data
    layer must find the store and statistics whatever the current directory is."""
    import json
    import shutil

    root = tmp_path / "project"
    run = root / "outputs" / tiny_run.run_name
    shutil.copytree(tiny_run.outputs_dir / "predictions", run / "predictions")
    zarr_rel = Path("data") / "processed" / tiny_run.zarr_path.name
    stats_rel = Path("data") / "processed" / tiny_run.stats_path.name
    (root / zarr_rel).parent.mkdir(parents=True)
    shutil.copytree(tiny_run.zarr_path, root / zarr_rel)
    shutil.copy(tiny_run.stats_path, root / stats_rel)
    meta = json.loads((tiny_run.outputs_dir / "run_meta.json").read_text(encoding="utf-8"))
    meta["paths"] = {
        "outputs_dir": (Path("outputs") / tiny_run.run_name).as_posix(),
        "zarr": zarr_rel.as_posix(),
        "stats": stats_rel.as_posix(),
        "argo_gridded": "data/processed/none.nc",
    }
    (run / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    day = da.available_dates(run)[0]
    assert da.load_target(run, day) is not None
    assert da.load_climatology(run, day).shape[0] == len(meta["grid"]["depths"])
    assert da._paths(run)["zarr"].resolve() == (root / zarr_rel).resolve()
