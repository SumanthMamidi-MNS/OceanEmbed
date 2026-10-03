"""Shared page shell and Streamlit components: CSS, navigation, run selector, synthetic-data
banner, metric cards, legends, empty states and the day / depth controls.

Selections (run, day, depth, point) are kept in ``st.session_state`` under plain keys so they
survive page switches; widgets mirror them through ``remember`` / ``sync``.
"""

from __future__ import annotations

import html
import sys
from collections.abc import Callable, Iterable, Sequence

import pandas as pd
import streamlit as st
from app.ui import data as D
from app.ui import theme as T
from app.ui import views as V
from app.ui.figures import PLOT_CONFIG

PAGES = [
    ("Home.py", "Overview"),
    ("pages/1_3D_Explorer.py", "3-D Explorer"),
    ("pages/2_Profiles_and_Sections.py", "Profiles & Sections"),
    ("pages/3_Validation.py", "Validation"),
    ("pages/4_Embeddings.py", "Embeddings"),
]

_CSS = f"""
<style>
:root {{
  --oe-bg: {T.BG}; --oe-surface: {T.SURFACE}; --oe-raised: {T.RAISED}; --oe-border: {T.BORDER};
  --oe-text: {T.TEXT}; --oe-text-2: {T.TEXT_2}; --oe-muted: {T.TEXT_MUTED};
  --oe-accent: {T.ACCENT}; --oe-warn: {T.WARN}; --oe-warn-bg: {T.WARN_BG};
  --oe-radius: {T.RADIUS}px;
}}
.stMainBlockContainer {{ padding-top: 3.25rem; padding-bottom: 5rem; max-width: 1320px; }}
@media (min-width: 900px) {{
  .stMainBlockContainer {{ padding-left: 2.5rem; padding-right: 2.5rem; }}
}}
[data-testid="stSidebarContent"] {{ padding-bottom: 3.5rem; }}
[data-testid="stSidebarHeader"] {{ height: 2.25rem; padding-bottom: 0; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}

.oe-brand {{ font-size: 1.25rem; font-weight: 700; color: var(--oe-text); line-height: 1.2; }}
.oe-brand small {{ display: block; font-size: 0.78rem; font-weight: 400;
  color: var(--oe-muted); margin: 3px 0 10px; line-height: 1.35; }}
.oe-side-label {{ font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--oe-muted); margin: 0.6rem 0 0.1rem; }}

.oe-header h1 {{ font-size: 1.7rem; font-weight: 700; margin: 0; padding: 0;
  color: var(--oe-text); }}
.oe-header p {{ margin: 0.2rem 0 0; color: var(--oe-text-2); font-size: 0.98rem;
  max-width: 78ch; }}
.oe-header {{ margin-bottom: 0.4rem; }}
.oe-context {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 0.55rem; }}

.oe-chip {{ display: inline-flex; align-items: center; gap: 6px; padding: 2px 10px;
  border: 1px solid var(--oe-border); border-radius: 999px; background: var(--oe-surface);
  color: var(--oe-text-2); font-size: 0.8rem; font-variant-numeric: tabular-nums;
  white-space: nowrap; }}
.oe-chip b {{ color: var(--oe-text); font-weight: 600; }}
.oe-chip.warn {{ border-color: var(--oe-warn); color: var(--oe-warn);
  background: var(--oe-warn-bg); font-weight: 600; }}
.oe-chip.off {{ color: var(--oe-muted); border-style: dashed; }}

.oe-banner {{ display: flex; gap: 12px; align-items: flex-start; padding: 12px 16px;
  border: 1px solid var(--oe-warn); border-left-width: 6px; border-radius: var(--oe-radius);
  background: var(--oe-warn-bg); color: var(--oe-text); margin: 0.2rem 0 0.6rem; }}
.oe-banner strong {{ color: var(--oe-warn); letter-spacing: 0.04em; }}
.oe-banner span {{ color: var(--oe-text); font-size: 0.95rem; }}
.oe-strip {{ position: fixed; left: 0; right: 0; bottom: 0; z-index: 1000010;
  padding: 6px 16px; text-align: center; font-size: 0.82rem; font-weight: 600;
  background: var(--oe-warn); color: #1A1200; letter-spacing: 0.02em; }}

.oe-cards {{ display: grid; gap: 12px; margin: 0.2rem 0 0.6rem;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); }}
.oe-cards.tight {{ grid-template-columns: repeat(auto-fit, minmax(112px, 1fr)); gap: 8px; }}
.oe-card {{ background: var(--oe-surface); border: 1px solid var(--oe-border);
  border-radius: var(--oe-radius); padding: 14px 16px; min-width: 0; }}
.oe-card.lead {{ border-top: 2px solid var(--oe-accent); }}
.oe-card .label {{ font-size: 0.74rem; font-weight: 600; letter-spacing: 0.07em;
  text-transform: uppercase; color: var(--oe-muted); }}
.oe-card .value {{ font-size: 1.9rem; font-weight: 700; line-height: 1.15; margin-top: 4px;
  color: var(--oe-text); font-variant-numeric: tabular-nums; }}
.oe-card .value small {{ font-size: 0.95rem; font-weight: 500; color: var(--oe-text-2);
  margin-left: 4px; }}
.oe-cards.tight .oe-card {{ padding: 10px 12px; }}
.oe-cards.tight .oe-card .value {{ font-size: 1.35rem; }}
.oe-card .sub {{ margin-top: 6px; color: var(--oe-text-2); font-size: 0.86rem;
  font-variant-numeric: tabular-nums; line-height: 1.45; }}
.oe-card .sub b {{ color: var(--oe-text); font-weight: 600; }}

.oe-flow {{ display: flex; flex-wrap: wrap; align-items: stretch; gap: 10px;
  margin: 0.4rem 0 0.8rem; }}
.oe-step {{ flex: 1 1 220px; background: var(--oe-surface); border: 1px solid var(--oe-border);
  border-radius: var(--oe-radius); padding: 12px 16px; min-width: 0; }}
.oe-step .n {{ font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--oe-accent); }}
.oe-step .t {{ font-size: 1.02rem; font-weight: 600; color: var(--oe-text); margin-top: 2px; }}
.oe-step .d {{ color: var(--oe-text-2); font-size: 0.88rem; margin-top: 4px; line-height: 1.45;
  font-variant-numeric: tabular-nums; }}
.oe-step .n span {{ float: right; color: var(--oe-muted); font-size: 1rem; line-height: 1; }}

.oe-panel {{ margin: 0.5rem 0 0.1rem; }}
.oe-panel .t {{ font-size: 1rem; font-weight: 600; color: var(--oe-text); }}
.oe-panel .c {{ font-size: 0.85rem; color: var(--oe-text-2); margin-top: 1px;
  font-variant-numeric: tabular-nums; }}

.oe-legend {{ display: flex; flex-wrap: wrap; gap: 4px 18px; margin: 0.2rem 0 0.3rem; }}
.oe-legend span {{ display: inline-flex; align-items: center; gap: 7px; color: var(--oe-text);
  font-size: 0.86rem; white-space: nowrap; }}

.oe-kv {{ display: grid; grid-template-columns: minmax(96px, auto) 1fr; gap: 6px 16px;
  font-size: 0.92rem; font-variant-numeric: tabular-nums; margin: 0.2rem 0 0.6rem; }}
.oe-kv dt {{ color: var(--oe-muted); }}
.oe-kv dd {{ color: var(--oe-text); margin: 0; overflow-wrap: anywhere; }}

.oe-empty {{ border: 1px dashed var(--oe-border); border-radius: var(--oe-radius);
  padding: 18px 20px; background: var(--oe-surface); margin: 0.3rem 0 0.4rem; }}
.oe-empty .t {{ font-weight: 600; color: var(--oe-text); font-size: 1rem; }}
.oe-empty .b {{ color: var(--oe-text-2); font-size: 0.92rem; margin-top: 4px; max-width: 72ch; }}

.oe-note {{ border-left: 3px solid var(--oe-accent); padding: 8px 14px; margin: 0.4rem 0;
  color: var(--oe-text-2); font-size: 0.92rem; background: var(--oe-surface);
  border-radius: 0 var(--oe-radius) var(--oe-radius) 0; max-width: 100ch; }}
.oe-note b {{ color: var(--oe-text); }}

[data-testid="stDataFrame"] {{ font-variant-numeric: tabular-nums; }}
/* columns wrap before a plot gets too narrow to read (the sidebar keeps its own columns) */
[data-testid="stMain"] [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; }}
[data-testid="stMain"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
  min-width: min(100%, 300px) !important; }}
[data-testid="stMain"]
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(4))
  > [data-testid="stColumn"] {{ min-width: min(100%, 210px) !important; }}
/* multiselect chips: quiet neutral instead of the accent colour */
[data-testid="stMultiSelectTagsContainer"] [role="group"] > span {{
  background-color: {T.AXIS} !important; }}
[data-testid="stMultiSelectTagsContainer"] [role="group"] > span,
[data-testid="stMultiSelectTagsContainer"] [role="group"] > span * {{
  color: var(--oe-text) !important; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
</style>
"""

_DASH = {
    "solid": "",
    "dot": "2 3",
    "dash": "6 4",
    "dashdot": "7 3 2 3",
    "longdash": "11 4",
    "longdashdot": "11 3 2 3",
}


def esc(value) -> str:
    return html.escape(str(value))


def _html(body: str) -> None:
    st.markdown(body, unsafe_allow_html=True)


# ----------------------------------------------------------------------------------------
# session state that survives page switches
# ----------------------------------------------------------------------------------------
def _wkey(name: str) -> str:
    return f"_w_{name}"


def sync(name: str) -> None:
    st.session_state[name] = st.session_state[_wkey(name)]


def remember(name: str, default, valid: Sequence | None = None) -> str:
    """Make ``st.session_state[name]`` hold a valid value and seed the widget key with it.
    Returns the widget key; pair it with ``on_change=sync, args=(name,)``."""
    current = st.session_state.get(name, default)
    if valid is not None and current not in valid:
        current = default
    st.session_state[name] = current
    st.session_state[_wkey(name)] = current
    return _wkey(name)


def _set(name: str, value) -> None:
    st.session_state[name] = value


# ----------------------------------------------------------------------------------------
# page shell
# ----------------------------------------------------------------------------------------
def setup_page(title: str) -> None:
    sys.path[:] = dict.fromkeys(sys.path)  # pages prepend the repo root on every rerun
    st.set_page_config(
        page_title=f"{title} · OceanEmbed",
        page_icon=":material/waves:",
        layout="wide",
        initial_sidebar_state="auto",  # expanded on a laptop, collapsed on a narrow window
    )
    _html(_CSS)
    with st.sidebar:
        _html(
            '<div class="oe-brand">OceanEmbed<small>Subsurface temperature from surface '
            "satellite fields</small></div>"
        )
        for path, label in PAGES:
            st.page_link(path, label=label)


def select_run() -> D.Run | None:
    """Sidebar run selector shared by all pages. Returns ``None`` (after rendering the empty
    state) when there is no run to show."""
    root = D.outputs_root()
    runs = D.list_runs(str(root))
    if not runs:
        empty_state(
            "No finished run found",
            f"The dashboard reads runs from {root}. Produce one with the pipeline (the synthetic "
            "demo needs no credentials), then reload this page. Set OCEANEMBED_OUTPUTS_ROOT if "
            "your runs live elsewhere.",
            "oceanembed run-all --config configs/synthetic.yaml",
        )
        return None
    by_name = {r["name"]: r for r in runs}
    names = list(by_name)
    with st.sidebar:
        _html('<div class="oe-side-label">Run</div>')
        key = remember("oe_run", names[0], names)
        name = st.selectbox(
            "Run",
            names,
            key=key,
            on_change=sync,
            args=("oe_run",),
            label_visibility="collapsed",
            help="A run is one pipeline output folder under the outputs root.",
        )
    run = D.load_run(by_name[name])
    with st.sidebar:
        source = run.data_source
        chips = chip("SYNTHETIC DATA", warn=True) if run.is_synthetic else chip(f"{source} data")
        _html(f'<div class="oe-context">{chips}</div>')
    return run


def synthetic_banner(run: D.Run) -> None:
    """Persistent notice (top of the page + a fixed strip) for synthetic runs."""
    if not run.is_synthetic:
        return
    _html(
        '<div class="oe-banner" role="alert"><div><strong>SYNTHETIC DATA</strong><br>'
        "<span>This run uses analytically generated ocean fields, not observations. Every number "
        "and map here only demonstrates that the pipeline works end to end; none of it is "
        "evidence of scientific skill.</span></div></div>"
        '<div class="oe-strip" role="status">SYNTHETIC DATA: pipeline demonstration only, '
        "not scientific skill</div>"
    )


def page_header(title: str, subtitle: str, run: D.Run | None = None, chips: str = "") -> None:
    _html(
        f'<div class="oe-header"><h1>{esc(title)}</h1><p>{esc(subtitle)}</p>'
        + (f'<div class="oe-context">{chips}</div>' if chips else "")
        + "</div>"
    )
    if run is not None:
        synthetic_banner(run)


def run_page(render: Callable[[], None]) -> None:
    """Run a page body; an unexpected failure becomes a readable message, never a traceback."""
    try:
        render()
    except Exception as exc:  # noqa: BLE001 - last-resort guard for the UI
        empty_state(
            "This view could not be drawn",
            f"{type(exc).__name__}: {exc}. The run's files may be incomplete or still being "
            "written; re-run the pipeline step that produces them and reload.",
        )


# ----------------------------------------------------------------------------------------
# small building blocks
# ----------------------------------------------------------------------------------------
def chip(text: str, value: str | None = None, *, warn: bool = False, off: bool = False) -> str:
    cls = "oe-chip" + (" warn" if warn else "") + (" off" if off else "")
    body = esc(text) + (f" <b>{esc(value)}</b>" if value is not None else "")
    return f'<span class="{cls}">{body}</span>'


def chips(items: Iterable[str]) -> None:
    _html(f'<div class="oe-context">{"".join(items)}</div>')


def cards(items: Sequence[T.Card], tight: bool = False) -> None:
    out = []
    for c in items:
        unit = f"<small>{esc(c.unit)}</small>" if c.unit else ""
        sub = f'<div class="sub">{c.sub}</div>' if c.sub else ""
        out.append(
            f'<div class="oe-card{" lead" if c.lead else ""}"><div class="label">{esc(c.label)}'
            f'</div><div class="value">{esc(c.value)}{unit}</div>{sub}</div>'
        )
    _html(f'<div class="oe-cards{" tight" if tight else ""}">{"".join(out)}</div>')


def panel_title(title: str, caption: str = "") -> None:
    cap = f'<div class="c">{esc(caption)}</div>' if caption else ""
    _html(f'<div class="oe-panel"><div class="t">{esc(title)}</div>{cap}</div>')


def note(body_html: str) -> None:
    _html(f'<div class="oe-note">{body_html}</div>')


def key_values(rows: Sequence[tuple[str, str]]) -> None:
    body = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in rows)
    _html(f'<dl class="oe-kv">{body}</dl>')


def empty_state(title: str, body: str, command: str | None = None) -> None:
    _html(
        f'<div class="oe-empty"><div class="t">{esc(title)}</div>'
        f'<div class="b">{esc(body)}</div></div>'
    )
    if command:
        st.code(command, language="powershell")


def command_for(run: D.Run, step: str) -> str:
    """The CLI command that (re)produces a missing product of this run."""
    return f"oceanembed {step} --config configs/{run.name}.yaml"


def _swatch(style: T.MethodStyle) -> str:
    c = style.color
    dash = _DASH.get(style.dash, "")
    marks = {
        "circle": f'<circle cx="17" cy="6" r="3.4" fill="{c}"/>',
        "circle-open": f'<circle cx="17" cy="6" r="3.2" fill="{T.BG}" stroke="{c}" '
        'stroke-width="1.6"/>',
        "square": f'<rect x="13.8" y="2.8" width="6.4" height="6.4" fill="{c}"/>',
        "diamond": f'<path d="M17 1.6 21.4 6 17 10.4 12.6 6Z" fill="{c}"/>',
        "x": f'<path d="M13.8 2.8 20.2 9.2M20.2 2.8 13.8 9.2" stroke="{c}" stroke-width="1.8"/>',
        "cross": f'<path d="M17 2v8M13 6h8" stroke="{c}" stroke-width="2"/>',
        "triangle-up": f'<path d="M17 2 21.2 9.4H12.8Z" fill="{c}"/>',
        "triangle-down": f'<path d="M17 10 21.2 2.6H12.8Z" fill="{c}"/>',
    }
    return (
        '<svg width="34" height="12" viewBox="0 0 34 12" aria-hidden="true">'
        f'<line x1="0" y1="6" x2="34" y2="6" stroke="{c}" stroke-width="2" '
        f'stroke-dasharray="{dash}"/>{marks.get(style.symbol, "")}</svg>'
    )


def legend(styles: Iterable[T.MethodStyle], labels: dict[str, str] | None = None) -> None:
    """Method legend shared by a row of small multiples: colour + dash + marker per method."""
    labels = labels or {}
    items = "".join(f"<span>{_swatch(s)}{esc(labels.get(s.key, s.label))}</span>" for s in styles)
    _html(f'<div class="oe-legend" role="list" aria-label="Methods">{items}</div>')


def plot(fig, key: str, **kw):
    """Render a Plotly figure with the dashboard's own theme (not Streamlit's)."""
    return st.plotly_chart(fig, theme=None, config=PLOT_CONFIG, key=key, **kw)


# ----------------------------------------------------------------------------------------
# day and depth controls (sidebar)
# ----------------------------------------------------------------------------------------
def _step(name: str, options: Sequence, delta: int) -> None:
    i = options.index(st.session_state[name]) + delta
    st.session_state[name] = options[max(0, min(len(options) - 1, i))]


def day_control(days: Sequence[pd.Timestamp], shortcuts: bool = True) -> pd.Timestamp:
    """Day slider with previous / next buttons (Left / Right keys). Returns the chosen day."""
    options = [D.day_key(d) for d in days]
    with st.sidebar:
        _html('<div class="oe-side-label">Day</div>')
        key = remember("oe_day", options[0], options)
        st.select_slider(
            "Day", options, key=key, on_change=sync, args=("oe_day",),
            label_visibility="collapsed",
        )  # fmt: skip
        i = options.index(st.session_state["oe_day"])
        left, right = st.columns(2)
        left.button(
            "Prev", icon=":material/chevron_left:", on_click=_step,
            args=("oe_day", options, -1), disabled=i == 0, width="stretch", key="oe_day_prev",
            shortcut="Left" if shortcuts else None, help="Previous day",
        )  # fmt: skip
        right.button(
            "Next", icon=":material/chevron_right:", icon_position="right", on_click=_step,
            args=("oe_day", options, 1), disabled=i == len(options) - 1, width="stretch",
            key="oe_day_next", shortcut="Right" if shortcuts else None, help="Next day",
        )  # fmt: skip
    return pd.Timestamp(st.session_state["oe_day"])


def depth_control(depths: Sequence[float], shortcuts: bool = True) -> float:
    """Depth slider over the standard levels with shallower / deeper buttons (Up / Down keys)."""
    options = [float(d) for d in depths]
    default = min(options, key=lambda d: abs(d - 100.0))
    with st.sidebar:
        _html('<div class="oe-side-label">Depth</div>')
        key = remember("oe_depth", default, options)
        st.select_slider(
            "Depth", options, key=key, on_change=sync, args=("oe_depth",),
            format_func=T.fmt_depth, label_visibility="collapsed",
        )  # fmt: skip
        i = options.index(st.session_state["oe_depth"])
        left, right = st.columns(2)
        left.button(
            "Up", icon=":material/keyboard_arrow_up:", on_click=_step,
            args=("oe_depth", options, -1), disabled=i == 0, width="stretch",
            key="oe_depth_up", shortcut="Up" if shortcuts else None, help="One level up",
        )  # fmt: skip
        right.button(
            "Down", icon=":material/keyboard_arrow_down:", on_click=_step,
            args=("oe_depth", options, 1), disabled=i == len(options) - 1, width="stretch",
            key="oe_depth_down", shortcut="Down" if shortcuts else None, help="One level down",
        )  # fmt: skip
    return float(st.session_state["oe_depth"])


def surface_grid(run: D.Run, surface, key_prefix: str, per_row: int = 4) -> None:
    """Small maps of every surface input field, or the empty state when the store is missing."""
    if surface is None:
        empty_state(
            "Surface inputs unavailable",
            "The harmonised store of this run was not found, or does not contain this day: "
            f"{run.meta.get('paths', {}).get('zarr', 'unknown path')}",
            command_for(run, "harmonize"),
        )
        return
    names = list(surface.data_vars)
    for start in range(0, len(names), per_row):
        cols = st.columns(per_row, gap="small")
        for col, name in zip(cols, names[start : start + per_row], strict=False):
            with col:
                spec = T.surface_field(name, dict(surface[name].attrs))
                panel_title(spec.label, spec.unit)
                plot(V.surface_map(surface, name), f"{key_prefix}_{name}")


def no_predictions(run: D.Run) -> None:
    empty_state(
        "This run has no predictions yet",
        "Temperature predictions are written as monthly NetCDF files by the predict step.",
        command_for(run, "predict"),
    )


def point_control(
    lat: Sequence[float], lon: Sequence[float], default: tuple[float, float]
) -> tuple[float, float]:
    """Latitude / longitude inputs for the selected point (shared with the map click)."""
    lat0, lat1, lon0, lon1 = float(min(lat)), float(max(lat)), float(min(lon)), float(max(lon))
    step = round(abs(float(lat[1]) - float(lat[0])), 6) if len(lat) > 1 else 0.25
    for name, value, lo, hi in (
        ("oe_lat", default[0], lat0, lat1),
        ("oe_lon", default[1], lon0, lon1),
    ):
        st.session_state[name] = float(min(max(float(st.session_state.get(name, value)), lo), hi))
        remember(name, value)
    with st.sidebar:
        _html('<div class="oe-side-label">Point</div>')
        st.number_input(
            "Latitude (°N)", lat0, lat1, step=step, format="%.3f", key=_wkey("oe_lat"),
            on_change=sync, args=("oe_lat",),
        )  # fmt: skip
        st.number_input(
            "Longitude (°E)", lon0, lon1, step=step, format="%.3f", key=_wkey("oe_lon"),
            on_change=sync, args=("oe_lon",),
        )  # fmt: skip
    return float(st.session_state["oe_lat"]), float(st.session_state["oe_lon"])


def clickable_plot(fig, key: str) -> None:
    """Render a ``clickable`` map; a new click moves the selected point (``oe_lat`` / ``oe_lon``)
    and reruns the page so the sidebar inputs and every view follow it."""
    event = plot(fig, key, on_select="rerun", selection_mode="points")
    try:
        points = event["selection"]["points"]
    except (KeyError, TypeError):
        points = []
    if not points:
        return
    click = (float(points[-1]["y"]), float(points[-1]["x"]))
    if st.session_state.get(f"{key}_seen") == click:
        return
    st.session_state[f"{key}_seen"] = click
    st.session_state["oe_lat"], st.session_state["oe_lon"] = click
    st.rerun()
