"""Plotly figure builders for the dashboard: pure functions of arrays, no Streamlit, no file IO.

Conventions (``docs/design.md``): depth increases downward on a square-root axis (resolves the
thermocline), units on every axis and colour bar, land / below-sea-floor cells are transparent so
the flat ``LAND`` plot background shows through, and every method is drawn with its fixed colour
plus a dash pattern and marker.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from app.ui import theme as T
from plotly.subplots import make_subplots

PLOT_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"],
    "scrollZoom": False,
    "responsive": True,
}

# Expected on-screen figure widths (px): they only set map heights. The page content is capped
# at 1320 px, so columns vary by about +-10 % between a 1366 px laptop and a wide monitor.
WIDTH_FULL, WIDTH_WIDE, WIDTH_HALF, WIDTH_QUARTER = 1150, 680, 560, 275
_NO_COLORBAR_PAD = 62  # px a colour bar would take: keeps bar-less maps aligned with the others
_DEPTH_TICKS = (0, 10, 50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000)


# ----------------------------------------------------------------------------------------
# colour ranges
# ----------------------------------------------------------------------------------------
def _finite(*arrays) -> np.ndarray:
    parts = [np.asarray(a, dtype=float).ravel() for a in arrays if a is not None]
    if not parts:
        return np.array([])
    v = np.concatenate(parts)
    return v[np.isfinite(v)]


def shared_range(*arrays, pct: float = 0.5) -> tuple[float, float]:
    """One (min, max) for several panels: the ``pct`` .. ``100 - pct`` percentile span of all
    finite values (a couple of extreme cells must not flatten the whole map)."""
    v = _finite(*arrays)
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(v, [pct, 100 - pct])
    if hi - lo < 1e-6:
        lo, hi = lo - 0.5, hi + 0.5
    return float(lo), float(hi)


def symmetric_limit(*arrays, pct: float = 99.0, floor: float = 1e-3) -> float:
    """Half-width of a zero-centred colour range: the ``pct`` percentile of ``|values|``."""
    v = _finite(*arrays)
    if v.size == 0:
        return 1.0
    return float(max(np.percentile(np.abs(v), pct), floor))


# ----------------------------------------------------------------------------------------
# shared layout
# ----------------------------------------------------------------------------------------
def _axis(title: str | None = None, **kw) -> dict:
    ax = {
        "title": {"text": title, "font": {"size": 12, "color": T.TEXT_2}, "standoff": 8},
        "gridcolor": T.GRID,
        "linecolor": T.AXIS,
        "zeroline": False,
        "ticks": "outside",
        "ticklen": 4,
        "tickcolor": T.AXIS,
        "tickfont": {"size": 12, "color": T.TEXT_2},
        "automargin": True,
    }
    ax.update(kw)
    return ax


def _layout(fig: go.Figure, height: int, **kw) -> go.Figure:
    fig.update_layout(
        height=height,
        font={"family": T.FONT, "size": T.FONT_SIZE, "color": T.TEXT_2},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=T.SURFACE,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        hoverlabel={
            "bgcolor": T.RAISED,
            "bordercolor": T.AXIS,
            "font": {"family": T.FONT, "size": 13, "color": T.TEXT},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"size": 12, "color": T.TEXT},
        },
        modebar={"bgcolor": "rgba(0,0,0,0)", "color": T.TEXT_MUTED, "activecolor": T.ACCENT},
        dragmode="zoom",
    )
    fig.update_layout(**kw)
    if fig.layout.showlegend is not False and len(fig.data) > 1:
        # room for the horizontal legend above the plot; the model (drawn last) is listed first
        fig.update_layout(margin_t=44, legend_traceorder="reversed")
    return fig


def _colorbar(title: str, **kw) -> dict:
    cb = {
        "title": {"text": title, "side": "right", "font": {"size": 12, "color": T.TEXT_2}},
        "thickness": 10,
        "len": 0.92,
        "outlinewidth": 0,
        "tickfont": {"size": 11, "color": T.TEXT_2},
        "ticks": "outside",
        "ticklen": 3,
        "tickcolor": T.AXIS,
        "xpad": 6,
    }
    cb.update(kw)
    return cb


def _unit(unit: str) -> str:
    return f" {unit}" if unit else ""


# ----------------------------------------------------------------------------------------
# depth axis (square-root, increasing downward)
# ----------------------------------------------------------------------------------------
def depth_pos(depth) -> np.ndarray:
    """Plot position of a depth in metres on the square-root depth axis."""
    return np.sqrt(np.clip(np.asarray(depth, dtype=float), 0.0, None))


def _depth_axis(depths, title: str | None = "depth (m)", **kw) -> dict:
    dmax = float(np.nanmax(depths))
    ticks = [t for t in _DEPTH_TICKS if t <= dmax + 1e-6]
    pad = 0.03 * float(np.sqrt(dmax)) if dmax > 0 else 0.5
    return _axis(
        title,
        tickmode="array",
        tickvals=[float(np.sqrt(t)) for t in ticks],
        ticktext=[f"{t:g}" for t in ticks],
        range=[float(np.sqrt(dmax)) + pad, -pad],
        **kw,
    )


# ----------------------------------------------------------------------------------------
# maps
# ----------------------------------------------------------------------------------------
def _map_height(lat: np.ndarray, lon: np.ndarray, width: int, chrome: int = 105) -> int:
    """Figure height for a lat/lon box in a figure about ``width`` px wide, of which ``chrome``
    px go to tick labels and the colour bar: degrees are square at that width. The map always
    fills its column, so other widths stretch it slightly instead of leaving blank bands."""
    span_lon = float(lon.max() - lon.min()) or 1.0
    span_lat = float(lat.max() - lat.min()) or 1.0
    return int(np.clip(max(width - chrome, 80) * span_lat / span_lon + 34, 110, 640))


def _geo_axes(fig: go.Figure, lat: np.ndarray, lon: np.ndarray, compact: bool = False) -> None:
    dlat = float(np.abs(np.diff(lat)).mean()) if lat.size > 1 else 1.0
    dlon = float(np.abs(np.diff(lon)).mean()) if lon.size > 1 else 1.0
    size = 10 if compact else 12
    fig.update_xaxes(
        **_axis(
            None,
            range=[float(lon.min()) - dlon / 2, float(lon.max()) + dlon / 2],
            ticksuffix="°E",
            showgrid=False,
            nticks=5 if compact else 8,
            tickfont={"size": size, "color": T.TEXT_2},
        )
    )
    fig.update_yaxes(
        **_axis(
            None,
            range=[float(lat.min()) - dlat / 2, float(lat.max()) + dlat / 2],
            ticksuffix="°N",
            showgrid=False,
            nticks=4 if compact else 6,
            tickfont={"size": size, "color": T.TEXT_2},
        )
    )


@dataclass(frozen=True)
class MapMark:
    """Optional annotations on a map: the selected point and / or a section line."""

    lat: float | None = None
    lon: float | None = None
    line_lat: float | None = None
    line_lon: float | None = None


def field_map(
    field: xr.DataArray,
    *,
    colorscale=T.TEMPERATURE_SCALE,
    zmin: float | None = None,
    zmax: float | None = None,
    unit: str = "°C",
    quantity: str = "temperature",
    diverging: bool = False,
    width: int = WIDTH_HALF,
    compact: bool = False,
    mark: MapMark | None = None,
    boxes: dict[str, Sequence[float]] | None = None,
    clickable: bool = False,
    decimals: int = 2,
) -> go.Figure:
    """Heat map of a ``(lat, lon)`` field. NaN cells stay transparent, so land shows as the flat
    ``LAND`` background. ``diverging`` centres the colour range on zero with symmetric limits.

    ``width`` is the expected on-screen width (px) and only sets the figure height; ``boxes``
    draws named ``[lat_min, lat_max, lon_min, lon_max]`` outlines (basins); ``clickable`` adds an
    invisible point per ocean cell so a click can be read back as a selection.
    """
    lat = np.asarray(field["lat"].values, dtype=float)
    lon = np.asarray(field["lon"].values, dtype=float)
    z = np.asarray(field.values, dtype=float)
    if diverging:
        lim = zmax if zmax is not None else symmetric_limit(z)
        zmin, zmax = -lim, lim
    elif zmin is None or zmax is None:
        zmin, zmax = shared_range(z)
    # compact maps name their unit in the panel caption: the colour bar keeps ticks only
    label = f"{quantity} ({unit})" if unit else quantity
    bar = _colorbar("", thickness=8, nticks=4) if compact else _colorbar(label)
    fig = go.Figure(
        go.Heatmap(
            x=lon,
            y=lat,
            z=z,
            zmin=zmin,
            zmax=zmax,
            colorscale=colorscale,
            hoverongaps=False,
            colorbar=bar,
            hoverinfo="skip" if clickable else None,
            hovertemplate=None
            if clickable
            else (
                "%{y:.2f}°N, %{x:.2f}°E<br>"
                f"<b>%{{z:.{decimals}f}}{_unit(unit)}</b><extra>{quantity}</extra>"
            ),
        )
    )
    if clickable:
        # one invisible point per ocean cell: it carries the hover label and makes a click a
        # selection Streamlit can read back (heat maps cannot be selected)
        ys, xs = np.nonzero(np.isfinite(z))
        fig.add_trace(
            go.Scattergl(
                x=lon[xs],
                y=lat[ys],
                customdata=z[ys, xs],
                mode="markers",
                marker={"size": 6, "opacity": 0, "color": T.RAISED},
                selected={"marker": {"opacity": 0}},
                unselected={"marker": {"opacity": 0}},
                hovertemplate=(
                    "%{y:.2f}°N, %{x:.2f}°E<br>"
                    f"<b>%{{customdata:.{decimals}f}}{_unit(unit)}</b>"
                    f"<extra>{quantity} · click to select</extra>"
                ),
                showlegend=False,
                name="cells",
            )
        )
    for name, (lat0, lat1, lon0, lon1) in (boxes or {}).items():
        fig.add_shape(
            type="rect", x0=lon0, x1=lon1, y0=lat0, y1=lat1,
            line={"color": T.TEXT, "width": 1, "dash": "dot"}, opacity=0.75,
        )  # fmt: skip
        fig.add_annotation(
            x=lon0, y=lat1, text=T.basin_label(name), showarrow=False, xanchor="left",
            yanchor="bottom", font={"size": 11, "color": T.TEXT}, bgcolor="rgba(7,13,26,0.6)",
        )  # fmt: skip
    if mark is not None:
        line = {"color": T.TEXT, "width": 1.5, "dash": "dash"}
        if mark.line_lat is not None:
            fig.add_shape(type="line", x0=lon.min(), x1=lon.max(), y0=mark.line_lat,
                          y1=mark.line_lat, line=line)  # fmt: skip
        if mark.line_lon is not None:
            fig.add_shape(type="line", x0=mark.line_lon, x1=mark.line_lon, y0=lat.min(),
                          y1=lat.max(), line=line)  # fmt: skip
        if mark.lat is not None and mark.lon is not None:
            fig.add_trace(
                go.Scatter(
                    x=[mark.lon],
                    y=[mark.lat],
                    mode="markers",
                    marker={
                        "size": 13,
                        "symbol": "circle-open",
                        "color": T.TEXT,
                        "line": {"width": 2.5, "color": T.TEXT},
                    },
                    hovertemplate="selected point<br>%{y:.2f}°N, %{x:.2f}°E<extra></extra>",
                    showlegend=False,
                    name="selected point",
                )
            )
    height = _map_height(lat, lon, width, 88 if compact else 105)
    _layout(fig, height, plot_bgcolor=T.LAND, showlegend=False)
    if clickable:
        fig.update_layout(clickmode="event+select", dragmode="zoom")
    _geo_axes(fig, lat, lon, compact)
    return fig


def points_map(
    ocean: xr.DataArray | None,
    lat,
    lon,
    hover: Sequence[str] | None = None,
    *,
    boxes: dict[str, Sequence[float]] | None = None,
    extent: tuple[float, float, float, float] | None = None,
    width: int = WIDTH_HALF,
) -> go.Figure:
    """Point locations (Argo profiles) over a flat land / ocean base map."""
    fig = go.Figure()
    if ocean is not None:
        glat = np.asarray(ocean["lat"].values, dtype=float)
        glon = np.asarray(ocean["lon"].values, dtype=float)
        fig.add_trace(
            go.Heatmap(
                x=glon,
                y=glat,
                z=np.where(np.asarray(ocean.values, dtype=bool), 1.0, np.nan),
                colorscale=[[0, T.SURFACE], [1, T.SURFACE]],
                showscale=False,
                hoverinfo="skip",
            )  # fmt: skip
        )
    else:
        lat0, lat1, lon0, lon1 = extent or (
            float(np.min(lat)), float(np.max(lat)), float(np.min(lon)), float(np.max(lon)),
        )  # fmt: skip
        glat, glon = np.array([lat0, lat1]), np.array([lon0, lon1])
    for name, (lat0, lat1, lon0, lon1) in (boxes or {}).items():
        fig.add_shape(
            type="rect", x0=lon0, x1=lon1, y0=lat0, y1=lat1,
            line={"color": T.TEXT_MUTED, "width": 1, "dash": "dot"},
        )  # fmt: skip
        fig.add_annotation(
            x=lon0, y=lat1, text=T.basin_label(name), showarrow=False, xanchor="left",
            yanchor="bottom", font={"size": 11, "color": T.TEXT_2},
        )  # fmt: skip
    argo = T.method_styles(["argo"])["argo"]
    fig.add_trace(
        go.Scattergl(
            x=np.asarray(lon, dtype=float),
            y=np.asarray(lat, dtype=float),
            mode="markers",
            marker={"size": 5, "color": argo.color, "opacity": 0.8},
            text=hover,
            hovertemplate=("%{y:.2f}°N, %{x:.2f}°E" + ("<br>%{text}" if hover is not None else ""))
            + "<extra>Argo profile</extra>",
            showlegend=False,
            name="Argo profiles",
        )
    )
    _layout(
        fig,
        _map_height(glat, glon, width, 50),
        plot_bgcolor=T.LAND if ocean is not None else T.SURFACE,
        showlegend=False,
    )
    _geo_axes(fig, glat, glon)
    return fig


def rgb_map(
    rgb: xr.DataArray, ocean: np.ndarray | None = None, *, width: int = WIDTH_HALF
) -> go.Figure:
    """Embedding PCA as an RGB image ``(y, x, 3)`` in [0, 1] with ``lat(y)`` / ``lon(x)``
    coordinates; cells where ``ocean`` is False are painted with the flat land colour."""
    lat = np.asarray(rgb["lat"].values, dtype=float)
    lon = np.asarray(rgb["lon"].values, dtype=float)
    img = np.clip(np.asarray(rgb.values, dtype=float), 0, 1)
    text = np.empty(img.shape[:2], dtype=object)
    for j in range(img.shape[0]):
        for i in range(img.shape[1]):
            r, g, b = img[j, i]
            text[j, i] = (
                f"{T.fmt_lat(lat[j])}, {T.fmt_lon(lon[i])}<br>"
                f"PC1 {r:.2f} · PC2 {g:.2f} · PC3 {b:.2f}"
            )
    pix = (img * 255).round().astype(np.uint8)
    if ocean is not None:
        land = ~np.asarray(ocean, dtype=bool)
        pix[land] = [int(T.LAND[i : i + 2], 16) for i in (1, 3, 5)]
        text[land] = "land"
    # go.Image draws row 0 at the top, so rows go north -> south and the y axis is -lat
    order = np.argsort(-lat)
    dlat = float(np.abs(np.diff(lat)).mean()) if lat.size > 1 else 1.0
    dlon = float(np.abs(np.diff(lon)).mean()) if lon.size > 1 else 1.0
    fig = go.Figure(
        go.Image(
            z=pix[order],
            x0=float(lon.min()),
            dx=dlon,
            y0=-float(lat.max()),
            dy=dlat,
            text=text[order],
            hovertemplate="%{text}<extra>embedding</extra>",
        )
    )
    edges = np.array([lat.min() - dlat / 2, lat.max() + dlat / 2])
    # the right margin stands in for a colour bar, so the image lines up with a map beside it
    _layout(fig, _map_height(edges, lon, width), plot_bgcolor=T.LAND, margin_r=_NO_COLORBAR_PAD)
    step = 5 if edges[1] - edges[0] <= 40 else 10
    ticks = np.arange(np.ceil(edges[0] / step) * step, edges[1] + 1e-9, step)
    fig.update_xaxes(
        **_axis(
            None,
            range=[float(lon.min()) - dlon / 2, float(lon.max()) + dlon / 2],
            ticksuffix="°E",
            showgrid=False,
        )
    )
    fig.update_yaxes(
        **_axis(
            None,
            range=[-float(edges[0]), -float(edges[1])],
            tickmode="array",
            tickvals=[-float(t) for t in ticks],
            ticktext=[f"{t:g}°N" for t in ticks],
            showgrid=False,
            scaleanchor=False,  # fill the frame like the heat maps do
        )
    )
    return fig


# ----------------------------------------------------------------------------------------
# lines against depth
# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Series:
    """One line: its method style and its values (aligned with the depth / time axis)."""

    style: T.MethodStyle
    values: Sequence[float]
    label: str | None = None

    @property
    def name(self) -> str:
        return self.label or self.style.label


def _line_trace(s: Series, **kw) -> dict:
    open_symbol = s.style.symbol.endswith("-open") or s.style.symbol.startswith("x")
    return {
        "name": s.name,
        "line": {"color": s.style.color, "width": s.style.width, "dash": s.style.dash},
        "marker": {
            "symbol": s.style.symbol,
            "size": 7,
            "color": s.style.color,
            "line": {"width": 1.5 if open_symbol else 1, "color": s.style.color if open_symbol
                     else T.SURFACE},
        },
        **kw,
    }  # fmt: skip


def depth_profiles(
    depths,
    series: Iterable[Series],
    *,
    x_title: str,
    unit: str = "°C",
    decimals: int = 2,
    zero_line: bool = False,
    x_range: tuple[float, float] | None = None,
    height: int = 440,
    showlegend: bool = True,
    y_title: str | None = "depth (m)",
) -> go.Figure:
    """Quantities against depth (downward, square-root axis): one line + marker per series.
    Series without any finite value are skipped."""
    depths = np.asarray(depths, dtype=float)
    fig = go.Figure()
    for s in series:
        v = T.to_array(s.values)
        if not np.isfinite(v).any():
            continue
        fig.add_trace(
            go.Scatter(
                x=v,
                y=depth_pos(depths),
                mode="lines+markers",
                customdata=depths,
                hovertemplate=(
                    f"%{{customdata:g}} m<br><b>%{{x:.{decimals}f}}{_unit(unit)}</b>"
                    f"<extra>{s.name}</extra>"
                ),
                **_line_trace(s),
            )
        )
    _layout(fig, height, showlegend=showlegend, hovermode="closest")
    fig.update_xaxes(**_axis(x_title, side="bottom", range=x_range))
    fig.update_yaxes(**_depth_axis(depths, y_title))
    if zero_line:
        fig.add_vline(x=0, line={"color": T.AXIS, "width": 1})
    return fig


def time_series(
    dates,
    series: Iterable[Series],
    *,
    y_title: str,
    unit: str = "°C",
    decimals: int = 2,
    height: int = 380,
) -> go.Figure:
    """Daily values per method; dash patterns distinguish the lines, hover compares them."""
    x = pd.DatetimeIndex(dates)
    fig = go.Figure()
    for s in series:
        v = T.to_array(s.values)
        if not np.isfinite(v).any():
            continue
        tr = _line_trace(s)
        tr.pop("marker")
        fig.add_trace(
            go.Scatter(
                x=x,
                y=v,
                mode="lines",
                hovertemplate=f"<b>%{{y:.{decimals}f}}{_unit(unit)}</b><extra>{s.name}</extra>",
                **tr,
            )
        )
    _layout(fig, height, hovermode="x unified")
    fig.update_xaxes(**_axis(None, hoverformat="%d %b %Y"))
    fig.update_yaxes(**_axis(y_title, rangemode="tozero"))
    return fig


# ----------------------------------------------------------------------------------------
# vertical sections
# ----------------------------------------------------------------------------------------
def section_figure(
    section: xr.Dataset, *, mark: float | None = None, row_height: int = 230
) -> go.Figure:
    """Stacked vertical sections on a shared horizontal axis.

    With a target: predicted and target (one shared colour range) and their difference (diverging,
    symmetric). Without one: predicted, climatology and the predicted anomaly. ``mark`` draws a
    vertical line at the selected point's coordinate.
    """
    along = "lon" if "lon" in section["predicted"].dims else "lat"
    x = np.asarray(section[along].values, dtype=float)
    depths = np.asarray(section["depth"].values, dtype=float)
    has_target = bool(np.isfinite(section["target"].values).any())
    pred = section["predicted"].values
    if has_target:
        second, second_name = section["target"].values, "GLORYS target"
        diff, diff_name = section["difference"].values, "Predicted − target"
    else:
        second, second_name = section["climatology"].values, "Climatology"
        diff, diff_name = pred - second, "Predicted − climatology"
    lo, hi = shared_range(pred, second)
    lim = symmetric_limit(diff)
    names = ["OceanEmbed prediction", second_name, diff_name]
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07, subplot_titles=names
    )
    suffix = "°E" if along == "lon" else "°N"
    custom = np.broadcast_to(depths[:, None], pred.shape)
    panels = [
        (pred, T.TEMPERATURE_SCALE, lo, hi, "temperature (°C)", 0.655, 0.69),
        (second, T.TEMPERATURE_SCALE, lo, hi, "", None, None),
        (diff, T.DIVERGING_SCALE, -lim, lim, "difference (°C)", 0.13, 0.30),
    ]
    for row, (z, scale, zmin, zmax, cb_title, cb_y, cb_len) in enumerate(panels, start=1):
        fig.add_trace(
            go.Heatmap(
                x=x,
                y=depth_pos(depths),
                z=np.asarray(z, dtype=float),
                customdata=custom,
                zmin=zmin,
                zmax=zmax,
                colorscale=scale,
                hoverongaps=False,
                showscale=cb_y is not None,
                colorbar=_colorbar(cb_title, y=cb_y, len=cb_len, yanchor="middle")
                if cb_y is not None
                else None,
                hovertemplate=(
                    f"%{{x:.2f}}{suffix}, %{{customdata:g}} m<br><b>%{{z:.2f}} °C</b>"
                    f"<extra>{names[row - 1]}</extra>"
                ),
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(**_depth_axis(depths, "depth (m)", showgrid=False), row=row, col=1)
        fig.update_xaxes(**_axis(None, showgrid=False, ticksuffix=suffix), row=row, col=1)
        if mark is not None:
            # light on the temperature panels, dark on the (light-centred) difference panel
            color = T.BG if scale is T.DIVERGING_SCALE else T.TEXT
            fig.add_vline(
                x=mark, line={"color": color, "width": 1.5, "dash": "dash"}, row=row, col=1
            )
    _layout(fig, 3 * row_height + 40, plot_bgcolor=T.LAND, margin={"l": 8, "r": 8, "t": 28, "b": 8})
    fig.update_annotations(font={"size": 13, "color": T.TEXT}, xanchor="left", x=0)
    return fig


# ----------------------------------------------------------------------------------------
# distributions and scatter
# ----------------------------------------------------------------------------------------
def difference_histogram(
    diff, *, label: str = "predicted − target", unit: str = "°C", height: int = 190
) -> go.Figure:
    """Distribution of a difference field, on a zero-centred axis."""
    v = _finite(diff)
    lim = symmetric_limit(v, pct=99.5)
    counts, edges = np.histogram(np.clip(v, -lim, lim), bins=41, range=(-lim, lim))
    centres = 0.5 * (edges[:-1] + edges[1:])
    share = 100 * counts / max(counts.sum(), 1)
    fig = go.Figure(
        go.Bar(
            x=centres,
            y=share,
            width=(edges[1] - edges[0]) * 0.86,
            marker={"color": T.TEXT_MUTED, "line": {"width": 0}},
            hovertemplate=(
                f"%{{x:+.2f}}{_unit(unit)}<br><b>%{{y:.1f}} %</b> of cells<extra></extra>"
            ),
        )
    )
    _layout(fig, height, showlegend=False, bargap=0)
    fig.add_vline(x=0, line={"color": T.TEXT, "width": 1})
    fig.update_xaxes(**_axis(f"{label} ({unit})", range=[-lim, lim]))
    fig.update_yaxes(**_axis("% of ocean cells"))
    return fig


def obs_scatter(obs, pred, depth, *, label: str, unit: str = "°C", height: int = 460) -> go.Figure:
    """Observed vs predicted temperature, coloured by depth, with the 1:1 line."""
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    depth = np.asarray(depth, dtype=float)
    lo, hi = shared_range(obs, pred, pct=0.0)
    pad = 0.03 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    dmax = float(depth.max()) if depth.size else 1.0
    ticks = [t for t in _DEPTH_TICKS if t <= dmax + 1e-6]
    fig = go.Figure(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="1:1",
            line={"color": T.TEXT_2, "width": 1, "dash": "dash"},
            hoverinfo="skip",
        )  # fmt: skip
    )
    fig.add_trace(
        go.Scattergl(
            x=obs,
            y=pred,
            mode="markers",
            customdata=depth,
            marker={
                "size": 4,
                "opacity": 0.7,
                "color": depth_pos(depth),
                "colorscale": T.DEPTH_SCALE,
                "cmin": 0,
                "cmax": float(np.sqrt(dmax)),
                "colorbar": _colorbar(
                    "depth (m)",
                    tickmode="array",
                    tickvals=[float(np.sqrt(t)) for t in ticks],
                    ticktext=[f"{t:g}" for t in ticks],
                ),
            },
            hovertemplate=(
                f"Argo %{{x:.2f}}{_unit(unit)}<br>{label} <b>%{{y:.2f}}{_unit(unit)}</b>"
                "<br>%{customdata:g} m<extra></extra>"
            ),
            name=label,
        )
    )
    _layout(fig, height, showlegend=False)
    fig.update_xaxes(**_axis(f"Argo observation ({unit})", range=[lo, hi], constrain="domain"))
    fig.update_yaxes(
        **_axis(f"{label} ({unit})", range=[lo, hi], scaleanchor="x", scaleratio=1,
                constrain="domain")
    )  # fmt: skip
    return fig
