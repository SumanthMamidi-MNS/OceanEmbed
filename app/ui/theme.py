"""Design tokens of the dashboard (recorded in ``docs/design.md``), plus small formatting helpers.

Nothing here imports Streamlit, so the tokens can be used by the figure builders and tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import plotly.express as px

# ----------------------------------------------------------------------------------------
# colour tokens (dark, deep-ocean)
# ----------------------------------------------------------------------------------------
BG = "#070D1A"  # page background
SIDEBAR = "#0A1222"
SURFACE = "#0E1728"  # cards and plot areas
RAISED = "#142037"  # hover labels, chips
BORDER = "#1F2C44"
GRID = "#1A2740"  # plot grid lines (one step off the surface)
AXIS = "#33435F"
TEXT = "#E6ECF5"
TEXT_2 = "#A9B6CA"
TEXT_MUTED = "#7C8BA3"
ACCENT = "#45C4D6"
WARN = "#F2B544"  # synthetic-data banner only
WARN_BG = "#231B07"
LAND = "#2B3547"  # land / below the sea floor: flat neutral, never a data colour

FONT = '"Source Sans", "Source Sans 3", "Source Sans Pro", "Segoe UI", system-ui, sans-serif'
FONT_SIZE = 13

# spacing scale (px) used by the CSS below and the figure margins
SPACE = {"xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 24, "xxl": 32}
RADIUS = 10

# ----------------------------------------------------------------------------------------
# colormaps
# ----------------------------------------------------------------------------------------
# Temperature: cmocean "thermal" (perceptually uniform), without its darkest 10 % so the coldest
# water never looks like land on the dark theme.
TEMPERATURE_SCALE = px.colors.sample_colorscale("thermal", list(np.linspace(0.10, 1.0, 24)))
SALINITY_SCALE = px.colors.sample_colorscale("haline", list(np.linspace(0.08, 1.0, 24)))
MAGNITUDE_SCALE = "Viridis"  # RMSE, correlation, skill
DIVERGING_SCALE = "RdBu_r"  # difference / bias / signed fields, always centred on zero
# depth as a colour (Argo scatter): light = shallow, darker = deep, never darker than readable
DEPTH_SCALE = px.colors.sample_colorscale("deep", list(np.linspace(0.05, 0.75, 16)))

# ----------------------------------------------------------------------------------------
# methods: one fixed colour + dash + marker each (never colour alone)
# ----------------------------------------------------------------------------------------
# Chromatic slots validated on SURFACE with the dataviz palette validator (lightness band,
# chroma floor, CVD separation >= 8, contrast >= 3:1). Climatology and Argo are deliberately
# neutral: they are the reference lines everything else is read against.


@dataclass(frozen=True)
class MethodStyle:
    key: str
    label: str
    color: str
    dash: str
    symbol: str
    width: float = 2.0


_FIXED: dict[str, tuple[str, str, str, str]] = {
    # key: (short label, colour, dash, marker symbol)
    "model": ("OceanEmbed", "#3987E5", "solid", "circle"),
    "ridge": ("Ridge regression", "#D95926", "dashdot", "square"),
    "glorys": ("GLORYS reanalysis", "#199E70", "longdash", "diamond"),
    "climatology": ("Climatology", "#8E9AAD", "dot", "x"),
    "argo": ("Argo observation", "#E6ECF5", "solid", "circle-open"),
    # the standard ablation (same network, encoder not pretrained)
    "model_scratch": ("OceanEmbed, no pretraining", "#C98500", "dash", "triangle-up"),
}
# any other model_<tag> ablation, assigned in sorted order of the keys present in the run
_ABLATION_SLOTS = [
    ("#D55181", "longdashdot", "triangle-down"),
    ("#9085E9", "dash", "cross"),
]
# back-to-front drawing order: references first, the model on top
_ORDER = {"climatology": 0, "ridge": 1, "glorys": 3, "model": 9}


def method_styles(keys, labels: dict[str, str] | None = None) -> dict[str, MethodStyle]:
    """Style for every method key, in drawing order (back to front).

    Known methods have one fixed style everywhere. Other ``model_<tag>`` ablations take the
    spare slots in sorted key order, so within a run a method keeps its style on every page.
    ``labels`` (e.g. from the metrics metadata) is only used for keys this module does not know.
    """
    labels = labels or {}
    keys = list(dict.fromkeys(keys))
    extra = sorted(k for k in keys if k not in _FIXED)
    out: dict[str, MethodStyle] = {}
    for k in sorted(keys, key=lambda k: (_ORDER.get(k, 2), k)):
        if k in _FIXED:
            label, color, dash, symbol = _FIXED[k]
        else:
            color, dash, symbol = _ABLATION_SLOTS[extra.index(k) % len(_ABLATION_SLOTS)]
            label = labels.get(k) or f"OceanEmbed, {k.removeprefix('model_')}"
        out[k] = MethodStyle(k, label, color, dash, symbol, 2.6 if k == "model" else 2.0)
    return out


def legend_order(styles: dict[str, MethodStyle]) -> list[str]:
    """Reading order for legends and tables: the model first, references last."""
    return sorted(styles, key=lambda k: (-_ORDER.get(k, 2), k))


# ----------------------------------------------------------------------------------------
# surface input fields (presentation only)
# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldSpec:
    label: str
    unit: str
    scale: object
    diverging: bool = False


SURFACE_FIELDS: dict[str, FieldSpec] = {
    "sst": FieldSpec("Sea surface temperature", "°C", TEMPERATURE_SCALE),
    "sss": FieldSpec("Sea surface salinity", "PSU", SALINITY_SCALE),
    "sla": FieldSpec("Sea level anomaly", "m", DIVERGING_SCALE, True),
    "uo": FieldSpec("Surface current, eastward", "m/s", DIVERGING_SCALE, True),
    "vo": FieldSpec("Surface current, northward", "m/s", DIVERGING_SCALE, True),
    "uw": FieldSpec("10 m wind, eastward", "m/s", DIVERGING_SCALE, True),
    "vw": FieldSpec("10 m wind, northward", "m/s", DIVERGING_SCALE, True),
}


def surface_field(name: str, attrs: dict | None = None) -> FieldSpec:
    if name in SURFACE_FIELDS:
        return SURFACE_FIELDS[name]
    attrs = attrs or {}
    return FieldSpec(str(attrs.get("long_name", name)), str(attrs.get("units", "")), "Viridis")


# ----------------------------------------------------------------------------------------
# metrics (presentation only)
# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MetricSpec:
    label: str
    axis: str
    unit: str
    decimals: int
    zero_line: bool = False
    note: str = ""


METRICS: dict[str, MetricSpec] = {
    "rmse": MetricSpec("RMSE", "RMSE (°C)", "°C", 2, note="Root-mean-square error."),
    "bias": MetricSpec("Bias", "bias (°C)", "°C", 2, True, "Method minus reference."),
    "mae": MetricSpec("MAE", "MAE (°C)", "°C", 2, note="Mean absolute error."),
    "corr_anom": MetricSpec(
        "Anomaly correlation",
        "anomaly correlation",
        "",
        2,
        note="Climatology removed from both sides.",
    ),
    "corr_raw": MetricSpec(
        "Raw correlation",
        "raw correlation",
        "",
        2,
        note="Inflated by gradients; for reference only.",
    ),
    "skill_vs_clim": MetricSpec(
        "Skill vs climatology",
        "skill (1 − MSE / MSE clim)",
        "",
        2,
        True,
        "0 = no better than climatology, 1 = perfect.",
    ),
}

BASIN_LABELS = {"arabian_sea": "Arabian Sea", "bay_of_bengal": "Bay of Bengal"}


def basin_label(key: str) -> str:
    return BASIN_LABELS.get(key, key.replace("_", " ").title())


@dataclass(frozen=True)
class Card:
    """A headline number: label, value (+ unit) and an optional line of context."""

    label: str
    value: str
    unit: str = ""
    sub: str = ""  # trusted HTML assembled from formatted numbers and fixed text
    lead: bool = False


# ----------------------------------------------------------------------------------------
# formatting
# ----------------------------------------------------------------------------------------
MISSING = "–"


def is_number(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and math.isfinite(float(v))


def fmt(v, decimals: int = 2, unit: str = "", signed: bool = False) -> str:
    """Number with fixed decimals and a real minus sign; an en dash when missing."""
    if not is_number(v):
        return MISSING
    s = f"{float(v):+.{decimals}f}" if signed else f"{float(v):.{decimals}f}"
    s = s.replace("-", "−")
    return f"{s} {unit}".strip() if unit else s


def fmt_int(v) -> str:
    return f"{int(v):,}" if is_number(v) else MISSING


def fmt_depth(d) -> str:
    return f"{float(d):g} m"


def fmt_lat(v: float) -> str:
    return f"{abs(v):.2f}°{'N' if v >= 0 else 'S'}"


def fmt_lon(v: float) -> str:
    return f"{abs(v):.2f}°{'E' if v >= 0 else 'W'}"


def to_array(values) -> np.ndarray:
    """JSON list (``None`` = missing) -> float array with NaN."""
    return np.array([np.nan if v is None else v for v in values], dtype=float)
