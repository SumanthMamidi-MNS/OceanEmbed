"""Derived physical quantities of a temperature profile / field, defined once for every product.

All functions take temperature with the **depth axis first**, ``(D, ...)`` (a single profile
``(D,)`` or a field ``(D, H, W)``), the matching ascending ``depths`` in metres, and return an
array of the trailing shape. A value is NaN whenever the definition cannot be evaluated from valid
standard levels; nothing is extrapolated and no level is invented.

Definitions
-----------
Isotherm depth (:func:`isotherm_depth`, D20 for 20 degC, D23 for 23 degC)
    The profile must be valid and at least as warm as the isotherm at the first (shallowest)
    level. The depth is that of the **shallowest downward crossing**: the first pair of adjacent
    valid levels ``k, k+1`` with ``T_k >= level > T_{k+1}``, linearly interpolated in temperature
    between the two level depths. Consequences: an isotherm that outcrops (``T_0 < level``) or is
    not bracketed above the deepest valid level (the water column ends, or the profile stays
    warmer than ``level``) gives NaN; a profile that crosses the level several times (inversions,
    or crossing back up) is assigned the shallowest downward crossing and the others are ignored;
    ``T_k == level`` exactly counts as warm enough, so a level at exactly ``level`` followed by
    a colder one returns that level's depth. With the 15 standard depths the search is confined
    to 0-1000 m.

Layer integral and layer mean (:func:`layer_integral`, :func:`layer_mean`)
    Trapezoid rule over the standard levels from 0 m to ``top`` (``top`` must be one of the
    levels). All levels in the layer must be valid ("the cell reaches ``top``"), else NaN.
    ``layer_mean = integral / top`` (degC).

Heat content (:func:`heat_content`)
    ``rho * cp * layer_integral(T, 0..top)`` in J m-2 with constant rho = 1025 kg m-3 and
    cp = 3990 J kg-1 K-1, temperature in degC (so the heat content is relative to 0 degC; the
    anomaly is taken against the climatology's own heat content, which makes the reference
    temperature irrelevant). ``heat_content / (rho * cp * top)`` is the equivalent layer-mean
    temperature (:func:`layer_mean`).
"""

from __future__ import annotations

import numpy as np

RHO = 1025.0  # kg m-3
CP = 3990.0  # J kg-1 K-1
HC_TOP_M = 300.0
MIXED_TOP_M = 30.0
ISOTHERMS = {"d20": 20.0, "d23": 23.0}


def isotherm_depth(temp, depths, level: float) -> np.ndarray:
    """Depth (m) of the shallowest downward crossing of ``level`` (see the module docstring)."""
    t = np.asarray(temp, dtype=np.float64)
    z = np.asarray(depths, dtype=np.float64)
    if t.shape[0] != len(z):
        raise ValueError(f"depth axis has {t.shape[0]} levels, depths has {len(z)}")
    out_shape = t.shape[1:]
    if len(z) < 2:
        return np.full(out_shape, np.nan)
    upper, lower = t[:-1], t[1:]
    with np.errstate(invalid="ignore"):
        cross = np.isfinite(upper) & np.isfinite(lower) & (upper >= level) & (lower < level)
        cross &= (t[0] >= level)[None]  # NaN compares False: a missing surface level gives NaN
    has = cross.any(axis=0)
    first = cross.argmax(axis=0)  # index k of the shallowest crossing pair
    ta = np.take_along_axis(upper, first[None], axis=0)[0]
    tb = np.take_along_axis(lower, first[None], axis=0)[0]
    za, zb = z[first], z[first + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = (ta - level) / (ta - tb)
    return np.where(has, za + frac * (zb - za), np.nan)


def layer_integral(temp, depths, top: float) -> np.ndarray:
    """Trapezoid integral of temperature over ``0 .. top`` m in degC m (NaN unless every standard
    level down to ``top`` is valid)."""
    t = np.asarray(temp, dtype=np.float64)
    z = np.asarray(depths, dtype=np.float64)
    if t.shape[0] != len(z):
        raise ValueError(f"depth axis has {t.shape[0]} levels, depths has {len(z)}")
    n = int(np.searchsorted(z, top, side="right"))
    if n < 2 or not np.isclose(z[n - 1], top) or not np.isclose(z[0], 0.0):
        raise ValueError(f"depths must contain 0 and {top:g} m exactly; got {z.tolist()}")
    seg = t[:n]
    dz = np.diff(z[:n]).reshape((-1,) + (1,) * (t.ndim - 1))
    area = (0.5 * (seg[:-1] + seg[1:]) * dz).sum(axis=0)
    return np.where(np.isfinite(seg).all(axis=0), area, np.nan)


def layer_mean(temp, depths, top: float) -> np.ndarray:
    """Depth-weighted (trapezoid) mean temperature over ``0 .. top`` m, degC."""
    return layer_integral(temp, depths, top) / float(top)


def heat_content(temp, depths, top: float = HC_TOP_M) -> np.ndarray:
    """Upper-ocean heat content ``rho cp integral(T dz)`` over ``0 .. top`` m, J m-2."""
    return RHO * CP * layer_integral(temp, depths, top)


# name -> (label, unit, function(temp, depths) -> array)
QUANTITIES = {
    "d20": ("Depth of the 20 degC isotherm", "m", lambda t, z: isotherm_depth(t, z, 20.0)),
    "d23": ("Depth of the 23 degC isotherm", "m", lambda t, z: isotherm_depth(t, z, 23.0)),
    "hc300": ("Heat content 0-300 m", "J m-2", lambda t, z: heat_content(t, z, HC_TOP_M)),
    "t300": (
        "Mean temperature 0-300 m (heat content / (rho cp 300 m))",
        "degC",
        lambda t, z: layer_mean(t, z, HC_TOP_M),
    ),
    "t030": (
        "Mean temperature 0-30 m (mixed-layer proxy)",
        "degC",
        lambda t, z: layer_mean(t, z, MIXED_TOP_M),
    ),
}


def derived_fields(temp, depths) -> dict[str, np.ndarray]:
    """Every quantity of :data:`QUANTITIES` for one temperature array ``(D, ...)``."""
    return {name: fn(temp, depths) for name, (_, _, fn) in QUANTITIES.items()}


# ----------------------------------------------------------------------------------------
# strata
# ----------------------------------------------------------------------------------------
SEASONS = {
    "winter_monsoon": (12, 1, 2),
    "pre_monsoon": (3, 4, 5),
    "summer_monsoon": (6, 7, 8, 9),
    "post_monsoon": (10, 11),
}
SEASON_LABELS = {
    "winter_monsoon": "winter monsoon (Dec-Feb)",
    "pre_monsoon": "pre-monsoon (Mar-May)",
    "summer_monsoon": "summer monsoon (Jun-Sep)",
    "post_monsoon": "post-monsoon (Oct-Nov)",
}


def season_of_month(month: int) -> str:
    for name, months in SEASONS.items():
        if month in months:
            return name
    raise ValueError(f"month {month} out of range")


def tercile_edges(values) -> tuple[float, float]:
    """Lower and upper tercile (33.3 and 66.7 percentiles) of the finite ``values``."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    q = np.quantile(v, [1 / 3, 2 / 3])
    return float(q[0]), float(q[1])


def tercile_class(values, edges: tuple[float, float]) -> np.ndarray:
    """0 / 1 / 2 for ``values < lo``, ``lo <= values < hi``, ``values >= hi``; -1 where not finite
    (those samples belong to no tercile)."""
    v = np.asarray(values, dtype=np.float64)
    lo, hi = edges
    cls = np.where(v < lo, 0, np.where(v < hi, 1, 2))
    return np.where(np.isfinite(v), cls, -1).astype(np.int8)
