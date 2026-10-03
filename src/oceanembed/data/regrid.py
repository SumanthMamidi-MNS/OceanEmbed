"""Horizontal, temporal and vertical regridding with plain numpy / xarray.

Design notes
------------
* Horizontal regridding is *separable*: ``out = Wy @ field @ Wx.T`` with one weight matrix per
  axis. Block means and bilinear interpolation are both expressed this way, so NaN handling
  (sum of values / sum of valid weights) is exact and fast.
* Block mean (source finer than target): every source cell whose centre lies inside a target
  cell (``[c - res/2, c + res/2)``) contributes; NaNs are ignored and a target cell is valid only
  if at least ``min_valid`` of the *nominal* number of source cells are valid.
* Bilinear (source ~ same resolution as target): NaN-aware, valid where at least ``min_valid``
  of the interpolation weight sits on valid neighbours.
* Source longitudes are wrapped to 0-360 (the canonical grid is in 0-360 too) and axes are
  sorted ascending, so descending latitudes and either lon convention are handled here.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

LAT_NAMES = ("lat", "latitude", "nav_lat", "y")
LON_NAMES = ("lon", "longitude", "nav_lon", "x")
TIME_NAMES = ("time", "valid_time", "t")
DEPTH_NAMES = ("depth", "lev", "level", "z", "deptht")

_EPS = 1e-6


# --------------------------------------------------------------------------------------
# coordinate handling
# --------------------------------------------------------------------------------------
def _find(names: tuple[str, ...], available, override: str | None) -> str | None:
    if override and override in available:
        return override
    for n in names:
        if n in available:
            return n
    return None


def fix_time(obj):
    """Make a ``time`` coordinate ``datetime64``: files such as OSCAR declare a ``julian``
    calendar, which xarray decodes to cftime objects that cannot be mixed with datetime64
    slices. The calendar date (y/m/d/h) of the stamp is what the product means (it matches the
    granule file names), so it is rebuilt directly from the cftime fields."""
    if "time" in obj.coords and obj["time"].dtype == object:
        t = obj["time"].values
        stamps = [
            np.datetime64(f"{x.year:04d}-{x.month:02d}-{x.day:02d}T{x.hour:02d}:{x.minute:02d}")
            for x in t.ravel()
        ]
        obj = obj.assign_coords(time=("time", np.array(stamps, dtype="datetime64[ns]")))
    return obj


def promote_coordinate_variables(obj):
    """Turn a 1-D coordinate *variable* that sits on a differently named dimension into a real
    coordinate (OSCAR: dimension ``longitude`` with the variable ``lon``); without this the
    dimension would be read as a plain 0..N-1 index."""
    if not isinstance(obj, xr.Dataset):
        return obj
    known = {*LAT_NAMES, *LON_NAMES, *TIME_NAMES, *DEPTH_NAMES}
    for dim in list(obj.dims):
        if dim in obj.coords:
            continue
        for name, var in obj.variables.items():
            units = str(var.attrs.get("units", "")).lower()
            geo = units.startswith(("degrees_east", "degrees_north", "degree_east", "degree_north"))
            if name != dim and var.dims == (dim,) and (name in known or geo):
                obj = obj.assign_coords({dim: (dim, var.values, var.attrs)}).drop_vars(name)
                break
    return obj


def standardize(obj, coords: dict[str, str] | None = None):
    """Rename coords to ``time/depth/lat/lon``, wrap lon to 0-360, sort axes ascending and
    transpose to ``(time, depth, lat, lon)`` order (whichever exist)."""
    coords = coords or {}
    obj = promote_coordinate_variables(obj)
    avail = set(obj.dims) | set(obj.coords)
    rename = {}
    for canon, names in (
        ("lat", LAT_NAMES),
        ("lon", LON_NAMES),
        ("time", TIME_NAMES),
        ("depth", DEPTH_NAMES),
    ):
        found = _find(names, avail, coords.get(canon))
        if found is None:
            if canon in ("lat", "lon"):
                raise ValueError(f"no {canon} coordinate found among {sorted(avail)}")
            continue
        if found != canon:
            rename[found] = canon
    obj = obj.rename(rename) if rename else obj
    obj = fix_time(obj)
    lon = obj["lon"].values
    if lon.min() < 0:
        obj = obj.assign_coords(lon=np.mod(lon, 360.0))
    obj = obj.sortby("lon").sortby("lat")
    order = [d for d in ("time", "depth", "lat", "lon") if d in obj.dims]
    if isinstance(obj, xr.DataArray):
        order += [d for d in obj.dims if d not in order]
        obj = obj.transpose(*order)
    return obj


def crop(obj, lat_range: tuple[float, float], lon_range: tuple[float, float]):
    """Crop a standardised object to ``[lat0, lat1] x [lon0, lon1]`` (inclusive)."""
    return obj.sel(lat=slice(*lat_range), lon=slice(*lon_range))


def to_celsius(da: xr.DataArray) -> xr.DataArray:
    """Kelvin -> degC based on the ``units`` attribute (fallback: values clearly in K)."""
    units = str(da.attrs.get("units", "")).strip().lower()
    is_k = units in {"kelvin", "k", "degk", "deg_k"}
    if not units and np.nanmedian(da.isel({d: 0 for d in da.dims[:-2]}).values) > 150:
        is_k = True
    if is_k:
        out = da - 273.15
        out.attrs = {**da.attrs, "units": "degC"}
        return out
    return da


# --------------------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------------------
def _step(c: np.ndarray) -> float:
    if len(c) < 2:
        raise ValueError("need at least two source coordinates")
    return float(np.median(np.diff(c)))


def block_weights(src: np.ndarray, tgt: np.ndarray, res: float) -> np.ndarray:
    """0/1 matrix ``(len(tgt), len(src))``: source centres inside each target cell."""
    lo = tgt[0] - res / 2
    idx = np.floor((src - lo) / res + _EPS).astype(int)
    ok = (idx >= 0) & (idx < len(tgt))
    w = np.zeros((len(tgt), len(src)))
    w[idx[ok], np.nonzero(ok)[0]] = 1.0
    return w


def bilinear_weights(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Linear interpolation matrix ``(len(tgt), len(src))``; rows outside the source are 0."""
    n = len(src)
    w = np.zeros((len(tgt), n))
    tol = 1e-6 * max(_step(src), 1e-9) * 1e3
    inside = (tgt >= src[0] - tol) & (tgt <= src[-1] + tol)
    i = np.clip(np.searchsorted(src, tgt, side="right") - 1, 0, n - 2)
    frac = np.clip((tgt - src[i]) / (src[i + 1] - src[i]), 0.0, 1.0)
    rows = np.nonzero(inside)[0]
    w[rows, i[rows]] = 1.0 - frac[rows]
    w[rows, i[rows] + 1] += frac[rows]
    return w


def _apply(arr: np.ndarray, wy: np.ndarray, wx: np.ndarray) -> np.ndarray:
    return np.matmul(np.matmul(wy, arr), wx.T)


# --------------------------------------------------------------------------------------
# horizontal regridding
# --------------------------------------------------------------------------------------
def regrid_horizontal(
    da: xr.DataArray,
    lat: np.ndarray,
    lon: np.ndarray,
    res: float,
    method: str = "auto",
    min_valid: float = 0.5,
) -> xr.DataArray:
    """Regrid a standardised DataArray (trailing dims lat, lon) onto the target centres."""
    src_lat, src_lon = da["lat"].values.astype(float), da["lon"].values.astype(float)
    dy, dx = _step(src_lat), _step(src_lon)
    if method == "auto":
        method = "block" if (res / dy > 1.5 and res / dx > 1.5) else "bilinear"
    arr = np.asarray(da.values, dtype=np.float64)
    valid = np.isfinite(arr)
    vals = np.where(valid, arr, 0.0)
    if method == "block":
        wy, wx = block_weights(src_lat, lat, res), block_weights(src_lon, lon, res)
        nominal = (res / dy) * (res / dx)
        s = _apply(vals, wy, wx)
        n = _apply(valid.astype(np.float64), wy, wx)
        ok = n >= min_valid * nominal - 1e-9
    elif method == "bilinear":
        wy, wx = bilinear_weights(src_lat, lat), bilinear_weights(src_lon, lon)
        s = _apply(vals, wy, wx)
        n = _apply(valid.astype(np.float64), wy, wx)
        ok = n >= min_valid - 1e-9
    else:
        raise ValueError(f"unknown regrid method {method!r}")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(ok & (n > 0), s / np.where(n > 0, n, 1.0), np.nan).astype(np.float32)
    coords = {
        k: v
        for k, v in da.coords.items()
        if k not in ("lat", "lon") and set(v.dims) <= set(da.dims[:-2])
    }
    coords["lat"] = lat
    coords["lon"] = lon
    return xr.DataArray(out, dims=da.dims, coords=coords, attrs=da.attrs, name=da.name)


# --------------------------------------------------------------------------------------
# temporal
# --------------------------------------------------------------------------------------
def to_daily(da: xr.DataArray) -> xr.DataArray:
    """Average sub-daily (or re-stamp daily, e.g. 12:00) samples to a 00:00 UTC daily axis."""
    t = da["time"].values
    days = t.astype("datetime64[D]")
    uniq = np.unique(days)
    arr = np.asarray(da.values, dtype=np.float64)
    tax = da.dims.index("time")
    out = []
    for d in uniq:
        sel = np.nonzero(days == d)[0]
        block = np.take(arr, sel, axis=tax)
        with np.errstate(all="ignore"):
            valid = np.isfinite(block)
            cnt = valid.sum(axis=tax)
            s = np.where(valid, block, 0.0).sum(axis=tax)
            out.append(np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan))
    res = np.stack(out, axis=tax).astype(np.float32)
    coords = {k: v for k, v in da.coords.items() if "time" not in v.dims}
    coords["time"] = uniq.astype("datetime64[ns]")
    return xr.DataArray(res, dims=da.dims, coords=coords, attrs=da.attrs, name=da.name)


# --------------------------------------------------------------------------------------
# vertical
# --------------------------------------------------------------------------------------
def vertical_weights(src: np.ndarray, tgt: np.ndarray) -> list[tuple[int, int, float] | None]:
    """For each target depth: ``(i_lo, i_hi, w_hi)`` or ``None`` when below the deepest level.
    Targets shallower than the shallowest source level take that level's value."""
    out: list[tuple[int, int, float] | None] = []
    for d in tgt:
        if d <= src[0] + 1e-9:
            out.append((0, 0, 0.0))
        elif d > src[-1] + 1e-9:
            out.append(None)
        else:
            i = int(np.searchsorted(src, d, side="right") - 1)
            i = min(i, len(src) - 2)
            w = float((d - src[i]) / (src[i + 1] - src[i]))
            if w <= 1e-9:
                out.append((i, i, 0.0))
            elif w >= 1 - 1e-9:
                out.append((i + 1, i + 1, 0.0))
            else:
                out.append((i, i + 1, w))
    return out


def interp_to_depths(da: xr.DataArray, depths: np.ndarray) -> xr.DataArray:
    """Linear vertical interpolation along ``depth`` to the target depths (NaN-propagating)."""
    src = da["depth"].values.astype(float)
    if np.any(np.diff(src) <= 0):
        raise ValueError("source depth levels must be strictly ascending")
    ax = da.dims.index("depth")
    arr = np.asarray(da.values, dtype=np.float32)
    plan = vertical_weights(src, np.asarray(depths, dtype=float))
    slabs = []
    for p in plan:
        if p is None:
            shape = list(arr.shape)
            shape[ax] = 1
            slabs.append(np.full(shape, np.nan, dtype=np.float32))
            continue
        lo, hi, w = p
        a = np.take(arr, [lo], axis=ax)
        if hi != lo:
            b = np.take(arr, [hi], axis=ax)
            a = ((1 - w) * a + w * b).astype(np.float32)
        slabs.append(a)
    out = np.concatenate(slabs, axis=ax)
    coords = {k: v for k, v in da.coords.items() if "depth" not in v.dims}
    coords["depth"] = np.asarray(depths, dtype=float)
    return xr.DataArray(out, dims=da.dims, coords=coords, attrs=da.attrs, name=da.name)
