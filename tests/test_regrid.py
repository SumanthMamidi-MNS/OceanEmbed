from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from oceanembed.data.regrid import (
    block_weights,
    interp_to_depths,
    regrid_horizontal,
    standardize,
    to_celsius,
    to_daily,
)

RES = 0.25
TLAT = np.array([10.125, 10.375])
TLON = np.array([60.125, 60.375, 60.625])


def _da(arr, lat, lon):
    return xr.DataArray(arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})


def _fine(res_native):
    """Native cell centres covering the target cells exactly (cell-centred grid)."""
    n_y = int(round(len(TLAT) * RES / res_native))
    n_x = int(round(len(TLON) * RES / res_native))
    lat = 10.0 + res_native * (np.arange(n_y) + 0.5)
    lon = 60.0 + res_native * (np.arange(n_x) + 0.5)
    return lat, lon


def test_block_mean_analytic_5x5():
    lat, lon = _fine(0.05)  # 5x5 native cells per target cell
    arr = np.zeros((len(lat), len(lon)))
    arr[:5, :5] = np.arange(1, 26).reshape(5, 5)  # mean 13
    arr[5:10, 10:15] = 7.0
    out = regrid_horizontal(_da(arr, lat, lon), TLAT, TLON, RES, "block")
    assert out.values[0, 0] == pytest.approx(13.0)
    assert out.values[1, 2] == pytest.approx(7.0)
    assert out.values[0, 1] == pytest.approx(0.0)


def test_block_mean_ignores_nan_and_applies_50pct_rule():
    lat, lon = _fine(0.05)
    arr = np.full((len(lat), len(lon)), 2.0)
    arr[:5, :5] = 4.0
    arr[0, :5] = np.nan  # 5 of 25 NaN: the other 20 are averaged
    arr[:5, 5:10] = np.nan
    arr[:3, 5:10] = 9.0  # 15 of 25 valid (60 %): kept
    arr[5:10, 0:5] = np.nan
    arr[5:7, 0:5] = 1.0  # 10 of 25 valid (40 %): rejected
    out = regrid_horizontal(_da(arr, lat, lon), TLAT, TLON, RES, "block").values
    assert out[0, 0] == pytest.approx(4.0)
    assert out[0, 1] == pytest.approx(9.0)
    assert np.isnan(out[1, 0])
    assert out[1, 1] == pytest.approx(2.0)


def test_block_mean_exactly_at_threshold():
    lat, lon = _fine(0.05)
    arr = np.full((len(lat), len(lon)), np.nan)
    flat = arr[:5, :5].reshape(-1)
    flat[:13] = 5.0  # 13/25 = 52 % valid
    arr[:5, :5] = flat.reshape(5, 5)
    assert regrid_horizontal(_da(arr, lat, lon), TLAT, TLON, RES, "block").values[0, 0] == 5.0
    flat[:12] = 5.0
    flat[12:] = np.nan  # 12/25 = 48 %
    arr[:5, :5] = flat.reshape(5, 5)
    assert np.isnan(regrid_horizontal(_da(arr, lat, lon), TLAT, TLON, RES, "block").values[0, 0])


def test_block_mean_3x3_glorys_like_node_aligned_grid():
    # nodes on multiples of 1/12 degree: target cell edges fall exactly on native nodes
    step = 1 / 12
    lat = np.round(10.0 + step * np.arange(0, 6), 10)
    lon = np.round(60.0 + step * np.arange(0, 9), 10)
    arr = np.arange(len(lat) * len(lon), dtype=float).reshape(len(lat), len(lon))
    w = block_weights(lat, TLAT, RES)
    assert (w.sum(axis=1) == 3).all()  # exactly 3 native rows per target row
    out = regrid_horizontal(_da(arr, lat, lon), TLAT, TLON, RES, "block").values
    assert out[0, 0] == pytest.approx(arr[:3, :3].mean())
    assert out[1, 2] == pytest.approx(arr[3:6, 6:9].mean())


def test_block_mean_2x2_sla_like_chosen_by_auto():
    lat, lon = _fine(0.125)
    arr = np.arange(len(lat) * len(lon), dtype=float).reshape(len(lat), len(lon))
    out = regrid_horizontal(_da(arr, lat, lon), TLAT, TLON, RES, "auto").values
    assert out[0, 0] == pytest.approx(arr[:2, :2].mean())


def test_bilinear_exact_on_linear_field_offset_grid():
    # source nodes on multiples of 0.25 (OSCAR-like, half-cell offset from the target centres)
    lat = np.arange(9.75, 11.01, 0.25)
    lon = np.arange(59.75, 61.01, 0.25)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    arr = 3.0 * la - 2.0 * lo + 5.0
    out = regrid_horizontal(_da(arr, lat, lon), TLAT, TLON, RES, "auto")
    tla, tlo = np.meshgrid(TLAT, TLON, indexing="ij")
    np.testing.assert_allclose(out.values, 3.0 * tla - 2.0 * tlo + 5.0, atol=1e-4)


def test_bilinear_identity_on_aligned_grid():
    lat = np.arange(9.875, 11.0, 0.25)
    lon = np.arange(59.875, 61.0, 0.25)
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(len(lat), len(lon)))
    out = regrid_horizontal(_da(arr, lat, lon), TLAT, TLON, RES, "auto").values
    iy = [np.argmin(abs(lat - v)) for v in TLAT]
    ix = [np.argmin(abs(lon - v)) for v in TLON]
    np.testing.assert_allclose(out, arr[np.ix_(iy, ix)], atol=1e-5)


def test_bilinear_nan_aware():
    lat = np.arange(9.75, 11.01, 0.25)
    lon = np.arange(59.75, 61.01, 0.25)
    arr = np.ones((len(lat), len(lon)))
    # target (0, 0) sits between source rows 1-2 and columns 1-2
    arr[1, 1] = np.nan  # one of four neighbours lost (weight 0.25): still valid
    out = regrid_horizontal(_da(arr, lat, lon), TLAT, TLON, RES, "bilinear").values
    assert out[0, 0] == pytest.approx(1.0)
    arr[1, 2] = np.nan
    arr[2, 1] = np.nan  # 3 of 4 neighbours NaN: only 25 % of the weight is valid
    out = regrid_horizontal(_da(arr, lat, lon), TLAT, TLON, RES, "bilinear").values
    assert np.isnan(out[0, 0])


def test_bilinear_outside_source_is_nan():
    lat = np.arange(10.0, 10.51, 0.25)
    lon = np.arange(60.0, 61.01, 0.25)
    arr = np.ones((len(lat), len(lon)))
    out = regrid_horizontal(_da(arr, lat, lon), np.array([10.125, 12.0]), TLON, RES, "bilinear")
    assert np.isnan(out.values[1]).all() and np.isfinite(out.values[0]).all()


def test_standardize_lon_convention_and_descending_lat():
    lat = np.array([11.0, 10.5, 10.0])  # descending
    lon = np.array([-170.0, 60.0, 61.0, 120.0])  # includes a negative longitude
    arr = np.arange(12.0).reshape(3, 4)
    ds = xr.Dataset(
        {"v": (("latitude", "longitude"), arr)}, coords={"latitude": lat, "longitude": lon}
    )
    out = standardize(ds)["v"]
    assert list(out.dims) == ["lat", "lon"]
    assert np.all(np.diff(out.lat.values) > 0) and np.all(np.diff(out.lon.values) > 0)
    assert out.lon.values.min() >= 0 and 190.0 in out.lon.values  # -170 wrapped to 190
    # values follow their coordinates
    assert out.sel(lat=10.0, lon=60.0).item() == arr[2, 1]
    assert out.sel(lat=11.0, lon=190.0).item() == arr[0, 0]


def test_standardize_transposes_oscar_order():
    arr = np.arange(2 * 3 * 4.0).reshape(2, 3, 4)  # (time, lon, lat)
    ds = xr.Dataset(
        {"u": (("time", "longitude", "latitude"), arr)},
        coords={"time": [0, 1], "longitude": [1.0, 2.0, 3.0], "latitude": [4.0, 3.0, 2.0, 1.0]},
    )
    da = standardize(ds)["u"].transpose("time", "lat", "lon")
    assert da.shape == (2, 4, 3)
    assert da.sel(time=1, lat=4.0, lon=2.0).item() == arr[1, 1, 0]


def test_to_daily_six_hourly_mean_and_noon_restamp():
    t6 = pd.date_range("2022-01-01", periods=8, freq="6h")
    vals = np.array([1, 2, 3, 4, 10, 20, 30, 40], float)
    da = xr.DataArray(
        vals[:, None, None] * np.ones((8, 2, 2)),
        dims=("time", "lat", "lon"),
        coords={"time": t6.values, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    out = to_daily(da)
    assert list(out.time.values) == list(pd.to_datetime(["2022-01-01", "2022-01-02"]).values)
    np.testing.assert_allclose(out.values[:, 0, 0], [2.5, 25.0])
    # a single sample stamped 12:00 lands on that day at 00:00
    noon = da.isel(time=[2, 6]).assign_coords(
        time=pd.to_datetime(["2022-01-01 12:00", "2022-01-02 12:00"])
    )
    o2 = to_daily(noon)
    assert list(o2.time.values) == list(pd.to_datetime(["2022-01-01", "2022-01-02"]).values)
    np.testing.assert_allclose(o2.values[:, 0, 0], [3.0, 30.0])


def test_to_daily_ignores_nan():
    t6 = pd.date_range("2022-01-01", periods=4, freq="6h")
    vals = np.array([1.0, np.nan, 3.0, np.nan])
    da = xr.DataArray(vals[:, None], dims=("time", "lat"), coords={"time": t6.values, "lat": [0.0]})
    assert to_daily(da).values[0, 0] == pytest.approx(2.0)


def _profile_da(levels, f):
    levels = np.asarray(levels, float)
    arr = f(levels)[None, :, None, None] * np.ones((1, len(levels), 1, 2))
    return xr.DataArray(
        arr,
        dims=("time", "depth", "lat", "lon"),
        coords={"time": [0], "depth": levels, "lat": [0.0], "lon": [0.0, 1.0]},
    )


def test_vertical_interp_linear_profile_is_exact():
    glorys = [0.494, 1.541, 5.078, 9.573, 21.599, 47.373, 109.729, 222.475, 453.938, 902.339]
    glorys += [1062.44]
    da = _profile_da(glorys, lambda z: 30.0 - 0.02 * z)
    tgt = np.array([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000.0])
    out = interp_to_depths(da, tgt)
    assert out.shape == (1, 15, 1, 2)
    expect = 30.0 - 0.02 * tgt
    expect[0] = 30.0 - 0.02 * 0.494  # depth 0 takes the shallowest level
    np.testing.assert_allclose(out.values[0, :, 0, 0], expect, atol=1e-4)


def test_vertical_interp_nan_below_floor_and_deepest():
    levels = [0.5, 10.0, 50.0, 200.0, 600.0]
    da = _profile_da(levels, lambda z: z * 0 + 10.0)
    da.values[:, 3:] = np.nan  # sea floor between 50 and 200 m
    out = interp_to_depths(da, np.array([0.0, 30.0, 100.0, 700.0])).values[0, :, 0, 0]
    assert out[0] == 10.0 and out[1] == 10.0
    assert np.isnan(out[2])  # between a valid and a NaN level
    assert np.isnan(out[3])  # deeper than the deepest source level


def test_kelvin_to_celsius():
    da = xr.DataArray(
        np.full((1, 2, 2), 300.15), dims=("time", "lat", "lon"), attrs={"units": "kelvin"}
    )
    out = to_celsius(da)
    assert out.values[0, 0, 0] == pytest.approx(27.0, abs=1e-4)
    assert out.attrs["units"] == "degC"
    da2 = xr.DataArray(
        np.full((1, 2, 2), 27.0), dims=("time", "lat", "lon"), attrs={"units": "degrees_C"}
    )
    assert to_celsius(da2).values[0, 0, 0] == 27.0


def test_process_variable_squeezes_singleton_depth_of_surface_fields():
    """CMEMS SSS has a depth axis of length one (0 m); it must not go through the vertical
    interpolation that the 3-D temperature uses."""
    from oceanembed.config import load_config
    from oceanembed.data.harmonize import process_variable
    from oceanembed.grid import build_grid

    tiny = Path(__file__).resolve().parents[1] / "configs" / "test_tiny.yaml"
    grid = build_grid(load_config(tiny))
    lat = np.arange(grid.lat[0] - 0.0625, grid.lat[-1] + 0.1, 0.125)
    lon = np.arange(grid.lon[0] - 0.0625, grid.lon[-1] + 0.1, 0.125)
    t = pd.date_range("2024-01-01", periods=2)
    raw = xr.DataArray(
        np.full((2, 1, len(lat), len(lon)), 35.0, dtype="float32"),
        dims=("time", "depth", "lat", "lon"),
        coords={"time": t, "depth": [0.0], "lat": lat, "lon": lon},
    )
    out = process_variable(raw, "sss", grid, t.values)
    assert out.shape == (2, *grid.shape)
    assert np.nanmax(abs(out - 35.0)) < 1e-4
    raw3 = raw.copy(data=raw.values.repeat(1, axis=1))
    bad = xr.concat([raw3, raw3], dim="depth").assign_coords(depth=[0.0, 10.0])
    with pytest.raises(ValueError, match="depth levels"):
        process_variable(bad, "sss", grid, t.values)
