"""Argo vertical interpolation rule, collocation and gridded-ARGO planning (hand-made profiles)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oceanembed.config import STANDARD_DEPTHS
from oceanembed.eval.argo_validation import (
    basin_of,
    collocate,
    interp_profile,
    max_gap,
    plan_gridded_matches,
    profile_table,
)
from oceanembed.grid import Grid, build_grid

Z = np.array(STANDARD_DEPTHS, dtype=float)


def _interp(depth, temp, targets=Z):
    return interp_profile(np.asarray(depth, float), np.asarray(temp, float), targets)


def test_max_gap_rule_values():
    assert max_gap(5) == 10 and max_gap(50) == 10
    assert max_gap(100) == pytest.approx(20) and max_gap(500) == pytest.approx(100)
    assert max_gap(1000) == pytest.approx(200)


def test_linear_interpolation_between_bracketing_levels():
    # profile T = 30 - 0.02 z sampled every 8 m -> interpolation is exact for a linear profile
    d = np.arange(4.0, 1100.0, 8.0)
    out = _interp(d, 30 - 0.02 * d)
    assert np.allclose(out[1:-1], 30 - 0.02 * Z[1:-1], atol=1e-9)
    assert out[0] == pytest.approx(30 - 0.02 * 4.0)  # 0 m: the surface rule uses the 4 m level
    # exact level is returned as is (no interpolation)
    out = _interp([0.5, 50.0, 60.0], [20.0, 10.0, 9.0], np.array([50.0]))
    assert out[0] == 10.0


def test_gap_tolerance_depends_on_depth():
    # 20 m gap around 100 m is exactly tolerated (max_gap(100) = 20) ...
    out = _interp([90.0, 110.0, 130.0], [20.0, 18.0, 16.0], np.array([100.0]))
    assert out[0] == pytest.approx(19.0)
    # ... but a 30 m gap there is not
    out = _interp([85.0, 115.0], [20.0, 18.0], np.array([100.0]))
    assert np.isnan(out[0])
    # the same 30 m gap is fine at 200 m (tolerance 40 m)
    out = _interp([185.0, 215.0], [12.0, 10.0], np.array([200.0]))
    assert out[0] == pytest.approx(11.0)
    # a 15 m hole at 50 m (tolerance 10 m) is rejected even though 15 < 0.2*100
    out = _interp([42.0, 57.0], [25.0, 22.0], np.array([50.0]))
    assert np.isnan(out[0])


def test_no_extrapolation_below_deepest_or_above_shallowest():
    out = _interp([5.0, 100.0, 400.0], [28.0, 22.0, 12.0], np.array([30.0, 300.0, 500.0, 1000.0]))
    assert np.isnan(out[0])  # 5..100 m spans 95 m > max_gap(30)
    assert np.isnan(out[1])  # 100..400 m spans 300 m > max_gap(300) = 60
    assert np.isnan(out[2]) and np.isnan(out[3])  # deeper than the deepest level
    # interior depth far from the shallowest level is not extrapolated upwards either
    out = _interp([40.0, 50.0], [25.0, 24.0], np.array([20.0]))
    assert np.isnan(out[0])


def test_surface_rule_for_zero_and_five_metres():
    # shallowest obs at 6 m: 0 m and 5 m use it (within 10 m)
    out = _interp([6.0, 16.0, 26.0], [28.0, 27.9, 27.8], np.array([0.0, 5.0, 10.0]))
    assert out[0] == 28.0 and out[1] == 28.0
    assert out[2] == pytest.approx(27.96)  # 10 m is bracketed -> interpolated, not the surface rule
    # shallowest obs at 12 m: 5 m is 7 m away -> allowed; 0 m is 12 m away -> not
    out = _interp([12.0, 22.0], [28.0, 27.0], np.array([0.0, 5.0]))
    assert np.isnan(out[0]) and out[1] == 28.0
    # shallowest obs at 11 m: neither 0 m nor 5 m? 5 m is 6 m away -> ok, 0 m is 11 m -> no
    out = _interp([11.0, 21.0], [28.0, 27.0], np.array([0.0, 5.0]))
    assert np.isnan(out[0]) and out[1] == 28.0
    # the surface exception does not apply to deeper levels
    out = _interp([20.0, 30.0], [26.0, 25.0], np.array([10.0]))
    assert np.isnan(out[0])
    # bracketed surface point with an unacceptable gap falls back to the shallowest level
    out = _interp([3.0, 25.0], [28.0, 27.0], np.array([5.0]))
    assert out[0] == 28.0


def test_unsorted_nan_and_empty_input():
    d = np.array([100.0, 10.0, np.nan, 50.0, 0.0])
    t = np.array([20.0, 28.0, 99.0, 25.0, 29.0])
    out = _interp(d, t, np.array([0.0, 10.0, 50.0, 100.0]))
    assert out.tolist() == [29.0, 28.0, 25.0, 20.0]
    assert np.isnan(_interp([], [])).all()


def test_profile_table_builds_matrix_per_profile():
    rows = []
    for pid, shift in (("a", 0.0), ("b", 5.0)):
        for z in (4.0, 12.0, 20.0, 45.0, 55.0, 100.0, 108.0):
            rows.append(
                {
                    "profile_id": pid,
                    "time": pd.Timestamp("2022-02-20 06:00"),
                    "latitude": 10.0,
                    "longitude": 75.0,
                    "depth": z,
                    "temp": 30 - 0.05 * z - shift,
                }  # fmt: skip
            )
    prof, obs = profile_table(pd.DataFrame(rows), Z)
    assert prof["profile_id"].tolist() == ["a", "b"] and obs.shape == (2, len(Z))
    i50, i0 = list(Z).index(50.0), 0
    assert obs[0, i50] == pytest.approx(30 - 0.05 * 50)
    assert obs[1, i50] == pytest.approx(30 - 0.05 * 50 - 5.0)
    assert obs[0, i0] == pytest.approx(30 - 0.05 * 4.0)  # surface rule: shallowest level
    assert np.isnan(obs[:, list(Z).index(200.0)]).all()  # no level below ~108 m
    empty, mat = profile_table(pd.DataFrame(columns=rows[0].keys()), Z)
    assert len(empty) == 0 and mat.shape == (0, len(Z))


# ----------------------------------------------------------------------- collocation
def _grid() -> tuple[Grid, np.ndarray, pd.DatetimeIndex]:
    from oceanembed.config import GridConfig

    g = build_grid(GridConfig(lat_min=8.0, lat_max=12.0, lon_min=72.0, lon_max=78.0))
    ocean = np.ones(g.shape, dtype=bool)
    ocean[0, :] = False  # southernmost row is land
    days = pd.date_range("2022-02-20", "2022-03-01", freq="D")
    return g, ocean, days


def test_collocation_picks_right_cell_and_day():
    g, ocean, days = _grid()
    t = ["2022-02-20 00:10", "2022-02-24 23:50", "2022-03-01 12:00"]
    lat = [10.12, 11.99, 8.30]
    lon = [75.00, 72.01, 77.74]
    day, j, k, status = collocate(pd.to_datetime(t), lat, lon, g, days, ocean)
    assert status.tolist() == ["ok", "ok", "ok"]
    assert day.tolist() == [0, 4, 9]
    # cell centres: lat 8.125 + 0.25 j, lon 72.125 + 0.25 k
    assert j.tolist() == [8, 15, 1] and k.tolist() == [12, 0, 22]
    for jj, kk, la, lo in zip(j, k, lat, lon, strict=True):
        assert abs(g.lat[jj] - la) <= 0.125 + 1e-9 and abs(g.lon[kk] - lo) <= 0.125 + 1e-9


def test_collocation_drops_out_of_domain_land_and_dates():
    g, ocean, days = _grid()
    t = pd.to_datetime(["2022-02-22"] * 6 + ["2022-03-05", "2022-02-10"])
    lat = [10.0, 7.9, 12.2, 8.05, 10.0, 12.0, 10.0, 10.0]  # 8.05 -> land row; 12.0 on the edge
    lon = [75.0, 75.0, 75.0, 75.0, 71.9, 75.0, 75.0, 75.0]
    day, j, k, status = collocate(t, lat, lon, g, days, ocean)
    assert status.tolist() == [
        "ok", "outside_domain", "outside_domain", "land", "outside_domain",
        "outside_domain", "outside_dates", "outside_dates",
    ]  # fmt: skip
    assert j[1] == -1 and (day[6:] == -1).all()


def test_basin_assignment():
    from oceanembed.config import GridConfig

    g = build_grid(GridConfig())  # full domain
    cells = [(10.0, 60.0), (10.0, 90.0), (28.0, 60.0), (10.0, 79.0)]
    j = np.array([int((la - 5.0) / 0.25) for la, _ in cells])
    k = np.array([int((lo - 45.0) / 0.25) for _, lo in cells])
    assert basin_of(j, k, g).tolist() == ["arabian_sea", "bay_of_bengal", "other", "other"]


# ----------------------------------------------------------------------- gridded ARGO plan
def test_plan_gridded_matches_daily_and_monthly():
    days = pd.date_range("2022-02-20", "2022-03-10", freq="D")
    daily = pd.to_datetime(["2022-02-21", "2022-02-22", "2022-02-23", "2022-04-01"])
    assert plan_gridded_matches(daily, days) == [(0, [1]), (1, [2]), (2, [3])]
    # monthly product: only months entirely inside the evaluated days are used
    days = pd.date_range("2022-01-01", "2022-03-31", freq="D")
    monthly = pd.to_datetime(["2022-01-16", "2022-02-15", "2022-03-16", "2022-04-16"])
    plan = plan_gridded_matches(monthly, days)
    assert [p[0] for p in plan] == [0, 1, 2]
    assert len(plan[1][1]) == 28 and plan[1][1][0] == 31
    # a period starting mid-month cannot be compared with that month's mean
    days = pd.date_range("2022-01-10", "2022-03-31", freq="D")
    assert [p[0] for p in plan_gridded_matches(monthly, days)] == [1, 2]
