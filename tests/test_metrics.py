"""Streaming metric accumulators against direct numpy computations (pure numpy, no data)."""

from __future__ import annotations

import numpy as np
import pytest

from oceanembed.eval.metrics import (
    MetricAccumulator,
    metrics_from_sums,
    point_metrics,
    skill_score,
    sums_from_arrays,
)

SHAPE = (4, 6, 8)  # (D, H, W)


def _data(seed=0, n=12):
    rng = np.random.default_rng(seed)
    ref = rng.normal(15, 5, (n, *SHAPE))
    pred = ref + rng.normal(0.3, 1.0, ref.shape)  # bias +0.3, noise 1
    valid = rng.random(ref.shape) > 0.25
    # some non-finite entries that must be skipped regardless of the mask
    pred[rng.random(pred.shape) > 0.97] = np.nan
    ref[rng.random(ref.shape) > 0.97] = np.inf
    return pred, ref, valid


def _direct(pred, ref, valid):
    ok = valid & np.isfinite(pred) & np.isfinite(ref)
    x, y = pred[ok], ref[ok]
    e = x - y
    return {
        "n": len(x),
        "rmse": np.sqrt(np.mean(e**2)),
        "bias": np.mean(e),
        "mae": np.mean(np.abs(e)),
        "corr": np.corrcoef(x, y)[0, 1],
    }


def test_total_matches_numpy_and_streaming_equals_one_shot():
    pred, ref, valid = _data()
    one = MetricAccumulator(SHAPE)
    one.update(pred, ref, valid)
    chunked = MetricAccumulator(SHAPE)
    for a, b in ((0, 1), (1, 5), (5, 12)):  # uneven chunks
        chunked.update(pred[a:b], ref[a:b], valid[a:b])
    d = _direct(pred, ref, valid)
    for acc in (one, chunked):
        m = acc.metrics_total()
        assert m["n"] == d["n"]
        for k in ("rmse", "bias", "mae", "corr"):
            assert m[k] == pytest.approx(d[k], rel=1e-10, abs=1e-12), k
    for f in one.sums:
        np.testing.assert_allclose(one.sums[f], chunked.sums[f], rtol=1e-12)


def test_merge_equals_single_accumulator():
    pred, ref, valid = _data(1)
    a, b, c = (MetricAccumulator(SHAPE) for _ in range(3))
    a.update(pred[:6], ref[:6], valid[:6])
    b.update(pred[6:], ref[6:], valid[6:])
    c.update(pred, ref, valid)
    a.merge(b)
    np.testing.assert_allclose(a.metrics_total()["rmse"], c.metrics_total()["rmse"], rtol=1e-12)


def test_by_depth_by_region_and_map():
    pred, ref, valid = _data(2)
    acc = MetricAccumulator(SHAPE)
    acc.update(pred, ref, valid)
    per_depth = acc.metrics_by_depth()
    for d in range(SHAPE[0]):
        want = _direct(pred[:, d], ref[:, d], valid[:, d])
        assert per_depth["rmse"][d] == pytest.approx(want["rmse"], rel=1e-10)
        assert per_depth["corr"][d] == pytest.approx(want["corr"], rel=1e-10)
        assert per_depth["bias"][d] == pytest.approx(want["bias"], rel=1e-10)
        assert per_depth["n"][d] == want["n"]
    region = np.zeros(SHAPE[1:], dtype=bool)
    region[1:4, 2:6] = True
    by_region = acc.metrics_by_depth(region)
    for d in range(SHAPE[0]):
        want = _direct(pred[:, d][:, region], ref[:, d][:, region], valid[:, d][:, region])
        assert by_region["rmse"][d] == pytest.approx(want["rmse"], rel=1e-10)
    # pooled over a subset of depths and a region
    tot = acc.metrics_total(region, [1, 2])
    want = _direct(pred[:, 1:3][..., region], ref[:, 1:3][..., region], valid[:, 1:3][..., region])
    assert tot["rmse"] == pytest.approx(want["rmse"], rel=1e-10)
    assert tot["mae"] == pytest.approx(want["mae"], rel=1e-10)
    # per-point map: temporal statistics at one grid point
    mp = acc.metrics_map()
    assert mp["rmse"].shape == SHAPE
    d, j, k = 2, 3, 4
    w = _direct(pred[:, d, j, k], ref[:, d, j, k], valid[:, d, j, k])
    assert mp["rmse"][d, j, k] == pytest.approx(w["rmse"], rel=1e-10)
    assert mp["bias"][d, j, k] == pytest.approx(w["bias"], rel=1e-10)
    assert mp["n"][d, j, k] == w["n"]


def test_single_day_3d_input_and_shape_check():
    pred, ref, valid = _data(3, n=1)
    acc = MetricAccumulator(SHAPE)
    acc.update(pred[0], ref[0], valid[0])
    assert acc.metrics_total()["n"] == _direct(pred, ref, valid)["n"]
    with pytest.raises(ValueError):
        acc.update(np.zeros((2, 3, 3, 3)), np.zeros((2, 3, 3, 3)))


def test_nan_where_no_data_and_few_samples():
    acc = MetricAccumulator((1, 2, 2))
    x = np.array([[[[1.0, np.nan], [2.0, 3.0]]]])
    y = np.array([[[[1.5, 1.0], [2.5, np.nan]]]])
    acc.update(x, y)
    m = acc.metrics_map()
    assert m["n"][0].tolist() == [[1, 0], [1, 0]]
    assert np.isnan(m["rmse"][0, 0, 1]) and np.isfinite(m["rmse"][0, 0, 0])
    assert np.isnan(m["corr"]).all()  # one sample per point: correlation undefined


def test_constant_series_has_nan_correlation_but_defined_rmse():
    m = point_metrics(np.full(10, 5.0), np.arange(10.0))
    assert np.isnan(m["corr"]) and m["rmse"] > 0


def test_point_metrics_matches_numpy_and_ignores_nan():
    rng = np.random.default_rng(5)
    y = rng.normal(size=200)
    x = 0.8 * y + rng.normal(scale=0.3, size=200)
    x[:7] = np.nan
    m = point_metrics(x, y)
    ok = np.isfinite(x)
    assert m["n"] == ok.sum()
    assert m["rmse"] == pytest.approx(np.sqrt(np.mean((x[ok] - y[ok]) ** 2)))
    assert m["corr"] == pytest.approx(np.corrcoef(x[ok], y[ok])[0, 1])
    assert m["bias"] == pytest.approx(np.mean(x[ok] - y[ok]))


def test_skill_score_against_climatology():
    rng = np.random.default_rng(6)
    truth = rng.normal(size=500)
    clim_err = rng.normal(scale=2.0, size=500)
    model_err = 0.5 * clim_err
    clim = truth + clim_err
    model = truth + model_err
    mse_m = np.mean((model - truth) ** 2)
    mse_c = np.mean((clim - truth) ** 2)
    assert skill_score(mse_m, mse_c) == pytest.approx(1 - mse_m / mse_c)
    assert skill_score(mse_m, mse_c) == pytest.approx(0.75, abs=1e-9)  # error halved -> 1 - 1/4
    assert skill_score(mse_c, mse_c) == pytest.approx(0.0)
    assert skill_score(5.0, 0.0) != skill_score(5.0, 0.0)  # NaN when the climatology is perfect
    out = skill_score(np.array([1.0, 2.0]), np.array([4.0, np.nan]))
    assert out[0] == pytest.approx(0.75) and np.isnan(out[1])


def test_anomaly_correlation_is_lower_than_raw_with_shared_gradient():
    """Documents why both exist: a shared climatological gradient inflates the raw correlation."""
    rng = np.random.default_rng(7)
    clim = np.linspace(5, 30, 400)  # strong shared gradient
    truth_anom = rng.normal(size=400)
    pred_anom = 0.3 * truth_anom + rng.normal(size=400)
    raw = point_metrics(clim + pred_anom, clim + truth_anom)["corr"]
    anom = point_metrics(pred_anom, truth_anom)["corr"]
    assert raw > 0.95 and anom < 0.5


def test_sums_from_arrays_axis_none_and_zero_dim():
    s = sums_from_arrays(np.array([1.0, 2.0, np.nan]), np.array([1.0, 4.0, 3.0]), axis=None)
    assert s["n"] == 2 and s["sxy"] == pytest.approx(9.0) and s["se2"] == pytest.approx(4.0)
    m = metrics_from_sums(s)
    assert m["rmse"] == pytest.approx(np.sqrt(2.0))
