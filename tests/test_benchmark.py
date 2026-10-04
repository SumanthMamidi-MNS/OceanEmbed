"""Comparison study: tree baselines (fit, save, predict round trip), the job plan, the scoring
sums and the benchmark report on constructed evaluation files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from oceanembed.config import INPUT_GROUPS, load_config
from oceanembed.eval.metrics import SUM_FIELDS, sums_from_arrays
from oceanembed.models.baselines import FEATURE_CHANNELS, N_FEATURES
from oceanembed.research import benchmark as B
from oceanembed.research import benchmark_report as BR

ROOT = Path(__file__).resolve().parents[1]
DEPTHS = np.array([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000.0])


# ------------------------------------------------------------------------------ plan and config
def test_plan_runs_cheap_jobs_first_then_seed_by_seed():
    jobs = B.plan_jobs([0, 1])
    assert jobs[0] == B.Job("ridge", "sst_sla", 0)  # deterministic: once, first
    names = [str(j) for j in jobs]
    assert names.count("ridge_sst_sla seed 0") == 1
    first_seed1 = next(i for i, j in enumerate(jobs) if j.seed == 1)
    assert all(j.seed == 0 for j in jobs[1:first_seed1])  # every family has seed 0 before seed 1
    assert {j.name for j in jobs} == {
        "gbt_all7", "rf_all7", "unet_all7",
        "ridge_sst_sla", "mlp_sst_sla", "gbt_sst_sla", "rf_sst_sla",
    }  # fmt: skip
    only = B.plan_jobs([0], ["unet"], ["all7"])
    assert only == [B.Job("unet", "all7", 0)]
    assert B.plan_jobs([0], ["unet"], ["sst_sla"]) == []  # the plain U-Net is an all-input job
    with pytest.raises(ValueError, match="family"):
        B.plan_jobs([0], ["xgboost"])
    with pytest.raises(ValueError, match="input set"):
        B.plan_jobs([0], None, ["winds"])


def test_input_sets_are_the_two_of_the_existing_results():
    cfg = load_config(ROOT / "configs" / "poc_long.yaml")
    assert list(B.SETS["all7"]) == list(INPUT_GROUPS) == cfg.model.input_groups
    ss = B.set_config(cfg, "sst_sla")
    assert ss.model.input_groups == ["sst", "sla"] and cfg.model.input_groups == list(INPUT_GROUPS)
    assert ss.model.input_variables == ["sst", "sla"]
    assert B.set_config(cfg, "all7") is cfg


# ------------------------------------------------------------------------------ tree baselines
def _sample(n=3000, seed=0):
    """Features of a nonlinear relation: anomaly at depth d depends on sst, sla and the day."""
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(n, N_FEATURES)).astype(np.float32)
    y = np.stack(
        [np.tanh(f[:, 0]) * (1 + 0.1 * d) + 0.5 * f[:, 2] * (d % 3 == 0) for d in range(15)], 1
    ).astype(np.float32)
    y += rng.normal(scale=0.05, size=y.shape).astype(np.float32)
    valid = np.ones_like(y, bool)
    valid[:, 13:] = rng.random((n, 2)) > 0.4  # the deep levels are below the sea floor at times
    return f, np.where(valid, y, 0), valid


def test_gbt_and_rf_learn_a_nonlinear_relation_and_round_trip(tmp_path):
    f, y, v = _sample(3000, 0)
    vf, vy, vv = _sample(800, 1)
    sd = np.ones(15, np.float32)
    zero = B.val_rmse(np.zeros_like(vy), vy, vv, sd)
    gbt, rounds = B.fit_gbt(
        f, y, v, vf, vy, vv, {"num_leaves": 15, "n_estimators": 40}, seed=0, n_threads=2
    )
    assert len(rounds) == 15 and gbt.n_trees == sum(rounds) > 0
    assert B.val_rmse(gbt.predict(vf), vy, vv, sd) < 0.5 * zero
    gbt.save(tmp_path / "gbt.joblib")
    again = B.GbtModel.load(tmp_path / "gbt.joblib", 2)
    np.testing.assert_allclose(again.predict(vf), gbt.predict(vf), rtol=1e-6)
    rf = B.fit_rf(f, y, v, {"min_samples_leaf": 10, "max_features": 1.0}, 0, 2, 20, 2000)
    assert rf.n_trees == 20 and rf.n_nodes > 20
    assert B.val_rmse(rf.predict(vf), vy, vv, sd) < 0.6 * zero
    # the same seed gives the same forest; another seed another one
    rf2 = B.fit_rf(f, y, v, {"min_samples_leaf": 10, "max_features": 1.0}, 0, 2, 20, 2000)
    np.testing.assert_array_equal(rf.predict(vf), rf2.predict(vf))


def test_a_depth_with_too_few_samples_predicts_zero_anomaly():
    f, y, v = _sample(600, 2)
    v[:, 14] = False
    v[:10, 14] = True
    vf, vy, vv = _sample(300, 3)
    gbt, rounds = B.fit_gbt(f, y, v, vf, vy, vv, {"num_leaves": 7, "n_estimators": 10}, 0, 1)
    assert rounds[14] == 0 and gbt.boosters[14] is None
    assert np.all(gbt.predict(vf)[:, 14] == 0)


def test_tree_predictor_matches_the_model_on_ocean_pixels_and_skips_land():
    f, y, v = _sample(1500, 0)
    vf, vy, vv = _sample(300, 1)
    gbt, _ = B.fit_gbt(f, y, v, vf, vy, vv, {"num_leaves": 7, "n_estimators": 15}, 0, 1)
    b, h, w = 2, 4, 5
    rng = np.random.default_rng(5)
    x = torch.zeros(b, 12, h, w)
    feats = rng.normal(size=(b, h, w, N_FEATURES)).astype(np.float32)
    for k, ch in enumerate(FEATURE_CHANNELS):
        x[:, ch] = torch.from_numpy(feats[..., k])
    ocean = torch.ones(h, w, dtype=torch.bool)
    ocean[:, 0] = False  # a land column
    x[:, 7] = ocean.float()
    out = B.TreePredictor(gbt)(x)
    assert out.shape == (b, 15, h, w) and out.dtype == torch.float32
    assert torch.all(out[:, :, :, 0] == 0)  # land is not predicted
    want = gbt.predict(feats.reshape(-1, N_FEATURES)).reshape(b, h, w, 15).transpose(0, 3, 1, 2)
    np.testing.assert_allclose(out[:, :, :, 1:].numpy(), want[:, :, :, 1:], rtol=1e-5)


def test_val_rmse_is_the_pooled_degc_error():
    y = np.zeros((4, 3), np.float32)
    pred = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 0], [0, 0, 5]], np.float32)
    valid = np.ones((4, 3), bool)
    valid[3, 2] = False  # the 5 does not count
    sd = np.array([1.0, 0.5, 2.0], np.float32)
    # errors in degC: 1, 1, 0 ... over 11 valid pairs
    assert B.val_rmse(pred, y, valid, sd) == pytest.approx(np.sqrt((1 + 1) / 11))


# ------------------------------------------------------------------------------ the report
def _eval_file(path: Path, rng, noise: float, dates, bias: float = 0.0):
    """An eval.npz of a method with the given error scale: per day, depth and region sums."""
    t, nd, h, w = len(dates), len(DEPTHS), 4, 6
    ref = rng.normal(size=(t, nd, h, w))
    clim = np.zeros_like(ref)
    pred = ref + rng.normal(scale=noise, size=ref.shape) + bias
    valid = np.ones_like(ref, bool)
    regions = [np.ones((h, w), bool), np.zeros((h, w), bool), np.zeros((h, w), bool)]
    regions[1][:, :3] = True
    regions[2][:, 3:] = True
    raw = np.zeros((3, len(SUM_FIELDS), t, nd))
    anom = np.zeros_like(raw)
    for r, m in enumerate(regions):
        v = valid & m[None, None]
        for kind, (x, yv) in {"raw": (pred, ref), "anom": (pred - clim, ref - clim)}.items():
            part = sums_from_arrays(x, yv, v, axis=(2, 3))
            for i, f in enumerate(SUM_FIELDS):
                (raw if kind == "raw" else anom)[r, i] = part[f]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        raw=raw,
        anom=anom,
        dates=np.array([str(d) for d in dates]),
        depths=DEPTHS,
        regions=np.array(["all", "arabian_sea", "bay_of_bengal"]),
        fields=np.array(SUM_FIELDS),
    )


def _done(d: Path, **train):
    (d / "done.json").write_text(
        json.dumps({"train": {"seconds": 60.0, **train}, "score": {"seconds": 5.0}}),
        encoding="utf-8",
    )


@pytest.fixture
def bench_tree(tmp_path, monkeypatch):
    monkeypatch.setenv("OCEANEMBED_OUTPUTS_ROOT", str(tmp_path / "outputs"))
    cfg = load_config(ROOT / "configs" / "poc_long.yaml")
    dates = [*np.arange("2023-12-01", "2024-01-31", dtype="datetime64[D]")]  # 2023 and 2024
    base = cfg.outputs_dir / "research"
    # error scale per family: the boosted trees are the best, the random forest ties the
    # Transformer, the rest are worse; ridge / climatology are single-seed
    noise = {"scratch": 0.50, "gbt": 0.40, "rf": 0.50, "mlp": 0.60, "ridge": 0.80, "unet": 0.58}
    seeds = (0, 1, 2)
    layout = {
        "all7": {
            "scratch": base / "r2" / "scratch",
            "mlp": base / "r2" / "mlp",
            "ridge": base / "r2" / "ridge",
            "gbt": base / "benchmark" / "gbt_all7",
            "rf": base / "benchmark" / "rf_all7",
            "unet": base / "benchmark" / "unet_all7",
        },
        "sst_sla": {
            "scratch": base / "final_inputs" / "scratch_sst_sla",
            "mlp": base / "benchmark" / "mlp_sst_sla",
            "ridge": base / "benchmark" / "ridge_sst_sla",
            "gbt": base / "benchmark" / "gbt_sst_sla",
            "rf": base / "benchmark" / "rf_sst_sla",
        },
    }
    for fams in layout.values():
        for fam, folder in fams.items():
            for s in (0,) if fam == "ridge" else seeds:
                d = folder / f"seed{s}"
                d.mkdir(parents=True, exist_ok=True)
                # the same truth and error draws for everything, scaled per family
                _eval_file(d / "eval.npz", np.random.default_rng(100), noise[fam] + 0.01 * s, dates)
                _done(d, **({"n_trees": 300, "n_nodes": 90000} if fam in ("gbt", "rf") else {}))
    clim = base / "r2" / "climatology" / "seed0"
    clim.mkdir(parents=True, exist_ok=True)
    _eval_file(clim / "eval.npz", np.random.default_rng(100), 1.2, dates)
    _done(clim)
    return cfg


def test_report_ranks_every_family_and_states_winner_or_tie(bench_tree):
    cfg = bench_tree
    out = BR.make_benchmark_report(cfg, n_boot=200, block_length=3)
    summary = json.loads(out.with_name("summary.json").read_text(encoding="utf-8"))
    assert set(summary["input_sets"]) == {"all7", "sst_sla"}
    a7 = summary["input_sets"]["all7"]
    assert set(a7["families"]) == {"scratch", "unet", "gbt", "rf", "mlp", "ridge", "climatology"}
    assert set(summary["input_sets"]["sst_sla"]["families"]) >= {"gbt", "rf", "mlp", "ridge"}
    rk = a7["ranking"]["pooled"]["all"]
    assert rk["order"][0] == "gbt" and rk["order"][-1] == "climatology"
    assert rk["top"] == "gbt" and "gbt" in rk["tie_group"]
    # the boosted trees (0.40) beat the Transformer (0.50) clearly; the random forest ties it
    d = a7["other_minus_transformer"]["pooled"]["all"]
    pool = "pooled_50_200m"
    assert d["gbt"][pool]["diff"] < 0 and d["gbt"][pool]["established"] is True
    assert d["ridge"][pool]["diff"] > 0 and d["ridge"][pool]["established"] is True
    assert d["rf"][pool]["established"] is False  # same error scale: only seed noise differs
    assert a7["ranking"]["pooled"]["all"]["vs_top"]["mlp"]["diff"] > 0
    bp = a7["best_pixel_vs_transformer"]
    assert bp["all"]["family"] == "gbt" and set(bp) == {"all", "arabian_sea", "bay_of_bengal"}
    assert set(a7["settings"]["periods"]) == {"2023", "2024", "pooled"}
    assert a7["families"]["gbt"]["cost"]["trees"] == 300
    assert a7["families"]["scratch"]["cost"]["parameters"] > 1_000_000
    assert a7["families"]["ridge"]["cost"]["n_seeds"] == 1
    md = out.read_text(encoding="utf-8")
    tokens = (
        "## Input set: all seven inputs",
        "## Input set: SST + sea level",
        "### Ranking",
        "Boosted trees, LightGBM (per pixel)",
        "Winner by period",
        "Training cost",
    )
    for token in tokens:
        assert token in md
    figs = sorted(p.name for p in (out.parent / "figures").glob("*.png"))
    assert figs == ["ranking_all7.png", "ranking_sst_sla.png",
                    "rmse_by_depth_all7.png", "rmse_by_depth_sst_sla.png"]  # fmt: skip


def test_report_refuses_runs_scored_on_different_samples(bench_tree):
    cfg = bench_tree
    f = cfg.outputs_dir / "research" / "benchmark" / "gbt_all7" / "seed0" / "eval.npz"
    z = dict(np.load(f))
    z["raw"] = z["raw"].copy()
    z["raw"][:, SUM_FIELDS.index("n")] *= 0.9  # another number of points per day
    np.savez_compressed(f, **z)
    with pytest.raises(ValueError, match="sample"):
        BR.load_results(cfg)


def test_report_needs_the_climatology_and_the_transformer(bench_tree):
    cfg = bench_tree
    (cfg.outputs_dir / "research" / "r2" / "climatology" / "seed0" / "done.json").unlink()
    with pytest.raises(FileNotFoundError, match="climatology"):
        BR.load_results(cfg)
