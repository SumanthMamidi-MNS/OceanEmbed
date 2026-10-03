"""Research stage R2: the on-disk array cache, per-year scoring, the runner, Argo and the report
(tiny synthetic config, CPU)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from oceanembed.cli import app
from oceanembed.config import Config, load_config
from oceanembed.data.dataset import cache_dir, make_dataset
from oceanembed.eval.metrics import SUM_FIELDS, sums_from_arrays
from oceanembed.research import r1
from oceanembed.research.r2 import (
    LEARNING_YEARS,
    Job,
    _bootstrap_rmse,
    job_dir,
    plan_jobs,
    r2_dir,
    run_r2,
    train_days,
)
from oceanembed.research.r2_report import build_summary, load_results, make_r2_report
from oceanembed.research.r3_report import POOLED

TINY = Path(__file__).resolve().parents[1] / "configs" / "test_tiny.yaml"
runner = CliRunner()


def test_train_days_counts_the_last_calendar_years(tiny_pipeline: Config):
    ds = make_dataset(tiny_pipeline, "train")
    end = tiny_pipeline.split.train.end
    assert train_days(ds, None, end) == len(ds)
    assert train_days(ds, 5, end) == len(ds)  # the tiny train split is shorter than a year
    dates = ds.dates()
    n = train_days(ds, 1, "2023-01-31")  # one year back from the end: nothing earlier than 2022-02
    assert n == int((dates >= "2022-02-01").sum())


# ----------------------------------------------------------------------------------------
# planning and the profile bootstrap
# ----------------------------------------------------------------------------------------
def test_plan_orders_cheap_first_then_seed_zero_argo_learning_curve_and_other_seeds():
    jobs = plan_jobs([0, 1, 2])
    names = [str(j) for j in jobs]
    assert names[:4] == ["ridge seed 0", "climatology seed 0", "scratch seed 0", "mlp seed 0"]
    assert names[4] == "argo" and names[5:7] == ["scratch_ty2 seed 0", "scratch_ty5 seed 0"]
    assert names[7:] == ["scratch seed 1", "mlp seed 1", "scratch seed 2", "mlp seed 2"]
    assert LEARNING_YEARS == (2, 5)
    assert [str(j) for j in plan_jobs([0], ["scratch", "climatology"], False, False)] == [
        "climatology seed 0",
        "scratch seed 0",
    ]
    assert "argo" not in [j.name for j in plan_jobs([0], ["scratch", "mlp"])]  # needs ridge + clim
    with pytest.raises(ValueError):
        plan_jobs([0], ["resnet"])
    assert Job("model", "scratch", 0, 5).name == "scratch_ty5"


def test_profile_bootstrap_interval_and_paired_difference():
    rng = np.random.default_rng(0)
    n = np.full(200, 6.0)
    good = rng.uniform(0.5, 1.5, 200)
    per = {"model": (n, good * n), "ridge": (n, 2.0 * good * n), "glorys": (n, good * n)}
    out, paired = _bootstrap_rmse(per, 400, 1)
    assert out["model"]["rmse"] == pytest.approx(np.sqrt(good.mean()), rel=1e-9)
    assert out["model"]["ci_lo"] < out["model"]["rmse"] < out["model"]["ci_hi"]
    assert paired["model-ridge"]["excludes_zero"] and paired["model-ridge"]["diff"] < 0
    assert not paired["model-glorys"]["excludes_zero"] and paired["model-glorys"]["diff"] == 0


# ----------------------------------------------------------------------------------------
# the runner (tiny config, memmap cache)
# ----------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def r2_env(tiny_pipeline: Config, tmp_path_factory):
    root = tmp_path_factory.mktemp("oe_r2_out")
    saved = os.environ.get("OCEANEMBED_OUTPUTS_ROOT")
    os.environ["OCEANEMBED_OUTPUTS_ROOT"] = str(root)
    raw = yaml.safe_load(TINY.read_text(encoding="utf-8"))
    raw["train"] = {**raw["train"], "cache": "memmap", "cache_dtype": "float16"}
    cfg_file = root / "tiny_memmap.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    try:
        yield load_config(cfg_file), cfg_file
    finally:
        if saved is None:
            os.environ.pop("OCEANEMBED_OUTPUTS_ROOT", None)
        else:
            os.environ["OCEANEMBED_OUTPUTS_ROOT"] = saved


@pytest.fixture(scope="module")
def r2_done(r2_env):
    cfg, _ = r2_env
    results = run_r2(cfg, [0, 1], device="cpu", progress=False)
    return cfg, results


def test_runner_writes_the_layout_and_the_cache(r2_done):
    cfg, results = r2_done
    assert {r["status"] for r in results} == {"done"} and len(results) == 2 + 2 * 2 + 1 + 2
    base = r2_dir(cfg)
    for name, files in {
        "scratch": ["recon.pt", "train.jsonl", "train.done.json"],
        "mlp": ["mlp.pt", "train.jsonl", "train.done.json"],
        "ridge": ["ridge.joblib", "train.done.json"],
        "climatology": [],
        "scratch_ty2": ["recon.pt"],
        "scratch_ty5": ["recon.pt"],
    }.items():
        seeds = (0, 1) if name in ("scratch", "mlp") else (0,)
        for s in seeds:
            d = base / name / f"seed{s}"
            for f in [*files, "eval.npz", "done.json"]:
                assert (d / f).exists(), (name, s, f)
    assert (base / "argo" / "summary.json").exists() and (
        base / "argo" / "matchups.parquet"
    ).exists()
    # the run's own memory-mapped cache was created for the train and validation splits
    assert (cache_dir(cfg, "train") / "meta.json").exists()
    assert (cache_dir(cfg, "val") / "meta.json").exists()
    # evaluation covers the whole test split of every job (here the tiny test split)
    z = r1.load_eval(base / "scratch" / "seed0" / "eval.npz")
    assert len(z["dates"]) == 10 and z["raw"].shape == (3, 8, 10, 15)
    info = json.loads((base / "scratch_ty2" / "seed0" / "done.json").read_text())
    assert info["train_years"] == 2 and info["train"]["n_train_days"] > 0
    assert (
        json.loads((base / "scratch" / "seed0" / "done.json").read_text())["train"]["n_train_days"]
        == 40
    )


def test_eval_sums_match_the_reference_scoring(r2_done):
    """The climatology job reproduces the reference sums of ``sums_from_arrays`` on the grid."""
    cfg, _ = r2_done
    from oceanembed.data.dataset import make_surface_dataset
    from oceanembed.data.harmonize import open_harmonized
    from oceanembed.eval.evaluate import read_target

    ds = make_surface_dataset(cfg, cfg.split.test.start, cfg.split.test.end)
    ref = read_target(open_harmonized(cfg), ds, 0, len(ds))
    clim = ds.stats.climatology(ds.dates().values)
    valid = ds.mask[None] & np.isfinite(ref)
    want = sums_from_arrays(clim, ref, valid, axis=(2, 3))
    got = r1.load_eval(r2_dir(cfg) / "climatology" / "seed0" / "eval.npz")["raw"][0]
    for i, f in enumerate(SUM_FIELDS):
        assert np.allclose(got[i], want[f], rtol=1e-4, atol=1e-3), f


def test_runner_skips_finished_jobs_and_resumes_interrupted_ones(r2_done):
    cfg, _ = r2_done
    again = run_r2(cfg, [0, 1], device="cpu", progress=False)
    assert {r["status"] for r in again} == {"skipped"}
    d = job_dir(cfg, Job("model", "scratch", 1))
    stamp = (d / "recon.pt").stat().st_mtime_ns
    (d / "done.json").unlink()
    (d / "eval.npz").unlink()
    res = run_r2(
        cfg, [1], ["scratch"], learning_curve=False, argo=False, device="cpu", progress=False
    )
    assert [r["status"] for r in res] == ["done"]
    assert (d / "recon.pt").stat().st_mtime_ns == stamp  # training kept, only re-scored
    assert (d / "done.json").exists() and (d / "eval.npz").exists()


def test_argo_summary_counts_and_floor(r2_done):
    cfg, _ = r2_done
    a = json.loads((r2_dir(cfg) / "argo" / "summary.json").read_text())
    assert a["methods"] == ["model", "ridge", "clim", "glorys"]
    y = a["years"]["pooled"]
    assert y["n_matchups"] > 0 and y["n_profiles"] > 0
    assert set(y["methods"]) == {"model", "ridge", "clim", "glorys"}
    assert (
        y["methods"]["glorys"]["pooled_50_200m"]["n"]
        == y["methods"]["model"]["pooled_50_200m"]["n"]
    )
    assert set(y["pooled_50_200m_bootstrap"]["paired"]) == {
        "model-ridge",
        "model-clim",
        "model-glorys",
    }
    assert "not an independent" in a["note"]


def test_argo_months_are_never_downloaded(r2_env):
    from oceanembed.research.r2 import argo_months_available

    cfg, _ = r2_env
    assert argo_months_available(cfg, "2030-01-01", "2030-02-01") == ["2030-01", "2030-02"]
    assert argo_months_available(cfg, cfg.split.test.start, cfg.split.test.end) == []


def test_report_files_numbers_and_figures(r2_done):
    cfg, _ = r2_done
    make_r2_report(cfg, cfg, n_boot=200, block_length=3, seed=1)
    out = r2_dir(cfg)
    assert (out / "summary.json").exists() and (out / "summary.md").exists()
    figs = {p.name for p in (out / "figures").glob("*.png")}
    assert {
        "rmse_by_depth.png",
        "years_bias_by_depth.png",
        "learning_curve.png",
        "argo_by_depth.png",
    } <= figs
    s = json.loads((out / "summary.json").read_text())
    assert set(s["settings"]["periods"]) == {"pooled"}  # the tiny test split is a single year
    assert s["settings"]["periods"]["pooled"]["n_days"] == 10
    sc = s["methods"]["scratch"]
    assert sc["n_seeds"] == 2 and sc["n_train_days"] == [40, 40]
    cell = sc["periods"]["pooled"]["regions"]["all"]["columns"][POOLED]["rmse"]
    z = [r1.load_eval(job_dir(cfg, Job("model", "scratch", sd)) / "eval.npz") for sd in (0, 1)]
    d = z[0]["depths"]
    sel = (d >= 50) & (d <= 200)

    def pooled(e):
        return float(
            np.sqrt(
                e["raw"][0, SUM_FIELDS.index("se2")][:, sel].sum()
                / e["raw"][0, SUM_FIELDS.index("n")][:, sel].sum()
            )
        )

    assert cell["seed_values"] == pytest.approx([pooled(e) for e in z], rel=1e-9)
    assert cell["ci_lo"] < cell["point"] < cell["ci_hi"]
    lc = s["learning_curve"]
    assert {"2", "5"} <= set(lc["points"]) and len(lc["points"]) == 3
    ref = s["transformer_vs_mlp"]["pooled"]["all"][POOLED]
    assert ref["established"] in (True, False) and "diff" in ref
    assert set(s["vs_climatology"]) == {"scratch", "mlp", "ridge"}
    text = (out / "summary.md").read_text(encoding="utf-8")
    assert "Learning curve" in text and "Argo profiles" in text


def test_cli_r2_commands(r2_done, r2_env):
    cfg, cfg_file = r2_done[0], r2_env[1]
    res = runner.invoke(
        app,
        ["research", "r2", "--config", str(cfg_file), "--seeds", "0,1", "--device", "cpu"],
    )
    assert res.exit_code == 0, res.output + str(res.exception)
    assert "0 job(s) run, 9 skipped" in res.output
    out = runner.invoke(
        app,
        [
            "research", "r2-report", "--config", str(cfg_file), "--n-boot", "100",
            "--block-length", "2", "--compare-config", str(cfg_file),
        ],
    )  # fmt: skip
    assert out.exit_code == 0, out.output + str(out.exception)
    s = json.loads((r2_dir(cfg) / "summary.json").read_text())
    assert s["settings"]["bootstrap"]["block_length_days"] == 2
    bad = runner.invoke(
        app, ["research", "r2", "--config", str(cfg_file), "--methods", "resnet", "--device", "cpu"]
    )
    assert bad.exit_code != 0


def test_report_needs_the_references(tiny_cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("OCEANEMBED_OUTPUTS_ROOT", str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError):
        load_results(tiny_cfg)
    res = runner.invoke(app, ["research", "r2-report", "--config", str(TINY)])
    assert res.exit_code == 2


# ----------------------------------------------------------------------------------------
# per-year logic and the comparison with the first run, on constructed two-year results
# ----------------------------------------------------------------------------------------
def _fake_run(rng, seeds, n_days, n_depth=15, bias=0.0, scale=1.0, start="2023-01-01"):
    """Per-day sums ``(S, R=3, F=8, T, D)`` of ``scale``-sized errors (plus ``bias``) over 50
    points per cell, with a seasonal reference."""
    t = np.arange(n_days)
    ref = (
        15
        + 5 * np.sin(2 * np.pi * t / 365)[None, None, :, None]
        + np.zeros((1, 3, n_days, n_depth))
    )
    out = []
    for _ in seeds:
        err = np.reshape(bias, (1, 1, -1, 1)) + scale * rng.normal(size=(50, 3, n_days, n_depth))
        pred = ref + err
        s = sums_from_arrays(pred, np.broadcast_to(ref, pred.shape), None, axis=0)
        out.append(np.stack([s[f] for f in SUM_FIELDS], axis=1))
    raw = np.stack(out)
    anom = raw.copy()
    dates = (np.datetime64(start) + np.arange(n_days)).astype("datetime64[D]")
    return {
        "seeds": list(seeds),
        "raw": raw,
        "anom": anom,
        "info": [{"train": {"n_train_days": 4018, "epoch": 3}} for _ in seeds],
        "dates": [str(d) for d in dates],
        "depths": np.array(
            [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000], float
        ),
    }


def test_year_comparison_poc_comparison_and_learning_curve_on_two_test_years():
    rng = np.random.default_rng(3)
    n = 120  # 2023-12-01 .. 2024-03-29: both years present
    start = "2023-12-01"
    runs = {
        "climatology": _fake_run(rng, [0], n, scale=2.0, start=start),
        "scratch": _fake_run(rng, [0, 1, 2], n, scale=1.0, start=start),
        "mlp": _fake_run(rng, [0, 1, 2], n, scale=1.3, start=start),
        "ridge": _fake_run(rng, [0], n, scale=1.6, start=start),
        "scratch_ty2": _fake_run(rng, [0], n, scale=1.8, start=start),
        "scratch_ty5": _fake_run(rng, [0], n, scale=1.2, start=start),
    }
    # the first run scored only 2024 (the same days as the long run's second year)
    d24 = np.flatnonzero(np.array(runs["climatology"]["dates"]) >= "2024-01-01")
    poc = {}
    for name, run in runs.items():
        if name.startswith("scratch_ty"):
            continue
        poc[name] = {
            **run,
            "raw": run["raw"][:, :, :, d24] * 1.0,
            "anom": run["anom"][:, :, :, d24] * 1.0,
            "dates": [run["dates"][i] for i in d24],
        }
    res = {"runs": runs, "poc": poc, "argo": None}
    s = build_summary(res, n_boot=200, block_length=7, seed=0)
    assert set(s["settings"]["periods"]) == {"2023", "2024", "pooled"}
    assert s["settings"]["periods"]["2023"]["n_days"] == 31
    assert s["settings"]["periods"]["2024"]["n_days"] == len(d24)
    # year comparison: independent resamples, so the difference interval is wider than either
    # year's own half-width; here both years are the same process so zero is inside
    yc = s["year_comparison"]["scratch"]["all"][POOLED]["rmse"]
    assert abs(yc["diff"]) < 0.2 and yc["ci_lo"] <= yc["diff"] <= yc["ci_hi"]
    assert set(s["year_comparison"]) == {"scratch", "mlp", "ridge", "climatology"}
    # long vs first run: the same days and the same points
    lvp = s["long_vs_poc"]
    assert lvp["sample"]["same_days"] and lvp["sample"]["same_points"]
    d = lvp["scratch"]["difference"]["all"][POOLED]["rmse"]
    assert d["diff"] == pytest.approx(0.0, abs=1e-12)  # identical sums: zero difference, paired
    assert d["ci_lo"] == pytest.approx(0.0, abs=1e-12) and d["ci_hi"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert "contrast_poc_2024" in lvp
    # learning curve: fewer years = larger error, on every period
    lc = s["learning_curve"]
    assert set(lc["points"]) == {"2", "5", "11"}
    r = {
        k: lc["points"][k]["periods"]["pooled"]["all"][POOLED]["rmse"]["point"]
        for k in lc["points"]
    }
    assert r["2"] > r["5"] > r["11"]
    assert lc["points"]["2"]["periods"]["pooled"]["all"][POOLED]["delta_vs_full"]["excludes_zero"]
    # the Transformer beats the MLP in every period here, and both beat the climatology
    for period in ("2023", "2024", "pooled"):
        c = s["transformer_vs_mlp"][period]["all"][POOLED]
        assert c["diff"] > 0 and c["excludes_zero"]
    assert s["vs_climatology"]["scratch"]["pooled"]["all"][POOLED]["diff"] < 0
    # a different set of days in the first run disables the comparison instead of misleading
    poc2 = {
        k: {**v, "dates": [str(np.datetime64(d) + 1) for d in v["dates"]]} for k, v in poc.items()
    }
    s2 = build_summary({"runs": runs, "poc": poc2, "argo": None}, n_boot=100, block_length=7)
    assert s2["long_vs_poc"]["sample"]["same_days"] is False and "scratch" not in s2["long_vs_poc"]


def test_year_bias_difference_detects_a_warm_year():
    rng = np.random.default_rng(5)
    n = 200
    days = np.datetime64("2023-09-01") + np.arange(n)
    bias = np.where(days >= np.datetime64("2024-01-01"), 0.8, 0.0)  # a warm second year
    warm = _fake_run(rng, [0, 1], n, scale=0.5, bias=bias, start="2023-09-01")
    clim = _fake_run(rng, [0], n, scale=1.0, start="2023-09-01")
    s = build_summary(
        {"runs": {"climatology": clim, "scratch": warm}, "poc": None, "argo": None},
        n_boot=200,
        block_length=7,
    )
    b = s["year_comparison"]["scratch"]["all"][POOLED]["bias"]
    assert b["diff"] == pytest.approx(0.8, abs=0.1) and b["excludes_zero"]
    c = s["year_comparison"]["climatology"]["all"][POOLED]["bias"]
    assert abs(c["diff"]) < 0.2
    assert not s["long_vs_poc"] and not s["learning_curve"]  # no first run, no short-period jobs
