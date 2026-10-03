"""Research stage R1: seeds, U-Net and MLP baselines, the resumable runner, the block bootstrap and
the report (tiny synthetic config, CPU)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from oceanembed.cli import app
from oceanembed.config import Config, ModelConfig, load_config
from oceanembed.data.dataset import make_dataset
from oceanembed.eval.metrics import SUM_FIELDS, point_metrics, sums_from_arrays
from oceanembed.infer.predict import load_recon_model, model_predictor, predict_batch
from oceanembed.models.baselines import FEATURE_CHANNELS
from oceanembed.models.encoder import TransformerBlock, unet_blocks
from oceanembed.models.pixel_mlp import PixelMLP, load_pixel_mlp
from oceanembed.models.recon import ReconModel
from oceanembed.research import bootstrap as bs
from oceanembed.research.r1 import (
    METHODS,
    job_dir,
    load_eval,
    plan_jobs,
    r1_dir,
    run_r1,
)
from oceanembed.research.r1_report import load_results, make_r1_report
from oceanembed.train.mlp import run_train_mlp
from oceanembed.train.pretrain import run_pretrain
from oceanembed.train.train import run_train

TINY = Path(__file__).resolve().parents[1] / "configs" / "test_tiny.yaml"
runner = CliRunner()


def _params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _same_weights(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# ----------------------------------------------------------------------------------------
# seeds
# ----------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_datasets(tiny_pipeline: Config):
    return make_dataset(tiny_pipeline, "train", preload=True), make_dataset(
        tiny_pipeline, "val", preload=True
    )


def test_training_seed_reproduces_and_differs(tiny_pipeline: Config, tiny_datasets, tmp_path):
    def train(seed, name):
        return run_train(
            tiny_pipeline,
            pretrained=False,
            device="cpu",
            progress=False,
            seed=seed,
            ckpt_path=tmp_path / f"{name}.pt",
            log_path=tmp_path / f"{name}.jsonl",
            datasets=tiny_datasets,
        )

    a, b, c = train(3, "a"), train(3, "b"), train(4, "c")
    wa, wb, wc = (torch.load(tmp_path / f"{n}.pt", weights_only=False)["model"] for n in "abc")
    assert a["seed"] == 3 and c["seed"] == 4
    assert _same_weights(wa, wb)  # CPU: the same seed is bit-identical
    assert a["val_rmse"] == b["val_rmse"]
    assert not _same_weights(wa, wc)


def test_pretrain_seed_reproduces_and_writes_only_where_told(
    tiny_pipeline: Config, tiny_datasets, tmp_path
):
    def pre(seed, name):
        return run_pretrain(
            tiny_pipeline,
            "cpu",
            False,
            seed=seed,
            ckpt_path=tmp_path / f"{name}.pt",
            log_path=tmp_path / f"{name}.jsonl",
            datasets=tiny_datasets,
        )

    pre(1, "a"), pre(1, "b"), pre(2, "c")
    w = {n: torch.load(tmp_path / f"{n}.pt", weights_only=False)["model"] for n in "abc"}
    assert _same_weights(w["a"], w["b"]) and not _same_weights(w["a"], w["c"])
    assert (tmp_path / "a.jsonl").exists()


def test_cli_seed_option_is_accepted(tiny_pipeline: Config, tmp_path, monkeypatch):
    monkeypatch.setenv("OCEANEMBED_OUTPUTS_ROOT", str(tmp_path / "out"))
    for args in (["pretrain", "--seed", "5"], ["train", "--no-pretrained", "--seed", "5"]):
        res = runner.invoke(app, [*args, "--config", str(TINY), "--device", "cpu"])
        assert res.exit_code == 0, res.output + str(res.exception)


# ----------------------------------------------------------------------------------------
# U-Net
# ----------------------------------------------------------------------------------------
def test_unet_has_no_attention_and_a_similar_parameter_budget():
    tr = ReconModel(ModelConfig())
    un = ReconModel(ModelConfig(arch="unet"))
    for model, expect in ((tr, True), (un, False)):
        has = any(isinstance(m, TransformerBlock) for m in model.modules())
        assert has is expect
    assert unet_blocks(ModelConfig()) == 4
    assert abs(_params(un) / _params(tr) - 1) < 0.05  # the same budget, no attention
    # the stem and the decoder are identical, only the bottleneck differs
    assert _params(un.decoder) == _params(tr.decoder)
    assert un.encoder.stem2.state_dict().keys() == tr.encoder.stem2.state_dict().keys()


def test_unet_forward_shape_and_training_smoke(tiny_pipeline: Config, tiny_datasets, tmp_path):
    mc = tiny_pipeline.model.model_copy(update={"arch": "unet"})
    out = ReconModel(mc)(torch.randn(2, 12, 16, 24))
    assert out.shape == (2, 15, 16, 24) and torch.isfinite(out).all()
    best = run_train(
        tiny_pipeline,
        pretrained=False,
        arch="unet",
        tag="unet",
        device="cpu",
        progress=False,
        seed=0,
        ckpt_path=tmp_path / "recon.pt",
        log_path=tmp_path / "train.jsonl",
        datasets=tiny_datasets,
    )
    assert np.isfinite(best["val_rmse"]) and best["n_params"] > 0
    model, cfg = load_recon_model(tmp_path / "recon.pt")
    assert cfg.model.arch == "unet" and not any(
        isinstance(m, TransformerBlock) for m in model.modules()
    )
    with pytest.raises(ValueError, match="transformer"):
        run_train(tiny_pipeline, pretrained=True, arch="unet", device="cpu", datasets=tiny_datasets)


# ----------------------------------------------------------------------------------------
# MLP
# ----------------------------------------------------------------------------------------
def test_pixel_mlp_shapes_and_per_pixel_independence():
    m = PixelMLP(hidden=8, layers=2)
    x = torch.randn(2, 12, 5, 6)
    y = m(x)
    assert y.shape == (2, 15, 5, 6)
    # one pixel's output depends on its own 11 features only (mask channel 7 is unused)
    x2 = x.clone()
    x2[:, 7] += 3.0
    x2[:, :, 0, 0] = torch.randn(2, 12)
    y2 = m(x2)
    assert torch.allclose(y[:, :, 1:, :], y2[:, :, 1:, :], atol=1e-6)
    assert len(FEATURE_CHANNELS) == 11


def test_mlp_training_smoke_and_predictor(tiny_pipeline: Config, tiny_datasets, tmp_path):
    cfg = tiny_pipeline.model_copy(
        update={
            "mlp": tiny_pipeline.mlp.model_copy(
                update={"hidden": 16, "epochs": 3, "batch_size": 256, "max_points": 5000}
            )
        }
    )
    best = run_train_mlp(
        cfg, 1, "cpu", False, ckpt_path=tmp_path / "mlp.pt", datasets=tiny_datasets
    )
    assert np.isfinite(best["val_rmse"]) and best["n_params"] == _params(
        load_pixel_mlp(tmp_path / "mlp.pt")[0]
    )
    again = run_train_mlp(
        cfg, 1, "cpu", False, ckpt_path=tmp_path / "mlp2.pt", datasets=tiny_datasets
    )
    assert again["val_rmse"] == best["val_rmse"]  # seeded: sample, init and batch order
    model, _ = load_pixel_mlp(tmp_path / "mlp.pt")
    test = make_dataset(tiny_pipeline, "test", preload=True)
    batch = {"x": test[0]["x"][None], "t": test[0]["t"].reshape(1)}
    temp = predict_batch(model_predictor(model, amp=False), batch, test, "cpu")
    assert temp.shape == (1, 15, *test.shape) and np.isnan(temp[0][~test.mask]).all()


# ----------------------------------------------------------------------------------------
# bootstrap
# ----------------------------------------------------------------------------------------
def _daily_from_points(xs, ys):
    """(F, T, 1) sums from lists of per-day point vectors."""
    per_day = [sums_from_arrays(x, y) for x, y in zip(xs, ys, strict=True)]
    return np.array([[float(d[f]) for d in per_day] for f in SUM_FIELDS])[:, :, None]


def test_block_counts_structure():
    c = bs.block_counts(10, 3, 50, seed=0)
    assert c.shape == (50, 10) and (c.sum(axis=1) == 10).all() and (c >= 0).all()
    # a block as long as the series is the series itself
    assert np.array_equal(bs.block_counts(10, 10, 5, seed=1), np.ones((5, 10)))
    # block length 1 = ordinary bootstrap; the same seed gives the same replicates
    assert np.array_equal(bs.block_counts(10, 1, 7, seed=2), bs.block_counts(10, 1, 7, seed=2))
    # consecutive days stay together: with blocks of 5 over 10 days, runs of 5 appear
    c5 = bs.block_counts(10, 5, 200, seed=3)
    assert (c5[:, 1:] - c5[:, :-1]).any()
    assert not np.array_equal(c5, bs.block_counts(10, 5, 200, seed=4))


def test_resampled_metrics_match_a_hand_computed_case():
    # 4 days, one depth; explicit replicates [day0, day2, day2, day3] and [day1 x4]
    xs = [np.array([1.0, 2.0]), np.array([0.0, 4.0]), np.array([3.0, 3.0]), np.array([5.0, 1.0])]
    ys = [np.array([1.0, 0.0]), np.array([1.0, 1.0]), np.array([0.0, 6.0]), np.array([2.0, 2.0])]
    raw = _daily_from_points(xs, ys)
    sel = np.ones((1, 1))
    w = np.array([[1.0, 0.0, 2.0, 1.0], [0.0, 4.0, 0.0, 0.0]])
    m = bs.metrics_from_daily(raw, raw, raw, sel, w)
    for row, picks in enumerate([[0, 2, 2, 3], [1, 1, 1, 1]]):
        x = np.concatenate([xs[i] for i in picks])
        y = np.concatenate([ys[i] for i in picks])
        ref = point_metrics(x, y)
        assert m["rmse"][row, 0] == pytest.approx(ref["rmse"])
        assert m["bias"][row, 0] == pytest.approx(ref["bias"])
        assert m["mae"][row, 0] == pytest.approx(ref["mae"])
        if np.isfinite(ref["corr"]):
            assert m["corr_raw"][row, 0] == pytest.approx(ref["corr"])
    # first replicate by hand: errors 0,2 | 3,-3 | 3,-3 | 3,-1 -> MSE (0+4+9+9+9+9+9+1)/8
    assert m["rmse"][0, 0] == pytest.approx(np.sqrt(50 / 8))
    # skill against itself is 0
    assert m["skill_vs_clim"][0, 0] == pytest.approx(0.0)


def test_columns_pool_depths():
    names, sel = bs.column_selector([0, 50, 100, 200, 500], (50.0, 200.0))
    assert names == ["0", "50", "100", "200", "500", "pooled_50_200m", "overall"]
    assert sel[5].tolist() == [0, 1, 1, 1, 0] and sel[6].sum() == 5
    daily = np.ones((len(SUM_FIELDS), 3, 5))
    cols = bs.column_sums(daily, np.ones((1, 3)), sel)
    assert cols["n"][0].tolist() == [3, 3, 3, 3, 3, 9, 15]


def test_paired_difference_uses_the_same_resamples():
    rng = np.random.default_rng(0)
    n_days = 60
    xs = [rng.normal(size=20) for _ in range(n_days)]
    ys = [rng.normal(size=20) for _ in range(n_days)]
    a = _daily_from_points(xs, ys)
    sel = np.ones((1, 1))
    counts = bs.block_counts(n_days, 6, 300, seed=0)
    ma = bs.metrics_from_daily(a, a, a, sel, counts)["rmse"]
    point = bs.metrics_from_daily(a, a, a, sel, np.ones((1, n_days)))["rmse"][0]
    # identical methods: the difference is exactly zero in every shared replicate
    same = bs.paired_difference(ma, ma, point, point)
    assert same["ci_lo"][0] == 0 and same["ci_hi"][0] == 0 and not same["excludes_zero"][0]
    assert same["p_two_sided"][0] == 1.0
    # a method that is better on every day by a fixed margin gives a difference that excludes 0
    # and a *tighter* interval than two independent bootstraps would (the common noise cancels)
    b = _daily_from_points([x + 0.0 for x in xs], [y + 0.5 for y in ys])
    mb = bs.metrics_from_daily(b, b, b, sel, counts)["rmse"]
    pb = bs.metrics_from_daily(b, b, b, sel, np.ones((1, n_days)))["rmse"][0]
    d = bs.paired_difference(ma, mb, point, pb)
    paired_width = d["ci_hi"][0] - d["ci_lo"][0]
    other = bs.metrics_from_daily(b, b, b, sel, bs.block_counts(n_days, 6, 300, seed=99))["rmse"]
    indep_width = np.diff(bs.percentile_ci(ma[:, 0] - other[:, 0])[0:2])[0]
    assert paired_width < indep_width
    with pytest.raises(ValueError):
        bs.paired_difference(ma, mb[:10], point, pb)


def test_bootstrap_standard_error_of_a_mean():
    # block length 1 on iid data reproduces the textbook standard error sigma / sqrt(n)
    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(size=(n, 400))
    daily = _daily_from_points(list(x), [np.zeros(400)] * n)  # rmse of x against 0
    counts = bs.block_counts(n, 1, 1500, seed=3)
    r = bs.metrics_from_daily(daily, daily, daily, np.ones((1, 1)), counts)["rmse"][:, 0]
    mse_days = (x**2).mean(axis=1)
    expected = np.sqrt(mse_days.var(ddof=0) / n) / (2 * np.sqrt(mse_days.mean()))
    assert r.std() == pytest.approx(expected, rel=0.15)


def test_decorrelation_time_of_known_series():
    rng = np.random.default_rng(0)
    noise = rng.normal(size=2000)
    ar = np.zeros(2000)
    for i in range(1, 2000):
        ar[i] = 0.9 * ar[i - 1] + noise[i]
    d_ar = bs.decorrelation_time(ar)
    assert 6 <= d_ar["efolding_days"] <= 14  # -1 / ln 0.9 = 9.5
    assert 12 <= d_ar["tau_int_days"] <= 30  # (1 + phi) / (1 - phi) = 19
    assert bs.decorrelation_time(noise)["efolding_days"] <= 2
    assert bs.auto_block_length([ar[:350], ar[350:700]]) <= 350 // 4
    st = bs.seed_stats(np.array([[1.0], [3.0]]))
    assert st["mean"][0] == 2.0 and st["sd"][0] == pytest.approx(np.sqrt(2)) and st["max"][0] == 3.0
    assert np.isnan(bs.seed_stats(np.array([[1.0]]))["sd"][0])


# ----------------------------------------------------------------------------------------
# runner, resumability, isolation from the main run
# ----------------------------------------------------------------------------------------
def _fingerprint(root: Path, names=("checkpoints", "metrics", "predictions", "embeddings")):
    """(size, mtime, sha1) of every file the research stage must never touch."""
    out = {}
    files = [p for n in names for p in sorted((root / n).rglob("*")) if p.is_file()]
    files += [root / "report.md", root / "run_meta.json"]
    for p in files:
        if p.exists():
            out[str(p.relative_to(root))] = (
                p.stat().st_size,
                p.stat().st_mtime_ns,
                hashlib.sha1(p.read_bytes()).hexdigest(),
            )
    return out


@pytest.fixture(scope="module")
def r1_env(_tiny_run_roots, tmp_path_factory):
    """A copy of the finished tiny run in its own outputs root, with the data root of the run."""
    data, out, _ = _tiny_run_roots
    root = tmp_path_factory.mktemp("oe_r1_out")
    shutil.copytree(out / "test_tiny", root / "test_tiny")
    keys = ("OCEANEMBED_DATA_ROOT", "OCEANEMBED_OUTPUTS_ROOT")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ["OCEANEMBED_DATA_ROOT"] = str(data)
    os.environ["OCEANEMBED_OUTPUTS_ROOT"] = str(root)
    try:
        yield load_config(TINY)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="module")
def r1_done(r1_env: Config):
    """Seeds 0 and 1 of every method (the MLP for seed 0 only), run once for the module."""
    before = _fingerprint(r1_env.outputs_dir)
    results = run_r1(r1_env, [0, 1], mlp_seeds=1, device="cpu", progress=False)
    return r1_env, before, results


def test_plan_orders_seed_by_seed_and_limits_the_mlp():
    jobs = plan_jobs([0, 1, 2, 3, 4], None, 3)
    assert jobs[:2] == [("ridge", 0), ("climatology", 0)]
    assert jobs[2:6] == [("oceanembed", 0), ("scratch", 0), ("unet", 0), ("mlp", 0)]
    assert [s for m, s in jobs if m == "mlp"] == [0, 1, 2]
    assert sum(m == "oceanembed" for m, _ in jobs) == 5 and len(jobs) == 2 + 5 * 3 + 3
    assert plan_jobs([7], ["scratch"], 3) == [("scratch", 7)]
    with pytest.raises(ValueError):
        plan_jobs([0], ["resnet"])
    assert set(METHODS) == {"oceanembed", "scratch", "unet", "mlp", "ridge", "climatology"}


def test_runner_writes_files_and_leaves_the_main_run_untouched(r1_done):
    cfg, before, results = r1_done
    assert {r["status"] for r in results} == {"done"} and len(results) == 2 + 2 * 3 + 1
    base = r1_dir(cfg)
    assert base == cfg.outputs_dir / "research" / "r1"
    for method, seed in [("oceanembed", 0), ("oceanembed", 1), ("scratch", 1), ("unet", 0)]:
        d = job_dir(cfg, method, seed)
        assert (d / "done.json").exists() and (d / "eval.npz").exists()
        assert (d / "recon.pt").exists() and (d / "train.jsonl").exists()
    assert (job_dir(cfg, "oceanembed", 0) / "pretrain.pt").exists()
    assert not (job_dir(cfg, "scratch", 0) / "pretrain.pt").exists()
    assert (job_dir(cfg, "mlp", 0) / "mlp.pt").exists() and not job_dir(cfg, "mlp", 1).exists()
    assert job_dir(cfg, "ridge", 0).joinpath("eval.npz").exists()
    # seeds differ, a seed's checkpoint is the one it trained
    w0 = torch.load(job_dir(cfg, "scratch", 0) / "recon.pt", weights_only=False)
    w1 = torch.load(job_dir(cfg, "scratch", 1) / "recon.pt", weights_only=False)
    assert not _same_weights(w0["model"], w1["model"])
    info = json.loads((job_dir(cfg, "unet", 0) / "done.json").read_text())
    assert info["train"]["seed"] == 0 and info["train"]["n_params"] > 0
    pre = json.loads((job_dir(cfg, "oceanembed", 1) / "done.json").read_text())["pretrain"]
    assert pre["seed"] == 1
    # compact evaluation file: sums for the domain and both basins, raw and anomaly
    e = load_eval(job_dir(cfg, "scratch", 0) / "eval.npz")
    assert e["raw"].shape == (3, len(SUM_FIELDS), 10, 15) and e["raw"].shape == e["anom"].shape
    assert list(e["regions"]) == ["all", "arabian_sea", "bay_of_bengal"]
    assert (e["raw"][0, SUM_FIELDS.index("n")] >= e["raw"][1, SUM_FIELDS.index("n")]).all()
    # nothing of the main run changed
    assert _fingerprint(cfg.outputs_dir) == before
    assert not any((cfg.outputs_dir / "predictions").rglob("*unet*"))


def test_runner_skips_finished_jobs_and_resumes_interrupted_ones(r1_done):
    cfg, _, _ = r1_done
    d = job_dir(cfg, "oceanembed", 1)
    stamp = {p.name: p.stat().st_mtime_ns for p in d.iterdir()}
    again = run_r1(cfg, [0, 1], mlp_seeds=1, device="cpu", progress=False)
    assert {r["status"] for r in again} == {"skipped"}
    assert {p.name: p.stat().st_mtime_ns for p in d.iterdir()} == stamp
    # an interrupted job: pretraining finished, training died half-way (partial files, no marker)
    (d / "done.json").unlink()
    (d / "eval.npz").unlink()
    (d / "train.done.json").unlink()
    (d / "recon.pt").write_bytes(b"truncated")
    (d / "train.jsonl").write_text("{}\n")
    pre_stamp = (d / "pretrain.done.json").stat().st_mtime_ns
    res = run_r1(cfg, [1], ["oceanembed"], device="cpu", progress=False)
    assert [r["status"] for r in res] == ["done"]
    assert (d / "pretrain.done.json").stat().st_mtime_ns == pre_stamp  # not repeated
    assert (d / "done.json").exists() and (d / "eval.npz").exists()
    load_recon_model(d / "recon.pt")  # a complete, loadable checkpoint
    # --no-skip-existing retrains everything of that job
    res = run_r1(cfg, [1], ["scratch"], skip_existing=False, device="cpu", progress=False)
    assert [r["status"] for r in res] == ["done"]


# ----------------------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------------------
def test_report_matches_the_main_evaluation_and_writes_everything(r1_done):
    cfg, _, _ = r1_done
    summary = make_r1_report(cfg, n_boot=300, seed=0)
    out = r1_dir(cfg)
    for name in ("summary.json", "summary.md"):
        assert (out / name).exists()
    for fig in ("rmse_by_depth", "paired_differences", "skill_by_depth"):
        assert (out / "figures" / f"{fig}.png").stat().st_size > 5000
    assert json.loads((out / "summary.json").read_text())["settings"]["n_days"] == 10
    st = summary["settings"]
    assert st["n_seeds"] == {
        "oceanembed": 2, "scratch": 2, "unet": 2, "mlp": 1, "ridge": 1, "climatology": 1
    }  # fmt: skip
    pn = "pooled_50_200m"
    # the deterministic methods reproduce the main run's metrics exactly (same sample, same maths)
    main = json.loads((cfg.outputs_dir / "metrics" / "metrics_glorys.json").read_text())
    for m, key in (("ridge", "ridge"), ("climatology", "climatology")):
        mine = summary["methods"][m]["regions"]["all"]["columns"][pn]
        ref = main["methods"][key]["pooled_50_200m"]
        assert mine["rmse"]["point"] == pytest.approx(ref["rmse"], rel=1e-6)
        assert mine["bias"]["point"] == pytest.approx(ref["bias"], abs=1e-6)
        if ref["corr_anom"] is not None and m != "climatology":
            assert mine["corr_anom"]["point"] == pytest.approx(ref["corr_anom"], rel=1e-5)
    # structure: seed spread for multi-seed methods, none for deterministic ones
    oe = summary["methods"]["oceanembed"]["regions"]["all"]["columns"][pn]["rmse"]
    assert len(oe["seed_values"]) == 2 and oe["seed_sd"] > 0
    assert oe["seed_mean"] == pytest.approx(np.mean(oe["seed_values"]))
    assert oe["ci_lo"] <= oe["point"] <= oe["ci_hi"]
    on_disk = json.loads((out / "summary.json").read_text())
    assert on_disk["methods"]["ridge"]["regions"]["all"]["columns"][pn]["rmse"]["seed_sd"] is None
    assert summary["methods"]["unet"]["n_params"] and summary["methods"]["mlp"]["n_params"]
    # sqrt of the seed-mean MSE lies between the seeds' own RMSEs
    assert oe["seed_min"] - 1e-9 <= oe["point"] <= oe["seed_max"] + 1e-9
    pairs = {(c["a"], c["b"]) for c in summary["comparisons"]}
    assert ("oceanembed", "scratch") in pairs and ("scratch", "unet") in pairs
    comp = next(
        c for c in summary["comparisons"] if (c["a"], c["b"]) == ("oceanembed", "climatology")
    )
    cell = comp["regions"]["all"]["columns"][pn]["rmse"]
    assert cell["diff"] == pytest.approx(oe["point"] - summary["methods"]["climatology"][
        "regions"]["all"]["columns"][pn]["rmse"]["point"])  # fmt: skip
    text = (out / "summary.md").read_text(encoding="utf-8")
    for needle in ("Block length", "Does pretraining matter?", "Paired comparisons", "Provenance"):
        assert needle in text
    assert "moving-block" in text.lower() and "n=2" in text


def test_cli_research_commands(r1_done):
    cfg, _, _ = r1_done
    out = runner.invoke(
        app,
        [
            "research",
            "r1",
            "--config",
            str(TINY),
            "--methods",
            "ridge,climatology",
            "--device",
            "cpu",
        ],
    )
    assert out.exit_code == 0, out.output + str(out.exception)
    assert "0 job(s) run, 2 skipped" in out.output
    out = runner.invoke(
        app,
        ["research", "r1-report", "--config", str(TINY), "--n-boot", "100", "--block-length", "3"],
    )
    assert out.exit_code == 0, out.output + str(out.exception)
    s = json.loads((r1_dir(cfg) / "summary.json").read_text())
    assert s["settings"]["bootstrap"]["block_length_days"] == 3
    bad = runner.invoke(
        app, ["research", "r1", "--config", str(TINY), "--methods", "resnet", "--device", "cpu"]
    )
    assert bad.exit_code != 0


def test_report_needs_the_climatology(tiny_cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("OCEANEMBED_OUTPUTS_ROOT", str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError):
        load_results(tiny_cfg)
    res = runner.invoke(app, ["research", "r1-report", "--config", str(TINY)])
    assert res.exit_code == 2
