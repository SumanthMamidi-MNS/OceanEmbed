"""Research stage R3: input groups, masking, history, permutation, the runner and the report
(tiny synthetic config, CPU)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from typer.testing import CliRunner

from oceanembed.cli import app
from oceanembed.config import Config, load_config
from oceanembed.data.dataset import make_dataset, make_surface_dataset
from oceanembed.eval.metrics import SUM_FIELDS, sums_from_arrays
from oceanembed.infer.predict import load_recon_model, model_predictor
from oceanembed.models.baselines import ClimatologyBaseline, sample_points
from oceanembed.models.pixel_mlp import PixelMLP, load_pixel_mlp
from oceanembed.research import r1
from oceanembed.research.inputs import (
    EXPERIMENTS,
    InputSpec,
    InputView,
    load_test_inputs,
    make_test_batch_fn,
    make_views,
    month_permutation,
    permute_batch,
    sample_points_view,
    zero_channels,
)
from oceanembed.research.r3 import (
    Job,
    job_dir,
    model_cfg,
    plan_jobs,
    r3_dir,
    region_matrix,
    region_sums,
    run_r3,
    score_passes,
)
from oceanembed.research.r3_report import POOLED, load_results, make_r3_report
from oceanembed.train.mlp import run_train_mlp
from oceanembed.train.train import run_train

TINY = Path(__file__).resolve().parents[1] / "configs" / "test_tiny.yaml"
runner = CliRunner()


def _same_weights(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


@pytest.fixture(scope="module")
def tiny_datasets(tiny_pipeline: Config):
    return make_dataset(tiny_pipeline, "train", preload=True), make_dataset(
        tiny_pipeline, "val", preload=True
    )


# ----------------------------------------------------------------------------------------
# groups and masking
# ----------------------------------------------------------------------------------------
def test_input_spec_channels_and_experiments():
    s = InputSpec(keep=("sst", "winds"), history=3)
    assert s.n_channels == 26 and s.offset == 2
    # dropped: sss (1), sla (2), currents (3, 4) -- in the current day and in lags 1 and 2
    assert s.zero_channels == sorted([1, 2, 3, 4, 13, 14, 15, 16, 20, 21, 22, 23])
    assert len(s.feature_channels) == 11 + 14 and s.feature_channels[:3] == [0, 1, 2]
    assert InputSpec().is_identity and InputSpec().zero_channels == []
    with pytest.raises(ValueError):
        InputSpec(keep=("sst", "salinity"))
    with pytest.raises(ValueError):
        InputSpec(history=0)
    # the eleven experiments: five removals, three reduced sets, two histories, the reference
    assert len(EXPERIMENTS) == 11
    assert EXPERIMENTS["no_currents"].zero_channels == [3, 4]
    assert EXPERIMENTS["no_winds"].zero_channels == [5, 6]
    assert EXPERIMENTS["sst_only"].zero_channels == [1, 2, 3, 4, 5, 6]
    assert EXPERIMENTS["sst_sla_winds"].dropped == ("sss", "currents")
    assert EXPERIMENTS["hist7"].n_channels == 12 + 7 * 6


def test_zero_channels_copies_and_zeroes_only_those_channels():
    x = torch.ones(2, 12, 4, 4)
    out = zero_channels(x, [3, 4])
    assert (out[:, 3:5] == 0).all() and (out[:, :3] == 1).all() and (out[:, 5:] == 1).all()
    assert (x == 1).all()  # the input is untouched
    assert zero_channels(x, []) is x


def test_identity_view_equals_the_dataset_and_masks_zero_only_their_group(tiny_datasets):
    train, _ = tiny_datasets
    ident = InputView(train, InputSpec())
    assert len(ident) == len(train)
    for i in (0, 5, len(train) - 1):
        a, b = ident[i], train[i]
        assert all(torch.equal(a[k], b[k]) for k in b)
    view = InputView(train, EXPERIMENTS["no_currents"])
    for i in (0, 7):
        x, ref = view[i]["x"], train[i]["x"]
        assert (x[3:5] == 0).all()
        keep = [c for c in range(12) if c not in (3, 4)]
        assert torch.equal(x[keep], ref[keep])
        assert torch.equal(view[i]["y"], train[i]["y"])  # the target is never touched
    assert ref[3:5].abs().sum() > 0  # the currents did carry a signal


def test_history_view_stacks_the_previous_days_from_the_same_arrays(
    tiny_pipeline: Config, tiny_datasets
):
    train, _ = tiny_datasets
    view = InputView(train, InputSpec(history=3))
    assert len(view) == len(train) - 2
    assert view.dates()[0] == train.dates()[2]
    surf = train.arrays()["surf"]
    for i in (0, 3, len(view) - 1):
        x = view[i]["x"]
        assert x.shape[0] == 26
        assert torch.equal(x[:12], train[i + 2]["x"])
        assert np.array_equal(x[12:19].numpy(), surf[i + 1])  # day t-1
        assert np.array_equal(x[19:26].numpy(), surf[i])  # day t-2
        assert int(view[i]["index"]) == i
    with pytest.raises(ValueError, match="preloaded"):
        InputView(make_dataset(tiny_pipeline, "train"), InputSpec(history=3))


def test_streamed_history_equals_the_training_view(tiny_pipeline: Config, tiny_datasets):
    """The test-time builder and the training view produce the same input for the same day."""
    cfg = tiny_pipeline
    _, val = tiny_datasets
    spec = InputSpec(keep=("sst", "sla"), history=3)
    view = InputView(val, spec)
    ds = make_surface_dataset(cfg, cfg.split.val.start, cfg.split.val.end)
    fn = make_test_batch_fn(cfg, ds, spec)
    batch = fn(2, 8)  # val days 2..7 = view items 0..5
    expected = torch.stack([view[i]["x"] for i in range(6)])
    assert torch.equal(batch["x"], expected)
    assert batch["x"].shape == (6, 26, *ds.shape)
    # the first val days take their history from the days before the split (inputs only)
    train, _ = tiny_datasets
    first = fn(0, 2)
    last = train.arrays()["surf"][-1]  # SST and sea level are channels 0 and 2 of a lag block
    assert np.array_equal(first["x"][0, [12, 14]].numpy(), last[[0, 2]])  # lag 1 of val day 0
    assert np.array_equal(first["x"][1, [19, 21]].numpy(), last[[0, 2]])  # lag 2 of val day 1
    # sst & sla are kept, everything else zeroed in every lag
    for ch in spec.zero_channels:
        assert (first["x"][:, ch] == 0).all()
    # a split with no days before it in the store cannot have a history
    early = make_surface_dataset(cfg, cfg.time.start, "2022-01-10")
    with pytest.raises(ValueError, match="before"):
        make_test_batch_fn(cfg, early, InputSpec(history=3))
    assert make_test_batch_fn(cfg, early, InputSpec())(0, 2)["x"].shape[1] == 12


def test_every_history_length_is_scored_on_the_identical_test_days(tiny_pipeline: Config):
    cfg = tiny_pipeline
    ctx = r1._Context(cfg, "cpu")
    ds, _ = ctx.test()
    specs = [EXPERIMENTS[k] for k in ("full", "hist3", "hist7", "no_sst")]
    fns = [make_test_batch_fn(cfg, ds, s) for s in specs]
    raw, anom = score_passes(ctx, ClimatologyBaseline(15), fns, batch_size=4, progress=False)
    assert raw.shape == (4, 3, 8, len(ds), 15) and anom.shape == raw.shape
    n = SUM_FIELDS.index("n")
    assert (raw[:, :, n] == raw[0:1, :, n]).all() and raw[0, 0, n].sum() > 0
    # the climatology predictor ignores x, so every pass must give the same sums
    assert np.allclose(raw, raw[0:1]) and np.allclose(anom, anom[0:1])
    for fn, s in zip(fns, specs, strict=True):
        assert fn(0, len(ds))["x"].shape[0] == len(ds)  # every test day, also the first ones
        assert fn(0, 3)["x"].shape[1] == s.n_channels


# ----------------------------------------------------------------------------------------
# scoring helpers
# ----------------------------------------------------------------------------------------
def test_region_sums_match_the_per_region_reference_sums(tiny_pipeline: Config):
    ctx = r1._Context(tiny_pipeline, "cpu")
    rng = np.random.default_rng(0)
    b, d, h, w = 3, 5, ctx.grid.lat.size, ctx.grid.lon.size
    x = rng.normal(size=(b, d, h, w)).astype(np.float32)
    y = rng.normal(size=(b, d, h, w)).astype(np.float32)
    x[0, 0, :2] = np.nan
    valid = rng.random((b, d, h, w)) > 0.3
    got = region_sums(x, y, valid, region_matrix(ctx))
    basins = ctx.grid.basin_masks()
    for ri, region in enumerate([None, *(basins[k] for k in r1.REGIONS[1:])]):
        v = valid if region is None else valid & region[None, None]
        ref = sums_from_arrays(x, y, v, axis=(2, 3))
        for fi, f in enumerate(SUM_FIELDS):
            assert np.allclose(got[ri, fi], ref[f], rtol=1e-10, atol=1e-9), f


def test_sample_points_view_identity_masks_and_history(tiny_datasets):
    train, _ = tiny_datasets
    f0, y0, v0 = sample_points(train, 500, 3)
    f1, y1, v1 = sample_points_view(InputView(train, InputSpec()), 500, 3)
    assert np.array_equal(f0, f1) and np.array_equal(y0, y1) and np.array_equal(v0, v1)
    # a removed group is zero in its feature columns and nothing else changes
    spec = EXPERIMENTS["no_winds"]
    f2, y2, _ = sample_points_view(InputView(train, spec), 500, 3)
    assert (f2[:, [5, 6]] == 0).all() and np.array_equal(
        np.delete(f2, [5, 6], 1), np.delete(f0, [5, 6], 1)
    )
    assert np.array_equal(y2, y0)
    # history: every sampled row is a row of the view's own inputs (per-pixel features)
    spec = InputSpec(keep=("sst", "sla", "currents"), history=3)
    view = InputView(train, spec)
    feats, _, _ = sample_points_view(view, 300, 5)
    assert feats.shape[1] == len(spec.feature_channels) == 25
    ocean = np.flatnonzero(view.mask[0].reshape(-1))
    rows = set()
    for i in range(len(view)):
        x = view[i]["x"][spec.feature_channels].numpy().reshape(25, -1)[:, ocean].T
        rows.update(r.tobytes() for r in np.ascontiguousarray(x, dtype=np.float32))
    assert all(r.tobytes() in rows for r in feats)


# ----------------------------------------------------------------------------------------
# permutation
# ----------------------------------------------------------------------------------------
def test_month_permutation_is_a_derangement_within_each_month():
    dates = pd.date_range("2022-02-20", "2022-03-01")  # nine February days, one March day
    for seed in range(5):
        donor = month_permutation(dates, np.random.default_rng(seed))
        feb = np.arange(9)
        assert (donor[feb] != feb).all()  # nobody keeps their own day
        assert sorted(donor[feb]) == list(feb)  # a permutation of the month's days
        assert donor[9] == 9  # a month with one day maps to itself
    a = month_permutation(dates, np.random.default_rng(1))
    b = month_permutation(dates, np.random.default_rng(1))
    assert np.array_equal(a, b)
    long = pd.date_range("2022-01-01", "2023-02-14")
    d = month_permutation(long, np.random.default_rng(0))
    assert (long.year * 100 + long.month == (long.year * 100 + long.month)[d]).all()
    assert (d != np.arange(len(long))).all()


def test_permute_batch_moves_a_group_together_and_nothing_else():
    g = torch.Generator().manual_seed(0)
    x_all = torch.randn(6, 12, 4, 5, generator=g)
    keep = x_all.clone()
    donor = np.array([1, 0, 3, 2, 5, 4])
    out = permute_batch(x_all, donor, [3, 4], 2, 6)
    assert torch.equal(out[:, 3], keep[donor[2:6]][:, 3])
    assert torch.equal(out[:, 4], keep[donor[2:6]][:, 4])  # u and v from the same donor day
    others = [c for c in range(12) if c not in (3, 4)]
    assert torch.equal(out[:, others], keep[2:6][:, others])
    assert torch.equal(x_all, keep)  # the stored inputs are never modified


# ----------------------------------------------------------------------------------------
# training through the views
# ----------------------------------------------------------------------------------------
def test_identity_view_trains_exactly_like_the_dataset(
    tiny_pipeline: Config, tiny_datasets, tmp_path
):
    """The code path of the retraining jobs is identical to R1's, so the R1 seeds are the
    full-input reference (on CPU the same seed reproduces a run bit for bit)."""
    train, val = tiny_datasets
    views = make_views(train, val, EXPERIMENTS["full"])

    def fit(datasets, name):
        run_train(
            tiny_pipeline, pretrained=False, device="cpu", progress=False, seed=2,
            ckpt_path=tmp_path / f"{name}.pt", log_path=tmp_path / f"{name}.jsonl",
            datasets=datasets,
        )  # fmt: skip
        return torch.load(tmp_path / f"{name}.pt", weights_only=False)["model"]

    assert _same_weights(fit((train, val), "a"), fit(views, "b"))

    def fit_mlp(datasets, name, **kw):
        run_train_mlp(
            tiny_pipeline,
            2,
            "cpu",
            False,
            ckpt_path=tmp_path / f"{name}.pt",
            datasets=datasets,
            **kw,
        )
        return torch.load(tmp_path / f"{name}.pt", weights_only=False)["model"]

    assert _same_weights(
        fit_mlp((train, val), "m1"), fit_mlp(views, "m2", sample_fn=sample_points_view)
    )


def test_history_and_masked_models_train_save_and_predict(
    tiny_pipeline: Config, tiny_datasets, tmp_path
):
    cfg = tiny_pipeline
    train, val = tiny_datasets
    spec = InputSpec(keep=("sst", "sla"), history=3)
    views = make_views(train, val, spec)
    mcfg = model_cfg(cfg, spec)
    assert mcfg.model.in_channels == 26 and cfg.model.in_channels == 12
    best = run_train(
        mcfg, pretrained=False, device="cpu", progress=False, seed=0,
        ckpt_path=tmp_path / "recon.pt", log_path=tmp_path / "t.jsonl", datasets=views,
    )  # fmt: skip
    assert np.isfinite(best["val_rmse"])
    model, loaded = load_recon_model(tmp_path / "recon.pt")
    assert loaded.model.in_channels == 26
    ds = make_surface_dataset(cfg, cfg.split.test.start, cfg.split.test.end)
    batch = make_test_batch_fn(cfg, ds, spec)(0, 3)
    out = model_predictor(model, amp=False)(batch["x"])
    assert out.shape == (3, 15, *ds.shape) and torch.isfinite(out).all()
    # per-pixel MLP with the history as extra features
    run_train_mlp(
        mcfg, 0, "cpu", False, ckpt_path=tmp_path / "mlp.pt", datasets=views,
        sample_fn=sample_points_view, model_kwargs={"channels": spec.feature_channels},
    )  # fmt: skip
    mlp, meta = load_pixel_mlp(tmp_path / "mlp.pt")
    assert mlp.channels == spec.feature_channels and mlp.spec["n_features"] == 25
    y = mlp(batch["x"])
    assert y.shape == (3, 15, *ds.shape)
    # the zeroed channels carry nothing: changing them cannot change the output
    other = batch["x"].clone()
    other[:, [1, 3, 4]] = 9.0
    assert not torch.equal(batch["x"], other)
    assert torch.allclose(mlp(zero_channels(other, spec.zero_channels)), y)
    # a default MLP is unchanged by the generalisation
    assert (
        PixelMLP().channels == [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11]
        and "channels" not in PixelMLP().spec
    )


# ----------------------------------------------------------------------------------------
# runner and report
# ----------------------------------------------------------------------------------------
def test_plan_runs_round_by_round_with_the_cheap_model_next_to_the_expensive_one():
    jobs = plan_jobs([0, 1], [0, 1, 2])
    assert jobs[0] == Job("train", "scratch", "no_sst", 0) and jobs[1] == Job(
        "train", "mlp", "no_sst", 0
    )
    first_round = [j for j in jobs if j.seed == 0]
    assert jobs[: len(first_round)] == first_round  # all of seed 0 before any second seed
    assert [j.kind for j in first_round[-2:]] == ["perm", "perm"]
    assert sum(j.kind == "train" and j.model == "scratch" for j in jobs) == 20
    assert sum(j.kind == "train" and j.model == "mlp" for j in jobs) == 30
    assert sum(j.kind == "perm" for j in jobs) == 5
    assert all(j.seed != 2 or j.model == "mlp" for j in jobs)
    assert plan_jobs([0], [0], ["mlp"], ["hist3"], permutation=False) == [
        Job("train", "mlp", "hist3", 0)
    ]
    assert Job("perm", "scratch", "permutation", 1).name == "perm_scratch"
    with pytest.raises(ValueError):
        plan_jobs([0], [0], ["resnet"])
    with pytest.raises(ValueError):
        plan_jobs([0], [0], None, ["no_salt"])


def _fingerprint(root: Path) -> dict[str, str]:
    out = {}
    for sub in ("checkpoints", "metrics", "predictions", "embeddings"):
        for p in sorted((root / sub).rglob("*")):
            if p.is_file():
                out[str(p)] = hashlib.md5(p.read_bytes()).hexdigest()
    rep = root / "report.md"
    if rep.exists():
        out[str(rep)] = hashlib.md5(rep.read_bytes()).hexdigest()
    return out


EXPS = ["no_sst", "no_sss", "no_winds", "hist3", "sst_only"]


@pytest.fixture(scope="module")
def r3_env(_tiny_run_roots, tmp_path_factory):
    data, out, _ = _tiny_run_roots
    root = tmp_path_factory.mktemp("oe_r3_out")
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
def r3_done(r3_env: Config):
    """R1 references (Transformer and MLP, seeds 0 and 1, climatology), then the R3 jobs."""
    cfg = r3_env
    r1.run_r1(
        cfg, [0, 1], ["scratch", "mlp", "climatology"], mlp_seeds=2, device="cpu", progress=False
    )
    before = _fingerprint(cfg.outputs_dir)
    r1_before = {
        str(p): hashlib.md5(p.read_bytes()).hexdigest()
        for p in sorted(r1.r1_dir(cfg).rglob("*"))
        if p.is_file()
    }
    results = run_r3(
        cfg, [0, 1], [0, 1], experiments=EXPS, perm_repeats=2, device="cpu", progress=False
    )
    return cfg, before, r1_before, results


def test_runner_writes_the_layout_and_leaves_the_main_run_and_r1_untouched(r3_done):
    cfg, before, r1_before, results = r3_done
    assert {r["status"] for r in results} == {"done"} and len(results) == 5 * 4 + 4
    for model, ckpt in (("scratch", "recon.pt"), ("mlp", "mlp.pt")):
        for exp in EXPS:
            for s in (0, 1):
                d = r3_dir(cfg) / f"{model}_{exp}" / f"seed{s}"
                for f in (ckpt, "train.jsonl", "train.done.json", "eval.npz", "done.json"):
                    assert (d / f).exists(), (d, f)
    for model in ("scratch", "mlp"):
        d = r3_dir(cfg) / f"perm_{model}" / "seed0"
        assert (d / "perm.npz").exists() and (d / "done.json").exists()
        z = np.load(d / "perm.npz")
        assert list(z["passes"]) == ["none", "sst", "sss", "sla", "currents", "winds"]
        assert z["raw"].shape[:2] == (6, 3) and int(z["repeats"]) == 2
    assert _fingerprint(cfg.outputs_dir) == before  # nothing of the main run changed
    after = {
        str(p): hashlib.md5(p.read_bytes()).hexdigest()
        for p in sorted(r1.r1_dir(cfg).rglob("*"))
        if p.is_file()
    }
    assert after == r1_before  # R1 is only read


def test_history_jobs_have_wider_inputs_and_the_same_test_days(r3_done):
    cfg, *_ = r3_done
    base = r1.load_eval(r1.job_dir(cfg, "scratch", 0) / "eval.npz")
    h3 = r1.load_eval(job_dir(cfg, Job("train", "scratch", "hist3", 0)) / "eval.npz")
    assert list(h3["dates"]) == list(base["dates"]) and len(h3["dates"]) == 10
    n = SUM_FIELDS.index("n")
    assert np.array_equal(h3["raw"][:, n], base["raw"][:, n])
    _, loaded = load_recon_model(job_dir(cfg, Job("train", "scratch", "hist3", 0)) / "recon.pt")
    assert loaded.model.in_channels == 12 + 14
    info = json.loads((job_dir(cfg, Job("train", "scratch", "hist3", 0)) / "done.json").read_text())
    assert info["history"] == 3 and info["n_input_channels"] == 26
    assert info["train"]["n_train_days"] == 40 - 2  # the first two days have no full history
    _, loaded = load_recon_model(job_dir(cfg, Job("train", "scratch", "no_sst", 0)) / "recon.pt")
    assert loaded.model.in_channels == 12


def test_unpermuted_pass_and_fast_sums_reproduce_the_r1_evaluation(r3_done):
    cfg, *_ = r3_done
    z = np.load(r3_dir(cfg) / "perm_scratch" / "seed0" / "perm.npz")
    ref = r1.load_eval(r1.job_dir(cfg, "scratch", 0) / "eval.npz")
    assert np.allclose(z["raw"][0], ref["raw"], rtol=1e-6, atol=1e-6)
    assert np.allclose(z["anom"][0], ref["anom"], rtol=1e-6, atol=1e-6)
    n = SUM_FIELDS.index("n")
    assert np.array_equal(z["raw"][1][:, n], ref["raw"][:, n])  # the same points are scored
    # permuting an input changes the prediction (SST certainly carries signal)
    se2 = SUM_FIELDS.index("se2")
    assert not np.allclose(z["raw"][1][:, se2], z["raw"][0][:, se2])


def test_full_through_r3_reproduces_the_r1_model(r3_done):
    cfg, *_ = r3_done
    run_r3(cfg, [0], [0], experiments=["full"], permutation=False, device="cpu", progress=False)
    for model, method in (("scratch", "scratch"), ("mlp", "mlp")):
        a = r1.load_eval(r1.job_dir(cfg, method, 0) / "eval.npz")
        b = r1.load_eval(r3_dir(cfg) / f"{model}_full" / "seed0" / "eval.npz")
        assert np.allclose(a["raw"], b["raw"], rtol=1e-6, atol=1e-6), model


def test_runner_skips_finished_jobs_and_resumes_interrupted_ones(r3_done):
    cfg, *_ = r3_done
    again = run_r3(
        cfg, [0, 1], [0, 1], experiments=EXPS, perm_repeats=2, device="cpu", progress=False
    )
    assert {r["status"] for r in again} == {"skipped"}
    # an interrupted job (scoring did not finish) keeps its training and only re-scores
    job = Job("train", "scratch", "no_sst", 1)
    d = job_dir(cfg, job)
    ckpt_before = (d / "recon.pt").stat().st_mtime_ns
    (d / "done.json").unlink()
    (d / "eval.npz").unlink()
    res = run_r3(
        cfg, [1], [], ["scratch"], ["no_sst"], permutation=False, device="cpu", progress=False
    )
    assert [r["status"] for r in res] == ["done"]
    assert (d / "recon.pt").stat().st_mtime_ns == ckpt_before  # not retrained
    assert (d / "done.json").exists() and (d / "eval.npz").exists()
    # a permutation job without its R1 model is reported, not an error
    res = run_r3(cfg, [5], [], ["scratch"], ["no_sst"], device="cpu", progress=False)
    assert res[-1] == {"job": "perm_scratch seed 5", "status": "missing", "seconds": 0.0}


def test_report_summary_numbers_figures_and_ranking(r3_done):
    cfg, *_ = r3_done
    summary = make_r3_report(cfg, n_boot=200, block_length=3, seed=1)
    out = r3_dir(cfg)
    for f in ("summary.json", "summary.md"):
        assert (out / f).exists()
    figs = {p.name for p in (out / "figures").glob("*.png")}
    assert {
        "ablation_heatmap_scratch.png",
        "ablation_heatmap_mlp.png",
        "permutation_by_depth.png",
        "history_rmse_by_depth.png",
    } <= figs
    s = json.loads((out / "summary.json").read_text())
    assert s["settings"]["bootstrap"]["block_length_days"] == 3
    assert s["settings"]["n_days"] == 10
    sc = s["experiments"]["scratch"]["experiments"]
    assert set(sc) == set(EXPS) and sc["no_sst"]["seeds"] == [0, 1]
    cell = sc["no_sst"]["regions"]["all"]["columns"][POOLED]
    # the reference is the R1 model of the same seeds; delta = experiment - reference
    assert cell["reference_rmse"]["seed_values"] == pytest.approx(
        [
            _pooled_rmse(r1.load_eval(r1.job_dir(cfg, "scratch", sd) / "eval.npz")["raw"])
            for sd in (0, 1)
        ],
        rel=1e-9,
    )
    exp_pt = _pooled_rmse(
        _mean_raw([job_dir(cfg, Job("train", "scratch", "no_sst", sd)) for sd in (0, 1)])
    )
    ref_pt = _pooled_rmse(_mean_raw([r1.job_dir(cfg, "scratch", sd) for sd in (0, 1)]))
    assert cell["rmse"]["point"] == pytest.approx(exp_pt, rel=1e-9)
    assert cell["delta"]["diff"] == pytest.approx(exp_pt - ref_pt, rel=1e-6)
    assert isinstance(cell["delta"]["established"], bool)
    assert sc["hist3"]["history"] == 3
    assert set(s["history_contrast"]) == {"1", "3"} and "change_vs_k1" in s["history_contrast"]["3"]
    # permutation: delta = permuted - unpermuted, from the stored sums
    z = {sd: np.load(r3_dir(cfg) / "perm_mlp" / f"seed{sd}" / "perm.npz")["raw"] for sd in (0, 1)}
    gp = s["permutation"]["mlp"]["groups"]["sst"]["regions"]["all"]["columns"][POOLED]
    mean = np.mean([z[0], z[1]], axis=0)
    want = _pooled_rmse(mean[1]) - _pooled_rmse(mean[0])
    assert gp["diff"] == pytest.approx(want, rel=1e-6) and len(gp["seed_values"]) == 2
    assert set(s["ranking"]) == {"scratch", "mlp"}
    rk = s["ranking"]["scratch"]["all"]
    assert set(rk["retrain_delta"]) == {"sst", "sss", "winds"}
    assert sorted(rk["rank_retrain"].values()) == [1, 2, 3]
    text = (out / "summary.md").read_text(encoding="utf-8")
    assert "retrain without" in text and "Permutation importance" in text
    assert summary["settings"]["pooled_range_m"] == [50.0, 200.0]


def _depths():
    return np.array([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000], float)


def _pooled_rmse(raw: np.ndarray) -> float:
    """Whole-domain pooled 50-200 m RMSE from per-day sums ``(R, F, T, D)``."""
    sel = (_depths() >= 50) & (_depths() <= 200)
    r = raw[0]
    n = r[SUM_FIELDS.index("n")][:, sel].sum()
    return float(np.sqrt(r[SUM_FIELDS.index("se2")][:, sel].sum() / n))


def _mean_raw(dirs: list[Path]) -> np.ndarray:
    return np.mean([r1.load_eval(d / "eval.npz")["raw"] for d in dirs], axis=0)


def test_cli_r3_commands(r3_done):
    cfg, *_ = r3_done
    res = runner.invoke(
        app,
        [
            "research", "r3", "--config", str(TINY), "--seeds", "0,1", "--mlp-seeds", "0,1",
            "--experiments", ",".join(EXPS), "--perm-repeats", "2", "--device", "cpu",
        ],
    )  # fmt: skip
    assert res.exit_code == 0, res.output + str(res.exception)
    assert "0 job(s) run, 24 skipped" in res.output
    out = runner.invoke(
        app,
        ["research", "r3-report", "--config", str(TINY), "--n-boot", "100", "--block-length", "2"],
    )
    assert out.exit_code == 0, out.output + str(out.exception)
    assert (
        json.loads((r3_dir(cfg) / "summary.json").read_text())["settings"]["bootstrap"][
            "block_length_days"
        ]
        == 2
    )
    bad = runner.invoke(
        app,
        ["research", "r3", "--config", str(TINY), "--experiments", "no_salt", "--device", "cpu"],
    )
    assert bad.exit_code != 0


def test_report_needs_the_references(tiny_cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("OCEANEMBED_OUTPUTS_ROOT", str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError):
        load_results(tiny_cfg)
    res = runner.invoke(app, ["research", "r3-report", "--config", str(TINY)])
    assert res.exit_code == 2


def test_load_test_inputs_matches_the_streamed_batches(tiny_pipeline: Config):
    ds = make_surface_dataset(
        tiny_pipeline, tiny_pipeline.split.test.start, tiny_pipeline.split.test.end
    )
    x = load_test_inputs(ds, batch_size=3)
    assert x.shape == (len(ds), 12, *ds.shape)
    assert torch.equal(x[2:6], ds.batch(2, 6)["x"])
