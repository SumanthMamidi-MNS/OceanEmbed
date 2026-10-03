"""Research stage R3: which surface variable matters at which depth, and does history help?

Three families of jobs, all scored on the test split against GLORYS exactly like R1 (per-day,
per-depth, per-basin sufficient statistics in ``eval.npz``), written under
``outputs/<run>/research/r3/`` only:

* **retrain-without ablations** ``<model>_<experiment>/seed<k>/`` -- the model (``scratch``: the
  headline Transformer trained from scratch; ``mlp``: the per-pixel MLP) is retrained with a
  variable group zeroed in its input (``no_sst`` ... ``no_winds``, ``sst_only``, ``sst_sla``,
  ``sst_sla_winds``; see :mod:`oceanembed.research.inputs`);
* **temporal context** ``<model>_hist3`` / ``<model>_hist7`` -- the surface fields of the previous
  ``k - 1`` days are extra input channels (``full`` is the k = 1 reference; the R1 ``scratch`` /
  ``mlp`` jobs of the same seeds are used instead of retraining it, the code path being identical);
* **permutation importance** ``perm_<model>/seed<k>/perm.npz`` -- no training: a finished R1 model
  is scored with one variable group permuted across the test days (within the calendar month),
  several repeats, plus an unpermuted pass from the same code.

Per training job folder: ``recon.pt`` (``mlp.pt``), ``train.jsonl``, ``train.done.json``,
``eval.npz`` and ``done.json`` (written last). A finished job is skipped; an interrupted one resumes
at its first stage without a ``*.done.json`` marker.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from oceanembed.config import Config
from oceanembed.eval.evaluate import read_target
from oceanembed.eval.metrics import SUM_FIELDS
from oceanembed.infer.predict import Predictor, predict_batch
from oceanembed.research import r1
from oceanembed.research.inputs import (
    EXPERIMENT_LABELS,
    EXPERIMENT_ORDER,
    EXPERIMENTS,
    GROUP_SURFACE_INDEX,
    GROUPS,
    BatchFn,
    InputSpec,
    load_test_inputs,
    make_test_batch_fn,
    make_views,
    month_permutation,
    permute_batch,
    sample_points_view,
)
from oceanembed.train.mlp import run_train_mlp
from oceanembed.train.train import run_train

log = logging.getLogger(__name__)

MODELS = ("scratch", "mlp")  # the headline Transformer (from scratch) and the per-pixel MLP
MODEL_LABELS = {"scratch": "OceanEmbed (no pretraining)", "mlp": "Per-pixel MLP"}
DEFAULT_SEEDS = (0, 1)
DEFAULT_MLP_SEEDS = (0, 1, 2)
DEFAULT_PERM_REPEATS = 3
REGIONS = r1.REGIONS
EVAL_FILE = r1.EVAL_FILE
DONE_FILE = r1.DONE_FILE
PERM_FILE = "perm.npz"
PERM_SEED = 20240


# ----------------------------------------------------------------------------------------
# jobs and paths
# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Job:
    kind: str  # "train" (retrain + score) or "perm" (score a finished R1 model, permuted)
    model: str  # scratch | mlp
    experiment: str  # an inputs.EXPERIMENTS key; "permutation" for kind == "perm"
    seed: int

    @property
    def name(self) -> str:
        return f"perm_{self.model}" if self.kind == "perm" else f"{self.model}_{self.experiment}"

    def __str__(self) -> str:
        return f"{self.name} seed {self.seed}"


def r3_dir(cfg: Config) -> Path:
    return cfg.outputs_dir / "research" / "r3"


def job_dir(cfg: Config, job: Job) -> Path:
    return r3_dir(cfg) / job.name / f"seed{job.seed}"


def is_done(cfg: Config, job: Job) -> bool:
    d = job_dir(cfg, job)
    return (d / DONE_FILE).exists() and (
        d / (PERM_FILE if job.kind == "perm" else EVAL_FILE)
    ).exists()


def plan_jobs(
    seeds: list[int],
    mlp_seeds: list[int],
    models: list[str] | None = None,
    experiments: list[str] | None = None,
    permutation: bool = True,
) -> list[Job]:
    """Round by round over the seeds: in every round the retraining jobs (Transformer and MLP of
    an experiment next to each other, experiments in order of interest), then the permutation jobs
    of that seed -- so one seed of everything arrives before any second seed."""
    wanted_models = list(models) if models else list(MODELS)
    wanted_exps = list(experiments) if experiments else list(EXPERIMENT_ORDER)
    for m in wanted_models:
        if m not in MODELS:
            raise ValueError(f"unknown model {m!r}; choose from {list(MODELS)}")
    for e in wanted_exps:
        if e not in EXPERIMENTS:
            raise ValueError(f"unknown experiment {e!r}; choose from {list(EXPERIMENTS)}")
    per_model = {"scratch": list(seeds), "mlp": list(mlp_seeds)}
    rounds = sorted({s for m in wanted_models for s in per_model[m]})
    jobs: list[Job] = []
    for s in rounds:
        for e in wanted_exps:
            jobs += [Job("train", m, e, s) for m in wanted_models if s in per_model[m]]
        if permutation:
            jobs += [Job("perm", m, "permutation", s) for m in wanted_models if s in per_model[m]]
    return jobs


def _say(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------------------------------------------------------------------
# scoring (several passes over the test split, one read of the reference)
# ----------------------------------------------------------------------------------------
def region_matrix(ctx: r1._Context) -> np.ndarray:
    """``(H*W, R)`` float64 0/1 matrix: column 0 the whole grid, then each basin."""
    basins = ctx.grid.basin_masks()
    h, w = ctx.grid.lat.size, ctx.grid.lon.size
    cols = [np.ones((h, w), bool), *(basins[r] for r in REGIONS[1:])]
    return np.stack([c.reshape(-1) for c in cols], axis=1).astype(np.float64)


def region_sums(x: np.ndarray, y: np.ndarray, valid: np.ndarray, rmat: np.ndarray) -> np.ndarray:
    """Sums of the eight :data:`SUM_FIELDS` over the grid points of every region of ``rmat``.

    ``x`` (prediction) and ``y`` (reference) are ``(B, D, H, W)``; a point counts when ``valid`` and
    both values are finite (as in :func:`oceanembed.eval.metrics.sums_from_arrays`). Returns
    ``(R, F, B, D)``. One matrix product per field instead of one pass per region."""
    ok = valid & np.isfinite(x) & np.isfinite(y)
    xs = np.where(ok, x, 0.0).astype(np.float64)
    ys = np.where(ok, y, 0.0).astype(np.float64)
    e = xs - ys
    fields = (ok.astype(np.float64), xs, ys, xs * xs, ys * ys, xs * ys, np.abs(e), e * e)
    b, d = ok.shape[:2]
    out = np.empty((rmat.shape[1], len(SUM_FIELDS), b, d))
    for i, arr in enumerate(fields):
        part = arr.reshape(b * d, -1) @ rmat  # (B*D, R)
        out[:, i] = part.T.reshape(rmat.shape[1], b, d)
    return out


def score_passes(
    ctx: r1._Context,
    predictor: Predictor,
    batch_fns: list[BatchFn],
    batch_size: int = 8,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-day, per-depth sums of ``predictor`` vs GLORYS for every pass in ``batch_fns``.

    Returns ``raw`` and ``anom`` of shape ``(pass, region, field, day, depth)``; ``anom`` has the
    climatology removed from both prediction and reference. The reference and the climatology of
    a batch are read once for all passes."""
    ds, zds = ctx.test()
    n_days, n_depth = len(ds), len(ds.depth)
    rmat = region_matrix(ctx)
    shape = (len(batch_fns), rmat.shape[1], len(SUM_FIELDS), n_days, n_depth)
    raw, anom = np.zeros(shape), np.zeros(shape)
    dates = ds.dates()
    bar = progress and sys.stderr.isatty()
    for a in tqdm(range(0, n_days, batch_size), desc="score", disable=not bar):
        b = min(a + batch_size, n_days)
        ref = read_target(zds, ds, a, b)
        clim = ds.stats.climatology(dates[a:b].values)
        valid = ds.mask[None] & np.isfinite(ref)
        for p, fn in enumerate(batch_fns):
            temp = predict_batch(predictor, fn(a, b), ds, ctx.dev)
            raw[p, :, :, a:b] = region_sums(temp, ref, valid, rmat)
            anom[p, :, :, a:b] = region_sums(temp - clim, ref - clim, valid, rmat)
    return raw, anom


def _save_npz(path: Path, raw: np.ndarray, anom: np.ndarray, ctx: r1._Context, **extra) -> None:
    ds, _ = ctx.test()
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp,
        raw=raw,
        anom=anom,
        dates=np.array([str(t.date()) for t in ds.dates()]),
        depths=np.asarray(ds.depth, dtype=float),
        regions=np.array(REGIONS),
        fields=np.array(SUM_FIELDS),
        **extra,
    )
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------
# one job
# ----------------------------------------------------------------------------------------
def model_cfg(cfg: Config, spec: InputSpec) -> Config:
    """``cfg`` with the network's input width of ``spec`` (extra channels for the history)."""
    if spec.n_channels == cfg.model.in_channels:
        return cfg
    return cfg.model_copy(
        update={"model": cfg.model.model_copy(update={"in_channels": spec.n_channels})}
    )


def _train_job(cfg: Config, job: Job, ctx: r1._Context, d: Path, progress: bool) -> dict:
    spec = EXPERIMENTS[job.experiment]
    info: dict = {
        "kind": "train",
        "model": job.model,
        "model_label": MODEL_LABELS[job.model],
        "experiment": job.experiment,
        "label": EXPERIMENT_LABELS[job.experiment],
        "seed": job.seed,
        "keep": list(spec.keep),
        "history": spec.history,
        "n_input_channels": spec.n_channels,
    }
    marker = d / "train.done.json"
    if marker.exists():
        info["train"] = json.loads(marker.read_text(encoding="utf-8"))
    else:
        ckpt = "mlp.pt" if job.model == "mlp" else "recon.pt"
        r1._remove(d / ckpt, d / "train.jsonl", d / "mlp.jsonl")
        t0 = time.time()
        train_ds, val_ds = make_views(*ctx.train_val(), spec)
        mcfg = model_cfg(cfg, spec)
        if job.model == "mlp":
            best = run_train_mlp(
                mcfg,
                job.seed,
                str(ctx.dev),
                progress,
                ckpt_path=d / ckpt,
                log_path=d / "train.jsonl",
                datasets=(train_ds, val_ds),
                sample_fn=sample_points_view,
                model_kwargs={"channels": spec.feature_channels} if spec.history > 1 else None,
            )
        else:
            best = run_train(
                mcfg,
                pretrained=False,
                tag=job.experiment,
                device=str(ctx.dev),
                progress=progress,
                seed=job.seed,
                ckpt_path=d / ckpt,
                log_path=d / "train.jsonl",
                datasets=(train_ds, val_ds),
            )
        info["train"] = {**best, "seconds": time.time() - t0, "n_train_days": len(train_ds)}
        r1._write_json(marker, info["train"])
    predictor, meta = r1._predictor(cfg, ctx, job.model, d)
    t0 = time.time()
    ds, _ = ctx.test()
    raw, anom = score_passes(ctx, predictor, [make_test_batch_fn(cfg, ds, spec)], progress=progress)
    _save_npz(d / EVAL_FILE, raw[0], anom[0], ctx)
    info["score"] = {**meta, "seconds": time.time() - t0}
    return info


def _perm_job(
    cfg: Config, job: Job, ctx: r1._Context, d: Path, repeats: int, progress: bool
) -> dict:
    src = r1.job_dir(cfg, job.model, job.seed)
    if not r1.is_done(src):
        raise FileNotFoundError(f"{src}: the R1 model to permute has not been trained")
    predictor, meta = r1._predictor(cfg, ctx, job.model, src)
    ds, _ = ctx.test()
    x_all = load_test_inputs(ds)
    t_all = torch.from_numpy(ds.time[ds.indices].astype("datetime64[D]").astype(np.int64))
    dates = ds.dates()
    names = list(GROUPS)
    fns: list[BatchFn] = [lambda a, b: {"x": x_all[a:b], "t": t_all[a:b]}]  # unpermuted pass
    for gi, g in enumerate(names):
        for rep in range(repeats):
            rng = np.random.default_rng([PERM_SEED, job.seed, gi, rep])
            donor = month_permutation(dates, rng)
            ch = list(GROUP_SURFACE_INDEX[g])

            def fn(a, b, donor=donor, ch=ch):
                return {"x": permute_batch(x_all, donor, ch, a, b), "t": t_all[a:b]}

            fns.append(fn)
    t0 = time.time()
    raw, anom = score_passes(ctx, predictor, fns, progress=progress)
    # repeats of a group -> their mean sums (the expected permuted score); pass 0 stays as it is
    shape = (len(names), repeats, *raw.shape[1:])
    raw_g = raw[1:].reshape(shape).mean(axis=1)
    anom_g = anom[1:].reshape(shape).mean(axis=1)
    _save_npz(
        d / PERM_FILE,
        np.concatenate([raw[:1], raw_g]),
        np.concatenate([anom[:1], anom_g]),
        ctx,
        passes=np.array(["none", *names]),
        repeats=np.array(repeats),
    )
    return {
        "kind": "perm",
        "model": job.model,
        "model_label": MODEL_LABELS[job.model],
        "seed": job.seed,
        "groups": names,
        "repeats": repeats,
        "scheme": "within calendar month, random derangement, all channels of a group together",
        "score": {**meta, "seconds": time.time() - t0},
    }


def run_job(
    cfg: Config,
    job: Job,
    ctx: r1._Context,
    repeats: int = DEFAULT_PERM_REPEATS,
    progress: bool = True,
) -> dict:
    d = job_dir(cfg, job)
    d.mkdir(parents=True, exist_ok=True)
    r1._remove(d / DONE_FILE, d / "eval.tmp.npz", d / EVAL_FILE, d / "perm.tmp.npz", d / PERM_FILE)
    info = (
        _perm_job(cfg, job, ctx, d, repeats, progress)
        if job.kind == "perm"
        else _train_job(cfg, job, ctx, d, progress)
    )
    info["created"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    r1._write_json(d / DONE_FILE, info)  # last: the job exists only once everything else does
    return info


def run_r3(
    cfg: Config,
    seeds: list[int] | None = None,
    mlp_seeds: list[int] | None = None,
    models: list[str] | None = None,
    experiments: list[str] | None = None,
    permutation: bool = True,
    perm_repeats: int = DEFAULT_PERM_REPEATS,
    skip_existing: bool = True,
    device: str | None = None,
    progress: bool = True,
) -> list[dict]:
    """Run every planned job (resumable); one ``{job, status, seconds}`` each, status ``done``,
    ``skipped`` (finished earlier) or ``missing`` (a permutation job whose R1 model does not
    exist). Stops at the first failure; re-running resumes."""
    jobs = plan_jobs(
        list(DEFAULT_SEEDS if seeds is None else seeds),
        list(DEFAULT_MLP_SEEDS if mlp_seeds is None else mlp_seeds),
        models,
        experiments,
        permutation,
    )
    ctx = r1._Context(cfg, device)
    results = []
    t_all = time.time()
    for i, job in enumerate(jobs, 1):
        d = job_dir(cfg, job)
        head = f"[{i}/{len(jobs)}] {job}"
        if skip_existing and is_done(cfg, job):
            _say(f"{head}: finished earlier, skipped")
            results.append({"job": str(job), "status": "skipped", "seconds": 0.0})
            continue
        if job.kind == "perm" and not r1.is_done(r1.job_dir(cfg, job.model, job.seed)):
            _say(f"{head}: no finished R1 {job.model} model for this seed, skipped")
            results.append({"job": str(job), "status": "missing", "seconds": 0.0})
            continue
        _say(f"{head} -> {d}")
        if not skip_existing:
            r1._remove(*d.glob("*.done.json"))  # a forced re-run retrains every stage too
        t0 = time.time()
        info = run_job(cfg, job, ctx, perm_repeats, progress)
        dt = time.time() - t0
        tr = info.get("train", {})
        _say(
            f"{head}: done in {dt:.0f}s"
            + (f" (best epoch {tr['epoch']}, val RMSE {tr['val_rmse']:.3f} degC)" if tr else "")
            + f"; total {time.time() - t_all:.0f}s"
        )
        results.append({"job": str(job), "status": "done", "seconds": dt})
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results
