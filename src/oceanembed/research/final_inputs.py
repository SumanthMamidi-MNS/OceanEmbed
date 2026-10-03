"""Input-set selection for the final model, on the long run, decided on the validation year only.

Candidates are the headline Transformer (trained from scratch) with a reduced set of input groups:
``sst_sla_winds`` ("all but salinity and currents", i.e. SST + sea level + winds) and ``sst_sla``
(SST + sea level). The reference is the full seven-input model of R2 (``research/r2/scratch``, same
seeds, same data and settings). Every candidate is trained with the R3 machinery
(:func:`oceanembed.research.r3._train_job`: the dropped groups are zeroed in every input
channel) for the same seeds as the reference, and scored on the whole test split exactly like R2
(``eval.npz``).

Everything is written under ``outputs/<run>/research/final_inputs/<model>_<experiment>/seed<k>/``
(``recon.pt``, ``train.jsonl``, ``train.done.json``, ``eval.npz``, ``done.json``); a finished job is
skipped and an interrupted one resumes at its first stage without a ``*.done.json`` marker.

The decision rule lives in :mod:`oceanembed.research.final_inputs_report` and uses the validation
RMSE of the saved checkpoint (the year-2022 error that early stopping already minimises) -- never a
test-year number.
"""

from __future__ import annotations

import gc
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from oceanembed.config import Config
from oceanembed.research import r1, r2, r3

log = logging.getLogger(__name__)

CANDIDATES = ("sst_sla_winds", "sst_sla")  # reduced input sets; "full" is the R2 headline model
DEFAULT_SEEDS = (0, 1, 2)
MODEL = "scratch"


def fi_dir(cfg: Config) -> Path:
    return cfg.outputs_dir / "research" / "final_inputs"


def job_dir(cfg: Config, experiment: str, seed: int) -> Path:
    return fi_dir(cfg) / f"{MODEL}_{experiment}" / f"seed{seed}"


def is_done(cfg: Config, experiment: str, seed: int) -> bool:
    d = job_dir(cfg, experiment, seed)
    return (d / r1.DONE_FILE).exists() and (d / r1.EVAL_FILE).exists()


def plan_jobs(seeds: list[int], experiments: list[str] | None = None) -> list[tuple[str, int]]:
    """Seed by seed, so one seed of every candidate exists before any second seed."""
    wanted = list(experiments) if experiments else list(CANDIDATES)
    bad = [e for e in wanted if e not in CANDIDATES]
    if bad:
        raise ValueError(f"unknown experiment(s) {bad}; choose from {list(CANDIDATES)}")
    return [(e, int(s)) for s in seeds for e in wanted]


def run_job(cfg: Config, experiment: str, seed: int, ctx: r1._Context, progress: bool = True):
    d = job_dir(cfg, experiment, seed)
    d.mkdir(parents=True, exist_ok=True)
    r1._remove(d / r1.DONE_FILE, d / "eval.tmp.npz", d / r1.EVAL_FILE)
    job = r3.Job("train", MODEL, experiment, seed)
    info = r3._train_job(cfg, job, ctx, d, progress)
    info["created"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    r1._write_json(d / r1.DONE_FILE, info)
    return info


def run_final_inputs(
    cfg: Config,
    seeds: list[int] | None = None,
    experiments: list[str] | None = None,
    skip_existing: bool = True,
    device: str | None = None,
    progress: bool = True,
) -> list[dict]:
    """Train and score every candidate (resumable); stops at the first failure."""
    jobs = plan_jobs(list(DEFAULT_SEEDS if seeds is None else seeds), experiments)
    for s in {s for _, s in jobs}:  # the reference must exist: same seed, same settings
        if not r2.is_done(cfg, r2.Job("model", "scratch", s)):
            raise FileNotFoundError(
                f"{r2.job_dir(cfg, r2.Job('model', 'scratch', s))}: run `research r2` first "
                "(the full-input reference of this seed)"
            )
    ctx = r1._Context(cfg, device)
    results = []
    t_all = time.time()
    for i, (exp, seed) in enumerate(jobs, 1):
        head = f"[{i}/{len(jobs)}] {MODEL}_{exp} seed {seed}"
        if skip_existing and is_done(cfg, exp, seed):
            print(f"{head}: finished earlier, skipped", flush=True)
            results.append({"job": head, "status": "skipped", "seconds": 0.0})
            continue
        print(f"{head} -> {job_dir(cfg, exp, seed)}", flush=True)
        if not skip_existing:
            r1._remove(*job_dir(cfg, exp, seed).glob("*.done.json"))
        t0 = time.time()
        info = run_job(cfg, exp, seed, ctx, progress)
        dt = time.time() - t0
        tr = info.get("train", {})
        print(
            f"{head}: done in {dt:.0f}s (best epoch {tr.get('epoch')}, val RMSE "
            f"{tr.get('val_rmse', float('nan')):.4f} degC); total {time.time() - t_all:.0f}s",
            flush=True,
        )
        results.append({"job": head, "status": "done", "seconds": dt})
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results
