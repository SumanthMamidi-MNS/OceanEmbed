"""Research stage R1 (rigour): train every method for several seeds and score it on the test split.

Everything is written under ``outputs/<run>/research/r1/<method>/seed<k>/`` and nothing under the
main run's ``checkpoints``, ``metrics``, ``predictions``, ``embeddings`` or ``report.md`` (the
ridge checkpoint and the climatology statistics are only read). Per ``(method, seed)`` folder:

* ``pretrain.pt`` / ``pretrain.jsonl`` + ``pretrain.done.json``   (``oceanembed`` only)
* ``recon.pt`` (``mlp.pt``) / ``train.jsonl`` + ``train.done.json`` (trained methods)
* ``eval.npz`` -- per-day, per-depth sufficient statistics of the test split vs GLORYS for
  the whole domain and each basin, raw and climatology-removed (what the bootstrap needs)
* ``done.json`` -- written last: provenance, timings, parameter counts, best epochs

A ``(method, seed)`` with ``done.json`` is skipped; an unfinished one restarts at its first stage
without a ``*.done.json`` marker, after deleting that stage's partial files.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from oceanembed.config import Config
from oceanembed.data.dataset import make_dataset, make_surface_dataset
from oceanembed.data.harmonize import open_harmonized
from oceanembed.eval.evaluate import json_clean, read_target, split_bounds
from oceanembed.eval.metrics import SUM_FIELDS, sums_from_arrays
from oceanembed.grid import build_grid
from oceanembed.infer.predict import Predictor, load_recon_model, model_predictor, predict_batch
from oceanembed.models.baselines import RIDGE_FILE, ClimatologyBaseline, RidgeBaseline
from oceanembed.models.pixel_mlp import load_pixel_mlp
from oceanembed.train.mlp import run_train_mlp
from oceanembed.train.pretrain import run_pretrain
from oceanembed.train.train import run_train
from oceanembed.train.utils import get_device

log = logging.getLogger(__name__)

METHODS = ("oceanembed", "scratch", "unet", "mlp", "ridge", "climatology")
TRAINED = ("oceanembed", "scratch", "unet", "mlp")  # one training per seed
DETERMINISTIC = ("ridge", "climatology")  # taken from the main run; one evaluation, no seeds
LABELS = {
    "oceanembed": "OceanEmbed (pretrained encoder)",
    "scratch": "OceanEmbed (no pretraining)",
    "unet": "Plain U-Net (no Transformer)",
    "mlp": "Per-pixel MLP",
    "ridge": "Ridge regression",
    "climatology": "Climatology",
}
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_MLP_SEEDS = 3
REGIONS = ("all", "arabian_sea", "bay_of_bengal")
EVAL_FILE = "eval.npz"
DONE_FILE = "done.json"


# ----------------------------------------------------------------------------------------
# paths and small helpers
# ----------------------------------------------------------------------------------------
def r1_dir(cfg: Config) -> Path:
    return cfg.outputs_dir / "research" / "r1"


def job_dir(cfg: Config, method: str, seed: int) -> Path:
    return r1_dir(cfg) / method / f"seed{seed}"


def is_done(d: Path) -> bool:
    return (d / DONE_FILE).exists() and (d / EVAL_FILE).exists()


def _write_json(path: Path, obj: dict) -> None:
    """Atomic: a marker file either exists complete or not at all."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(json_clean(obj), indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _remove(*paths: Path) -> None:
    for p in paths:
        if p.exists():
            p.unlink()


def _say(msg: str) -> None:
    print(msg, flush=True)


def plan_jobs(
    seeds: list[int], methods: list[str] | None = None, mlp_seeds: int = DEFAULT_MLP_SEEDS
) -> list[tuple[str, int]]:
    """Cheap deterministic methods first, then seed by seed over the trained methods, so the most
    decision-relevant results (one seed of every method) arrive first. The MLP runs for the
    first ``mlp_seeds`` of ``seeds`` only."""
    wanted = list(methods) if methods else list(METHODS)
    bad = [m for m in wanted if m not in METHODS]
    if bad:
        raise ValueError(f"unknown method(s) {bad}; choose from {list(METHODS)}")
    jobs = [(m, 0) for m in DETERMINISTIC if m in wanted]
    for i, s in enumerate(seeds):
        for m in TRAINED:
            if m in wanted and (m != "mlp" or i < mlp_seeds):
                jobs.append((m, int(s)))
    return jobs


# ----------------------------------------------------------------------------------------
# evaluation file
# ----------------------------------------------------------------------------------------
class _Context:
    """Datasets shared by every job of one process: trained jobs reuse the preloaded train / val
    data (about 7 GB for the real run), scoring reuses one test dataset and store handle."""

    def __init__(self, cfg: Config, device: str | None):
        self.cfg = cfg
        self.dev = get_device(device)
        self._train_val: tuple | None = None
        self._test = None
        self._zds = None
        self.grid = build_grid(cfg)

    def train_val(self) -> tuple:
        if self._train_val is None:
            self._train_val = (
                make_dataset(self.cfg, "train", preload=True),
                make_dataset(self.cfg, "val", preload=True),
            )
        return self._train_val

    def test(self):
        if self._test is None:
            start, end = split_bounds(self.cfg, "test")
            self._test = make_surface_dataset(self.cfg, start, end)
            self._zds = open_harmonized(self.cfg)
        return self._test, self._zds


def evaluate_to_file(
    ctx: _Context, predictor: Predictor, path: Path, batch_size: int = 8, progress: bool = True
) -> None:
    """Stream the test split and save the per-day, per-depth sums of ``predictor`` vs GLORYS.

    ``eval.npz`` holds ``raw`` and ``anom`` of shape ``(region, field, day, depth)`` (regions
    ``REGIONS``; fields ``SUM_FIELDS``), ``dates``, ``depths``. ``anom`` is the sums of
    (prediction - climatology, reference - climatology). A point counts when the static mask and
    the GLORYS value are finite, exactly as in ``oceanembed evaluate``."""
    ds, zds = ctx.test()
    n_days, n_depth = len(ds), len(ds.depth)
    basins = ctx.grid.basin_masks()
    regions: list[np.ndarray | None] = [None, *(basins[r] for r in REGIONS[1:])]
    out = {
        kind: np.zeros((len(REGIONS), len(SUM_FIELDS), n_days, n_depth)) for kind in ("raw", "anom")
    }
    dates = ds.dates()
    bar = progress and sys.stderr.isatty()
    for a in tqdm(range(0, n_days, batch_size), desc="score", disable=not bar):
        b = min(a + batch_size, n_days)
        ref = read_target(zds, ds, a, b)
        clim = ds.stats.climatology(dates[a:b].values)
        valid = ds.mask[None] & np.isfinite(ref)
        temp = predict_batch(predictor, ds.batch(a, b), ds, ctx.dev)
        for r, region in enumerate(regions):
            v = valid if region is None else valid & region[None, None]
            for kind, (x, y) in {"raw": (temp, ref), "anom": (temp - clim, ref - clim)}.items():
                part = sums_from_arrays(x, y, v, axis=(2, 3))
                for i, f in enumerate(SUM_FIELDS):
                    out[kind][r, i, a:b] = part[f]
    tmp = path.with_name("eval.tmp.npz")
    np.savez_compressed(
        tmp,
        raw=out["raw"],
        anom=out["anom"],
        dates=np.array([str(t.date()) for t in dates]),
        depths=np.asarray(ds.depth, dtype=float),
        regions=np.array(REGIONS),
        fields=np.array(SUM_FIELDS),
    )
    os.replace(tmp, path)


def load_eval(path: Path) -> dict:
    """The arrays of an ``eval.npz`` as a plain dict."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------------------
# one (method, seed)
# ----------------------------------------------------------------------------------------
def _predictor(cfg: Config, ctx: _Context, method: str, d: Path) -> tuple[Predictor, dict]:
    if method == "ridge":
        path = cfg.checkpoints_dir / RIDGE_FILE
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; the main run's ridge baseline is required")
        return RidgeBaseline.load(path), {"checkpoint": str(path)}
    if method == "climatology":
        return ClimatologyBaseline(cfg.model.n_depths), {}
    if method == "mlp":
        model, meta = load_pixel_mlp(d / "mlp.pt", ctx.dev)
        return model_predictor(model, amp=False), meta
    model, _ = load_recon_model(d / "recon.pt", ctx.dev)
    return model_predictor(model, amp=cfg.train.amp), dict(model.ckpt_meta)


def run_job(cfg: Config, method: str, seed: int, ctx: _Context, progress: bool = True) -> dict:
    """Train (if the method trains) and score one ``(method, seed)``; return its ``done.json``."""
    d = job_dir(cfg, method, seed)
    d.mkdir(parents=True, exist_ok=True)
    _remove(d / DONE_FILE, d / "eval.tmp.npz", d / EVAL_FILE)  # scoring always starts afresh
    info: dict = {"method": method, "label": LABELS[method], "seed": seed}
    seconds: dict[str, float] = {}

    def stage(name: str, files: list[str], fn):
        marker = d / f"{name}.done.json"
        if marker.exists():
            info[name] = json.loads(marker.read_text(encoding="utf-8"))
            return
        _remove(*(d / f for f in files))  # partial files of an interrupted attempt
        t0 = time.time()
        best = fn()
        seconds[name] = time.time() - t0
        info[name] = {**best, "seconds": seconds[name]}
        _write_json(marker, info[name])

    if method == "oceanembed":
        stage(
            "pretrain",
            ["pretrain.pt", "pretrain.jsonl"],
            lambda: run_pretrain(
                cfg,
                str(ctx.dev),
                progress,
                seed=seed,
                ckpt_path=d / "pretrain.pt",
                log_path=d / "pretrain.jsonl",
                datasets=ctx.train_val(),
            ),
        )
    if method in ("oceanembed", "scratch", "unet"):
        stage(
            "train",
            ["recon.pt", "train.jsonl"],
            lambda: run_train(
                cfg,
                pretrained=method == "oceanembed",
                tag=method,
                device=str(ctx.dev),
                progress=progress,
                seed=seed,
                arch="unet" if method == "unet" else None,
                ckpt_path=d / "recon.pt",
                log_path=d / "train.jsonl",
                pretrain_path=d / "pretrain.pt",
                datasets=ctx.train_val(),
            ),
        )
    elif method == "mlp":
        stage(
            "train",
            ["mlp.pt", "mlp.jsonl", "train.jsonl"],
            lambda: run_train_mlp(
                cfg,
                seed,
                str(ctx.dev),
                progress,
                ckpt_path=d / "mlp.pt",
                log_path=d / "train.jsonl",
                datasets=ctx.train_val(),
            ),
        )
    predictor, meta = _predictor(cfg, ctx, method, d)
    t0 = time.time()
    evaluate_to_file(ctx, predictor, d / EVAL_FILE, progress=progress)
    seconds["score"] = time.time() - t0
    info["score"] = {**meta, "seconds": seconds["score"]}
    info["created"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_json(d / DONE_FILE, info)  # last: the job exists only once everything else does
    return info


def run_r1(
    cfg: Config,
    seeds: list[int] | None = None,
    methods: list[str] | None = None,
    mlp_seeds: int = DEFAULT_MLP_SEEDS,
    skip_existing: bool = True,
    device: str | None = None,
    progress: bool = True,
) -> list[dict]:
    """Run every planned ``(method, seed)``; returns one ``{method, seed, status, seconds}`` each
    (status ``done`` or ``skipped``). Stops at the first failure (re-running resumes)."""
    jobs = plan_jobs(list(seeds or DEFAULT_SEEDS), methods, mlp_seeds)
    ctx = _Context(cfg, device)
    results = []
    t_all = time.time()
    for i, (method, seed) in enumerate(jobs, 1):
        d = job_dir(cfg, method, seed)
        head = f"[{i}/{len(jobs)}] {method} seed {seed}"
        if skip_existing and is_done(d):
            _say(f"{head}: finished earlier, skipped")
            results.append({"method": method, "seed": seed, "status": "skipped", "seconds": 0.0})
            continue
        _say(f"{head} -> {d}")
        if not skip_existing:  # a forced re-run retrains every stage too
            _remove(*d.glob("*.done.json"))
        t0 = time.time()
        info = run_job(cfg, method, seed, ctx, progress)
        dt = time.time() - t0
        tr = info.get("train", {})
        _say(
            f"{head}: done in {dt:.0f}s"
            + (f" (best epoch {tr['epoch']}, val RMSE {tr['val_rmse']:.3f} degC)" if tr else "")
            + f"; total {time.time() - t_all:.0f}s"
        )
        results.append({"method": method, "seed": seed, "status": "done", "seconds": dt})
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results
