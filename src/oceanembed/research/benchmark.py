"""Comparison study (phase 13): the model families of the literature under the identical protocol.

Most published work on reconstructing subsurface temperature from satellite data uses tree
ensembles. This stage adds them -- per-pixel **gradient-boosted trees** (LightGBM) and a **random
forest** (scikit-learn) -- next to the families the project already measured, and re-trains the
**plain U-Net** on the eleven-year data. Run on ``configs/poc_long.yaml`` (train 2011-2021,
validation 2022, test 2023 and 2024), for the two input sets of the existing results: all seven
surface inputs (beside the R2 Transformer, per-pixel MLP, ridge, climatology) and SST + sea level
(beside the final model of ``research/final_inputs``).

The trees use exactly the per-pixel features of the per-pixel MLP (the 7 standardised surface
values, sin / cos day of year, normalised lat / lon) and the same training sample:
:func:`oceanembed.models.baselines.sample_points` -- up to ``mlp.max_points`` random (day, ocean
pixel) points of the train split, the seed deciding the draw and the model's own randomness. A
dropped input group is zeroed in the features, as everywhere else. The hyper-parameters are chosen
on the validation year only (the same fixed validation sample as the MLP) from a small documented
grid (:data:`GBT_GRID`, :data:`RF_GRID`) on a smaller training sample; the boosting rounds of every
depth are chosen by early stopping on the same validation sample.

Per job folder ``outputs/<run>/research/benchmark/<family>_<set>/seed<k>/``: the model file (trees
of boosted models; the random forest is not stored, it is large and refits in minutes),
``train.done.json``, ``eval.npz`` (the sufficient statistics of R1/R2: per day, depth and basin, the
whole test split) and ``done.json`` (written last). A finished job is skipped; an interrupted one
restarts at its first unfinished stage.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import torch

from oceanembed.config import INPUT_GROUPS, Config
from oceanembed.infer.predict import Predictor, load_recon_model
from oceanembed.models.baselines import (
    FEATURE_CHANNELS,
    MIN_FIT_POINTS,
    N_FEATURES,
    RidgeBaseline,
    fit_ridge,
    sample_points,
)
from oceanembed.research import r1
from oceanembed.research.r3 import _save_npz, score_passes
from oceanembed.train.mlp import VAL_SAMPLE_SEED, run_train_mlp
from oceanembed.train.train import run_train

log = logging.getLogger(__name__)

SETS: dict[str, tuple[str, ...]] = {"all7": tuple(INPUT_GROUPS), "sst_sla": ("sst", "sla")}
SET_LABELS = {"all7": "all seven inputs", "sst_sla": "SST + sea level"}
FAMILIES = ("rf", "gbt", "unet", "mlp", "ridge")
FAMILY_LABELS = {
    "rf": "Random forest (per pixel)",
    "gbt": "Boosted trees, LightGBM (per pixel)",
    "unet": "Plain U-Net (no Transformer)",
    "mlp": "Per-pixel MLP",
    "ridge": "Ridge regression",
    "scratch": "CNN + Transformer",
    "climatology": "Climatology",
}
# the jobs this stage runs, per input set (the other families are read from R2 / final_inputs)
NEW_JOBS: dict[str, tuple[str, ...]] = {
    "sst_sla": ("ridge", "mlp", "gbt", "rf"),
    "all7": ("gbt", "rf", "unet"),
}
DETERMINISTIC = ("ridge",)
DEFAULT_SEEDS = (0, 1, 2)
EVAL_FILE = r1.EVAL_FILE
DONE_FILE = r1.DONE_FILE
TUNE_POINTS = 300_000  # training sample of the grid search (the final fit uses mlp.max_points)
RF_TUNE_POINTS = 300_000
RF_TUNE_TREES = 40
RF_TREES = 100
RF_MAX_SAMPLES = 400_000  # rows each tree of the forest draws from the training sample

GBT_FIXED = {
    "objective": "regression",
    "learning_rate": 0.1,
    "n_estimators": 600,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
}
GBT_GRID = [{"num_leaves": n} for n in (15, 63, 255)]
RF_GRID = [
    {"min_samples_leaf": leaf, "max_features": mf} for leaf in (5, 20, 60) for mf in (0.5, 1.0)
]
EARLY_STOPPING = 30


@dataclass(frozen=True)
class Job:
    family: str
    set: str
    seed: int = 0

    @property
    def name(self) -> str:
        return f"{self.family}_{self.set}"

    def __str__(self) -> str:
        return f"{self.name} seed {self.seed}"


def bench_dir(cfg: Config) -> Path:
    return cfg.outputs_dir / "research" / "benchmark"


def job_dir(cfg: Config, job: Job) -> Path:
    return bench_dir(cfg) / job.name / f"seed{job.seed}"


def is_done(cfg: Config, job: Job) -> bool:
    d = job_dir(cfg, job)
    return (d / DONE_FILE).exists() and (d / EVAL_FILE).exists()


def set_config(cfg: Config, name: str) -> Config:
    """``cfg`` with the input groups of the set (all seven inputs, or SST + sea level)."""
    groups = list(SETS[name])
    if cfg.model.input_groups == groups:
        return cfg
    return cfg.model_copy(update={"model": cfg.model.model_copy(update={"input_groups": groups})})


def plan_jobs(
    seeds: list[int], families: list[str] | None = None, sets: list[str] | None = None
) -> list[Job]:
    """Cheap deterministic jobs first, then seed by seed, so one seed of every family exists
    before any second seed."""
    wanted = list(families) if families else list(FAMILIES)
    bad = [f for f in wanted if f not in FAMILIES]
    if bad:
        raise ValueError(f"unknown family {bad}; choose from {list(FAMILIES)}")
    use_sets = list(sets) if sets else list(SETS)
    bad = [s for s in use_sets if s not in SETS]
    if bad:
        raise ValueError(f"unknown input set {bad}; choose from {list(SETS)}")
    pairs = [(f, s) for s in use_sets for f in NEW_JOBS[s] if f in wanted]
    jobs = [Job(f, s) for f, s in pairs if f in DETERMINISTIC]
    for seed in seeds:
        jobs += [Job(f, s, int(seed)) for f, s in pairs if f not in DETERMINISTIC]
    return jobs


def _say(msg: str) -> None:
    print(msg, flush=True)


def hardware(device: torch.device | None = None) -> str:
    cpu = f"CPU {platform.processor() or platform.machine()}, {os.cpu_count()} logical cores"
    if device is not None and device.type == "cuda":
        return f"GPU {torch.cuda.get_device_name(device)}; {cpu}"
    return cpu


# ----------------------------------------------------------------------------------------
# tree models: one object that maps features (N, 11) to standardised anomalies (N, 15)
# ----------------------------------------------------------------------------------------
class GbtModel:
    """One LightGBM regressor per depth (the rounds chosen by early stopping)."""

    kind = "gbt"

    def __init__(self, boosters: list, n_threads: int = 0):
        self.boosters = boosters
        self.n_threads = n_threads

    @property
    def n_trees(self) -> int:
        return int(sum(b.current_iteration() for b in self.boosters if b is not None))

    def predict(self, f: np.ndarray) -> np.ndarray:
        out = np.zeros((len(f), len(self.boosters)), np.float32)
        for z, b in enumerate(self.boosters):
            if b is not None:
                out[:, z] = b.predict(f, num_threads=self.n_threads or None)
        return out

    def save(self, path: Path) -> None:
        strings = [None if b is None else b.model_to_string() for b in self.boosters]
        iters = [None if b is None else b.current_iteration() for b in self.boosters]
        joblib.dump({"models": strings, "iterations": iters}, path, compress=3)

    @classmethod
    def load(cls, path: Path, n_threads: int = 0) -> GbtModel:
        import lightgbm as lgb

        d = joblib.load(path)
        return cls(
            [None if s is None else lgb.Booster(model_str=s) for s in d["models"]], n_threads
        )


class RfModel:
    """One multi-output random forest for all depths (not stored: large, refits in minutes)."""

    kind = "rf"

    def __init__(self, forest):
        self.forest = forest

    @property
    def n_trees(self) -> int:
        return len(self.forest.estimators_)

    @property
    def n_nodes(self) -> int:
        return int(sum(t.tree_.node_count for t in self.forest.estimators_))

    def predict(self, f: np.ndarray) -> np.ndarray:
        return self.forest.predict(f).astype(np.float32)


class TreePredictor:
    """Wraps a tree model as a predictor ``x (B, 12, H, W) -> (B, 15, H, W)`` like every other
    method. Land (the mask channel) is not predicted."""

    def __init__(self, model, n_depths: int = 15):
        self.model, self.n_depths = model, n_depths

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        feats = x[:, FEATURE_CHANNELS].permute(0, 2, 3, 1).reshape(-1, N_FEATURES)
        ocean = (x[:, 7].reshape(-1) > 0.5).cpu().numpy()
        f = feats.float().cpu().numpy()
        out = np.zeros((f.shape[0], self.n_depths), np.float32)
        if ocean.any():
            out[ocean] = self.model.predict(f[ocean])
        res = torch.from_numpy(out).reshape(b, h, w, self.n_depths).permute(0, 3, 1, 2)
        return res.contiguous().to(x.device)


def val_rmse(pred: np.ndarray, y: np.ndarray, valid: np.ndarray, anom_std: np.ndarray) -> float:
    """Pooled RMSE in degC over the valid (sample, depth) pairs (anomaly error x depth std), the
    selection criterion of the per-pixel MLP."""
    err = np.where(valid, (pred - y) * anom_std[None], 0.0).astype(np.float64)
    return float(np.sqrt((err**2).sum() / max(valid.sum(), 1)))


def fit_gbt(
    feats, y, valid, vf, vy, vv, params: dict, seed: int, n_threads: int
) -> tuple[GbtModel, list[int]]:
    import lightgbm as lgb

    boosters, rounds = [], []
    for z in range(y.shape[1]):
        v, vz = valid[:, z], vv[:, z]
        if v.sum() < MIN_FIT_POINTS or vz.sum() < 10:
            boosters.append(None)
            rounds.append(0)
            continue
        m = lgb.LGBMRegressor(
            **(GBT_FIXED | params), random_state=seed, n_jobs=n_threads, verbose=-1
        )
        m.fit(
            feats[v], y[v, z], eval_X=vf[vz], eval_y=vy[vz, z],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
        )  # fmt: skip
        boosters.append(m.booster_)
        rounds.append(int(m.booster_.current_iteration()))
    return GbtModel(boosters, n_threads), rounds


def fit_rf(feats, y, valid, params: dict, seed: int, n_threads: int, n_trees: int, max_samples):
    from sklearn.ensemble import RandomForestRegressor

    # a depth below the sea floor has no target: it is filled with 0, the climatological anomaly
    # (the network's loss ignores such depths; a forest has no mask)
    target = np.where(valid, y, 0.0).astype(np.float32)
    forest = RandomForestRegressor(
        n_estimators=n_trees,
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        max_samples=min(max_samples, len(feats)),
        n_jobs=n_threads,
        random_state=seed,
    )
    forest.fit(feats, target)
    return RfModel(forest)


# ----------------------------------------------------------------------------------------
# hyper-parameters: a small grid, chosen on the validation year only
# ----------------------------------------------------------------------------------------
def tuning_file(cfg: Config, family: str, set_name: str) -> Path:
    return bench_dir(cfg) / "tuning" / f"{family}_{set_name}.json"


def tune(cfg: Config, family: str, set_name: str, ctx, n_threads: int) -> dict:
    """Grid search of the family on the validation sample; stored in ``tuning/``. The training
    sample is a smaller draw (seed 0) than the final fit, the criterion is the pooled validation
    RMSE in degC. Returns ``{"best": params, "grid": [...]}``."""
    path = tuning_file(cfg, family, set_name)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    train_ds, val_ds = ctx.train_val()
    mc = cfg.mlp
    anom_std = train_ds.stats.anom_std.astype(np.float32)
    n = TUNE_POINTS if family == "gbt" else RF_TUNE_POINTS
    feats, y, valid = sample_points(train_ds, min(n, mc.max_points), 0)
    vf, vy, vv = sample_points(val_ds, mc.val_points, VAL_SAMPLE_SEED)
    rows = []
    for params in GBT_GRID if family == "gbt" else RF_GRID:
        t0 = time.time()
        if family == "gbt":
            model, rounds = fit_gbt(feats, y, valid, vf, vy, vv, params, 0, n_threads)
            extra = {"rounds_per_depth": rounds}
        else:
            model = fit_rf(feats, y, valid, params, 0, n_threads, RF_TUNE_TREES, RF_MAX_SAMPLES)
            extra = {}
        rmse = val_rmse(model.predict(vf), vy, vv, anom_std)
        rows.append({"params": params, "val_rmse": rmse, "seconds": time.time() - t0, **extra})
        _say(
            f"tune {family}/{set_name} {params}: val RMSE {rmse:.4f} degC "
            f"({rows[-1]['seconds']:.0f}s)"
        )
    best = min(rows, key=lambda r: r["val_rmse"])
    out = {
        "family": family,
        "input_set": set_name,
        "criterion": "pooled validation-year RMSE (degC) of a fixed random validation sample",
        "train_points": int(len(feats)),
        "validation_points": int(len(vf)),
        "grid": rows,
        "best": best["params"],
        "fixed": GBT_FIXED if family == "gbt" else {"n_estimators": RF_TREES,
                                                    "max_samples": RF_MAX_SAMPLES},
    }  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    r1._write_json(path, out)
    return out


# ----------------------------------------------------------------------------------------
# one job
# ----------------------------------------------------------------------------------------
def _train_trees(cfg: Config, job: Job, ctx, d: Path, n_threads: int) -> tuple[dict, object]:
    train_ds, val_ds = ctx.train_val()
    mc = cfg.mlp
    best = tune(cfg, job.family, job.set, ctx, n_threads)["best"]
    feats, y, valid = sample_points(train_ds, mc.max_points, job.seed)
    vf, vy, vv = sample_points(val_ds, mc.val_points, VAL_SAMPLE_SEED)
    anom_std = train_ds.stats.anom_std.astype(np.float32)
    if job.family == "gbt":
        model, rounds = fit_gbt(feats, y, valid, vf, vy, vv, best, job.seed, n_threads)
        model.save(d / "gbt.joblib")
        info = {"rounds_per_depth": rounds, "n_trees": model.n_trees}
    else:
        model = fit_rf(feats, y, valid, best, job.seed, n_threads, RF_TREES, RF_MAX_SAMPLES)
        info = {"n_trees": model.n_trees, "n_nodes": model.n_nodes}
    return {
        "val_rmse": val_rmse(model.predict(vf), vy, vv, anom_std),
        "params": best,
        "n_samples": int(len(feats)),
        "n_train_days": len(train_ds),
        **info,
    }, model


def _predictor(cfg: Config, job: Job, ctx, d: Path, model=None, n_threads: int = 0):
    if job.family == "ridge":
        return RidgeBaseline.load(d / "ridge.joblib"), {}
    if job.family == "mlp":
        return r1._predictor(cfg, ctx, "mlp", d)
    if job.family == "unet":
        return r1._predictor(cfg, ctx, "unet", d)
    if model is None:  # resumed after training: boosted models are on disk
        model = GbtModel.load(d / "gbt.joblib", n_threads)
    return TreePredictor(model, cfg.model.n_depths), {"checkpoint": str(d / "gbt.joblib")}


def run_job(cfg_all: Config, job: Job, ctxs: dict, n_threads: int, progress: bool = True) -> dict:
    cfg = set_config(cfg_all, job.set)
    ctx = ctxs[job.set]
    d = job_dir(cfg_all, job)
    d.mkdir(parents=True, exist_ok=True)
    r1._remove(d / DONE_FILE, d / "eval.tmp.npz", d / EVAL_FILE)
    info: dict = {
        "family": job.family,
        "label": FAMILY_LABELS[job.family],
        "input_set": job.set,
        "input_groups": list(SETS[job.set]),
        "seed": job.seed,
        "hardware": hardware(ctx.dev if job.family in ("unet", "mlp") else None),
    }
    marker = d / "train.done.json"
    model = None
    if marker.exists():
        info["train"] = json.loads(marker.read_text(encoding="utf-8"))
    else:
        r1._remove(d / "recon.pt", d / "mlp.pt", d / "ridge.joblib", d / "gbt.joblib",
                   d / "train.jsonl")  # fmt: skip
        t0 = time.time()
        train_ds, val_ds = ctx.train_val()
        if job.family == "ridge":
            ridge = fit_ridge(cfg, train_ds)
            ridge.save(d / "ridge.joblib")
            best = {"n_samples": ridge.meta["n_samples"], "n_train_days": len(train_ds)}
        elif job.family == "mlp":
            best = run_train_mlp(
                cfg, job.seed, str(ctx.dev), progress, ckpt_path=d / "mlp.pt",
                log_path=d / "train.jsonl", datasets=(train_ds, val_ds),
            )  # fmt: skip
            best["n_train_days"] = len(train_ds)
        elif job.family == "unet":
            best = run_train(
                cfg, pretrained=False, tag="unet", device=str(ctx.dev), progress=progress,
                seed=job.seed, arch="unet", ckpt_path=d / "recon.pt", log_path=d / "train.jsonl",
                datasets=(train_ds, val_ds),
            )  # fmt: skip
            m, _ = load_recon_model(d / "recon.pt")
            best["n_params"] = int(sum(p.numel() for p in m.parameters()))
            best["n_train_days"] = len(train_ds)
        else:
            best, model = _train_trees(cfg, job, ctx, d, n_threads)
        info["train"] = {**best, "seconds": time.time() - t0}
        r1._write_json(marker, info["train"])
    predictor, meta = _predictor(cfg, job, ctx, d, model, n_threads)
    ds, _ = ctx.test()
    t0 = time.time()
    raw, anom = score_passes(ctx, predictor, [lambda a, b: ds.batch(a, b)], progress=progress)
    _save_npz(d / EVAL_FILE, raw[0], anom[0], ctx)
    info["score"] = {**meta, "seconds": time.time() - t0}
    info["created"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    r1._write_json(d / DONE_FILE, info)
    return info


def run_benchmark(
    cfg: Config,
    seeds: list[int] | None = None,
    families: list[str] | None = None,
    sets: list[str] | None = None,
    skip_existing: bool = True,
    device: str | None = None,
    threads: int | None = None,
    progress: bool = True,
) -> list[dict]:
    """Run every planned job (resumable); one ``{job, status, seconds}`` each. Stops at the first
    failure; re-running resumes."""
    if list(cfg.model.input_groups) != list(INPUT_GROUPS):
        raise ValueError("the benchmark config must use all input groups (configs/poc_long.yaml)")
    jobs = plan_jobs(list(DEFAULT_SEEDS if seeds is None else seeds), families, sets)
    n_threads = threads or max(1, (os.cpu_count() or 4) - 2)
    ctxs: dict[str, r1._Context] = {}
    results = []
    t_all = time.time()
    for i, job in enumerate(jobs, 1):
        d = job_dir(cfg, job)
        head = f"[{i}/{len(jobs)}] {job}"
        if skip_existing and is_done(cfg, job):
            _say(f"{head}: finished earlier, skipped")
            results.append({"job": str(job), "status": "skipped", "seconds": 0.0})
            continue
        if job.set not in ctxs:
            ctxs[job.set] = r1._Context(set_config(cfg, job.set), device)
        _say(f"{head} -> {d}")
        if not skip_existing:
            r1._remove(*d.glob("*.done.json"))
        t0 = time.time()
        info = run_job(cfg, job, ctxs, n_threads, progress)
        dt = time.time() - t0
        tr = info.get("train", {})
        _say(
            f"{head}: done in {dt:.0f}s (train {tr.get('seconds', 0):.0f}s"
            + (f", val RMSE {tr['val_rmse']:.3f} degC" if "val_rmse" in tr else "")
            + f"); total {time.time() - t_all:.0f}s"
        )
        results.append({"job": str(job), "status": "done", "seconds": dt})
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


__all__ = ["Predictor", "TreePredictor", "run_benchmark"]
