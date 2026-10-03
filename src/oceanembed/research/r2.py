"""Research stage R2: a longer training period and two test years.

Run on a long configuration (``configs/poc_long.yaml``: train 2011-2021, validation 2022, test
2023 and 2024) whose arrays do not fit in RAM; ``train.cache: memmap`` makes the train / validation
arrays on-disk caches read by day (see :mod:`oceanembed.data.dataset`). Everything is written under
``outputs/<run>/research/r2/`` and nothing else of the run is touched.

Per job folder ``<name>/seed<k>/`` (``name``: ``scratch`` and ``mlp`` -- the headline Transformer
trained from scratch and the per-pixel MLP --, ``ridge``, ``climatology``, and the learning-curve
jobs ``scratch_ty<N>`` = the headline model trained on the last ``N`` years only): the model file,
``train.jsonl`` + ``train.done.json``, ``eval.npz`` (per-day, per-depth, per-basin sufficient
statistics of the **whole** test split against GLORYS -- both test years; the report splits them by
year) and ``done.json`` (written last). The ``argo`` job scores the 2023 and 2024 Argo profiles with
the headline model (seed 0), ridge, climatology and GLORYS into ``argo/``.

A finished job is skipped; an interrupted one resumes at its first stage without a
``*.done.json`` marker. The normalisation and the climatology are those fitted on the full train
period for every job, also the learning-curve ones (only the training days differ).
"""

from __future__ import annotations

import gc
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from oceanembed.config import Config
from oceanembed.data.providers.base import months, raw_file
from oceanembed.eval.argo_validation import (
    _argo_block,
    basin_of,
    collocate,
    load_argo_period,
    profile_table,
)
from oceanembed.eval.evaluate import read_target, split_bounds
from oceanembed.infer.predict import Predictor, predict_batch
from oceanembed.models.baselines import ClimatologyBaseline, RidgeBaseline, fit_ridge
from oceanembed.research import r1
from oceanembed.research.r3 import _save_npz, score_passes
from oceanembed.train.mlp import run_train_mlp
from oceanembed.train.train import run_train

log = logging.getLogger(__name__)

METHODS = ("scratch", "mlp", "ridge", "climatology")
LABELS = {
    "scratch": "OceanEmbed (no pretraining)",
    "mlp": "Per-pixel MLP",
    "ridge": "Ridge regression",
    "climatology": "Climatology",
}
DEFAULT_SEEDS = (0, 1, 2)
LEARNING_YEARS = (2, 5)  # the headline model on the last N years (the full period is the main run)
EVAL_FILE = r1.EVAL_FILE
DONE_FILE = r1.DONE_FILE
ARGO_DIR = "argo"
ARGO_SUMMARY = "summary.json"
ARGO_MATCHUPS = "matchups.parquet"
ARGO_METHODS = ("model", "ridge", "clim", "glorys")  # matchup columns


@dataclass(frozen=True)
class Job:
    kind: str  # "model" (train / fit + score) or "argo"
    method: str  # scratch | mlp | ridge | climatology | argo
    seed: int = 0
    years: int | None = None  # train on the last N years only (learning curve); None = all

    @property
    def name(self) -> str:
        if self.kind == "argo":
            return ARGO_DIR
        return self.method if self.years is None else f"{self.method}_ty{self.years}"

    def __str__(self) -> str:
        return self.name if self.kind == "argo" else f"{self.name} seed {self.seed}"


def r2_dir(cfg: Config) -> Path:
    return cfg.outputs_dir / "research" / "r2"


def job_dir(cfg: Config, job: Job) -> Path:
    base = r2_dir(cfg) / job.name
    return base if job.kind == "argo" else base / f"seed{job.seed}"


def is_done(cfg: Config, job: Job) -> bool:
    d = job_dir(cfg, job)
    need = ARGO_SUMMARY if job.kind == "argo" else EVAL_FILE
    return (d / DONE_FILE).exists() and (d / need).exists()


def plan_jobs(
    seeds: list[int],
    methods: list[str] | None = None,
    learning_curve: bool = True,
    argo: bool = True,
) -> list[Job]:
    """Cheap deterministic methods first, then seed 0 of the trained methods, the Argo scoring (it
    needs only seed 0), the learning curve, and the remaining seeds -- the most decision-relevant
    results arrive first."""
    wanted = list(methods) if methods else list(METHODS)
    bad = [m for m in wanted if m not in METHODS]
    if bad:
        raise ValueError(f"unknown method(s) {bad}; choose from {list(METHODS)}")
    jobs = [Job("model", m) for m in ("ridge", "climatology") if m in wanted]
    first = seeds[0] if seeds else 0
    jobs += [Job("model", m, first) for m in ("scratch", "mlp") if m in wanted]
    if argo and {"scratch", "ridge", "climatology"} <= set(wanted):
        jobs.append(Job("argo", "argo", first))
    if learning_curve and "scratch" in wanted:
        jobs += [Job("model", "scratch", first, y) for y in LEARNING_YEARS]
    for s in seeds[1:]:
        jobs += [Job("model", m, s) for m in ("scratch", "mlp") if m in wanted]
    return jobs


def _say(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------------------
def _predictor(cfg: Config, ctx: r1._Context, job: Job, d: Path) -> tuple[Predictor, dict]:
    if job.method == "ridge":
        return RidgeBaseline.load(d / "ridge.joblib"), {"checkpoint": str(d / "ridge.joblib")}
    if job.method == "climatology":
        return ClimatologyBaseline(cfg.model.n_depths), {}
    return r1._predictor(cfg, ctx, job.method, d)


def train_days(ds, years: int | None, train_end) -> int:
    """Number of days of the last ``years`` calendar years of the train split (all when None)."""
    if years is None:
        return len(ds)
    first = pd.Timestamp(train_end) - pd.DateOffset(years=years) + pd.Timedelta(days=1)
    return int((ds.dates() >= first).sum())


def _fit(cfg: Config, job: Job, ctx: r1._Context, d: Path, progress: bool) -> dict:
    train_ds, val_ds = ctx.train_val()
    if job.years is not None:
        train_ds = train_ds.tail(train_days(train_ds, job.years, cfg.split.train.end))
    if job.method == "ridge":
        ridge = fit_ridge(cfg, train_ds)
        ridge.save(d / "ridge.joblib")
        return {"n_samples": ridge.meta["n_samples"], "n_train_days": len(train_ds)}
    if job.method == "mlp":
        best = run_train_mlp(
            cfg,
            job.seed,
            str(ctx.dev),
            progress,
            ckpt_path=d / "mlp.pt",
            log_path=d / "train.jsonl",
            datasets=(train_ds, val_ds),
        )
    else:
        best = run_train(
            cfg,
            pretrained=False,
            tag=job.name,
            device=str(ctx.dev),
            progress=progress,
            seed=job.seed,
            ckpt_path=d / "recon.pt",
            log_path=d / "train.jsonl",
            datasets=(train_ds, val_ds),
        )
    return {**best, "n_train_days": len(train_ds)}


def _model_job(cfg: Config, job: Job, ctx: r1._Context, d: Path, progress: bool) -> dict:
    info: dict = {"kind": "model", "method": job.method, "label": LABELS[job.method]}
    info |= {"seed": job.seed, "train_years": job.years}
    trained = job.method != "climatology"
    marker = d / "train.done.json"
    if trained and marker.exists():
        info["train"] = json.loads(marker.read_text(encoding="utf-8"))
    elif trained:
        r1._remove(d / "recon.pt", d / "mlp.pt", d / "ridge.joblib", d / "train.jsonl")
        t0 = time.time()
        info["train"] = {**_fit(cfg, job, ctx, d, progress), "seconds": time.time() - t0}
        r1._write_json(marker, info["train"])
    predictor, meta = _predictor(cfg, ctx, job, d)
    ds, _ = ctx.test()
    t0 = time.time()
    raw, anom = score_passes(ctx, predictor, [lambda a, b: ds.batch(a, b)], progress=progress)
    _save_npz(d / EVAL_FILE, raw[0], anom[0], ctx)
    info["score"] = {**meta, "seconds": time.time() - t0}
    return info


# ----------------------------------------------------------------------------------------
# Argo
# ----------------------------------------------------------------------------------------
def argo_months_available(cfg: Config, start, end) -> list[str]:
    """The months of ``[start, end]`` without a stored Argo file (R2 never downloads)."""
    first = [f for f, _ in months(pd.Timestamp(start).date(), pd.Timestamp(end).date())]
    return [f"{f:%Y-%m}" for f in first if not raw_file(cfg, "argo", f).exists()]


def _bootstrap_rmse(per_profile: dict[str, tuple[np.ndarray, np.ndarray]], n_boot: int, seed: int):
    """Profile bootstrap of the pooled RMSE: ``{method: (point, lo, hi)}`` and the paired
    difference of every method with the model, from per-profile ``(n, se2)`` of one sample."""
    keys = list(per_profile)
    n_prof = len(per_profile[keys[0]][0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_prof, size=(n_boot, n_prof))
    out = {}
    boots = {}
    for k, (n, se2) in per_profile.items():
        pt = float(np.sqrt(se2.sum() / max(n.sum(), 1)))
        bt = np.sqrt(se2[idx].sum(axis=1) / np.maximum(n[idx].sum(axis=1), 1))
        boots[k] = (pt, bt)
        lo, hi = np.percentile(bt, [2.5, 97.5])
        out[k] = {"rmse": pt, "ci_lo": float(lo), "ci_hi": float(hi)}
    paired = {}
    if "model" in boots:
        for k in keys:
            if k == "model":
                continue
            d = boots["model"][1] - boots[k][1]
            lo, hi = np.percentile(d, [2.5, 97.5])
            paired[f"model-{k}"] = {
                "diff": boots["model"][0] - boots[k][0],
                "ci_lo": float(lo),
                "ci_hi": float(hi),
                "excludes_zero": bool(lo > 0 or hi < 0),
            }
    return out, paired


def _argo_job(cfg: Config, job: Job, ctx: r1._Context, d: Path, progress: bool) -> dict:
    """Score the Argo profiles of the test period with the headline model (seed ``job.seed``),
    ridge, climatology and GLORYS, per test year; matchups to ``matchups.parquet``."""
    start, end = split_bounds(cfg, "test")
    missing = argo_months_available(cfg, start, end)
    if missing:
        raise FileNotFoundError(f"no stored Argo file for {missing}; R2 does not download them")
    ds, zds = ctx.test()
    grid, days, depths = ctx.grid, ds.dates(), ds.depth
    sources = {
        "model": job_dir(cfg, Job("model", "scratch", job.seed)),
        "ridge": job_dir(cfg, Job("model", "ridge")),
    }
    for k, p in sources.items():
        if not is_done(cfg, Job("model", "scratch" if k == "model" else k, job.seed)):
            raise FileNotFoundError(f"{p}: the {k} model has not been run (run R2 first)")
    preds = {
        "model": _predictor(cfg, ctx, Job("model", "scratch", job.seed), sources["model"])[0],
        "ridge": _predictor(cfg, ctx, Job("model", "ridge"), sources["ridge"])[0],
    }
    raw = load_argo_period(cfg, start, end)
    t = pd.to_datetime(raw["time"])
    raw = raw[
        ((t >= pd.Timestamp(start)) & (t < pd.Timestamp(end) + pd.Timedelta(days=1))).to_numpy()
    ]
    prof, obs = profile_table(raw, depths)
    day_idx, j, k, status = collocate(
        prof["time"], prof["lat"], prof["lon"], grid, days, ds.mask[0]
    )
    ok_idx = np.flatnonzero(status == "ok")
    n_p, n_d = len(ok_idx), len(depths)
    cols = {c: np.full((n_p, n_d), np.nan, np.float32) for c in (*preds, "glorys", "clim")}
    if n_p:
        pj, pk, pdi = j[ok_idx], k[ok_idx], day_idx[ok_idx]
        for di in np.unique(pdi):
            sel = np.flatnonzero(pdi == di)
            batch = ds.batch(int(di), int(di) + 1)
            clim_day = ds.stats.climatology(days[int(di) : int(di) + 1].values)[0]
            cols["clim"][sel] = clim_day[:, pj[sel], pk[sel]].T
            cols["glorys"][sel] = read_target(zds, ds, int(di), int(di) + 1)[0][
                :, pj[sel], pk[sel]
            ].T
            for name, pr in preds.items():
                cols[name][sel] = predict_batch(pr, batch, ds, ctx.dev)[0][:, pj[sel], pk[sel]].T
    obs_ok = obs[ok_idx] if n_p else np.empty((0, n_d))
    mask_pt = ds.mask[:, j[ok_idx], k[ok_idx]].T if n_p else np.zeros((0, n_d), bool)
    valid = np.isfinite(obs_ok) & mask_pt
    for c in cols.values():
        valid &= np.isfinite(c)
    pi, di_ = np.nonzero(valid)
    prof_ok = prof.iloc[ok_idx].reset_index(drop=True)
    basins = basin_of(j[ok_idx], k[ok_idx], grid) if n_p else np.array([], dtype=str)
    table = pd.DataFrame(
        {
            "profile_id": prof_ok["profile_id"].to_numpy()[pi],
            "time": prof_ok["time"].to_numpy()[pi],
            "lat": prof_ok["lat"].to_numpy()[pi],
            "lon": prof_ok["lon"].to_numpy()[pi],
            "depth": depths[di_].astype(np.float32),
            "obs": obs_ok[pi, di_].astype(np.float32),
            **{c: cols[c][pi, di_] for c in ("model", "ridge", "clim", "glorys")},
            "basin": basins[pi],
        }
    )
    d.mkdir(parents=True, exist_ok=True)
    table.to_parquet(d / ARGO_MATCHUPS, index=False)
    year = pd.DatetimeIndex(table["time"]).year
    result: dict = {
        "depths": [float(z) for z in depths],
        "methods": list(ARGO_METHODS),
        "years": {},
    }
    for label, sub_mask in [
        *((str(y), year == y) for y in sorted(set(year))),
        ("pooled", np.ones(len(table), bool)),
    ]:
        sub = table[np.asarray(sub_mask)]
        entry = {
            "n_profiles": int(sub["profile_id"].nunique()),
            "n_matchups": int(len(sub)),
            "methods": {c: _argo_block(sub, c, depths) for c in ARGO_METHODS},
            "per_basin": {
                b: {c: _argo_block(sub[sub["basin"] == b], c, depths) for c in ARGO_METHODS}
                for b in ("arabian_sea", "bay_of_bengal")
            },
        }
        s50 = sub[(sub["depth"] >= 50) & (sub["depth"] <= 200)]
        if len(s50):
            per = {}
            for c in ARGO_METHODS:
                g = s50.assign(e2=(s50[c] - s50["obs"]) ** 2).groupby("profile_id")["e2"]
                per[c] = (g.size().to_numpy(float), g.sum().to_numpy(float))
            ci, paired = _bootstrap_rmse(per, 1000, 0)
            entry["pooled_50_200m_bootstrap"] = {"rmse": ci, "paired": paired}
        result["years"][label] = entry
    result["n_profiles_collocated"] = int(n_p)
    result["interpolation"] = "as validate-argo (eval/argo_validation.py); same cell, same UTC day"
    result["note"] = (
        "GLORYS assimilates Argo and the model is trained on GLORYS: not an independent "
        "error estimate; GLORYS-vs-Argo is the reference floor."
    )
    r1._write_json(d / ARGO_SUMMARY, result)
    return {"kind": "argo", "n_profiles": int(table["profile_id"].nunique()), "score": {}}


# ----------------------------------------------------------------------------------------
# runner
# ----------------------------------------------------------------------------------------
def run_job(cfg: Config, job: Job, ctx: r1._Context, progress: bool = True) -> dict:
    d = job_dir(cfg, job)
    d.mkdir(parents=True, exist_ok=True)
    r1._remove(d / DONE_FILE, d / "eval.tmp.npz", d / EVAL_FILE, d / ARGO_SUMMARY)
    info = (
        _argo_job(cfg, job, ctx, d, progress)
        if job.kind == "argo"
        else _model_job(cfg, job, ctx, d, progress)
    )
    info["created"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    r1._write_json(d / DONE_FILE, info)
    return info


def run_r2(
    cfg: Config,
    seeds: list[int] | None = None,
    methods: list[str] | None = None,
    learning_curve: bool = True,
    argo: bool = True,
    skip_existing: bool = True,
    device: str | None = None,
    progress: bool = True,
) -> list[dict]:
    """Run every planned job (resumable); one ``{job, status, seconds}`` each. Stops at the first
    failure; re-running resumes."""
    jobs = plan_jobs(list(DEFAULT_SEEDS if seeds is None else seeds), methods, learning_curve, argo)
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
        _say(f"{head} -> {d}")
        if not skip_existing:
            r1._remove(*d.glob("*.done.json"))
        t0 = time.time()
        info = run_job(cfg, job, ctx, progress)
        dt = time.time() - t0
        tr = info.get("train", {})
        _say(
            f"{head}: done in {dt:.0f}s"
            + (
                f" (best epoch {tr['epoch']}, val RMSE {tr['val_rmse']:.3f} degC)"
                if "epoch" in tr
                else ""
            )
            + f"; total {time.time() - t_all:.0f}s"
        )
        results.append({"job": str(job), "status": "done", "seconds": dt})
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results
