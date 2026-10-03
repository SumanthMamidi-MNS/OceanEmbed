"""R3 analysis: retrain-without ablations, temporal context and permutation importance.

Reads the ``eval.npz`` / ``perm.npz`` files written by :mod:`oceanembed.research.r3` (and the R1
climatology and full-input models, which are the references) and writes
``outputs/<run>/research/r3/summary.json``, ``summary.md`` and ``figures/*.png``.

Conventions are those of R1 (:mod:`oceanembed.research.r1_report`): the moving-block bootstrap
over test days with replicates shared by every method, depth, basin and metric; a paired
difference ``A - B`` is formed inside each replicate from the seed-mean per-day sums of the two
methods; and a difference is called *established* only when its paired interval excludes zero
**and** the per-seed RMSE ranges of the two methods do not overlap.

* ``delta`` of an experiment is ``RMSE(experiment) - RMSE(full-input model of the same seeds)``:
  positive = the removal / the change made the model worse. The full-input reference is the R1
  ``scratch`` / ``mlp`` result of the seeds the experiment has.
* The permutation importance of a group is ``RMSE(permuted) - RMSE(unpermuted)`` of the same
  trained models (seed-mean sums for the interval, per-seed values for the seed spread).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from scipy import stats as scistats  # noqa: E402

from oceanembed.config import Config  # noqa: E402
from oceanembed.eval.evaluate import write_json  # noqa: E402
from oceanembed.eval.metrics import SUM_FIELDS  # noqa: E402
from oceanembed.research import r1  # noqa: E402
from oceanembed.research.bootstrap import (  # noqa: E402
    column_selector,
    decorrelation_time,
    metrics_from_daily,
    paired_difference,
    percentile_ci,
)
from oceanembed.research.common import choose_block_length, make_counts, md_table  # noqa: E402
from oceanembed.research.inputs import (  # noqa: E402
    EXPERIMENT_LABELS,
    GROUP_LABELS,
    GROUPS,
    HISTORY_EXPERIMENTS,
    REDUCED_EXPERIMENTS,
    REMOVAL_EXPERIMENTS,
)
from oceanembed.research.r3 import (  # noqa: E402
    DONE_FILE,
    EVAL_FILE,
    MODEL_LABELS,
    MODELS,
    PERM_FILE,
    r3_dir,
)

log = logging.getLogger(__name__)

POOLED_RANGE = (50.0, 200.0)
POOLED = f"pooled_{POOLED_RANGE[0]:g}_{POOLED_RANGE[1]:g}m"
DEEP_RANGE = (500.0, 1000.0)
DEEP = f"pooled_{DEEP_RANGE[0]:g}_{DEEP_RANGE[1]:g}m"
REGIONS = r1.REGIONS
REGION_LABELS = {
    "all": "whole domain",
    "arabian_sea": "Arabian Sea",
    "bay_of_bengal": "Bay of Bengal",
}
KEPT_METRICS = ("rmse", "skill_vs_clim", "corr_anom", "bias")
DPI = 130
GROUP_COLORS = {  # Okabe-Ito
    "sst": "#D55E00",
    "sss": "#0072B2",
    "sla": "#009E73",
    "currents": "#CC79A7",
    "winds": "#E69F00",
}
K_COLORS = {1: "#555555", 3: "#0072B2", 7: "#D55E00"}


# ----------------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------------
def _seed_dirs(base: Path, marker: str) -> list[Path]:
    return sorted(
        (
            p
            for p in base.glob("seed*")
            if p.name[4:].isdigit() and (p / DONE_FILE).exists() and (p / marker).exists()
        ),
        key=lambda p: int(p.name[4:]),
    )


def _load_runs(dirs: list[Path]) -> dict | None:
    if not dirs:
        return None
    evals = [r1.load_eval(p / EVAL_FILE) for p in dirs]
    return {
        "seeds": [int(p.name[4:]) for p in dirs],
        "raw": np.stack([e["raw"] for e in evals]),
        "anom": np.stack([e["anom"] for e in evals]),
        "info": [json.loads((p / DONE_FILE).read_text(encoding="utf-8")) for p in dirs],
        "dates": [str(x) for x in evals[0]["dates"]],
        "depths": evals[0]["depths"],
    }


def load_results(cfg: Config) -> dict:
    """``{"climatology": run, "runs": {(model, experiment): run}, "perm": {model: perm}}``.

    ``runs[(model, "full")]`` is the R1 ``scratch`` / ``mlp`` result (the R3 ``<model>_full``
    reproduction when R1 has none). Raises when the climatology or every reference is missing, or
    when the evaluated samples differ between jobs."""
    r1_base, base = r1.r1_dir(cfg), r3_dir(cfg)
    clim = _load_runs(_seed_dirs(r1_base / "climatology", EVAL_FILE))
    if clim is None:
        raise FileNotFoundError(
            f"no finished climatology job under {r1_base}; run `oceanembed research r1` first"
        )
    runs: dict[tuple[str, str], dict] = {}
    for model in MODELS:
        full = _load_runs(_seed_dirs(r1_base / model, EVAL_FILE)) or _load_runs(
            _seed_dirs(base / f"{model}_full", EVAL_FILE)
        )
        if full is not None:
            runs[model, "full"] = full
        for exp in [*REMOVAL_EXPERIMENTS, *HISTORY_EXPERIMENTS, *REDUCED_EXPERIMENTS]:
            run = _load_runs(_seed_dirs(base / f"{model}_{exp}", EVAL_FILE))
            if run is not None:
                runs[model, exp] = run
    if not any(k[1] != "full" for k in runs):
        raise FileNotFoundError(f"no finished R3 retraining job under {base}")
    perm: dict[str, dict] = {}
    for model in MODELS:
        dirs = _seed_dirs(base / f"perm_{model}", PERM_FILE)
        if not dirs:
            continue
        zs = []
        for p in dirs:
            with np.load(p / PERM_FILE, allow_pickle=False) as z:
                zs.append({k: z[k] for k in z.files})
        perm[model] = {
            "seeds": [int(p.name[4:]) for p in dirs],
            "raw": np.stack([z["raw"] for z in zs]),  # (S, P, R, F, T, D)
            "passes": [str(x) for x in zs[0]["passes"]],
            "repeats": int(zs[0]["repeats"]),
            "dates": [str(x) for x in zs[0]["dates"]],
            "depths": zs[0]["depths"],
        }
    n_idx = SUM_FIELDS.index("n")
    ref_n = clim["raw"][0:1, :, n_idx]
    for key, r in [*runs.items(), *((("perm", m), p) for m, p in perm.items())]:
        if r["dates"] != clim["dates"] or not np.array_equal(r["depths"], clim["depths"]):
            raise ValueError(f"{key}: evaluated days / depths differ from the climatology's")
        n = r["raw"][..., n_idx, :, :] if key[0] == "perm" else r["raw"][:, :, n_idx]
        if not np.allclose(n, ref_n if key[0] != "perm" else ref_n[:, None]):
            raise ValueError(f"{key}: the evaluated sample (n per day, depth, region) differs")
    return {"climatology": clim, "runs": runs, "perm": perm}


# ----------------------------------------------------------------------------------------
# computation
# ----------------------------------------------------------------------------------------
def columns(depths) -> tuple[list[str], np.ndarray]:
    """R1's columns (every depth, pooled 50-200 m, overall) plus pooled 500-1000 m (the depths
    where no model has skill in R1)."""
    names, sel = column_selector(depths, POOLED_RANGE)
    d = np.asarray(depths, dtype=float)
    deep = ((d >= DEEP_RANGE[0]) & (d <= DEEP_RANGE[1])).astype(float)
    return [*names, DEEP], np.vstack([sel, deep[None]])


def _bundle(run: dict, idx: list[int], c_raw: np.ndarray, sel, counts, ones) -> dict:
    """Point estimates, bootstrap replicates and per-seed values of every region for the seeds
    ``idx`` (seed-mean per-day sums)."""
    out = {}
    for ri, region in enumerate(REGIONS):
        mean_raw = run["raw"][idx][:, ri].mean(axis=0)
        mean_anom = run["anom"][idx][:, ri].mean(axis=0)
        pt = metrics_from_daily(mean_raw, mean_anom, c_raw[ri], sel, ones)
        bt = metrics_from_daily(mean_raw, mean_anom, c_raw[ri], sel, counts)
        per = [
            metrics_from_daily(run["raw"][s, ri], run["anom"][s, ri], c_raw[ri], sel, ones)
            for s in idx
        ]
        out[region] = {
            "pt": {m: pt[m] for m in KEPT_METRICS},
            "bt": {m: bt[m] for m in KEPT_METRICS},
            "seed": {m: np.concatenate([p[m] for p in per]) for m in KEPT_METRICS},
        }
    return out


def _f(x) -> float | None:
    return float(x) if np.isfinite(x) else None


def _stat(b: dict, metric: str, ci: int) -> dict:
    """Seed spread and bootstrap interval of one metric in one column of a bundle region."""
    sv = b["seed"][metric][:, ci]
    boot = b["bt"][metric][:, ci]
    lo, hi = percentile_ci(boot[:, None]) if np.isfinite(boot).any() else (np.nan, np.nan)
    sd = float(sv.std(ddof=1)) if len(sv) > 1 else float("nan")
    return {
        "seed_mean": _f(sv.mean()),
        "seed_sd": _f(sd),
        "seed_min": _f(sv.min()),
        "seed_max": _f(sv.max()),
        "seed_values": [_f(v) for v in sv],
        "point": _f(b["pt"][metric][0, ci]),
        "ci_lo": _f(np.ravel(lo)[0]),
        "ci_hi": _f(np.ravel(hi)[0]),
    }


def _delta(a: dict, b: dict, metric: str, ci: int, with_seeds: bool = True) -> dict:
    """``A - B`` of a metric in one column: paired bootstrap interval, plus the seed comparison
    (per-seed values paired by position, ranges overlap, Welch p) for RMSE."""
    pdiff = paired_difference(
        a["bt"][metric][:, ci : ci + 1],
        b["bt"][metric][:, ci : ci + 1],
        a["pt"][metric][0, ci : ci + 1],
        b["pt"][metric][0, ci : ci + 1],
    )
    ok = bool(np.isfinite(pdiff["diff"][0]))
    cell = {
        "diff": _f(pdiff["diff"][0]) if ok else None,
        "ci_lo": _f(pdiff["ci_lo"][0]) if ok else None,
        "ci_hi": _f(pdiff["ci_hi"][0]) if ok else None,
        "excludes_zero": bool(pdiff["excludes_zero"][0]) if ok else None,
        "p_two_sided": _f(pdiff["p_two_sided"][0]) if ok else None,
    }
    if with_seeds:
        va, vb = a["seed"][metric][:, ci], b["seed"][metric][:, ci]
        welch = None
        if len(va) > 1 and len(vb) > 1 and (va.std() > 0 or vb.std() > 0):
            welch = _f(scistats.ttest_ind(va, vb, equal_var=False).pvalue)
        overlap = bool(va.max() >= vb.min() and vb.max() >= va.min())
        cell |= {
            "seed_mean_diff": _f(va.mean() - vb.mean()),
            "ranges_overlap": overlap,
            "welch_p": welch,
            "established": bool(cell["excludes_zero"] and not overlap) if ok else None,
        }
    return cell


def _daily_pooled_mse(raw_region: np.ndarray, depths: np.ndarray) -> np.ndarray:
    pooled = (depths >= POOLED_RANGE[0]) & (depths <= POOLED_RANGE[1])
    se2 = raw_region[SUM_FIELDS.index("se2")][:, pooled].sum(axis=1)
    n = raw_region[SUM_FIELDS.index("n")][:, pooled].sum(axis=1)
    return se2 / np.maximum(n, 1)


def build_summary(
    res: dict, n_boot: int = 2000, block_length: int | None = None, seed: int = 0
) -> dict:
    clim, runs, perm = res["climatology"], res["runs"], res["perm"]
    depths = np.asarray(clim["depths"], dtype=float)
    n_days = len(clim["dates"])
    col_names, sel = columns(depths)
    ones = np.ones((1, n_days))
    c_raw = clim["raw"][0]  # (R, F, T, D)

    series = [_daily_pooled_mse(clim["raw"][0, 0], depths)]
    series += [
        _daily_pooled_mse(runs[m, "full"]["raw"][:, 0].mean(axis=0), depths)
        for m in MODELS
        if (m, "full") in runs
    ]
    length, auto = choose_block_length(series, n_days, block_length)
    length = length or auto
    counts = make_counts(n_days, length, n_boot, seed)
    decor = {
        f"{m}_full": decorrelation_time(s)
        for m, s in zip([m for m in MODELS if (m, "full") in runs], series[1:], strict=True)
    }

    # --- retraining experiments against the full-input reference of the same seeds ----------
    experiments: dict = {}
    bundles: dict[tuple, dict] = {}  # (model, experiment, seeds tuple) -> bundle
    contrast_bundles: dict[tuple, dict] = {}

    def bundle(model: str, exp: str, seeds: list[int]) -> dict:
        key = (model, exp, tuple(seeds))
        if key not in bundles:
            run = runs[model, exp]
            idx = [run["seeds"].index(s) for s in seeds]
            bundles[key] = _bundle(run, idx, c_raw, sel, counts, ones)
        return bundles[key]

    for model in MODELS:
        if (model, "full") not in runs:
            continue
        full = runs[model, "full"]
        entry: dict = {"label": MODEL_LABELS[model], "full_seeds": full["seeds"], "experiments": {}}
        for exp in [*REMOVAL_EXPERIMENTS, *HISTORY_EXPERIMENTS, *REDUCED_EXPERIMENTS]:
            if (model, exp) not in runs:
                continue
            run = runs[model, exp]
            seeds = [s for s in run["seeds"] if s in full["seeds"]]
            if not seeds:
                continue
            e, f = bundle(model, exp, seeds), bundle(model, "full", seeds)
            tr = [i.get("train", {}) for i in run["info"]]
            out = {
                "label": EXPERIMENT_LABELS[exp],
                "seeds": seeds,
                "n_seeds": len(seeds),
                "keep": run["info"][0].get("keep"),
                "history": run["info"][0].get("history"),
                "best_epoch": [t.get("epoch") for t in tr],
                "epochs_run": [t.get("epochs_run") for t in tr],
                "val_rmse": [t.get("val_rmse") for t in tr],
                "regions": {},
            }
            for region in REGIONS:
                cols = {}
                for ci, col in enumerate(col_names):
                    cols[col] = {
                        "rmse": _stat(e[region], "rmse", ci),
                        "skill_vs_clim": _stat(e[region], "skill_vs_clim", ci),
                        "reference_rmse": _stat(f[region], "rmse", ci),
                        "reference_skill_vs_clim": _stat(f[region], "skill_vs_clim", ci),
                        "delta": _delta(e[region], f[region], "rmse", ci),
                        "delta_skill": _delta(
                            e[region], f[region], "skill_vs_clim", ci, with_seeds=False
                        ),
                    }
                out["regions"][region] = {"columns": cols}
            entry["experiments"][exp] = out
        experiments[model] = entry

    # --- does history change the basin contrast with the per-pixel MLP? -----------------------
    def contrast_bundle(model: str, exp: str) -> dict:
        key = (model, exp)
        if key not in contrast_bundles:
            run = runs[model, exp]
            contrast_bundles[key] = _bundle(
                run, list(range(len(run["seeds"]))), c_raw, sel, counts, ones
            )
        return contrast_bundles[key]

    ci_pool = col_names.index(POOLED)
    contrast: dict = {}
    if all((m, "full") in runs for m in MODELS):
        for k, exp in [(1, "full"), *((v, e) for e, v in HISTORY_EXPERIMENTS.items())]:
            if not all((m, exp) in runs for m in MODELS):
                continue
            tb, mb = contrast_bundle("scratch", exp), contrast_bundle("mlp", exp)
            contrast[str(k)] = {
                "seeds": {m: runs[m, exp]["seeds"] for m in MODELS},
                "regions": {
                    region: _delta(mb[region], tb[region], "rmse", ci_pool, with_seeds=False)
                    for region in REGIONS
                },
            }
        base = contrast.get("1")
        if base:
            for k, c in contrast.items():
                if k == "1":
                    continue
                c["change_vs_k1"] = {}
                for region in REGIONS:
                    tk, mk = (
                        contrast_bundle("scratch", f"hist{k}"),
                        contrast_bundle("mlp", f"hist{k}"),
                    )
                    t1, m1 = contrast_bundle("scratch", "full"), contrast_bundle("mlp", "full")
                    # (MLP - Transformer) at k minus the same at k = 1, inside each replicate
                    ak = {
                        "bt": {"rmse": mk[region]["bt"]["rmse"] - tk[region]["bt"]["rmse"]},
                        "pt": {"rmse": mk[region]["pt"]["rmse"] - tk[region]["pt"]["rmse"]},
                    }
                    a1 = {
                        "bt": {"rmse": m1[region]["bt"]["rmse"] - t1[region]["bt"]["rmse"]},
                        "pt": {"rmse": m1[region]["pt"]["rmse"] - t1[region]["pt"]["rmse"]},
                    }
                    c["change_vs_k1"][region] = _delta(ak, a1, "rmse", ci_pool, with_seeds=False)

    # --- permutation importance --------------------------------------------------------------
    permutation: dict = {}
    for model, p in perm.items():
        names = p["passes"]
        raw = p["raw"]  # (S, P, R, F, T, D)
        n_seeds = raw.shape[0]
        groups = {}
        for gi, g in enumerate(names[1:], start=1):
            regions = {}
            for ri, region in enumerate(REGIONS):

                def rm(arr, ri=ri):
                    return metrics_from_daily(arr, arr, c_raw[ri], sel, ones)["rmse"]

                def rm_b(arr, ri=ri):
                    return metrics_from_daily(arr, arr, c_raw[ri], sel, counts)["rmse"]

                base_mean = raw[:, 0, ri].mean(axis=0)
                perm_mean = raw[:, gi, ri].mean(axis=0)
                pt_d = rm(perm_mean) - rm(base_mean)
                bt_d = rm_b(perm_mean) - rm_b(base_mean)
                lo, hi = percentile_ci(bt_d)
                seed_d = np.concatenate(
                    [rm(raw[s, gi, ri]) - rm(raw[s, 0, ri]) for s in range(n_seeds)]
                )
                cols = {}
                for ci, col in enumerate(col_names):
                    sd = float(seed_d[:, ci].std(ddof=1)) if n_seeds > 1 else float("nan")
                    cols[col] = {
                        "diff": _f(pt_d[0, ci]),
                        "ci_lo": _f(lo[ci]),
                        "ci_hi": _f(hi[ci]),
                        "excludes_zero": bool(lo[ci] > 0 or hi[ci] < 0),
                        "seed_mean": _f(seed_d[:, ci].mean()),
                        "seed_sd": _f(sd),
                        "seed_values": [_f(v) for v in seed_d[:, ci]],
                        "baseline_rmse": _f(rm(base_mean)[0, ci]),
                    }
                regions[region] = {"columns": cols}
            groups[g] = {"label": GROUP_LABELS[g], "regions": regions}
        permutation[model] = {
            "label": MODEL_LABELS[model],
            "seeds": p["seeds"],
            "repeats": p["repeats"],
            "groups": groups,
        }

    ranking = _ranking(experiments, permutation, col_names)

    return {
        "settings": {
            "reference": "GLORYS reanalysis, harmonised to the 0.25 degree grid",
            "n_days": n_days,
            "start": clim["dates"][0],
            "end": clim["dates"][-1],
            "depths": [float(z) for z in depths],
            "columns": col_names,
            "pooled_range_m": list(POOLED_RANGE),
            "deep_range_m": list(DEEP_RANGE),
            "regions": list(REGIONS),
            "bootstrap": {
                "kind": "moving-block bootstrap over test days (overlapping blocks, no wrap)",
                "n_replicates": n_boot,
                "block_length_days": length,
                "block_length_auto_days": auto,
                "ci": "95 % percentile interval of the replicates",
                "seed": seed,
                "estimator": "per-day sufficient statistics averaged over the seeds of the "
                "experiment (and of its full-input reference), metrics from the resampled sums",
            },
            "established": "paired interval excludes zero AND the per-seed RMSE ranges of the "
            "two methods do not overlap",
            "permutation": "within the calendar month (random derangement of each month's days; "
            "all channels of a group from the same donor day), averaged over repeats",
        },
        "autocorrelation": {"daily_pooled_mse": decor},
        "experiments": experiments,
        "history_contrast": contrast,
        "permutation": permutation,
        "ranking": ranking,
    }


def _ranks(values: dict[str, float]) -> dict[str, int]:
    """1 = largest increase (most important)."""
    order = sorted(values, key=lambda g: -values[g])
    return {g: i + 1 for i, g in enumerate(order)}


def _ranking(experiments: dict, permutation: dict, col_names: list[str]) -> dict:
    """Retrain-without vs permutation ranking of the five groups in the pooled 50-200 m RMSE."""
    out: dict = {}
    for model in MODELS:
        if model not in experiments or model not in permutation:
            continue
        per_region = {}
        for region in REGIONS:
            retrain, permute = {}, {}
            for g in GROUPS:
                exp = experiments[model]["experiments"].get(f"no_{g}")
                pg = permutation[model]["groups"].get(g)
                if exp is None or pg is None:
                    continue
                d1 = exp["regions"][region]["columns"][POOLED]["delta"]["diff"]
                d2 = pg["regions"][region]["columns"][POOLED]["diff"]
                if d1 is not None and d2 is not None:
                    retrain[g], permute[g] = d1, d2
            if len(retrain) < 3:
                continue
            r1_, r2_ = _ranks(retrain), _ranks(permute)
            rho = float(
                scistats.spearmanr(list(retrain.values()), list(permute.values())).statistic
            )
            per_region[region] = {
                "retrain_delta": retrain,
                "permutation_delta": permute,
                "rank_retrain": r1_,
                "rank_permutation": r2_,
                "spearman": rho if np.isfinite(rho) else None,
                "rank_differs_by_2_or_more": [g for g in retrain if abs(r1_[g] - r2_[g]) >= 2],
            }
        if per_region:
            out[model] = per_region
    return out


# ----------------------------------------------------------------------------------------
# markdown
# ----------------------------------------------------------------------------------------
def _fin(x) -> bool:
    return x is not None and bool(np.isfinite(x))


def _num(x, nd: int = 3) -> str:
    return f"{x:.{nd}f}" if _fin(x) else "n/a"


def _sgn(x, nd: int = 3) -> str:
    return f"{x:+.{nd}f}" if _fin(x) else "n/a"


def _pm(c: dict, nd: int = 3) -> str:
    if not _fin(c.get("seed_mean")):
        return "n/a"
    if not _fin(c.get("seed_sd")):
        return f"{c['seed_mean']:.{nd}f}"
    return f"{c['seed_mean']:.{nd}f} ± {c['seed_sd']:.{nd}f}"


def _delta_text(d: dict, nd: int = 3) -> str:
    if not _fin(d.get("diff")):
        return "n/a"
    star = " *" if d.get("established") else ""
    return f"{d['diff']:+.{nd}f} [{d['ci_lo']:+.{nd}f}, {d['ci_hi']:+.{nd}f}]{star}"


def _col_label(col: str) -> str:
    if col.startswith("pooled_"):
        lo, hi = col[len("pooled_") : -1].split("_")
        return f"{lo}-{hi} m pooled"
    return f"{col} m"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [*md_table(header, rows), ""]


def _verdict(d: dict) -> str:
    if not _fin(d.get("diff")):
        return "n/a"
    if d["excludes_zero"] and not d["ranges_overlap"]:
        return "established"
    if d["excludes_zero"]:
        return "interval excludes 0, seed ranges overlap (not established)"
    return "not distinguishable from zero"


def write_markdown(summary: dict, path: Path, run_name: str) -> None:
    st = summary["settings"]
    bs = st["bootstrap"]
    depths = st["depths"]
    cols = [f"{z:g}" for z in depths]
    L = [
        f"# R3 - which surface variable matters where, and does history help ({run_name})",
        "",
        f"Test period {st['start']} .. {st['end']} ({st['n_days']} days), reference GLORYS. "
        "Intervals: "
        f"{bs['kind']}, {bs['n_replicates']} replicates, blocks of {bs['block_length_days']} days, "
        "95 % percentile interval of the paired difference. `*` marks an *established* difference "
        f"({st['established']}). Delta = experiment minus the full-input model of the same seeds; "
        "positive = worse.",
        "",
    ]
    for entry in summary["experiments"].values():
        L += [f"## {entry['label']}: retrain without a variable group", ""]
        L += [f"Full-input reference seeds: {entry['full_seeds']} (R1).", ""]
        rows = []
        for exp, e in entry["experiments"].items():
            if exp in HISTORY_EXPERIMENTS:
                continue
            c = e["regions"]["all"]["columns"][POOLED]
            ref = c["reference_rmse"]
            cells = [_delta_text(e["regions"][r]["columns"][POOLED]["delta"]) for r in REGIONS]
            rows.append(
                [
                    e["label"],
                    str(e["n_seeds"]),
                    _pm(c["rmse"]),
                    _pm(ref),
                    *cells,
                    _verdict(c["delta"]),
                ]
            )
        L += _table(
            [
                "Experiment",
                "Seeds",
                "RMSE 50-200 m",
                "Full-input RMSE",
                "Delta whole domain [95 % CI]",
                "Delta Arabian Sea",
                "Delta Bay of Bengal",
                "Whole domain verdict",
            ],
            rows,
        )
        for region in REGIONS:
            L += [f"### {entry['label']}: delta RMSE by depth, {REGION_LABELS[region]}", ""]
            exps = [
                x for x in (*REMOVAL_EXPERIMENTS, *REDUCED_EXPERIMENTS) if x in entry["experiments"]
            ]
            rows = []
            for col in [*cols, POOLED, DEEP]:
                row = [_col_label(col)]
                for x in exps:
                    d = entry["experiments"][x]["regions"][region]["columns"][col]["delta"]
                    row.append(
                        "n/a"
                        if not _fin(d["diff"])
                        else f"{d['diff']:+.3f}" + ("*" if d["excludes_zero"] else "")
                    )
                rows.append(row)
            L += [
                "`*` = the paired bootstrap interval excludes zero.",
                "",
                *_table(["Depth", *[EXPERIMENT_LABELS[x] for x in exps]], rows),
            ]
        hist = {k: v for k, v in entry["experiments"].items() if k in HISTORY_EXPERIMENTS}
        if hist:
            L += [f"### {entry['label']}: temporal context", ""]
            rows = []
            for e in hist.values():
                for region in REGIONS:
                    cc = e["regions"][region]["columns"]
                    rows.append(
                        [
                            e["label"],
                            REGION_LABELS[region],
                            _pm(cc[POOLED]["rmse"]),
                            _delta_text(cc[POOLED]["delta"]),
                            _verdict(cc[POOLED]["delta"]),
                            _pm(cc[DEEP]["skill_vs_clim"]),
                            _pm(cc[DEEP]["reference_skill_vs_clim"]),
                            _sgn(cc[DEEP]["delta_skill"]["diff"]),
                        ]
                    )
            L += _table(
                [
                    "History",
                    "Region",
                    "RMSE 50-200 m",
                    "Delta vs k=1 [95 % CI]",
                    "Verdict",
                    "Skill 500-1000 m",
                    "k=1 skill",
                    "Skill change",
                ],
                rows,
            )
            for region in ("all",):
                L += [f"RMSE by depth, {REGION_LABELS[region]} (seed mean; delta vs k = 1):", ""]
                rows = []
                for z in cols:
                    row = [f"{z} m"]
                    first = next(iter(hist.values()))["regions"][region]["columns"][z]
                    row.append(_num(first["reference_rmse"]["seed_mean"]))
                    for e in hist.values():
                        c = e["regions"][region]["columns"][z]
                        row.append(
                            f"{_num(c['rmse']['seed_mean'])} ({_sgn(c['delta']['diff'])}"
                            + ("*" if c["delta"]["excludes_zero"] else "")
                            + ")"
                        )
                    rows.append(row)
                L += _table(["Depth", "k = 1", *[e["label"] for e in hist.values()]], rows)
    if summary["history_contrast"]:
        L += ["## Does history change the Bay of Bengal / Arabian Sea contrast?", ""]
        L += [
            "Advantage of the Transformer over the per-pixel MLP with the same inputs: "
            "RMSE(MLP) minus RMSE(Transformer), pooled 50-200 m (positive = the Transformer is "
            "better). Seed means of every finished seed of each model.",
            "",
        ]
        rows = []
        for k, c in summary["history_contrast"].items():
            row = [f"k = {k}"]
            for region in REGIONS:
                row.append(_delta_text(c["regions"][region]))
            row.append(
                "; ".join(
                    f"{REGION_LABELS[r]} {_delta_text(c['change_vs_k1'][r])}"
                    for r in REGIONS
                    if "change_vs_k1" in c
                )
                or "reference"
            )
            rows.append(row)
        L += _table(
            [
                "History",
                "Whole domain",
                "Arabian Sea",
                "Bay of Bengal",
                "Change of the advantage vs k = 1",
            ],
            rows,
        )
    if summary["permutation"]:
        L += ["## Permutation importance", ""]
        L += [
            "RMSE increase when one variable group is permuted across the test days within the "
            "same "
            f"calendar month ({st['permutation']}); trained R1 models, no retraining.",
            "",
        ]
        for pm in summary["permutation"].values():
            L += [f"### {pm['label']} (seeds {pm['seeds']}, {pm['repeats']} repeats)", ""]
            rows = []
            for gd in pm["groups"].values():
                row = [gd["label"]]
                for region in REGIONS:
                    c = gd["regions"][region]["columns"][POOLED]
                    row.append(
                        f"{c['diff']:+.3f} [{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}]"
                        if _fin(c["diff"])
                        else "n/a"
                    )
                row.append(_pm(gd["regions"]["all"]["columns"][POOLED]))
                rows.append(row)
            L += _table(
                [
                    "Permuted group",
                    "Whole domain 50-200 m",
                    "Arabian Sea",
                    "Bay of Bengal",
                    "Whole domain, seed mean ± SD",
                ],
                rows,
            )
    if summary["ranking"]:
        L += ["## Retrain-without vs permutation ranking (pooled 50-200 m)", ""]
        L += [
            "Rank 1 = the largest RMSE increase. Retraining lets the network re-learn from the "
            "remaining inputs, so a variable that is largely redundant (its information also sits "
            "in other inputs) costs little to remove; permutation breaks the learned link without "
            "retraining, so a redundant variable the network relies on still costs a lot. Rank "
            "differences of two or more are listed.",
            "",
        ]
        for model, per_region in summary["ranking"].items():
            L += [f"### {MODEL_LABELS[model]}", ""]
            rows = []
            for region, rk in per_region.items():
                for g in rk["retrain_delta"]:
                    rows.append(
                        [
                            REGION_LABELS[region],
                            GROUP_LABELS[g],
                            _sgn(rk["retrain_delta"][g]),
                            str(rk["rank_retrain"][g]),
                            _sgn(rk["permutation_delta"][g]),
                            str(rk["rank_permutation"][g]),
                            "differs" if g in rk["rank_differs_by_2_or_more"] else "",
                        ]
                    )
                rows.append(
                    [
                        REGION_LABELS[region],
                        "Spearman rank correlation",
                        "",
                        "",
                        _num(rk["spearman"], 2),
                        "",
                        "",
                    ]
                )
            L += _table(
                ["Region", "Group", "Delta (retrain)", "Rank", "Delta (permute)", "Rank", ""], rows
            )
    L += [
        "Figures: `figures/ablation_heatmap_<model>.png`, `figures/permutation_by_depth.png`, "
        "`figures/history_rmse_by_depth.png`. Numbers: `summary.json`.",
        "",
    ]
    path.write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------
def _depth_axis(ax, depths) -> None:
    ax.set_yscale("function", functions=(lambda d: np.sqrt(np.clip(d, 0, None)), lambda s: s**2))
    ax.set_ylim(max(depths), 0)
    ticks = [z for z in depths if z in (0, 50, 100, 200, 500, 1000)]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{z:g}" for z in ticks])
    ax.set_ylabel("depth (m)")
    ax.grid(alpha=0.25)


def _heatmap(summary: dict, model: str, path: Path) -> None:
    entry = summary["experiments"][model]
    depths = summary["settings"]["depths"]
    exps = [x for x in (*REMOVAL_EXPERIMENTS, *REDUCED_EXPERIMENTS) if x in entry["experiments"]]
    mats, stars = [], []
    for region in REGIONS:
        m = np.full((len(depths), len(exps)), np.nan)
        s = np.zeros_like(m, dtype=bool)
        for j, x in enumerate(exps):
            cc = entry["experiments"][x]["regions"][region]["columns"]
            for i, z in enumerate(depths):
                d = cc[f"{z:g}"]["delta"]
                if _fin(d["diff"]):
                    m[i, j], s[i, j] = d["diff"], bool(d["excludes_zero"])
        mats.append(m)
        stars.append(s)
    vmax = max(0.05, float(np.nanmax(np.abs(np.stack(mats)))))
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2), sharey=True)
    im = None
    for ax, m, s, region in zip(axes, mats, stars, REGIONS, strict=True):
        im = ax.imshow(m, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax))
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if np.isfinite(m[i, j]):
                    ax.text(
                        j, i, f"{m[i, j]:+.2f}" + ("*" if s[i, j] else ""),
                        ha="center", va="center", fontsize=6.5,
                    )  # fmt: skip
        ax.set_xticks(range(len(exps)))
        ax.set_xticklabels(
            [EXPERIMENT_LABELS[x] for x in exps], rotation=40, ha="right", fontsize=8
        )
        ax.set_title(REGION_LABELS[region])
    axes[0].set_yticks(range(len(depths)))
    axes[0].set_yticklabels([f"{z:g}" for z in depths])
    axes[0].set_ylabel("depth (m)")
    fig.colorbar(im, ax=axes, shrink=0.8, label="change in RMSE (°C), retrained - full input")
    fig.suptitle(
        f"{entry['label']}: RMSE increase when the input is removed (* = paired 95 % interval "
        "excludes 0)",
        fontsize=10,
    )
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _permutation_figure(summary: dict, path: Path) -> None:
    perm = summary["permutation"]
    depths = summary["settings"]["depths"]
    fig, axes = plt.subplots(
        len(perm), 3, figsize=(13, 4.6 * len(perm)), sharey=True, squeeze=False
    )
    for row, (_model, pm) in zip(axes, perm.items(), strict=True):
        for ax, region in zip(row, REGIONS, strict=True):
            for g, gd in pm["groups"].items():
                cc = gd["regions"][region]["columns"]
                y = np.array([cc[f"{z:g}"]["diff"] for z in depths], dtype=float)
                lo = np.array([cc[f"{z:g}"]["ci_lo"] for z in depths], dtype=float)
                hi = np.array([cc[f"{z:g}"]["ci_hi"] for z in depths], dtype=float)
                ax.plot(y, depths, color=GROUP_COLORS[g], lw=1.8, label=gd["label"])
                ax.fill_betweenx(depths, lo, hi, color=GROUP_COLORS[g], alpha=0.15, lw=0)
            ax.axvline(0, color="k", lw=0.8)
            _depth_axis(ax, depths)
            ax.set_title(f"{pm['label']}, {REGION_LABELS[region]}", fontsize=9)
            ax.set_xlabel("RMSE increase when permuted (°C)")
    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Permutation importance by depth (within-month permutation; 95 % bands)", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _history_figure(summary: dict, path: Path) -> None:
    depths = summary["settings"]["depths"]
    models = [m for m in MODELS if m in summary["experiments"]]
    fig, axes = plt.subplots(
        len(models), 3, figsize=(13, 4.6 * len(models)), sharey=True, squeeze=False
    )
    for row, model in zip(axes, models, strict=True):
        entry = summary["experiments"][model]["experiments"]
        hist = {HISTORY_EXPERIMENTS[k]: v for k, v in entry.items() if k in HISTORY_EXPERIMENTS}
        for ax, region in zip(row, REGIONS, strict=True):
            for k, e in [(1, None), *sorted(hist.items())]:
                src = next(iter(hist.values())) if e is None else e
                if e is None:
                    cc = src["regions"][region]["columns"]
                    y = [cc[f"{z:g}"]["reference_rmse"]["seed_mean"] for z in depths]
                else:
                    cc = e["regions"][region]["columns"]
                    y = [cc[f"{z:g}"]["rmse"]["seed_mean"] for z in depths]
                ax.plot(
                    np.array(y, dtype=float), depths, color=K_COLORS[k], lw=1.8, label=f"k = {k}"
                )
            _depth_axis(ax, depths)
            ax.set_title(
                f"{summary['experiments'][model]['label']}, {REGION_LABELS[region]}", fontsize=9
            )
            ax.set_xlabel("RMSE vs GLORYS (°C)")
    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.suptitle("RMSE by depth for 1, 3 and 7 days of surface history (seed means)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def write_figures(summary: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for model, entry in summary["experiments"].items():
        if any(x in entry["experiments"] for x in (*REMOVAL_EXPERIMENTS, *REDUCED_EXPERIMENTS)):
            paths.append(out_dir / f"ablation_heatmap_{model}.png")
            _heatmap(summary, model, paths[-1])
    if summary["permutation"]:
        paths.append(out_dir / "permutation_by_depth.png")
        _permutation_figure(summary, paths[-1])
    if any(
        k in e["experiments"] for e in summary["experiments"].values() for k in HISTORY_EXPERIMENTS
    ):
        paths.append(out_dir / "history_rmse_by_depth.png")
        _history_figure(summary, paths[-1])
    return paths


# ----------------------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------------------
def make_r3_report(
    cfg: Config, n_boot: int = 2000, block_length: int | None = None, seed: int = 0
) -> dict:
    """Write ``summary.json``, ``summary.md`` and the figures under ``research/r3/``."""
    res = load_results(cfg)
    summary = build_summary(res, n_boot=n_boot, block_length=block_length, seed=seed)
    out = r3_dir(cfg)
    write_json(out / "summary.json", summary)
    write_markdown(summary, out / "summary.md", cfg.run_name)
    write_figures(summary, out / "figures")
    return summary
