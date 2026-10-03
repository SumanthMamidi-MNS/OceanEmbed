"""R1 analysis: seed spread, block-bootstrap confidence intervals and paired comparisons.

Reads the ``eval.npz`` files written by :mod:`oceanembed.research.r1` and writes
``outputs/<run>/research/r1/summary.json``, ``summary.md`` and ``figures/*.png``.

What is estimated and what is resampled
---------------------------------------
* **Seed spread** -- for every method, each seed's metric is computed from its own per-day sums;
  the table gives mean, sample SD (``n - 1``), min and max over seeds.
* **Test-day sampling uncertainty** -- the per-day sums (``n, sum x, sum y, sum x^2, sum y^2,
  sum xy, sum |e|, sum e^2``) are averaged over seeds (a method's seed-mean daily statistics),
  then a moving-block bootstrap over the test days resamples blocks of consecutive days and the
  metric is recomputed from the resampled sums. The 95 % interval is the 2.5-97.5 percentile range
  of the replicates. It describes how much the seed-mean model's score would change with a
  different draw of test days; seeds are not resampled.
* **Paired comparison** -- ``A - B`` is formed *inside* each replicate, with the same resampled
  days for both methods (the replicates are generated once and shared by everything).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats as scistats  # noqa: E402

from oceanembed.config import Config  # noqa: E402
from oceanembed.eval.evaluate import write_json  # noqa: E402
from oceanembed.eval.metrics import SUM_FIELDS  # noqa: E402
from oceanembed.research.bootstrap import (  # noqa: E402
    METRIC_NAMES,
    auto_block_length,
    block_counts,
    column_selector,
    decorrelation_time,
    metrics_from_daily,
    paired_difference,
    percentile_ci,
    seed_stats,
)
from oceanembed.research.r1 import (  # noqa: E402
    DETERMINISTIC,
    EVAL_FILE,
    LABELS,
    METHODS,
    REGIONS,
    is_done,
    load_eval,
    r1_dir,
)

log = logging.getLogger(__name__)

POOLED_RANGE = (50.0, 200.0)
# (A, B): the difference reported is A - B in RMSE (negative = A is better)
PAIRS = [
    ("oceanembed", "scratch"),
    ("oceanembed", "unet"),
    ("oceanembed", "mlp"),
    ("oceanembed", "ridge"),
    ("oceanembed", "climatology"),
    ("scratch", "unet"),
    ("scratch", "ridge"),
    ("unet", "ridge"),
    ("mlp", "ridge"),
]
KEY_PAIRS = [("oceanembed", "scratch"), ("scratch", "unet"), ("oceanembed", "unet")]
PAIRED_METRICS = ("rmse", "corr_anom", "skill_vs_clim")
SENSITIVITY_LENGTHS = (1, 7, 15, 30, 45)
COLORS = {  # Okabe-Ito, one colour per method in every figure
    "oceanembed": "#0072B2",
    "scratch": "#009E73",
    "unet": "#D55E00",
    "mlp": "#56B4E9",
    "ridge": "#E69F00",
    "climatology": "#555555",
}
PAIR_COLORS = {
    ("oceanembed", "scratch"): "#0072B2",
    ("oceanembed", "unet"): "#D55E00",
    ("scratch", "unet"): "#009E73",
}
REGION_LABELS = {
    "all": "whole domain",
    "arabian_sea": "Arabian Sea",
    "bay_of_bengal": "Bay of Bengal",
}
DPI = 130


# ----------------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------------
def load_results(cfg: Config) -> dict[str, dict]:
    """Every finished ``(method, seed)`` of the run, grouped by method.

    ``{method: {seeds, raw (S,R,F,T,D), anom, info [done.json ...], dates, depths}}``. Raises when
    the climatology is missing or the evaluated samples differ between jobs."""
    base = r1_dir(cfg)
    res: dict[str, dict] = {}
    for m in METHODS:
        dirs = sorted(
            (p for p in (base / m).glob("seed*") if p.name[4:].isdigit() and is_done(p)),
            key=lambda p: int(p.name[4:]),
        )
        if not dirs:
            continue
        evals = [load_eval(p / EVAL_FILE) for p in dirs]
        res[m] = {
            "seeds": [int(p.name[4:]) for p in dirs],
            "raw": np.stack([e["raw"] for e in evals]),
            "anom": np.stack([e["anom"] for e in evals]),
            "info": [json.loads((p / "done.json").read_text(encoding="utf-8")) for p in dirs],
            "dates": [str(x) for x in evals[0]["dates"]],
            "depths": evals[0]["depths"],
        }
    if "climatology" not in res:
        raise FileNotFoundError(
            f"no finished climatology job under {base}; run `oceanembed research r1` first"
        )
    ref = res["climatology"]
    n_idx = SUM_FIELDS.index("n")
    for m, r in res.items():
        if r["dates"] != ref["dates"] or not np.array_equal(r["depths"], ref["depths"]):
            raise ValueError(f"{m}: evaluated days / depths differ from the climatology's")
        if not np.allclose(r["raw"][:, :, n_idx], ref["raw"][0:1, :, n_idx]):
            raise ValueError(f"{m}: the evaluated sample (n per day, depth, region) differs")
    return res


# ----------------------------------------------------------------------------------------
# computation
# ----------------------------------------------------------------------------------------
def _daily_pooled_mse(raw_region: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """``(T,)`` pooled-range MSE per day from ``(F, T, D)`` sums."""
    pooled = (depths >= POOLED_RANGE[0]) & (depths <= POOLED_RANGE[1])
    se2 = raw_region[SUM_FIELDS.index("se2")][:, pooled].sum(axis=1)
    n = raw_region[SUM_FIELDS.index("n")][:, pooled].sum(axis=1)
    return se2 / np.maximum(n, 1)


def _ci_cell(boot_col: np.ndarray) -> tuple[float, float]:
    lo, hi = percentile_ci(boot_col[:, None])
    return float(lo[0]), float(hi[0])


def build_summary(
    res: dict[str, dict],
    n_boot: int = 2000,
    block_length: int | None = None,
    seed: int = 0,
    n_boot_sensitivity: int = 1000,
) -> dict:
    """The full statistical summary (also the content of ``summary.json``)."""
    clim = res["climatology"]
    depths = np.asarray(clim["depths"], dtype=float)
    n_days = len(clim["dates"])
    col_names, sel = column_selector(depths, POOLED_RANGE)
    k_pooled = col_names.index(f"pooled_{POOLED_RANGE[0]:g}_{POOLED_RANGE[1]:g}m")
    ones = np.ones((1, n_days))

    # --- decorrelation of the daily pooled MSE (whole domain, seed-mean sums) -> block length
    daily = {m: _daily_pooled_mse(r["raw"][:, 0].mean(axis=0), depths) for m, r in res.items()}
    decor = {m: decorrelation_time(s) for m, s in daily.items()}
    diff_decor = {}
    for a, b in PAIRS:
        if a in daily and b in daily:
            diff_decor[f"{a}-{b}"] = decorrelation_time(daily[a] - daily[b])
    auto_len = auto_block_length(list(daily.values()))
    length = int(block_length) if block_length else auto_len
    counts = block_counts(n_days, length, n_boot, seed)

    # --- per method and region: point, seeds, bootstrap
    point: dict[tuple, dict] = {}
    boot: dict[tuple, dict] = {}
    seedpt: dict[tuple, dict] = {}
    for m, r in res.items():
        for ri, region in enumerate(REGIONS):
            c_raw = clim["raw"][0, ri]
            mean_raw, mean_anom = r["raw"][:, ri].mean(axis=0), r["anom"][:, ri].mean(axis=0)
            point[m, region] = metrics_from_daily(mean_raw, mean_anom, c_raw, sel, ones)
            boot[m, region] = metrics_from_daily(mean_raw, mean_anom, c_raw, sel, counts)
            per_seed = [
                metrics_from_daily(r["raw"][s, ri], r["anom"][s, ri], c_raw, sel, ones)
                for s in range(len(r["seeds"]))
            ]
            seedpt[m, region] = {k: np.concatenate([p[k] for p in per_seed]) for k in per_seed[0]}

    methods_out: dict = {}
    for m, r in res.items():
        infos = r["info"]
        tr = [i.get("train", {}) for i in infos]
        pre = [i.get("pretrain", {}) for i in infos]
        entry = {
            "label": LABELS[m],
            "n_seeds": len(r["seeds"]),
            "seeds": r["seeds"],
            "deterministic": m in DETERMINISTIC,
            "n_params": tr[0].get("n_params") if tr and tr[0] else None,
            "n_params_encoder": tr[0].get("n_params_encoder") if tr and tr[0] else None,
            "n_params_pretraining_model": pre[0].get("n_params") if pre and pre[0] else None,
            "best_epoch": [t.get("epoch") for t in tr] if tr[0] else None,
            "epochs_run": [t.get("epochs_run") for t in tr] if tr[0] else None,
            "val_rmse": [t.get("val_rmse") for t in tr] if tr[0] else None,
            "regions": {},
        }
        for region in REGIONS:
            cols = {}
            for ci, col in enumerate(col_names):
                cell = {}
                for name in METRIC_NAMES:
                    sv = seedpt[m, region][name][:, ci]
                    st = seed_stats(sv[:, None])
                    lo, hi = _ci_cell(boot[m, region][name][:, ci])
                    cell[name] = {
                        "seed_mean": float(st["mean"][0]),
                        "seed_sd": float(st["sd"][0]),
                        "seed_min": float(st["min"][0]),
                        "seed_max": float(st["max"][0]),
                        "seed_values": [float(v) for v in sv],
                        "point": float(point[m, region][name][0, ci]),
                        "ci_lo": lo,
                        "ci_hi": hi,
                    }
                cols[col] = cell
            entry["regions"][region] = {"columns": cols}
        methods_out[m] = entry

    # --- paired comparisons
    comparisons = []
    for a, b in PAIRS:
        if a not in res or b not in res:
            continue
        comp = {"a": a, "b": b, "difference": "A - B", "regions": {}}
        for region in REGIONS:
            cols = {}
            seed_a, seed_b = seedpt[a, region]["rmse"], seedpt[b, region]["rmse"]
            for ci, col in enumerate(col_names):
                cell = {}
                for name in PAIRED_METRICS:
                    pd_ = paired_difference(
                        boot[a, region][name][:, ci : ci + 1],
                        boot[b, region][name][:, ci : ci + 1],
                        point[a, region][name][0, ci : ci + 1],
                        point[b, region][name][0, ci : ci + 1],
                    )
                    ok = bool(np.isfinite(pd_["diff"][0]))
                    cell[name] = {
                        "diff": float(pd_["diff"][0]) if ok else None,
                        "ci_lo": float(pd_["ci_lo"][0]) if ok else None,
                        "ci_hi": float(pd_["ci_hi"][0]) if ok else None,
                        "excludes_zero": bool(pd_["excludes_zero"][0]) if ok else None,
                        "p_two_sided": float(pd_["p_two_sided"][0]) if ok else None,
                    }
                va, vb = seed_a[:, ci], seed_b[:, ci]
                welch = None
                if len(va) > 1 and len(vb) > 1 and (va.std() > 0 or vb.std() > 0):
                    welch = float(scistats.ttest_ind(va, vb, equal_var=False).pvalue)
                cell["seeds_rmse"] = {
                    "mean_a": float(va.mean()),
                    "mean_b": float(vb.mean()),
                    "mean_diff": float(va.mean() - vb.mean()),
                    "ranges_overlap": bool(va.max() >= vb.min() and vb.max() >= va.min()),
                    "welch_p": welch,
                }
                cols[col] = cell
            comp["regions"][region] = {"columns": cols}
        comparisons.append(comp)

    # --- how the interval depends on the block length (pooled RMSE, whole domain)
    sens_sel = sel[k_pooled : k_pooled + 1]
    sensitivity: dict = {}
    usable = [n for n in SENSITIVITY_LENGTHS if n <= max(1, n_days // 2)]
    for length_s in sorted({*usable, length}):
        cnt = block_counts(n_days, length_s, n_boot_sensitivity, seed + 1)
        b_rmse = {}
        for m, r in res.items():
            b_rmse[m] = metrics_from_daily(
                r["raw"][:, 0].mean(axis=0),
                r["anom"][:, 0].mean(axis=0),
                clim["raw"][0, 0],
                sens_sel,
                cnt,
            )["rmse"]
        row = {"rmse_ci_halfwidth": {}, "diff_ci_halfwidth": {}}
        for m, v in b_rmse.items():
            lo, hi = percentile_ci(v)
            row["rmse_ci_halfwidth"][m] = float((hi[0] - lo[0]) / 2)
        for a, b in KEY_PAIRS:
            if a in b_rmse and b in b_rmse:
                lo, hi = percentile_ci(b_rmse[a] - b_rmse[b])
                row["diff_ci_halfwidth"][f"{a}-{b}"] = float((hi[0] - lo[0]) / 2)
        sensitivity[str(length_s)] = row

    return {
        "settings": {
            "reference": "GLORYS reanalysis, harmonised to the 0.25 degree grid",
            "n_days": n_days,
            "start": clim["dates"][0],
            "end": clim["dates"][-1],
            "depths": [float(z) for z in depths],
            "columns": col_names,
            "pooled_range_m": list(POOLED_RANGE),
            "regions": list(REGIONS),
            "metrics": list(METRIC_NAMES),
            "bootstrap": {
                "kind": "moving-block bootstrap over test days (overlapping blocks, no wrap)",
                "n_replicates": n_boot,
                "block_length_days": length,
                "block_length_auto_days": auto_len,
                "block_length_rule": "median over methods of the e-folding time (first lag with "
                "autocorrelation < 1/e) of the daily pooled 50-200 m MSE, capped at n_days/4",
                "ci": "95 % percentile interval of the replicates",
                "seed": seed,
                "resampled": "test days (blocks of consecutive days), identical for every "
                "method, depth, basin and metric; seeds are not resampled",
                "estimator": "per-day sufficient statistics averaged over seeds, then metrics "
                "from the resampled sums",
            },
            "n_seeds": {m: len(r["seeds"]) for m, r in res.items()},
        },
        "autocorrelation": {
            "daily_pooled_mse": decor,
            "daily_pooled_mse_difference": diff_decor,
        },
        "block_sensitivity": sensitivity,
        "methods": methods_out,
        "comparisons": comparisons,
    }


# ----------------------------------------------------------------------------------------
# markdown
# ----------------------------------------------------------------------------------------
def _fin(x) -> bool:
    return x is not None and bool(np.isfinite(x))


def _signed(x, nd: int = 3) -> str:
    return f"{x:+.{nd}f}" if _fin(x) else "n/a"


def _yn(x) -> str:
    return "n/a" if x is None else ("yes" if x else "no")


def _pm(c: dict, nd: int = 3) -> str:
    if not _fin(c["seed_mean"]):
        return "n/a"
    sd = c["seed_sd"]
    if not _fin(sd):
        return f"{c['seed_mean']:.{nd}f}"
    return f"{c['seed_mean']:.{nd}f} ± {sd:.{nd}f}"


def _ci(lo, hi, nd: int = 3) -> str:
    if not (_fin(lo) and _fin(hi)):
        return "n/a"
    return f"[{lo:.{nd}f}, {hi:.{nd}f}]"


def _num(x, nd: int = 3) -> str:
    return f"{x:.{nd}f}" if _fin(x) else "n/a"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return [*out, ""]


def _verdict(comp: dict, metric: str = "rmse") -> dict:
    c = comp["regions"]["all"]["columns"][_pooled_name(comp)][metric]
    s = comp["regions"]["all"]["columns"][_pooled_name(comp)]["seeds_rmse"]
    return {"cell": c, "seeds": s}


def _pooled_name(comp: dict) -> str:
    return f"pooled_{POOLED_RANGE[0]:g}_{POOLED_RANGE[1]:g}m"


def _answer(comp: dict, summary: dict, question: str) -> str:
    v = _verdict(comp)
    c, s = v["cell"], v["seeds"]
    a, b = comp["a"], comp["b"]
    pn = _pooled_name(comp)
    ma = summary["methods"][a]["regions"]["all"]["columns"][pn]["rmse"]
    mb = summary["methods"][b]["regions"]["all"]["columns"][pn]["rmse"]
    day_clear = c["excludes_zero"]
    if day_clear and not s["ranges_overlap"]:
        verdict = "distinguishable from zero on both counts"
    elif day_clear:
        verdict = (
            "the day-resampling interval excludes zero but the seed ranges overlap, so the "
            "difference is not established independently of the training seed"
        )
    else:
        verdict = "not distinguishable from zero"
    welch = "" if s["welch_p"] is None else f"; Welch t-test across seeds p = {s['welch_p']:.3f}"
    few = min(summary["methods"][a]["n_seeds"], summary["methods"][b]["n_seeds"])
    caution = f" Only {few} seed(s) on one side: indicative only." if few < 3 else ""
    return (
        f"- **{question}** pooled 50-200 m RMSE, {LABELS[a]} minus {LABELS[b]}: "
        f"{c['diff']:+.3f} °C, 95 % CI {_ci(c['ci_lo'], c['ci_hi'])} over test days "
        f"(bootstrap p = {c['p_two_sided']:.3f}). Seeds: {_pm(ma)} "
        f"(n={summary['methods'][a]['n_seeds']}) vs {_pm(mb)} "
        f"(n={summary['methods'][b]['n_seeds']}), seed ranges "
        f"{'overlap' if s['ranges_overlap'] else 'do not overlap'}{welch}. "
        f"Verdict: {verdict}.{caution}"
    )


def write_markdown(summary: dict, path: Path, run_name: str) -> None:
    st = summary["settings"]
    bs = st["bootstrap"]
    methods = summary["methods"]
    order = [m for m in METHODS if m in methods]
    pn = f"pooled_{POOLED_RANGE[0]:g}_{POOLED_RANGE[1]:g}m"
    depths = st["depths"]
    L: list[str] = []
    L += [
        f"# R1 rigour: seeds, confidence intervals and paired comparisons ({run_name})",
        "",
        f"Test period {st['start']} .. {st['end']} ({st['n_days']} days), reference GLORYS "
        "reanalysis. Temperatures in °C; bias is method minus GLORYS; `skill` is "
        "`1 - MSE / MSE_climatology`; `anomaly corr` is the correlation of anomalies (climatology "
        "removed from both).",
        "",
        "## Method and uncertainty",
        "",
        "Seeds per method: "
        + ", ".join(f"{LABELS[m]} n={st['n_seeds'][m]}" for m in order)
        + ". Ridge and climatology are deterministic (one evaluation, taken from the main run).",
        "",
        "- **Seed spread** (`mean ± SD`): each seed's metric from its own run, sample SD over "
        "seeds (n - 1); for `oceanembed` every seed repeats pretraining as well as fine-tuning.",
        f"- **95 % CI over test days**: {bs['kind']}; {bs['n_replicates']} replicates, block "
        f"length {bs['block_length_days']} days, {bs['ci']}. The per-day sufficient statistics are "
        "averaged over seeds and the metric is recomputed from the resampled days, so the interval "
        "reflects test-day sampling of the seed-mean model, not seed variation.",
        "- **Paired difference** `A - B` (negative = A better): computed inside each replicate "
        "with the same resampled days for both methods.",
        "",
        "### Block length",
        "",
        f"Daily errors are strongly autocorrelated. Block length = {bs['block_length_rule']}: "
        f"{bs['block_length_auto_days']} days"
        + (
            ""
            if bs["block_length_days"] == bs["block_length_auto_days"]
            else f" (overridden: {bs['block_length_days']} days used)"
        )
        + ".",
        "",
    ]
    dec = summary["autocorrelation"]["daily_pooled_mse"]
    L += _table(
        [
            "daily pooled 50-200 m MSE of",
            "e-folding time (days)",
            "integrated autocorrelation time (days)",
        ],
        [
            [LABELS[m], f"{dec[m]['efolding_days']:.0f}", f"{dec[m]['tau_int_days']:.1f}"]
            for m in order
        ],
    )
    sens = summary["block_sensitivity"]
    lengths = sorted(sens, key=int)
    keys = [f"{a}-{b}" for a, b in KEY_PAIRS if f"{a}-{b}" in sens[lengths[0]]["diff_ci_halfwidth"]]
    L += [
        "Half-width of the 95 % interval (°C) of the pooled 50-200 m RMSE, and of the key "
        "differences, as a function of block length (length 1 = ignoring autocorrelation):",
        "",
    ]
    L += _table(
        ["block length (days)", *[LABELS[m] for m in order], *[f"{k} (diff)" for k in keys]],
        [
            [
                n + (" *" if int(n) == bs["block_length_days"] else ""),
                *[f"{sens[n]['rmse_ci_halfwidth'][m]:.4f}" for m in order],
                *[f"{sens[n]['diff_ci_halfwidth'][k]:.4f}" for k in keys],
            ]
            for n in lengths
        ],
    )
    L += ["(* = block length used)", ""]

    # ---- answers
    comps = {(c["a"], c["b"]): c for c in summary["comparisons"]}
    L += ["## The two questions", ""]
    if ("oceanembed", "scratch") in comps:
        L.append(_answer(comps["oceanembed", "scratch"], summary, "Does pretraining matter?"))
    if ("scratch", "unet") in comps:
        L.append(
            _answer(
                comps["scratch", "unet"], summary, "Does the Transformer add anything over a U-Net?"
            )
        )
    if ("oceanembed", "unet") in comps:
        L.append(_answer(comps["oceanembed", "unet"], summary, "Pretrained Transformer vs U-Net?"))
    L.append("")

    # ---- tables: pooled
    def metric_row(m: str, region: str, col: str) -> list[str]:
        c = methods[m]["regions"][region]["columns"][col]
        return [
            LABELS[m],
            str(methods[m]["n_seeds"]),
            _pm(c["rmse"]),
            _ci(c["rmse"]["ci_lo"], c["rmse"]["ci_hi"]),
            _pm(c["mae"]),
            _pm(c["bias"]),
            _pm(c["corr_anom"]) + " " + _ci(c["corr_anom"]["ci_lo"], c["corr_anom"]["ci_hi"], 2)
            if np.isfinite(c["corr_anom"]["point"])
            else "n/a",
            _pm(c["skill_vs_clim"], 3)
            + " "
            + _ci(c["skill_vs_clim"]["ci_lo"], c["skill_vs_clim"]["ci_hi"], 3),
        ]

    head = [
        "method",
        "seeds",
        "RMSE seed mean ± SD",
        "RMSE 95 % CI (days)",
        "MAE",
        "bias",
        "anomaly corr [95 % CI]",
        "skill vs clim [95 % CI]",
    ]
    L += ["## Pooled 50-200 m (the thermocline)", ""]
    for region in REGIONS:
        L += [f"### {REGION_LABELS[region]}", ""]
        L += _table(head, [metric_row(m, region, pn) for m in order])
    L += ["### All depths pooled (0-1000 m), whole domain", ""]
    L += _table(head, [metric_row(m, "all", "overall") for m in order])

    # ---- per depth
    for region in REGIONS:
        L += [f"## RMSE by depth, {REGION_LABELS[region]} (seed mean ± SD)", ""]
        L += _table(
            ["depth (m)", *[LABELS[m] for m in order]],
            [
                [
                    f"{z:g}",
                    *[
                        _pm(methods[m]["regions"][region]["columns"][f"{z:g}"]["rmse"])
                        for m in order
                    ],
                ]
                for z in depths
            ],
        )
    L += ["## RMSE by depth, whole domain: 95 % CI over test days", ""]
    L += _table(
        ["depth (m)", *[LABELS[m] for m in order]],
        [
            [
                f"{z:g}",
                *[
                    _ci(
                        methods[m]["regions"]["all"]["columns"][f"{z:g}"]["rmse"]["ci_lo"],
                        methods[m]["regions"]["all"]["columns"][f"{z:g}"]["rmse"]["ci_hi"],
                        2,
                    )
                    for m in order
                ],
            ]
            for z in depths
        ],
    )
    L += ["## Skill vs climatology by depth, whole domain (seed mean ± SD)", ""]
    L += _table(
        ["depth (m)", *[LABELS[m] for m in order if m != "climatology"]],
        [
            [
                f"{z:g}",
                *[
                    _pm(methods[m]["regions"]["all"]["columns"][f"{z:g}"]["skill_vs_clim"])
                    for m in order
                    if m != "climatology"
                ],
            ]
            for z in depths
        ],
    )
    L += ["## Anomaly correlation by depth, whole domain (seed mean ± SD)", ""]
    L += _table(
        ["depth (m)", *[LABELS[m] for m in order if m != "climatology"]],
        [
            [
                f"{z:g}",
                *[
                    _pm(methods[m]["regions"]["all"]["columns"][f"{z:g}"]["corr_anom"], 2)
                    for m in order
                    if m != "climatology"
                ],
            ]
            for z in depths
        ],
    )

    # ---- paired
    L += [
        "## Paired comparisons (A - B in RMSE, °C; negative = A better)",
        "",
        "Bootstrap interval and p-value from the shared replicates; the seed columns compare the "
        "per-seed pooled RMSEs (ranges overlap = the spread of A and B intersect; Welch t-test "
        "needs at least 2 seeds on each side).",
        "",
    ]
    for region in REGIONS:
        L += [f"### Pooled 50-200 m, {REGION_LABELS[region]}", ""]
        rows = []
        for c in summary["comparisons"]:
            cell = c["regions"][region]["columns"][pn]
            r, s = cell["rmse"], cell["seeds_rmse"]
            rows.append(
                [
                    f"{LABELS[c['a']]} - {LABELS[c['b']]}",
                    _signed(r["diff"]),
                    _ci(r["ci_lo"], r["ci_hi"]),
                    _yn(r["excludes_zero"]),
                    _num(r["p_two_sided"]),
                    f"{_num(s['mean_a'])} / {_num(s['mean_b'])}",
                    _yn(s["ranges_overlap"]) if _fin(s["mean_diff"]) else "n/a",
                    _num(s["welch_p"]),
                ]
            )
        L += _table(
            [
                "A - B",
                "Δ RMSE",
                "95 % CI",
                "excludes 0",
                "p (bootstrap)",
                "seed means A / B",
                "seed ranges overlap",
                "Welch p",
            ],
            rows,
        )
    L += ["### Anomaly correlation difference, pooled 50-200 m, whole domain", ""]
    rows = []
    for c in summary["comparisons"]:
        r = c["regions"]["all"]["columns"][pn]["corr_anom"]
        if r["diff"] is None:
            continue
        rows.append(
            [
                f"{LABELS[c['a']]} - {LABELS[c['b']]}",
                _signed(r["diff"]),
                _ci(r["ci_lo"], r["ci_hi"]),
                _yn(r["excludes_zero"]),
            ]
        )
    L += _table(["A - B", "Δ anomaly corr", "95 % CI", "excludes 0"], rows)
    for a, b in KEY_PAIRS:
        c = comps.get((a, b))
        if c is None:
            continue
        L += [
            f"### Δ RMSE by depth, {LABELS[a]} - {LABELS[b]}, whole domain (* = CI excludes 0)",
            "",
        ]
        rows = []
        for z in depths:
            r = c["regions"]["all"]["columns"][f"{z:g}"]["rmse"]
            rows.append(
                [
                    f"{z:g}",
                    _signed(r["diff"]) + ("*" if r["excludes_zero"] else ""),
                    _ci(r["ci_lo"], r["ci_hi"]),
                ]
            )
        L += _table(["depth (m)", "Δ RMSE", "95 % CI"], rows)

    L += [
        "## Provenance",
        "",
        "| method | parameters | best epoch per seed | val RMSE per seed (°C) |",
        "|---|---|---|---|",
    ]
    for m in order:
        e = methods[m]
        if e["n_params"] is None:
            L.append(f"| {LABELS[m]} | n/a | n/a | n/a |")
        else:
            vr = ", ".join(f"{v:.3f}" for v in e["val_rmse"])
            ep = ", ".join(str(v) for v in e["best_epoch"])
            L.append(f"| {LABELS[m]} | {e['n_params']:,} | {ep} | {vr} |")
    L += [
        "",
        "Figures: `figures/rmse_by_depth.png`, `figures/paired_differences.png`, "
        "`figures/skill_by_depth.png`. Numbers: `summary.json`.",
        "",
    ]
    path.write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------
def _depth_axis(ax, depths) -> None:
    ax.set_yscale("function", functions=(lambda d: np.sqrt(np.clip(d, 0, None)), lambda s: s**2))
    ax.set_ylim(max(depths), 0)
    ax.set_yticks([z for z in depths if z in (0, 50, 100, 200, 500, 1000)])
    ax.set_yticklabels([f"{z:g}" for z in depths if z in (0, 50, 100, 200, 500, 1000)])
    ax.set_ylabel("depth (m)")
    ax.grid(alpha=0.25)


def _by_depth_figure(summary: dict, metric: str, xlabel: str, path: Path, skip=()) -> None:
    depths = summary["settings"]["depths"]
    methods = summary["methods"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), sharey=True)
    for ax, region in zip(axes, REGIONS, strict=True):
        for m in [x for x in METHODS if x in methods and x not in skip]:
            cols = methods[m]["regions"][region]["columns"]
            mean = np.array([cols[f"{z:g}"][metric]["seed_mean"] for z in depths])
            sd = np.array([cols[f"{z:g}"][metric]["seed_sd"] for z in depths], dtype=float)
            ax.plot(
                mean,
                depths,
                color=COLORS[m],
                lw=1.8,
                ls="--" if m == "climatology" else "-",
                label=f"{LABELS[m]} (n={methods[m]['n_seeds']})",
            )
            if np.isfinite(sd).any():
                ax.fill_betweenx(depths, mean - sd, mean + sd, color=COLORS[m], alpha=0.25, lw=0)
        _depth_axis(ax, depths)
        ax.set_title(REGION_LABELS[region])
        ax.set_xlabel(xlabel)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Bands: ±1 SD across training seeds (test year, vs GLORYS)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _paired_figure(summary: dict, path: Path) -> None:
    pn = f"pooled_{POOLED_RANGE[0]:g}_{POOLED_RANGE[1]:g}m"
    comps = summary["comparisons"]
    depths = summary["settings"]["depths"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.2, 1]})
    markers = {"all": "o", "arabian_sea": "^", "bay_of_bengal": "s"}
    offsets = {"all": 0.0, "arabian_sea": -0.22, "bay_of_bengal": 0.22}
    labels = []
    for i, c in enumerate(comps):
        labels.append(f"{LABELS[c['a']]}\nminus {LABELS[c['b']]}")
        for region in REGIONS:
            r = c["regions"][region]["columns"][pn]["rmse"]
            if not _fin(r["diff"]):
                continue  # a basin without points (tiny test grid)
            y = i + offsets[region]
            ax.errorbar(
                r["diff"],
                y,
                xerr=[[r["diff"] - r["ci_lo"]], [r["ci_hi"] - r["diff"]]],
                fmt=markers[region],
                color=COLORS[c["a"]],
                mfc=COLORS[c["a"]] if r["excludes_zero"] else "white",
                capsize=2.5,
                ms=5,
                label=REGION_LABELS[region] if i == 0 else None,
            )
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(comps)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Δ pooled 50-200 m RMSE (°C), A minus B; negative = A better")
    ax.set_title("Paired differences, 95 % block-bootstrap CI (filled = excludes 0)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, axis="x")
    for c in comps:
        if (c["a"], c["b"]) not in KEY_PAIRS:
            continue
        rs = [c["regions"]["all"]["columns"][f"{z:g}"]["rmse"] for z in depths]
        d, lo, hi = (
            np.array([np.nan if r[k] is None else r[k] for r in rs], dtype=float)
            for k in ("diff", "ci_lo", "ci_hi")
        )
        col = PAIR_COLORS[c["a"], c["b"]]
        label = f"{LABELS[c['a']]} - {LABELS[c['b']]}"
        bx.plot(d, depths, color=col, lw=1.6, label=label)
        bx.fill_betweenx(depths, lo, hi, color=col, alpha=0.18, lw=0)
    bx.axvline(0, color="k", lw=0.8)
    _depth_axis(bx, depths)
    bx.set_xlabel("Δ RMSE (°C), whole domain, with 95 % CI band")
    bx.set_title("Key differences by depth")
    bx.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def write_figures(summary: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        out_dir / "rmse_by_depth.png",
        out_dir / "paired_differences.png",
        out_dir / "skill_by_depth.png",
    ]
    _by_depth_figure(summary, "rmse", "RMSE vs GLORYS (°C)", paths[0])
    _paired_figure(summary, paths[1])
    _by_depth_figure(
        summary,
        "skill_vs_clim",
        "skill vs climatology (1 - MSE / MSE_clim)",
        paths[2],
        skip=("climatology",),
    )
    return paths


# ----------------------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------------------
def make_r1_report(
    cfg: Config, n_boot: int = 2000, block_length: int | None = None, seed: int = 0
) -> dict:
    """Write ``summary.json``, ``summary.md`` and the figures under ``research/r1/``."""
    res = load_results(cfg)
    summary = build_summary(res, n_boot=n_boot, block_length=block_length, seed=seed)
    out = r1_dir(cfg)
    write_json(out / "summary.json", summary)
    write_markdown(summary, out / "summary.md", cfg.run_name)
    write_figures(summary, out / "figures")
    return summary
