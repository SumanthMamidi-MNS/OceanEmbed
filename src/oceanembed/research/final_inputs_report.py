"""Input-set selection: the decision (validation year only) and the test-year scores of every
candidate.

Reads the finished jobs of :mod:`oceanembed.research.final_inputs` (the reduced input sets) and of
R2
(the full-input headline model ``research/r2/scratch`` and the climatology), and writes
``outputs/<run>/research/final_inputs/summary.json`` and ``summary.md``.

**Selection rule (uses no test-year number).** For every input set the validation-year (2022)
RMSE of
the saved checkpoint -- the pooled RMSE over all 15 depths and the whole validation year, at the
epoch early stopping kept -- is taken for every seed. A reduced set is *adopted* only if its mean
validation RMSE is lower than that of the full seven-input set by **more than the seed spread**, the
larger of the two sample standard deviations over seeds; among the adopted sets the lowest wins.
Otherwise all seven inputs are kept. The test years are scored afterwards, for every candidate, with
the R2 moving-block bootstrap (per year and pooled, whole domain and per basin), only to report the
outcome.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from oceanembed.config import Config
from oceanembed.eval.evaluate import write_json
from oceanembed.eval.metrics import SUM_FIELDS
from oceanembed.research.common import choose_block_length, make_counts
from oceanembed.research.final_inputs import CANDIDATES, fi_dir
from oceanembed.research.inputs import EXPERIMENT_LABELS, EXPERIMENTS, GROUP_LABELS
from oceanembed.research.r2 import r2_dir
from oceanembed.research.r2_report import METRICS, _seed_runs, _slice
from oceanembed.research.r3_report import (
    POOLED,
    REGION_LABELS,
    REGIONS,
    _bundle,
    _daily_pooled_mse,
    _delta,
    _delta_text,
    _num,
    _pm,
    _stat,
    _table,
    columns,
)

log = logging.getLogger(__name__)

FULL = "full"
THERMOCLINE = (50.0, 200.0)


def label(name: str) -> str:
    return "All seven inputs" if name == FULL else EXPERIMENT_LABELS[name]


def inputs_text(name: str) -> str:
    keep = EXPERIMENTS["full" if name == FULL else name].keep
    return ", ".join(GROUP_LABELS[g] for g in keep)


# ----------------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------------
def load_results(cfg: Config) -> dict:
    """``{"runs": {name: run}, "climatology": run}``; ``runs["full"]`` is the R2 headline model."""
    r2base = r2_dir(cfg)
    full = _seed_runs(r2base / "scratch")
    clim = _seed_runs(r2base / "climatology")
    if full is None or clim is None:
        raise FileNotFoundError(
            f"no finished R2 scratch / climatology jobs under {r2base}; run `research r2` first"
        )
    runs = {FULL: full}
    for exp in CANDIDATES:
        run = _seed_runs(fi_dir(cfg) / f"scratch_{exp}")
        if run is not None:
            runs[exp] = run
    if len(runs) < 2:
        raise FileNotFoundError(
            f"no finished candidate under {fi_dir(cfg)}; run `research final-inputs` first"
        )
    n_idx = SUM_FIELDS.index("n")
    for name, r in runs.items():
        if r["dates"] != clim["dates"] or not np.array_equal(r["depths"], clim["depths"]):
            raise ValueError(f"{name}: evaluated days / depths differ from the climatology's")
        if not np.allclose(r["raw"][:, :, n_idx], clim["raw"][0:1, :, n_idx]):
            raise ValueError(f"{name}: the evaluated sample differs from the climatology's")
    return {"runs": runs, "climatology": clim}


# ----------------------------------------------------------------------------------------
# the decision: validation only
# ----------------------------------------------------------------------------------------
def _validation(run: dict, depths: np.ndarray) -> dict:
    trains = [i.get("train", {}) for i in run["info"]]
    val = np.array([t["val_rmse"] for t in trains], dtype=float)
    pd_ = np.array([t.get("val_rmse_per_depth") for t in trains], dtype=float)  # (S, D)
    sel = (depths >= THERMOCLINE[0]) & (depths <= THERMOCLINE[1])
    thermo = (
        np.sqrt((pd_[:, sel] ** 2).mean(axis=1)) if pd_.ndim == 2 else np.full(len(val), np.nan)
    )
    return {
        "seeds": run["seeds"],
        "val_rmse": [float(v) for v in val],
        "mean": float(val.mean()),
        "sd": float(val.std(ddof=1)) if len(val) > 1 else None,
        "best_epoch": [t.get("epoch") for t in trains],
        "val_rmse_thermocline_unweighted": [float(v) for v in thermo],
        "thermocline_mean": float(np.nanmean(thermo)) if np.isfinite(thermo).any() else None,
    }


def decide(validation: dict[str, dict]) -> dict:
    """Apply the selection rule to ``{name: _validation(...)}`` (``full`` must be present)."""
    ref = validation[FULL]
    rows = {}
    adopted = []
    for name, v in validation.items():
        if name == FULL:
            continue
        sds = [x for x in (ref["sd"], v["sd"]) if x is not None]
        spread = max(sds) if sds else 0.0
        gain = ref["mean"] - v["mean"]  # positive = the candidate is better on validation
        ok = bool(gain > spread)
        rows[name] = {
            "mean_val_rmse": v["mean"],
            "gain_vs_full": gain,
            "seed_spread": spread,
            "adopted": ok,
        }
        if ok:
            adopted.append(name)
    chosen = min(adopted, key=lambda n: validation[n]["mean"]) if adopted else FULL
    return {
        "rule": (
            "a reduced input set is adopted only if its mean validation-year RMSE is lower than "
            "the "
            "full set's by more than the seed spread (the larger sample SD over seeds of the two "
            "sets); the lowest adopted set wins, otherwise all seven inputs are kept"
        ),
        "criterion": "validation-year (2022) RMSE of the kept checkpoint, pooled over all depths",
        "spread": "max(SD over seeds of the full set, SD over seeds of the candidate)",
        "candidates": rows,
        "chosen": chosen,
        "chosen_label": label(chosen),
        "chosen_input_groups": list(EXPERIMENTS["full" if chosen == FULL else chosen].keep),
    }


# ----------------------------------------------------------------------------------------
# test-year scores (reported after the decision)
# ----------------------------------------------------------------------------------------
def build_summary(
    res: dict, n_boot: int = 2000, block_length: int | None = None, seed: int = 0
) -> dict:
    runs, clim = res["runs"], res["climatology"]
    depths = np.asarray(clim["depths"], dtype=float)
    dates = np.array(clim["dates"], dtype="datetime64[D]")
    col_names, sel = columns(depths)
    years = sorted({int(str(d)[:4]) for d in dates})
    periods = {
        str(y): np.flatnonzero(dates.astype("datetime64[Y]") == np.datetime64(str(y)))
        for y in years
    }
    periods["pooled"] = np.arange(len(dates))
    if len(years) < 2:
        periods = {"pooled": periods["pooled"]}

    series = [_daily_pooled_mse(runs[FULL]["raw"][:, 0].mean(axis=0), depths)]
    length, auto = choose_block_length(series, len(dates), block_length)
    length = length or auto
    counts = {
        p: make_counts(len(idx), length, n_boot, seed + 17 * i)
        for i, (p, idx) in enumerate(periods.items())
    }
    ones = {p: np.ones((1, len(idx))) for p, idx in periods.items()}
    cache: dict = {}

    def bundle(name: str, period: str) -> dict:
        key = (name, period)
        if key not in cache:
            idx = periods[period]
            run = _slice(runs[name], idx)
            cache[key] = _bundle(
                run,
                list(range(len(run["seeds"]))),
                clim["raw"][0][:, :, idx],
                sel,
                counts[period],
                ones[period],
            )
        return cache[key]

    validation = {name: _validation(r, depths) for name, r in runs.items()}
    decision = decide(validation)

    test: dict = {}
    for name in runs:
        test[name] = {
            "label": label(name),
            "inputs": inputs_text(name),
            "n_seeds": len(runs[name]["seeds"]),
            "periods": {},
        }
        for period in periods:
            b = bundle(name, period)
            test[name]["periods"][period] = {
                r: {
                    "columns": {
                        col: {m: _stat(b[r], m, ci) for m in METRICS}
                        for ci, col in enumerate(col_names)
                    }
                }
                for r in REGIONS
            }
    vs_full: dict = {}
    for name in runs:
        if name == FULL:
            continue
        vs_full[name] = {
            period: {
                r: {
                    col: _delta(bundle(name, period)[r], bundle(FULL, period)[r], "rmse", ci)
                    for ci, col in enumerate(col_names)
                    if col in (POOLED, "overall")
                }
                for r in REGIONS
            }
            for period in periods
        }
    return {
        "settings": {
            "reference": "GLORYS reanalysis, harmonised to the 0.25 degree grid",
            "periods": {
                p: {
                    "n_days": int(len(idx)),
                    "start": str(dates[idx[0]]),
                    "end": str(dates[idx[-1]]),
                }
                for p, idx in periods.items()
            },
            "depths": [float(z) for z in depths],
            "pooled_column": POOLED,
            "regions": list(REGIONS),
            "bootstrap": {
                "kind": "moving-block bootstrap over the days of each period (as R2)",
                "n_replicates": n_boot,
                "block_length_days": length,
                "block_length_auto_days": auto,
                "ci": "95 % percentile interval of the replicates",
                "seed": seed,
            },
        },
        "decision": {**decision, "validation": validation},
        "test": test,
        "test_vs_full": vs_full,
        "note": (
            "The decision uses validation numbers only. The test-year scores below are reported "
            "for every candidate after the fact and played no part in it."
        ),
    }


# ----------------------------------------------------------------------------------------
# markdown
# ----------------------------------------------------------------------------------------
def _cell(summary: dict, name: str, period: str, region: str, col: str, metric: str) -> dict:
    return summary["test"][name]["periods"][period][region]["columns"][col][metric]


def write_markdown(summary: dict, path: Path, run_name: str) -> None:
    dec = summary["decision"]
    val = dec["validation"]
    st = summary["settings"]
    bs = st["bootstrap"]
    L = [
        f"# Input-set selection for the final model ({run_name})",
        "",
        f"**Decision: {dec['chosen_label']}** ({inputs_text(dec['chosen'])}).",
        "",
        "## Rule (validation year only)",
        "",
        dec["rule"] + ".",
        "",
        f"Criterion: {dec['criterion']}. Spread: {dec['spread']}.",
        "",
        "## Validation (2022), the headline model trained from scratch",
        "",
    ]
    rows = []
    for name, v in val.items():
        c = dec["candidates"].get(name)
        rows.append(
            [
                label(name),
                inputs_text(name),
                ", ".join(f"{x:.4f}" for x in v["val_rmse"]),
                f"{v['mean']:.4f}",
                "-" if v["sd"] is None else f"{v['sd']:.4f}",
                "-" if v["thermocline_mean"] is None else f"{v['thermocline_mean']:.4f}",
                "reference" if c is None else f"{c['gain_vs_full']:+.4f}",
                "-" if c is None else f"{c['seed_spread']:.4f}",
                "-" if c is None else ("yes" if c["adopted"] else "no"),
            ]
        )
    L += _table(
        [
            "Input set",
            "Inputs",
            "Val. RMSE per seed (°C)",
            "Mean",
            "SD",
            "Mean over 50-200 m levels (unweighted)",
            "Gain vs full (positive = better)",
            "Seed spread",
            "Adopted",
        ],
        rows,
    )
    L += [
        "Test periods: "
        + "; ".join(
            f"{p} ({v['n_days']} days, {v['start']} .. {v['end']})"
            for p, v in st["periods"].items()
        )
        + f". Reference GLORYS. Intervals: {bs['kind']}, {bs['n_replicates']} replicates, blocks "
        f"of {bs['block_length_days']} days, 95 % percentile.",
        "",
        "## Test-year scores of every candidate (reported after the decision)",
        "",
        "RMSE over 50-200 m against GLORYS, mean ± SD over seeds.",
        "",
    ]
    rows = []
    for name in summary["test"]:
        for p in st["periods"]:
            rows.append(
                [label(name), str(summary["test"][name]["n_seeds"]), p]
                + [_pm(_cell(summary, name, p, r, POOLED, "rmse")) for r in REGIONS]
                + [
                    _num(_cell(summary, name, p, "all", POOLED, "skill_vs_clim")["seed_mean"], 3),
                    _num(_cell(summary, name, p, "all", "overall", "rmse")["seed_mean"], 3),
                ]
            )
    L += _table(
        [
            "Input set",
            "Seeds",
            "Period",
            "Whole domain",
            "Arabian Sea",
            "Bay of Bengal",
            "Skill vs climatology",
            "RMSE all depths",
        ],
        rows,
    )
    L += [
        "Candidate minus full set, RMSE over 50-200 m (negative = candidate better, "
        "`*` established):",
        "",
    ]
    rows = []
    for name, byp in summary["test_vs_full"].items():
        for p, rr in byp.items():
            rows.append([label(name), p] + [_delta_text(rr[r][POOLED]) for r in REGIONS])
    L += _table(
        ["Input set", "Period"] + [REGION_LABELS[r] for r in REGIONS],
        rows,
    )
    L += [summary["note"], ""]
    path.write_text("\n".join(L), encoding="utf-8")


def make_final_inputs_report(
    cfg: Config, n_boot: int = 2000, block_length: int | None = None, seed: int = 0
) -> dict:
    """Write ``summary.json`` and ``summary.md`` under ``research/final_inputs/``."""
    res = load_results(cfg)
    summary = build_summary(res, n_boot=n_boot, block_length=block_length, seed=seed)
    out = fi_dir(cfg)
    write_json(out / "summary.json", summary)
    write_markdown(summary, out / "summary.md", cfg.run_name)
    return summary
