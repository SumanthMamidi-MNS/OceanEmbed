"""R2 analysis: more training years, two test years, learning curve, Argo.

Reads the ``eval.npz`` files of :mod:`oceanembed.research.r2` (whole test split, both years) and,
for the comparison with the shorter training period, the R1 results of the first run (read only),
and writes ``outputs/<run>/research/r2/summary.json``, ``summary.md`` and ``figures/*.png``.

Conventions are those of R1 / R3: moving-block bootstrap over days with replicates shared by every
method, depth, basin and metric; a paired difference ``A - B`` is formed inside each replicate; a
difference is *established* only if its interval excludes zero **and** the per-seed RMSE ranges do
not overlap. Periods: each test year on its own (own replicates) and both pooled.

* ``long_vs_poc``: the long-period models against the models of the first (shorter) run on the same
  test year -- the same days, the same GLORYS reference, the same replicates. Absolute RMSE is
  compared; the skill against each run's *own* climatology is reported too, because the two
  climatologies are fitted on different periods.
* ``year_comparison``: 2024 minus 2023 for each method. The two years are different days, so the
  replicates of the two years are independent resamples (not paired).
* ``learning_curve``: the headline model trained on the last 2, 5 and 11 years, scored on the same
  days (normalisation and climatology are those of the full train period for every point).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from oceanembed.config import Config  # noqa: E402
from oceanembed.eval.evaluate import write_json  # noqa: E402
from oceanembed.eval.metrics import SUM_FIELDS  # noqa: E402
from oceanembed.research import r1  # noqa: E402
from oceanembed.research.common import choose_block_length, make_counts  # noqa: E402
from oceanembed.research.r2 import (  # noqa: E402
    ARGO_DIR,
    ARGO_SUMMARY,
    DONE_FILE,
    EVAL_FILE,
    LABELS,
    LEARNING_YEARS,
    METHODS,
    r2_dir,
)
from oceanembed.research.r3_report import (  # noqa: E402
    DEEP,
    POOLED,
    REGION_LABELS,
    REGIONS,
    _bundle,
    _daily_pooled_mse,
    _delta,
    _delta_text,
    _depth_axis,
    _num,
    _pm,
    _sgn,
    _stat,
    _table,
    columns,
)

log = logging.getLogger(__name__)

METRICS = ("rmse", "skill_vs_clim", "bias", "corr_anom")
DPI = 130
COLORS = {  # Okabe-Ito, one colour per method in every figure
    "scratch": "#009E73",
    "mlp": "#56B4E9",
    "ridge": "#E69F00",
    "climatology": "#555555",
}
ARGO_COLORS = {"model": "#009E73", "ridge": "#E69F00", "clim": "#555555", "glorys": "#CC79A7"}
ARGO_LABELS = {
    "model": "OceanEmbed (no pretraining)",
    "ridge": "Ridge regression",
    "clim": "Climatology",
    "glorys": "GLORYS",
}


# ----------------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------------
def _seed_runs(base: Path) -> dict | None:
    dirs = sorted(
        (
            p
            for p in base.glob("seed*")
            if p.name[4:].isdigit() and (p / DONE_FILE).exists() and (p / EVAL_FILE).exists()
        ),
        key=lambda p: int(p.name[4:]),
    )
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


def load_results(cfg: Config, poc_cfg: Config | None = None) -> dict:
    """``{"runs": {name: run}, "poc": {method: run} | None, "argo": dict | None}``.

    ``runs`` holds ``scratch``, ``mlp``, ``ridge``, ``climatology`` and the learning-curve jobs
    ``scratch_ty<N>``; raises when the climatology or the headline model is missing or when the
    evaluated samples differ between jobs."""
    base = r2_dir(cfg)
    runs: dict[str, dict] = {}
    for name in [*METHODS, *(f"scratch_ty{y}" for y in LEARNING_YEARS)]:
        run = _seed_runs(base / name)
        if run is not None:
            runs[name] = run
    for need in ("climatology", "scratch"):
        if need not in runs:
            raise FileNotFoundError(
                f"no finished {need} job under {base}; run `oceanembed research r2` first"
            )
    ref = runs["climatology"]
    n_idx = SUM_FIELDS.index("n")
    for name, r in runs.items():
        if r["dates"] != ref["dates"] or not np.array_equal(r["depths"], ref["depths"]):
            raise ValueError(f"{name}: evaluated days / depths differ from the climatology's")
        if not np.allclose(r["raw"][:, :, n_idx], ref["raw"][0:1, :, n_idx]):
            raise ValueError(f"{name}: the evaluated sample (n per day, depth, region) differs")
    poc = None
    if poc_cfg is not None:
        poc = {}
        for m in METHODS:
            run = _seed_runs(r1.r1_dir(poc_cfg) / m)
            if run is not None:
                poc[m] = run
        poc = poc or None
    argo = None
    f = base / ARGO_DIR / ARGO_SUMMARY
    if f.exists():
        argo = json.loads(f.read_text(encoding="utf-8"))
    return {"runs": runs, "poc": poc, "argo": argo}


# ----------------------------------------------------------------------------------------
# computation
# ----------------------------------------------------------------------------------------
def _slice(run: dict, idx: np.ndarray) -> dict:
    """The run restricted to the days ``idx`` (along the day axis of the sums)."""
    return {**run, "raw": run["raw"][:, :, :, idx], "anom": run["anom"][:, :, :, idx]}


def _regions_stat(b: dict, ci: int) -> dict:
    return {r: {m: _stat(b[r], m, ci) for m in METRICS} for r in REGIONS}


def build_summary(
    res: dict, n_boot: int = 2000, block_length: int | None = None, seed: int = 0
) -> dict:
    runs, poc, argo = res["runs"], res["poc"], res["argo"]
    clim = runs["climatology"]
    depths = np.asarray(clim["depths"], dtype=float)
    dates = np.array(clim["dates"], dtype="datetime64[D]")
    col_names, sel = columns(depths)
    years = sorted({int(str(d)[:4]) for d in dates})
    periods: dict[str, np.ndarray] = {
        str(y): np.flatnonzero(dates.astype("datetime64[Y]") == np.datetime64(str(y)))
        for y in years
    }
    periods["pooled"] = np.arange(len(dates))
    if len(years) < 2:
        periods = {"pooled": periods["pooled"]}
    ci_pool = col_names.index(POOLED)

    series = [_daily_pooled_mse(runs["scratch"]["raw"][:, 0].mean(axis=0), depths)]
    length, auto = choose_block_length(series, len(dates), block_length)
    length = length or auto
    counts = {
        p: make_counts(len(idx), length, n_boot, seed + 17 * i)
        for i, (p, idx) in enumerate(periods.items())
    }
    ones = {p: np.ones((1, len(idx))) for p, idx in periods.items()}

    cache: dict[tuple, dict] = {}

    def bundle(name: str, period: str, seeds: list[int] | None = None) -> dict:
        run = runs[name]
        use = seeds if seeds is not None else run["seeds"]
        key = (name, period, tuple(use))
        if key not in cache:
            idx = periods[period]
            sub = _slice(run, idx)
            c_raw = clim["raw"][0][:, :, idx]
            cache[key] = _bundle(
                sub, [run["seeds"].index(s) for s in use], c_raw, sel, counts[period], ones[period]
            )
        return cache[key]

    # --- every method, every period ------------------------------------------------------
    methods_out: dict = {}
    for name, run in runs.items():
        entry = {
            "label": LABELS.get(name, f"OceanEmbed (no pretraining), last {name[-1]} years"),
            "n_seeds": len(run["seeds"]),
            "seeds": run["seeds"],
            "n_train_days": [i.get("train", {}).get("n_train_days") for i in run["info"]],
            "best_epoch": [i.get("train", {}).get("epoch") for i in run["info"]],
            "epochs_run": [i.get("train", {}).get("epochs_run") for i in run["info"]],
            "val_rmse": [i.get("train", {}).get("val_rmse") for i in run["info"]],
            "periods": {},
        }
        for period in periods:
            b = bundle(name, period)
            entry["periods"][period] = {
                "regions": {
                    r: {
                        "columns": {
                            col: {m: _stat(b[r], m, ci) for m in METRICS}
                            for ci, col in enumerate(col_names)
                        }
                    }
                    for r in REGIONS
                }
            }
        methods_out[name] = entry

    # --- 2024 minus 2023 -------------------------------------------------------------------
    year_cmp: dict = {}
    if "2023" in periods and "2024" in periods:
        for name in METHODS:
            if name not in runs:
                continue
            a, b = bundle(name, "2024"), bundle(name, "2023")
            year_cmp[name] = {
                r: {
                    col: {
                        m: _delta(a[r], b[r], m, ci, with_seeds=(m == "rmse"))
                        for m in ("rmse", "skill_vs_clim", "bias")
                    }
                    for ci, col in enumerate(col_names)
                }
                for r in REGIONS
            }

    # --- Transformer vs per-pixel MLP, and against the climatology -----------------------
    contrast: dict = {}
    if "mlp" in runs:
        for period in periods:
            t, m = bundle("scratch", period), bundle("mlp", period)
            contrast[period] = {
                r: {col: _delta(m[r], t[r], "rmse", ci) for ci, col in enumerate(col_names)}
                for r in REGIONS
            }
    vs_clim: dict = {}
    for name in ("scratch", "mlp", "ridge"):
        if name not in runs:
            continue
        for period in periods:
            a, c = bundle(name, period), bundle("climatology", period)
            vs_clim.setdefault(name, {})[period] = {
                r: {col: _delta(a[r], c[r], "rmse", ci) for ci, col in enumerate(col_names)}
                for r in REGIONS
            }

    # --- long period vs the shorter first run, same test year --------------------------------
    long_vs_poc: dict = {}
    if poc and "2024" in periods:
        p_ref = poc.get("climatology")
        p_dates = np.array(p_ref["dates"], dtype="datetime64[D]") if p_ref else None
        idx24 = periods["2024"]
        same_days = p_dates is not None and np.array_equal(dates[idx24], p_dates)
        n_i = SUM_FIELDS.index("n")
        long_vs_poc["sample"] = {
            "same_days": bool(same_days),
            "n_long": float(clim["raw"][0, 0, n_i][idx24].sum()),
            "n_poc": float(p_ref["raw"][0, 0, n_i].sum()) if p_ref else None,
        }
        long_vs_poc["sample"]["same_points"] = bool(
            same_days and np.allclose(clim["raw"][0, :, n_i][:, idx24], p_ref["raw"][0, :, n_i])
        )
        if same_days:
            sel24 = idx24
            for name in METHODS:
                if name not in runs or name not in poc:
                    continue
                pr = poc[name]
                c_poc = p_ref["raw"][0]
                pb = _bundle(
                    pr, list(range(len(pr["seeds"]))), c_poc, sel, counts["2024"], ones["2024"]
                )
                lb = bundle(name, "2024")
                long_vs_poc[name] = {
                    "n_seeds_long": len(runs[name]["seeds"]),
                    "n_seeds_poc": len(pr["seeds"]),
                    "poc": {
                        r: {
                            col: {m: _stat(pb[r], m, ci) for m in ("rmse", "skill_vs_clim")}
                            for ci, col in enumerate(col_names)
                        }
                        for r in REGIONS
                    },
                    "difference": {
                        r: {
                            col: {
                                m: _delta(lb[r], pb[r], m, ci, with_seeds=(m == "rmse"))
                                for m in ("rmse", "skill_vs_clim")
                            }
                            for ci, col in enumerate(col_names)
                        }
                        for r in REGIONS
                    },
                }
            del sel24
            # does the Transformer-over-MLP contrast persist in the first run?
            if "scratch" in poc and "mlp" in poc:
                tb = _bundle(
                    poc["scratch"],
                    list(range(len(poc["scratch"]["seeds"]))),
                    p_ref["raw"][0],
                    sel,
                    counts["2024"],
                    ones["2024"],
                )
                mb = _bundle(
                    poc["mlp"],
                    list(range(len(poc["mlp"]["seeds"]))),
                    p_ref["raw"][0],
                    sel,
                    counts["2024"],
                    ones["2024"],
                )
                long_vs_poc["contrast_poc_2024"] = {
                    r: {col: _delta(mb[r], tb[r], "rmse", ci) for ci, col in enumerate(col_names)}
                    for r in REGIONS
                }

    # --- learning curve -----------------------------------------------------------------------
    learning: dict = {}
    lc = {y: f"scratch_ty{y}" for y in LEARNING_YEARS if f"scratch_ty{y}" in runs}
    if lc:
        n_full = methods_out["scratch"]["n_train_days"][0]
        full_years = round((n_full or 0) / 365.25) if n_full else None
        ref_seed = runs["scratch"]["seeds"][0]
        pts = {
            **{y: (name, [runs[name]["seeds"][0]]) for y, name in lc.items()},
            "full": ("scratch", [ref_seed]),
        }
        learning = {
            "years": [*lc, full_years],
            "full_years": full_years,
            "reference": f"scratch seed {ref_seed} (full train period)",
            "points": {},
        }
        for y, (name, seeds) in pts.items():
            key = str(full_years if y == "full" else y)
            learning["points"][key] = {
                "n_train_days": n_full if y == "full" else None,
                "periods": {},
            }
            for period in periods:
                b, ref = bundle(name, period, seeds), bundle("scratch", period, [ref_seed])
                learning["points"][key]["periods"][period] = {
                    r: {
                        col: {
                            "rmse": _stat(b[r], "rmse", ci),
                            "delta_vs_full": _delta(b[r], ref[r], "rmse", ci, with_seeds=False),
                        }
                        for ci, col in enumerate(col_names)
                    }
                    for r in REGIONS
                }
        if len(runs["scratch"]["seeds"]) > 1:
            learning["full_all_seeds"] = {
                period: _regions_stat(bundle("scratch", period), ci_pool) for period in periods
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
            "columns": col_names,
            "pooled_range_m": [50.0, 200.0],
            "deep_range_m": [500.0, 1000.0],
            "regions": list(REGIONS),
            "bootstrap": {
                "kind": "moving-block bootstrap over the days of each period",
                "n_replicates": n_boot,
                "block_length_days": length,
                "block_length_auto_days": auto,
                "ci": "95 % percentile interval of the replicates",
                "seed": seed,
                "year_difference": "2024 minus 2023 uses independent resamples of the two years",
            },
            "established": "interval excludes zero AND the per-seed RMSE ranges do not overlap",
        },
        "methods": methods_out,
        "year_comparison": year_cmp,
        "transformer_vs_mlp": contrast,
        "vs_climatology": vs_clim,
        "long_vs_poc": long_vs_poc,
        "learning_curve": learning,
        "argo": argo,
        "pooled_column": POOLED,
        "deep_column": DEEP,
    }


# ----------------------------------------------------------------------------------------
# markdown
# ----------------------------------------------------------------------------------------
def _cell(summary: dict, name: str, period: str, region: str, col: str, metric: str) -> dict:
    return summary["methods"][name]["periods"][period]["regions"][region]["columns"][col][metric]


def _n_or_dash(n) -> str:
    return "-" if n is None else f"{n:.0f}"


def write_markdown(summary: dict, path: Path, run_name: str) -> None:
    st = summary["settings"]
    bs = st["bootstrap"]
    periods = list(st["periods"])
    depths_cols = [f"{z:g}" for z in st["depths"]]
    L = [
        f"# R2 - more training years, two test years ({run_name})",
        "",
        "Test periods: "
        + "; ".join(
            f"{p} ({v['n_days']} days, {v['start']} .. {v['end']})"
            for p, v in st["periods"].items()
        )
        + f". Reference GLORYS. Intervals: {bs['kind']}, {bs['n_replicates']} replicates, "
        f"blocks of {bs['block_length_days']} days, 95 % percentile. `*` = established "
        f"({st['established']}).",
        "",
    ]
    L += ["## Skill of every method", ""]
    rows = []
    for name, m in summary["methods"].items():
        for p in periods:
            row = [m["label"], str(m["n_seeds"]), p]
            for region in REGIONS:
                row.append(_pm(_cell(summary, name, p, region, POOLED, "rmse")))
            row += [
                _pm(_cell(summary, name, p, "all", POOLED, "skill_vs_clim")),
                _pm(_cell(summary, name, p, "all", POOLED, "corr_anom")),
                _pm(_cell(summary, name, p, "all", DEEP, "skill_vs_clim")),
            ]
            rows.append(row)
    L += _table(
        [
            "Method",
            "Seeds",
            "Period",
            "RMSE 50-200 m whole domain",
            "Arabian Sea",
            "Bay of Bengal",
            "Skill 50-200 m",
            "Anomaly corr.",
            "Skill 500-1000 m",
        ],
        rows,
    )
    if summary["year_comparison"]:
        L += ["## Is the skill stable from year to year? (2024 minus 2023)", ""]
        rows = []
        for name, yc in summary["year_comparison"].items():
            for region in REGIONS:
                c = yc[region][POOLED]
                rows.append(
                    [
                        LABELS[name],
                        REGION_LABELS[region],
                        _delta_text(c["rmse"]),
                        _delta_text(c["skill_vs_clim"], 3),
                    ]
                )
        L += _table(["Method", "Region", "Delta RMSE 50-200 m [95 % CI]", "Delta skill"], rows)
        L += ["Bias (method minus GLORYS, whole domain, seed mean) by depth:", ""]
        rows = []
        for z in depths_cols:
            row = [f"{z} m"]
            for name in ("scratch", "ridge", "climatology"):
                if name in summary["methods"]:
                    for p in ("2023", "2024"):
                        if p in periods:
                            row.append(
                                _sgn(_cell(summary, name, p, "all", z, "bias")["seed_mean"], 2)
                            )
            rows.append(row)
        head = ["Depth"]
        for name in ("scratch", "ridge", "climatology"):
            if name in summary["methods"]:
                head += [f"{LABELS[name]} {p}" for p in ("2023", "2024") if p in periods]
        L += _table(head, rows)
    if summary["transformer_vs_mlp"]:
        L += ["## Does the Transformer still beat the per-pixel MLP, and where?", ""]
        L += ["RMSE(MLP) minus RMSE(Transformer), 50-200 m; positive = Transformer better.", ""]
        rows = []
        for p, rr in summary["transformer_vs_mlp"].items():
            row = [p]
            for region in REGIONS:
                row.append(_delta_text(rr[region][POOLED]))
            rows.append(row)
        poc_c = summary["long_vs_poc"].get("contrast_poc_2024")
        if poc_c:
            rows.append(
                ["2024, first run (5-year models)"]
                + [_delta_text(poc_c[r][POOLED]) for r in REGIONS]
            )
        L += _table(["Period", "Whole domain", "Arabian Sea", "Bay of Bengal"], rows)
    if summary["vs_climatology"]:
        L += ["## Skill below about 300 m", ""]
        L += [
            "RMSE of each method minus the climatology's, pooled 500-1000 m, whole domain "
            "(negative = better than climatology).",
            "",
        ]
        rows = []
        for name, byp in summary["vs_climatology"].items():
            for p, rr in byp.items():
                rows.append([LABELS[name], p, _delta_text(rr["all"][DEEP])])
        L += _table(["Method", "Period", "Delta RMSE vs climatology [95 % CI]"], rows)
    lvp = summary["long_vs_poc"]
    if lvp:
        L += ["## Eleven training years against the first run's five (same test year, 2024)", ""]
        smp = lvp["sample"]
        L += [
            f"Same days: {smp['same_days']}; same scored points: {smp.get('same_points')} "
            f"(n = {smp['n_long']:.0f} vs {_n_or_dash(smp['n_poc'])}). "
            "RMSE differences are absolute; the two climatologies are fitted on different "
            "periods, so skill is relative to each run's own climatology.",
            "",
        ]
        rows = []
        for name in METHODS:
            if name not in lvp:
                continue
            e = lvp[name]
            for region in REGIONS:
                d = e["difference"][region][POOLED]
                rows.append(
                    [
                        LABELS[name],
                        REGION_LABELS[region],
                        _pm(_cell(summary, name, "2024", region, POOLED, "rmse")),
                        _pm(e["poc"][region][POOLED]["rmse"]),
                        _delta_text(d["rmse"]),
                        _sgn(d["skill_vs_clim"]["diff"]),
                    ]
                )
        L += _table(
            [
                "Method",
                "Region",
                f"RMSE {st['pooled_range_m'][0]:g}-{st['pooled_range_m'][1]:g} m, long",
                "first run",
                "Delta (long - first) [95 % CI]",
                "Delta skill",
            ],
            rows,
        )
    lc = summary["learning_curve"]
    if lc:
        L += ["## Learning curve (headline model, one seed, last N training years)", ""]
        rows = []
        for key, pt in lc["points"].items():
            for p in periods:
                c = pt["periods"][p]["all"][POOLED]
                rows.append(
                    [
                        key,
                        p,
                        _num(c["rmse"]["point"]),
                        f"[{_num(c['rmse']['ci_lo'])}, {_num(c['rmse']['ci_hi'])}]",
                        _delta_text(c["delta_vs_full"]),
                    ]
                )
        L += _table(
            ["Training years", "Test period", "RMSE 50-200 m", "95 % CI", "Delta vs full period"],
            rows,
        )
    argo = summary["argo"]
    if argo:
        L += ["## Argo profiles (whole domain, 50-200 m pooled)", ""]
        L += [argo["note"], ""]
        rows = []
        for y, e in argo["years"].items():
            bt = e.get("pooled_50_200m_bootstrap")
            for c, lab in ARGO_LABELS.items():
                m = e["methods"][c]["pooled_50_200m"]
                ci = bt["rmse"][c] if bt else None
                rows.append(
                    [
                        y,
                        lab,
                        str(e["n_matchups"]),
                        _num(m["rmse"]),
                        "-" if not ci else f"[{_num(ci['ci_lo'])}, {_num(ci['ci_hi'])}]",
                        _sgn(m["bias"]),
                    ]
                )
        L += _table(
            ["Year", "Method", "Matchups", "RMSE vs Argo", "95 % CI (profiles)", "Bias"], rows
        )
    L += [
        "Figures: `figures/rmse_by_depth.png`, `long_vs_first_run_2024.png`, "
        "`years_bias_by_depth.png`, `learning_curve.png`, `argo_by_depth.png`. "
        "Numbers: `summary.json`.",
        "",
    ]
    path.write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------
def _series(summary: dict, name: str, period: str, region: str, metric: str, key: str):
    depths = summary["settings"]["depths"]
    return np.array(
        [
            np.nan
            if (v := _cell(summary, name, period, region, f"{z:g}", metric)[key]) is None
            else v
            for z in depths
        ],
        dtype=float,
    )


def _rmse_figure(summary: dict, path: Path) -> None:
    depths = summary["settings"]["depths"]
    periods = [p for p in summary["settings"]["periods"] if p != "pooled"] or ["pooled"]
    fig, axes = plt.subplots(
        len(periods), 3, figsize=(13, 4.6 * len(periods)), sharey=True, squeeze=False
    )
    for row, p in zip(axes, periods, strict=True):
        for ax, region in zip(row, REGIONS, strict=True):
            for name in METHODS:
                if name not in summary["methods"]:
                    continue
                y = _series(summary, name, p, region, "rmse", "seed_mean")
                sd = _series(summary, name, p, region, "rmse", "seed_sd")
                ax.plot(
                    y,
                    depths,
                    color=COLORS[name],
                    lw=1.8,
                    ls="--" if name == "climatology" else "-",
                    label=LABELS[name],
                )
                if np.isfinite(sd).any():
                    ax.fill_betweenx(depths, y - sd, y + sd, color=COLORS[name], alpha=0.25, lw=0)
            _depth_axis(ax, depths)
            ax.set_title(f"{p}, {REGION_LABELS[region]}", fontsize=9)
            ax.set_xlabel("RMSE vs GLORYS (°C)")
    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Long-period models by test year (bands: ±1 SD across seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _long_vs_poc_figure(summary: dict, path: Path) -> None:
    lvp = summary["long_vs_poc"]
    depths = summary["settings"]["depths"]
    names = [n for n in METHODS if n in lvp]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.3, 1]})
    for name in names:
        cc = lvp[name]["difference"]["all"]
        d = np.array(
            [
                np.nan if cc[f"{z:g}"]["rmse"]["diff"] is None else cc[f"{z:g}"]["rmse"]["diff"]
                for z in depths
            ]
        )
        lo = np.array(
            [
                np.nan if cc[f"{z:g}"]["rmse"]["ci_lo"] is None else cc[f"{z:g}"]["rmse"]["ci_lo"]
                for z in depths
            ]
        )
        hi = np.array(
            [
                np.nan if cc[f"{z:g}"]["rmse"]["ci_hi"] is None else cc[f"{z:g}"]["rmse"]["ci_hi"]
                for z in depths
            ]
        )
        ax.plot(d, depths, color=COLORS[name], lw=1.8, label=LABELS[name])
        ax.fill_betweenx(depths, lo, hi, color=COLORS[name], alpha=0.15, lw=0)
    ax.axvline(0, color="k", lw=0.8)
    _depth_axis(ax, depths)
    ax.set_xlabel("Δ RMSE on 2024 (°C), long-period minus first-run model; negative = long better")
    ax.set_title("By depth, whole domain (95 % bands)")
    ax.legend(fontsize=8, loc="lower left")
    offs = {"all": 0.0, "arabian_sea": -0.2, "bay_of_bengal": 0.2}
    marks = {"all": "o", "arabian_sea": "^", "bay_of_bengal": "s"}
    for i, name in enumerate(names):
        for region in REGIONS:
            d = lvp[name]["difference"][region][POOLED]["rmse"]
            if d["diff"] is None:
                continue
            bx.errorbar(
                d["diff"],
                i + offs[region],
                xerr=[[d["diff"] - d["ci_lo"]], [d["ci_hi"] - d["diff"]]],
                fmt=marks[region],
                color=COLORS[name],
                mfc=COLORS[name] if d.get("established") else "white",
                capsize=2.5,
                ms=5,
                label=REGION_LABELS[region] if i == 0 else None,
            )
    bx.axvline(0, color="k", lw=0.8)
    bx.set_yticks(range(len(names)))
    bx.set_yticklabels([LABELS[n] for n in names], fontsize=8)
    bx.invert_yaxis()
    bx.set_xlabel("Δ pooled 50-200 m RMSE (°C); filled = established")
    bx.legend(fontsize=8)
    bx.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _years_figure(summary: dict, path: Path) -> None:
    depths = summary["settings"]["depths"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.2, 1]})
    for name in ("scratch", "ridge", "climatology"):
        if name not in summary["methods"]:
            continue
        for p, ls in (("2023", "-"), ("2024", "--")):
            if p in summary["settings"]["periods"]:
                ax.plot(
                    _series(summary, name, p, "all", "bias", "seed_mean"),
                    depths,
                    color=COLORS[name],
                    ls=ls,
                    lw=1.6,
                    label=f"{LABELS[name]} {p}",
                )
    ax.axvline(0, color="k", lw=0.8)
    _depth_axis(ax, depths)
    ax.set_xlabel("bias vs GLORYS (°C), method minus GLORYS, whole domain")
    ax.set_title("Bias by depth, 2023 (solid) and 2024 (dashed)")
    ax.legend(fontsize=7.5, loc="lower left")
    ps = [p for p in ("2023", "2024") if p in summary["settings"]["periods"]]
    width = 0.8 / max(len(summary["methods"]), 1)
    for k, name in enumerate(n for n in METHODS if n in summary["methods"]):
        vals = [_cell(summary, name, p, "all", POOLED, "rmse") for p in ps]
        x = np.arange(len(ps)) + k * width
        bx.bar(
            x,
            [v["seed_mean"] for v in vals],
            width,
            color=COLORS[name],
            label=LABELS[name],
            yerr=[
                [v["seed_mean"] - v["ci_lo"] for v in vals],
                [v["ci_hi"] - v["seed_mean"] for v in vals],
            ],
            capsize=2,
        )
    bx.set_xticks(np.arange(len(ps)) + 0.4 - width / 2)
    bx.set_xticklabels(ps)
    bx.set_ylabel("pooled 50-200 m RMSE (°C), whole domain")
    bx.set_title("RMSE by test year (95 % bootstrap interval)")
    bx.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _learning_figure(summary: dict, path: Path) -> None:
    lc = summary["learning_curve"]
    periods = list(summary["settings"]["periods"])
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    keys = sorted(lc["points"], key=int)
    for p, color in zip(periods, ("#0072B2", "#D55E00", "#555555"), strict=False):
        y = [lc["points"][k]["periods"][p]["all"][POOLED]["rmse"]["point"] for k in keys]
        lo = [lc["points"][k]["periods"][p]["all"][POOLED]["rmse"]["ci_lo"] for k in keys]
        hi = [lc["points"][k]["periods"][p]["all"][POOLED]["rmse"]["ci_hi"] for k in keys]
        ax.errorbar(
            [int(k) for k in keys],
            y,
            yerr=[np.subtract(y, lo), np.subtract(hi, y)],
            marker="o",
            color=color,
            capsize=3,
            label=p,
        )
    ax.set_xscale("log")
    ax.set_xticks([int(k) for k in keys])
    ax.set_xticklabels(keys)
    ax.set_xlabel("training years (last N years before validation)")
    ax.set_ylabel("pooled 50-200 m RMSE vs GLORYS (°C)")
    ax.set_title(
        "Learning curve of the headline model (one seed; 95 % day-bootstrap bars)", fontsize=9
    )
    ax.grid(alpha=0.25)
    ax.legend(title="test period", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _argo_figure(summary: dict, path: Path) -> None:
    argo = summary["argo"]
    depths = argo["depths"]
    years = [y for y in argo["years"] if y != "pooled"] or ["pooled"]
    fig, axes = plt.subplots(
        1, len(years), figsize=(5.5 * len(years), 5.2), sharey=True, squeeze=False
    )
    for ax, y in zip(axes[0], years, strict=True):
        for c, lab in ARGO_LABELS.items():
            per = argo["years"][y]["methods"][c]["per_depth"]["rmse"]
            ax.plot(
                np.array([np.nan if v is None else v for v in per]),
                depths,
                color=ARGO_COLORS[c],
                lw=1.8,
                label=lab,
            )
        _depth_axis(ax, depths)
        ax.set_title(f"{y} ({argo['years'][y]['n_profiles']} profiles)")
        ax.set_xlabel("RMSE vs Argo (°C)")
    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Argo: not independent of the training target (GLORYS assimilates Argo)", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def write_figures(summary: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "rmse_by_depth.png"]
    _rmse_figure(summary, paths[0])
    if summary["long_vs_poc"] and any(n in summary["long_vs_poc"] for n in METHODS):
        paths.append(out_dir / "long_vs_first_run_2024.png")
        _long_vs_poc_figure(summary, paths[-1])
    paths.append(out_dir / "years_bias_by_depth.png")
    _years_figure(summary, paths[-1])
    if summary["learning_curve"]:
        paths.append(out_dir / "learning_curve.png")
        _learning_figure(summary, paths[-1])
    if summary["argo"]:
        paths.append(out_dir / "argo_by_depth.png")
        _argo_figure(summary, paths[-1])
    return paths


# ----------------------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------------------
def make_r2_report(
    cfg: Config,
    poc_cfg: Config | None = None,
    n_boot: int = 2000,
    block_length: int | None = None,
    seed: int = 0,
) -> dict:
    """Write ``summary.json``, ``summary.md`` and the figures under ``research/r2/``."""
    res = load_results(cfg, poc_cfg)
    summary = build_summary(res, n_boot=n_boot, block_length=block_length, seed=seed)
    out = r2_dir(cfg)
    write_json(out / "summary.json", summary)
    write_markdown(summary, out / "summary.md", cfg.run_name)
    write_figures(summary, out / "figures")
    return summary
