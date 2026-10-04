"""The benchmark report: every model family under one protocol, ranked, with intervals.

Reads the ``eval.npz`` files of the benchmark jobs (:mod:`oceanembed.research.benchmark`) and of the
earlier stages -- climatology, ridge, per-pixel MLP and the CNN + Transformer of
``research/r2`` (all seven inputs) and the Transformer of ``research/final_inputs`` (SST + sea
level) -- and writes ``outputs/<run>/research/benchmark/summary.{json,md}`` and ``figures/``.

Conventions are those of R1 / R2 / R3: the reference is GLORYS on the 0.25 degree grid; a
moving-block
bootstrap over days with replicates shared by every family, depth, basin and metric (so a paired
difference is formed inside each replicate); a difference is *established* only if its paired
interval excludes zero **and** the per-seed RMSE ranges do not overlap. Periods: 2023, 2024
and both.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from oceanembed.config import Config  # noqa: E402
from oceanembed.eval.evaluate import write_json  # noqa: E402
from oceanembed.eval.metrics import SUM_FIELDS  # noqa: E402
from oceanembed.research import benchmark as B  # noqa: E402
from oceanembed.research import final_inputs, r2  # noqa: E402
from oceanembed.research.common import choose_block_length, make_counts  # noqa: E402
from oceanembed.research.r2_report import _seed_runs, _slice  # noqa: E402
from oceanembed.research.r3_report import (  # noqa: E402
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
    _stat,
    _table,
    columns,
)

log = logging.getLogger(__name__)

DPI = 130
PERIOD_LABELS = {"2023": "2023", "2024": "2024", "pooled": "2023 + 2024"}
COLORS = {  # Okabe-Ito, one colour per family in every figure
    "scratch": "#009E73",
    "unet": "#0072B2",
    "gbt": "#D55E00",
    "rf": "#CC79A7",
    "mlp": "#56B4E9",
    "ridge": "#E69F00",
    "climatology": "#555555",
}
PIXEL_FAMILIES = ("rf", "gbt", "mlp")  # per-pixel models: no spatial context
ORDER = ("scratch", "unet", "gbt", "rf", "mlp", "ridge", "climatology")
SET_SOURCES = {
    "all7": "R2 (`research/r2`): CNN + Transformer, per-pixel MLP, ridge, climatology; this stage: "
    "boosted trees, random forest, plain U-Net",
    "sst_sla": "`research/final_inputs`: CNN + Transformer (the final model); this stage: ridge, "
    "per-pixel MLP, boosted trees, random forest; climatology from R2",
}


# ----------------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------------
def _hardware_gpu() -> str:
    return f"GPU {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "GPU"


def load_results(cfg: Config) -> dict[str, dict[str, dict]]:
    """``{input set: {family: run}}``; raises when the climatology or the Transformer of a set is
    missing or the evaluated samples differ between runs."""
    bench = B.bench_dir(cfg)
    base = r2.r2_dir(cfg)
    out: dict[str, dict[str, dict]] = {}
    for set_name in B.SETS:
        runs: dict[str, dict] = {}
        clim = _seed_runs(base / "climatology")
        if clim is not None:
            runs["climatology"] = clim
        if set_name == "all7":
            for fam in ("scratch", "mlp", "ridge"):
                r = _seed_runs(base / fam)
                if r is not None:
                    runs[fam] = r
        else:
            r = _seed_runs(final_inputs.fi_dir(cfg) / "scratch_sst_sla")
            if r is not None:
                runs["scratch"] = r
        for fam in B.FAMILIES:
            if fam in runs:
                continue
            r = _seed_runs(bench / f"{fam}_{set_name}")
            if r is not None:
                runs[fam] = r
        for need in ("climatology", "scratch"):
            if need not in runs:
                raise FileNotFoundError(f"no finished {need} job for the input set {set_name}")
        ref = runs["climatology"]
        n_idx = SUM_FIELDS.index("n")
        for name, r in runs.items():
            if r["dates"] != ref["dates"] or not np.array_equal(r["depths"], ref["depths"]):
                raise ValueError(f"{set_name}/{name}: evaluated days / depths differ")
            if not np.allclose(r["raw"][:, :, n_idx], ref["raw"][0:1, :, n_idx]):
                raise ValueError(f"{set_name}/{name}: the evaluated sample differs")
        out[set_name] = runs
    return out


def _n_params_transformer(cfg: Config) -> int:
    from oceanembed.models.recon import ReconModel

    return int(sum(p.numel() for p in ReconModel(cfg.model).parameters()))


def cost_of(family: str, run: dict, cfg: Config) -> dict:
    """Training cost per family from the ``done.json`` files: mean seconds per seed, model size,
    hardware."""
    tr = [i.get("train", {}) for i in run["info"]]
    secs = [t.get("seconds") for t in tr if t.get("seconds") is not None]
    sizes: dict = {}
    if family == "gbt":
        sizes = {"trees": int(np.mean([t.get("n_trees", 0) for t in tr]))}
    elif family == "rf":
        sizes = {"trees": int(np.mean([t.get("n_trees", 0) for t in tr])),
                 "nodes": int(np.mean([t.get("n_nodes", 0) for t in tr]))}  # fmt: skip
    elif family == "scratch":
        sizes = {"parameters": _n_params_transformer(cfg)}
    elif family == "unet":
        sizes = {"parameters": int(np.mean([t.get("n_params", 0) for t in tr]))}
    elif family == "mlp":
        sizes = {"parameters": int(np.mean([t.get("n_params", 0) for t in tr]))}
    elif family == "ridge":
        sizes = {"parameters": 15 * (B.N_FEATURES + 1)}
    hw = [i.get("hardware") for i in run["info"] if i.get("hardware")]
    if family in ("scratch", "mlp", "unet") and not hw:
        hw = [_hardware_gpu()]
    elif family in ("gbt", "rf", "ridge") and not hw:
        hw = ["CPU"]
    return {
        "train_seconds_mean": float(np.mean(secs)) if secs else None,
        "train_seconds_min": float(np.min(secs)) if secs else None,
        "train_seconds_max": float(np.max(secs)) if secs else None,
        "score_seconds_mean": float(
            np.mean([i.get("score", {}).get("seconds", np.nan) for i in run["info"]])
        ),
        "epochs": [t.get("epochs_run") for t in tr if t.get("epochs_run")],
        "n_seeds": len(run["seeds"]),
        **sizes,
        "hardware": hw[0] if hw else None,
    }


# ----------------------------------------------------------------------------------------
# computation
# ----------------------------------------------------------------------------------------
def _family_label(f: str) -> str:
    return B.FAMILY_LABELS.get(f, f)


def _rank(entries: dict[str, float]) -> list[str]:
    return sorted(entries, key=lambda k: entries[k])


def build_set_summary(
    runs: dict[str, dict], cfg: Config, n_boot: int, block_length: int | None, seed: int
) -> dict:
    clim = runs["climatology"]
    depths = np.asarray(clim["depths"], dtype=float)
    dates = np.array(clim["dates"], dtype="datetime64[D]")
    col_names, sel = columns(depths)
    years = sorted({int(str(d)[:4]) for d in dates})
    periods = {
        str(y): np.flatnonzero(dates.astype("datetime64[Y]") == np.datetime64(str(y)))
        for y in years
    }
    periods["pooled"] = np.arange(len(dates))
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

    def bundle(name: str, period: str) -> dict:
        key = (name, period)
        if key not in cache:
            idx = periods[period]
            run = runs[name]
            cache[key] = _bundle(
                _slice(run, idx),
                list(range(len(run["seeds"]))),
                clim["raw"][0][:, :, idx],
                sel,
                counts[period],
                ones[period],
            )
        return cache[key]

    names = [f for f in ORDER if f in runs]
    out: dict = {
        "families": {},
        "settings": {
            "periods": {
                p: {"n_days": int(len(i)), "start": str(dates[i[0]]), "end": str(dates[i[-1]])}
                for p, i in periods.items()
            },
            "depths": [float(z) for z in depths],
            "columns": col_names,
            "block_length_days": length,
            "n_boot": n_boot,
            "seed": seed,
        },
    }
    for name in names:
        run = runs[name]
        entry = {
            "label": _family_label(name),
            "seeds": run["seeds"],
            "cost": cost_of(name, run, cfg),
            "periods": {},
        }
        for period in periods:
            b = bundle(name, period)
            entry["periods"][period] = {
                r: {col: _stat(b[r], "rmse", ci) for ci, col in enumerate(col_names)}
                for r in REGIONS
            }
        out["families"][name] = entry

    # ranking by seed-mean pooled 50-200 m RMSE, winner or tie, per period and region
    ranking: dict = {}
    for period in periods:
        ranking[period] = {}
        for r in REGIONS:
            means = {
                n: out["families"][n]["periods"][period][r][POOLED]["seed_mean"] for n in names
            }
            order = _rank(means)
            top = order[0]
            vs_top = {
                n: _delta(bundle(n, period)[r], bundle(top, period)[r], "rmse", ci_pool)
                for n in order[1:]
            }
            tied = [top, *[n for n in order[1:] if not vs_top[n]["established"]]]
            ranking[period][r] = {
                "order": order,
                "top": top,
                "vs_top": vs_top,
                "tie_group": tied,
                "outright_winner": top if len(tied) == 1 else None,
            }
    out["ranking"] = ranking

    # Transformer against every other family
    tvs: dict = {}
    for period in periods:
        tvs[period] = {}
        for r in REGIONS:
            tvs[period][r] = {
                n: {
                    col: _delta(bundle(n, period)[r], bundle("scratch", period)[r], "rmse", ci)
                    for ci, col in enumerate(col_names)
                }
                for n in names
                if n != "scratch"
            }
    out["other_minus_transformer"] = tvs

    # the best per-pixel model against the Transformer, per basin (best chosen on the pooled period)
    pix = [n for n in PIXEL_FAMILIES if n in runs]
    best_pixel: dict = {}
    if pix:
        for r in REGIONS:
            best = min(
                pix, key=lambda n: out["families"][n]["periods"]["pooled"][r][POOLED]["seed_mean"]
            )
            best_pixel[r] = {
                "family": best,
                "periods": {
                    p: _delta(bundle(best, p)[r], bundle("scratch", p)[r], "rmse", ci_pool)
                    for p in periods
                },
            }
    out["best_pixel_vs_transformer"] = best_pixel
    return out


def build_summary(
    results: dict, cfg: Config, n_boot: int = 2000, block_length: int | None = None, seed: int = 0
) -> dict:
    return {
        "protocol": {
            "reference": "GLORYS reanalysis, harmonised to the 0.25 degree grid",
            "data": "eleven training years (2011-2021), validation 2022, test 2023 and 2024",
            "metric": "RMSE pooled over 50-200 m, whole test period and by basin and depth",
            "bootstrap": "moving-block bootstrap over days, 95 % percentile intervals, replicates "
            "shared by every family so differences are paired",
            "established": "paired interval excludes zero AND the per-seed RMSE ranges do not "
            "overlap",
            "seeds": "3 per trained family (ridge and climatology are deterministic)",
        },
        "input_sets": {
            s: {
                "label": B.SET_LABELS[s],
                "groups": list(B.SETS[s]),
                "sources": SET_SOURCES[s],
                **build_set_summary(results[s], cfg, n_boot, block_length, seed),
            }
            for s in B.SETS
        },
    }


# ----------------------------------------------------------------------------------------
# markdown
# ----------------------------------------------------------------------------------------
def _cell(s: dict, fam: str, period: str, region: str, col: str = POOLED) -> dict:
    return s["families"][fam]["periods"][period][region][col]


def _ci(c: dict) -> str:
    return f"[{_num(c['ci_lo'])}, {_num(c['ci_hi'])}]"


def _winner_text(rk: dict) -> str:
    if rk["outright_winner"]:
        return f"{_family_label(rk['top'])} (established)"
    tied = " = ".join(_family_label(n) for n in rk["tie_group"])
    return f"tie: {tied}"


def write_markdown(summary: dict, path: Path, run_name: str) -> None:
    p = summary["protocol"]
    L = [
        f"# Benchmark - every model family under one protocol ({run_name})",
        "",
        f"Reference: {p['reference']}. Data: {p['data']}. Metric: {p['metric']}. "
        f"Intervals: {p['bootstrap']}. A difference is *established* only if "
        f"{p['established']} (marked `*`). Seeds: {p['seeds']}.",
        "",
    ]
    for sname, s in summary["input_sets"].items():
        st = s["settings"]
        L += [
            f"## Input set: {s['label']}",
            "",
            f"Inputs: {', '.join(s['groups'])}. Sources: {s['sources']}. Block bootstrap: "
            f"{st['block_length_days']}-day blocks, {st['n_boot']} replicates.",
            "",
            "### Ranking, RMSE pooled over 50-200 m (degC)",
            "",
            "Seed mean +- SD; the block-bootstrap interval of the seed-mean predictions in "
            "brackets.",
            "",
        ]
        header = ["Rank", "Family"]
        for per in st["periods"]:
            header.append(PERIOD_LABELS[per])
        header += [f"{REGION_LABELS[r]} (both years)" for r in REGIONS[1:]]
        order = s["ranking"]["pooled"]["all"]["order"]
        rows = []
        for i, fam in enumerate(order, 1):
            row = [str(i), _family_label(fam)]
            for per in st["periods"]:
                c = _cell(s, fam, per, "all")
                row.append(f"{_pm(c)} {_ci(c)}")
            for r in REGIONS[1:]:
                c = _cell(s, fam, "pooled", r)
                row.append(f"{_pm(c)} {_ci(c)}")
            rows.append(row)
        L += _table(header, rows)
        L += [
            "Winner by period and region (ties: families not established worse than the best):",
            "",
        ]
        rows = [
            [PERIOD_LABELS[per]] + [_winner_text(s["ranking"][per][r]) for r in REGIONS]
            for per in st["periods"]
        ]
        L += _table(["Period", *[REGION_LABELS[r] for r in REGIONS]], rows)

        L += [
            "### Each family minus the CNN + Transformer (pooled 50-200 m, degC; positive = the "
            "family is worse)",
            "",
        ]
        fams = [n for n in order if n != "scratch"]
        rows = []
        for fam in fams:
            row = [_family_label(fam)]
            for per in st["periods"]:
                row.append(_delta_text(s["other_minus_transformer"][per]["all"][fam][POOLED]))
            for r in REGIONS[1:]:
                row.append(_delta_text(s["other_minus_transformer"]["pooled"][r][fam][POOLED]))
            rows.append(row)
        L += _table(
            ["Family", *[PERIOD_LABELS[per] for per in st["periods"]],
             *[f"{REGION_LABELS[r]} (both years)" for r in REGIONS[1:]]],
            rows,
        )  # fmt: skip
        if s["best_pixel_vs_transformer"]:
            L += ["### The best per-pixel model minus the CNN + Transformer, by basin", ""]
            rows = []
            for r in REGIONS:
                bp = s["best_pixel_vs_transformer"][r]
                rows.append(
                    [REGION_LABELS[r], _family_label(bp["family"])]
                    + [_delta_text(bp["periods"][per]) for per in st["periods"]]
                )
            L += _table(
                ["Region", "Best per-pixel family (chosen on both years)",
                 *[PERIOD_LABELS[per] for per in st["periods"]]],
                rows,
            )  # fmt: skip
        depths = st["depths"]
        L += ["### RMSE by depth, both test years, whole domain (seed mean, degC)", ""]
        rows = []
        for fam in order:
            row = [_family_label(fam)]
            for z in depths:
                row.append(_num(_cell(s, fam, "pooled", "all", f"{z:g}")["seed_mean"], 2))
            rows.append(row)
        L += _table(["Family", *[f"{z:g} m" for z in depths]], rows)
        L += ["### Training cost (per seed)", ""]
        rows = []
        for fam in order:
            c = s["families"][fam]["cost"]
            size = (
                f"{c['parameters']:,} parameters"
                if "parameters" in c
                else f"{c.get('trees', 0):,} trees"
            )
            t = c["train_seconds_mean"]
            rows.append(
                [
                    _family_label(fam),
                    "n/a (no training)" if t is None else f"{t / 60:.1f} min",
                    size if fam != "climatology" else "none",
                    str(c["n_seeds"]),
                    c["hardware"] or "-",
                ]
            )
        L += _table(["Family", "Training time", "Size", "Seeds", "Hardware"], rows)
        L += [
            f"Figures: `figures/rmse_by_depth_{sname}.png`, `figures/ranking_{sname}.png`.",
            "",
        ]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------
def _fig_depth(s: dict, sname: str, out: Path) -> None:
    st = s["settings"]
    depths = st["depths"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    for ax, r in zip(axes, REGIONS, strict=True):
        for fam in s["ranking"]["pooled"]["all"]["order"]:
            y = [_cell(s, fam, "pooled", r, f"{z:g}")["seed_mean"] for z in depths]
            ax.plot(y, depths, "o-", ms=3.5, lw=1.4, color=COLORS[fam], label=_family_label(fam))
        ax.set_title(REGION_LABELS[r])
        ax.set_xlabel("RMSE against GLORYS (degC)")
        _depth_axis(ax, depths)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"RMSE by depth, 2023 + 2024, inputs: {s['label']}")
    fig.tight_layout()
    fig.savefig(out / f"rmse_by_depth_{sname}.png", dpi=DPI)
    plt.close(fig)


def _fig_ranking(s: dict, sname: str, out: Path) -> None:
    periods = list(s["settings"]["periods"])
    order = s["ranking"]["pooled"]["all"]["order"]
    fig, axes = plt.subplots(1, len(periods), figsize=(4.2 * len(periods), 4.2), sharey=True)
    for ax, per in zip(np.atleast_1d(axes), periods, strict=True):
        for i, fam in enumerate(order):
            c = _cell(s, fam, per, "all")
            ax.barh(i, c["seed_mean"], color=COLORS[fam], alpha=0.85)
            ax.plot([c["ci_lo"], c["ci_hi"]], [i, i], color="black", lw=1.2)
            ax.plot(c["seed_values"], [i] * len(c["seed_values"]), "k.", ms=4)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([_family_label(f) for f in order], fontsize=8)
        ax.invert_yaxis()
        lo = min(_cell(s, f, per, "all")["ci_lo"] for f in order)
        ax.set_xlim(max(0, lo - 0.1), None)
        ax.set_title(PERIOD_LABELS[per])
        ax.set_xlabel("pooled 50-200 m RMSE (degC)")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle(
        f"Ranking, inputs: {s['label']} (bars: seed mean; line: bootstrap 95 %; dots: seeds)"
    )
    fig.tight_layout()
    fig.savefig(out / f"ranking_{sname}.png", dpi=DPI)
    plt.close(fig)


def write_figures(summary: dict, out_dir: Path) -> list[Path]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for sname, s in summary["input_sets"].items():
        _fig_depth(s, sname, fig_dir)
        _fig_ranking(s, sname, fig_dir)
    return sorted(fig_dir.glob("*.png"))


def make_benchmark_report(
    cfg: Config, n_boot: int = 2000, block_length: int | None = None, seed: int = 0
) -> Path:
    """Write ``summary.json``, ``summary.md`` and the figures; returns the markdown path."""
    results = load_results(cfg)
    summary = build_summary(results, cfg, n_boot, block_length, seed)
    out = B.bench_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", summary)
    write_markdown(summary, out / "summary.md", cfg.run_name)
    write_figures(summary, out)
    return out / "summary.md"


__all__ = ["make_benchmark_report", "build_summary", "load_results", "json"]
