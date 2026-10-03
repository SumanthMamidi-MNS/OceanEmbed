"""R5 analysis: skill of derived physical quantities and stratified skill, with block-bootstrap CIs.

Reads ``research/r5/sums.npz`` (written by :mod:`oceanembed.research.r5`) and writes
``summary.json``, ``summary.md`` and ``figures/*.png`` in the same folder.

Everything is a function of per-day sums over cells, so a moving-block bootstrap over the test days
(the R1 estimator, block length from the same rule, 2 000 replicates shared by every product,
quantity and stratum) gives each interval, and any ``A - B`` is formed inside each replicate.
Strata that are sets of *days* (seasons) restrict a full-year block-bootstrap replicate to the
days of the season; strata that are sets of *cells* (basins, |SLA| and salinity terciles) use all
days. The ``gain`` of method A over a reference B is ``RMSE_B - RMSE_A`` (positive: A better); a
``contrast`` is the gain in the upper tercile minus the gain in the lower one, with its own
interval, which is what tests "is the gain concentrated where the stratifier is large".
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from oceanembed.config import Config  # noqa: E402
from oceanembed.eval.evaluate import write_json  # noqa: E402
from oceanembed.eval.metrics import SUM_FIELDS  # noqa: E402
from oceanembed.research.common import (  # noqa: E402
    SCORES,
    choose_block_length,
    ci_text,
    make_counts,
    md_table,
    metric_cells,
    num,
    paired,
    score_series,
)
from oceanembed.research.physical import (  # noqa: E402
    CP,
    HC_TOP_M,
    QUANTITIES,
    RHO,
    SEASON_LABELS,
    SEASONS,
    season_of_month,
)
from oceanembed.research.r5 import (  # noqa: E402
    PRODUCT_LABELS,
    REGIONS,
    SUMS_FILE,
    TERCILES,
    load_sums,
    r5_dir,
)

log = logging.getLogger(__name__)

PROD_COLORS = {
    "model": "#0072B2",
    "scratch": "#009E73",
    "mlp": "#56B4E9",
    "ridge": "#E69F00",
    "climatology": "#555555",
}
REGION_LABELS = {
    "all": "whole domain",
    "arabian_sea": "Arabian Sea",
    "bay_of_bengal": "Bay of Bengal",
}
UNIT_SCALE = {"hc300": 1e-9}  # J m-2 -> 1e9 J m-2 in tables
DPI = 130
# (A, B) pairs reported as RMSE(A) - RMSE(B) (negative: A better)
PAIRS = [
    ("model", "climatology"),
    ("model", "ridge"),
    ("model", "mlp"),
    ("model", "scratch"),
    ("scratch", "climatology"),
    ("scratch", "ridge"),
    ("scratch", "mlp"),
]
GAIN_METHODS = ("model", "scratch")
GAIN_REFERENCES = ("climatology", "ridge", "mlp")
F_N, F_SE2 = SUM_FIELDS.index("n"), SUM_FIELDS.index("se2")


class _Ctx:
    """The loaded sums, the replicates and small lookups shared by the builders."""

    def __init__(self, S: dict, n_boot: int, block_length: int | None, seed: int):
        self.S = S
        self.products = [str(p) for p in S["products"]]
        self.pi = {p: i for i, p in enumerate(self.products)}
        self.classes = {str(c): i for i, c in enumerate(S["classes"])}
        self.dates = pd.DatetimeIndex([str(d) for d in S["dates"]])
        self.T = len(self.dates)
        ts = S["temp_sums"]
        c_all = self.classes["region:all"]
        mse = [
            ts[c_all, p, 0, F_SE2] / np.maximum(ts[c_all, p, 0, F_N], 1) for p in range(ts.shape[1])
        ]
        self.block, self.block_auto = choose_block_length(mse, self.T, block_length)
        self.n_boot, self.seed = n_boot, seed
        self.counts = make_counts(self.T, self.block, n_boot, seed)
        self.ones = np.ones((1, self.T))
        self.clim = self.pi["climatology"]

    def pairs(self, wanted=PAIRS):
        return [(a, b) for a, b in wanted if a in self.pi and b in self.pi]

    def day_weights(self, season: str | None):
        if season is None:
            return self.ones, self.counts
        dm = np.array([season_of_month(m) == season for m in self.dates.month], dtype=float)[None]
        return dm, self.counts * dm

    def stratum(self, name: str, season: str | None = None) -> dict:
        """``{product: {score: (point, replicates)}}`` for a class of ``temp_sums``."""
        c = self.classes[name]
        wp, wb = self.day_weights(season)
        ts = self.S["temp_sums"]
        return {
            p: score_series(ts[c, i, 0], ts[c, i, 1], ts[c, self.clim, 0], wp, wb)
            for p, i in self.pi.items()
        }

    def n_in(self, name: str, season: str | None = None) -> int:
        c = self.classes[name]
        wp, _ = self.day_weights(season)
        return int((self.S["temp_sums"][c, self.clim, 0, F_N] * wp[0]).sum())


def _products_block(sc: dict) -> dict:
    return {p: metric_cells(s) for p, s in sc.items()}


def _paired_block(ctx: _Ctx, sc: dict, score: str = "rmse") -> dict:
    return {f"{a}-{b}": paired(sc[a][score], sc[b][score]) for a, b in ctx.pairs()}


def _gains(ctx: _Ctx, by_t: dict[str, dict]) -> dict:
    """Gain of each method over each reference per tercile and the high-minus-low contrast."""
    out = {}
    for a in GAIN_METHODS:
        for b in GAIN_REFERENCES:
            if a not in ctx.pi or b not in ctx.pi:
                continue
            g = {t: paired(by_t[t][b]["rmse"], by_t[t][a]["rmse"]) for t in TERCILES}
            gt = {t: (by_t[t][b]["rmse"][0] - by_t[t][a]["rmse"][0],
                      by_t[t][b]["rmse"][1] - by_t[t][a]["rmse"][1]) for t in TERCILES}  # fmt: skip
            contrast = paired(gt["high"], gt["low"])
            out[f"{a}_over_{b}"] = {"by_tercile": g, "contrast_high_minus_low": contrast}
    return out


# ----------------------------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------------------------
def build_summary(S: dict, n_boot: int = 2000, block_length: int | None = None, seed: int = 0):
    ctx = _Ctx(S, n_boot, block_length, seed)
    quantities = [str(q) for q in S["quantities"]]
    dq = S["dq_sums"]
    n_ocean = float(np.asarray(S["ocean"]).sum() * ctx.T)

    derived: dict = {}
    for qi, q in enumerate(quantities):
        label, unit, _ = QUANTITIES[q]
        entry = {"label": label, "unit": unit, "regions": {}}
        entry["share_of_ocean_cell_days"] = {
            "glorys_defined": float(S["map_finite_glorys"][qi].sum() / n_ocean),
            "common_sample": float(S["map_n"][qi].sum() / n_ocean),
            **{p: float(S["map_finite"][qi, i].sum() / n_ocean) for p, i in ctx.pi.items()},
        }
        for ri, region in enumerate(REGIONS):
            sc = {
                p: score_series(dq[qi, ri, i, 0], dq[qi, ri, i, 1], dq[qi, ri, ctx.clim, 0],
                                ctx.ones, ctx.counts)
                for p, i in ctx.pi.items()
            }  # fmt: skip
            entry["regions"][region] = {
                "n_cell_days": int(dq[qi, ri, ctx.clim, 0, F_N].sum()),
                "products": _products_block(sc),
                "paired_rmse": _paired_block(ctx, sc),
            }
        derived[q] = entry

    seasons: dict = {}
    for s in [None, *SEASONS]:
        key = "all_year" if s is None else s
        seasons[key] = {"n_days": int(ctx.day_weights(s)[0].sum()), "regions": {}}
        for region in REGIONS:
            sc = ctx.stratum(f"region:{region}", s)
            seasons[key]["regions"][region] = {
                "n_samples": ctx.n_in(f"region:{region}", s),
                "products": _products_block(sc),
                "paired_rmse": _paired_block(ctx, sc),
            }

    sla: dict = {}
    for region in REGIONS:
        by_t = {t: ctx.stratum(f"sla:{region}:{t}") for t in TERCILES}
        sla[region] = {
            "terciles": {
                t: {
                    "n_samples": ctx.n_in(f"sla:{region}:{t}"),
                    "products": _products_block(by_t[t]),
                    "paired_rmse": _paired_block(ctx, by_t[t]),
                }
                for t in TERCILES
            },
            "gains": _gains(ctx, by_t),
        }
    by_t = {t: ctx.stratum(f"sss:bay_of_bengal:{t}") for t in TERCILES}
    sss = {
        "terciles": {
            t: {
                "n_samples": ctx.n_in(f"sss:bay_of_bengal:{t}"),
                "products": _products_block(by_t[t]),
                "paired_rmse": _paired_block(ctx, by_t[t]),
            }
            for t in TERCILES
        },
        "gains": _gains(ctx, by_t),
    }
    # is the salinity effect a season effect in disguise? composition and a within-season repeat
    ts = S["temp_sums"]
    month_season = np.array([season_of_month(m) for m in ctx.dates.month])
    sss["season_composition"] = {}
    for t in TERCILES:
        n_day = ts[ctx.classes[f"sss:bay_of_bengal:{t}"], ctx.clim, 0, F_N]
        sss["season_composition"][t] = {
            s: float(n_day[month_season == s].sum() / max(n_day.sum(), 1)) for s in SEASONS
        }
    sss["gains_within_season"] = {}
    for s in SEASONS:
        by_ts = {t: ctx.stratum(f"sss:bay_of_bengal:{t}", s) for t in TERCILES}
        sss["gains_within_season"][s] = {
            "n_samples": {t: ctx.n_in(f"sss:bay_of_bengal:{t}", s) for t in TERCILES},
            "gains": _gains(ctx, by_ts),
        }
    return {
        "settings": {
            "reference": "GLORYS reanalysis, harmonised to the 0.25 degree grid",
            "n_days": ctx.T,
            "start": str(ctx.dates[0].date()),
            "end": str(ctx.dates[-1].date()),
            "products": {p: PRODUCT_LABELS[p] for p in ctx.products},
            "depths": [float(z) for z in S["depths"]],
            "pooled_range_m": [50.0, 200.0],
            "scores": [m for m in SCORES if m != "mse"],
            "constants": {"rho_kg_m3": RHO, "cp_J_kg_K": CP, "heat_content_top_m": HC_TOP_M},
            "bootstrap": {
                "kind": "moving-block bootstrap over test days (overlapping blocks, no wrap)",
                "n_replicates": n_boot,
                "block_length_days": ctx.block,
                "block_length_auto_days": ctx.block_auto,
                "ci": "95 % percentile interval",
                "seed": seed,
                "seasons": "full-year replicates restricted to the days of the season",
            },
            "edges": {
                "abs_sla_m": [float(x) for x in S["sla_edges"]],
                "bob_surface_salinity": [float(x) for x in S["sss_edges"]],
            },
        },
        "derived": derived,
        "seasons": seasons,
        "sla": sla,
        "sss_bay_of_bengal": sss,
    }


# ----------------------------------------------------------------------------------------
# markdown
# ----------------------------------------------------------------------------------------
def _prod_rows(block: dict, products: list[str], metric_cols, scale: float = 1.0, nd: int = 3):
    rows = []
    for p in products:
        m = block["products"][p]
        row = [PRODUCT_LABELS[p]]
        for name in metric_cols:
            cell = m[name]
            if name == "rmse" or name == "bias":
                row.append(
                    ci_text(cell, nd, scale)
                    if name == "rmse"
                    else num(None if cell["point"] is None else cell["point"] * scale, nd)
                )
            else:
                row.append(num(cell["point"], 3))
        rows.append(row)
    return rows


def write_markdown(summary: dict, path: Path, run_name: str) -> None:
    st = summary["settings"]
    products = list(st["products"])
    L: list[str] = [
        f"# R5 physical metrics - run `{run_name}`",
        "",
        f"Test period {st['start']} .. {st['end']} ({st['n_days']} days), reference GLORYS. "
        "Intervals are "
        f"95 % moving-block bootstrap over days "
        f"(block {st['bootstrap']['block_length_days']} days, "
        f"{st['bootstrap']['n_replicates']} replicates). Derived quantities are evaluated on "
        "the cells where every "
        "product and GLORYS have a defined value (one common sample).",
        "",
        "## Derived quantities, whole domain",
        "",
    ]
    for q, entry in summary["derived"].items():
        scale = UNIT_SCALE.get(q, 1.0)
        unit = "1e9 J m-2" if q in UNIT_SCALE else entry["unit"]
        reg = entry["regions"]["all"]
        sh = entry["share_of_ocean_cell_days"]
        L += [f"### {entry['label']} ({unit})", ""]
        L.append(
            f"Defined in {100 * sh['glorys_defined']:.1f} % of GLORYS ocean cell-days; "
            f"{100 * sh['common_sample']:.1f} % in every product (the scored sample, "
            f"{reg['n_cell_days']:,} cell-days)."
        )
        L.append("")
        rows = []
        for p in products:
            m = reg["products"][p]
            rows.append([
                PRODUCT_LABELS[p],
                ci_text(m["rmse"], 3, scale),
                num(None if m["bias"]["point"] is None else m["bias"]["point"] * scale, 3),
                ci_text(m["corr_anom"], 3),
                ci_text(m["skill_vs_clim"], 3),
            ])  # fmt: skip
        L += md_table(
            ["Product", f"RMSE ({unit})", "bias", "anomaly corr.", "skill vs clim."], rows
        )
        L.append("")
    L += ["## D20, D23 and heat content by basin (RMSE)", ""]
    for q in ("d20", "d23", "hc300"):
        entry = summary["derived"].get(q)
        if not entry:
            continue
        scale = UNIT_SCALE.get(q, 1.0)
        unit = "1e9 J m-2" if q in UNIT_SCALE else entry["unit"]
        rows = [
            [PRODUCT_LABELS[p]]
            + [ci_text(entry["regions"][r]["products"][p]["rmse"], 3, scale) for r in REGIONS[1:]]
            for p in products
        ]
        L += [f"**{entry['label']}** (RMSE, {unit})", ""]
        L += md_table(["Product", *(REGION_LABELS[r] for r in REGIONS[1:])], rows) + [""]

    L += ["## Pooled 50-200 m temperature by season and basin", ""]
    for region in REGIONS:
        L += [f"**{REGION_LABELS[region]}**: RMSE (degC) [95 % CI] / anomaly correlation", ""]
        header = ["Season (days)", *(PRODUCT_LABELS[p] for p in products)]
        rows = []
        for key, blk in summary["seasons"].items():
            name = "all year" if key == "all_year" else SEASON_LABELS[key]
            b = blk["regions"][region]
            rows.append([
                f"{name} ({blk['n_days']})",
                *(f"{ci_text(b['products'][p]['rmse'], 2)} / "
                  f"{num(b['products'][p]['corr_anom']['point'], 2)}"
                  for p in products),
            ])  # fmt: skip
        L += md_table(header, rows) + [""]

    def tercile_tables(block: dict, title: str, labels: dict) -> None:
        L.append(f"## {title}")
        L.append("")
        for t in TERCILES:
            b = block["terciles"][t]
            L.append(f"- {labels[t]}: {b['n_samples']:,} cell-day-depth samples")
        L.append("")
        header = ["Tercile", *(PRODUCT_LABELS[p] for p in products)]
        rows = [
            [
                labels[t],
                *(ci_text(block["terciles"][t]["products"][p]["rmse"], 2) for p in products),
            ]
            for t in TERCILES
        ]
        L.extend(["Pooled 50-200 m RMSE (degC)", ""])
        L.extend(md_table(header, rows))
        L.append("")
        rows = [
            [
                labels[t],
                *(
                    num(block["terciles"][t]["products"][p]["corr_anom"]["point"], 2)
                    for p in products
                ),
            ]
            for t in TERCILES
        ]
        L.extend(["Anomaly correlation", ""])
        L.extend(md_table(header, rows))
        L.append("")
        L.extend(
            [
                "Gain of OceanEmbed over a reference "
                "(RMSE_ref - RMSE_method, degC, positive = better)",
                "",
            ]
        )
        rows = []
        for key, g in block["gains"].items():
            a, b = key.split("_over_")
            c = g["contrast_high_minus_low"]
            rows.append([
                f"{PRODUCT_LABELS[a]} over {PRODUCT_LABELS[b]}",
                *(ci_text(g["by_tercile"][t], 3) for t in TERCILES),
                f"{ci_text(c, 3)}{' *' if c.get('excludes_zero') else ''}",
            ])  # fmt: skip
        L.extend(
            md_table(["Method over reference", *(labels[t] for t in TERCILES), "high - low"], rows)
        )
        L.extend(["", "`*`: the interval of the high-minus-low contrast excludes zero.", ""])

    lo, hi = summary["settings"]["edges"]["abs_sla_m"]
    sla_labels = {
        "low": f"abs(SLA) < {lo:.3f} m",
        "mid": f"{lo:.3f} - {hi:.3f} m",
        "high": f"abs(SLA) >= {hi:.3f} m",
    }
    for region in REGIONS:
        tercile_tables(
            summary["sla"][region],
            f"Eddy regime (abs(SLA) terciles), {REGION_LABELS[region]}",
            sla_labels,
        )
    lo, hi = summary["settings"]["edges"]["bob_surface_salinity"]
    sss_labels = {
        "low": f"SSS < {lo:.2f}",
        "mid": f"{lo:.2f} - {hi:.2f}",
        "high": f"SSS >= {hi:.2f}",
    }
    tercile_tables(
        summary["sss_bay_of_bengal"],
        "Bay of Bengal by surface-salinity tercile (barrier-layer proxy, low salinity = fresh)",
        sss_labels,
    )
    sss = summary["sss_bay_of_bengal"]
    L += ["### Is it a season effect? Share of each salinity tercile's samples by season", ""]
    rows = [
        [sss_labels[t], *(f"{100 * sss['season_composition'][t][s]:.0f} %" for s in SEASONS)]
        for t in TERCILES
    ]
    L += md_table(["Tercile", *(SEASON_LABELS[s] for s in SEASONS)], rows) + [""]
    L += [
        "Gain over the per-pixel MLP (RMSE_MLP - RMSE_method, degC) by salinity tercile, "
        "within each season",
        "",
    ]
    rows = []
    for method in GAIN_METHODS:
        for s in SEASONS:
            g = sss["gains_within_season"][s]["gains"].get(f"{method}_over_mlp")
            if g is None:
                continue
            rows.append(
                [PRODUCT_LABELS[method], SEASON_LABELS[s]]
                + [ci_text(g["by_tercile"][t], 2) for t in TERCILES]
            )
    L += md_table(["Method", "Season", *(sss_labels[t] for t in TERCILES)], rows) + [""]
    path.write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------
def _cell(block: dict, p: str, metric: str) -> tuple[float, float, float]:
    c = block["products"][p][metric]
    if c["point"] is None:
        return np.nan, np.nan, np.nan
    return c["point"], c["ci_lo"], c["ci_hi"]


def _grouped_bars(
    ax, groups: list[str], blocks: list[dict], products: list[str], metric: str
) -> None:
    w = 0.8 / len(products)
    for k, p in enumerate(products):
        vals = np.array([_cell(b, p, metric) for b in blocks])
        x = np.arange(len(groups)) + (k - (len(products) - 1) / 2) * w
        ax.bar(
            x,
            vals[:, 0],
            w,
            color=PROD_COLORS[p],
            label=PRODUCT_LABELS[p],
            yerr=[vals[:, 0] - vals[:, 1], vals[:, 2] - vals[:, 0]],
            error_kw={"lw": 0.8, "capsize": 1.5},
        )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=8)
    ax.grid(alpha=0.25, axis="y")


def fig_season(summary: dict, path: Path) -> None:
    products = list(summary["settings"]["products"])
    keys = [k for k in summary["seasons"] if k != "all_year"]
    names = [SEASON_LABELS[k].replace(" (", "\n(") for k in keys]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    for ax, region in zip(axes, REGIONS, strict=True):
        _grouped_bars(
            ax, names, [summary["seasons"][k]["regions"][region] for k in keys], products, "rmse"
        )
        ax.set_title(REGION_LABELS[region])
    axes[0].set_ylabel("pooled 50-200 m RMSE vs GLORYS (degC)")
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Thermocline error by season and basin (95 % block-bootstrap CI)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _tercile_figure(
    block: dict, labels: dict, title: str, xlabel: str, path: Path, products: list[str]
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    names = [labels[t] for t in TERCILES]
    _grouped_bars(axes[0], names, [block["terciles"][t] for t in TERCILES], products, "rmse")
    axes[0].set_ylabel("pooled 50-200 m RMSE (degC)")
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.45)
    axes[0].set_title("Error")
    axes[0].legend(fontsize=7, loc="upper left")
    for ax, method in zip(axes[1:], GAIN_METHODS, strict=True):
        for k, ref in enumerate(GAIN_REFERENCES):
            g = block["gains"].get(f"{method}_over_{ref}")
            if g is None:
                continue
            vals = np.array(
                [[g["by_tercile"][t][c] for c in ("point", "ci_lo", "ci_hi")] for t in TERCILES],
                float,
            )
            x = np.arange(3) + (k - 1) * 0.25
            ax.errorbar(
                x, vals[:, 0], yerr=[vals[:, 0] - vals[:, 1], vals[:, 2] - vals[:, 0]],
                fmt="o-", color=PROD_COLORS[ref], capsize=2, label=f"over {PRODUCT_LABELS[ref]}",
            )  # fmt: skip
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(3))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_title(f"Gain of {PRODUCT_LABELS[method]}")
        ax.set_ylabel("RMSE_ref - RMSE_method (degC), >0 = better")
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=7)
    for ax in axes:
        ax.set_xlabel(xlabel)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_skill_table(summary: dict, path: Path) -> None:
    products = list(summary["settings"]["products"])
    qs = [q for q in ("d20", "d23", "hc300") if q in summary["derived"]]
    fig, axes = plt.subplots(len(qs), 1, figsize=(11, 2.3 * len(qs) + 0.6))
    axes = np.atleast_1d(axes)
    for ax, q in zip(axes, qs, strict=True):
        entry = summary["derived"][q]
        scale = UNIT_SCALE.get(q, 1.0)
        unit = "1e9 J m-2" if q in UNIT_SCALE else entry["unit"]
        reg = entry["regions"]["all"]
        rows = []
        for p in products:
            m = reg["products"][p]
            rows.append([
                PRODUCT_LABELS[p],
                ci_text(m["rmse"], 2, scale),
                num(None if m["bias"]["point"] is None else m["bias"]["point"] * scale, 2),
                num(m["corr_anom"]["point"], 2),
                num(m["skill_vs_clim"]["point"], 2),
            ])  # fmt: skip
        ax.axis("off")
        ax.set_title(f"{entry['label']} vs GLORYS, whole domain", loc="left", fontsize=10)
        tab = ax.table(
            cellText=rows,
            colLabels=["Product", f"RMSE ({unit}) [95 % CI]", "bias", "anom. corr.", "skill"],
            loc="center", cellLoc="center",
        )  # fmt: skip
        tab.auto_set_font_size(False)
        tab.set_fontsize(8.5)
        tab.scale(1, 1.25)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_d20_maps(S: dict, path: Path, product: str = "model", quantity: str = "d20") -> None:
    qi = [str(q) for q in S["quantities"]].index(quantity)
    products = [str(p) for p in S["products"]]
    pi = products.index(product)
    n = S["map_n"][qi]
    ok = n >= max(10, 0.1 * len(S["dates"]))
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.where(ok, S["map_glorys"][qi] / n, np.nan)
        m = np.where(ok, S["map_sum"][qi, pi] / n, np.nan)
        rmse = np.where(ok, np.sqrt(S["map_se2"][qi, pi] / n), np.nan)
    lon, lat = S["lon"], S["lat"]
    ext = (lon[0], lon[-1], lat[0], lat[-1])
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), sharey=True)
    vmax = np.nanpercentile(np.concatenate([g[ok], m[ok]]), 98)
    vmin = np.nanpercentile(np.concatenate([g[ok], m[ok]]), 2)
    panels = [
        (g, "GLORYS", "viridis_r", vmin, vmax, "m"),
        (m, PRODUCT_LABELS[product], "viridis_r", vmin, vmax, "m"),
        (m - g, f"{PRODUCT_LABELS[product]} - GLORYS", "RdBu_r", -15, 15, "m"),
        (rmse, "RMSE of daily D20", "magma_r", 0, 40, "m"),
    ]
    for ax, (arr, title, cmap, lo, hi, unit) in zip(axes, panels, strict=True):
        im = ax.imshow(arr, origin="lower", extent=ext, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("longitude (E)")
        fig.colorbar(im, ax=ax, shrink=0.85, label=unit)
    axes[0].set_ylabel("latitude (N)")
    fig.suptitle(
        "Depth of the 20 degC isotherm, 2024 mean over the days where defined in every product",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def write_figures(summary: dict, S: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    products = list(summary["settings"]["products"])
    paths = {
        "d20_maps": out_dir / "d20_maps.png",
        "skill_table": out_dir / "derived_skill_table.png",
        "season": out_dir / "skill_by_season_basin.png",
        "sla": out_dir / "skill_by_sla_tercile.png",
        "sss": out_dir / "bob_skill_by_salinity_tercile.png",
    }
    fig_d20_maps(S, paths["d20_maps"])
    fig_skill_table(summary, paths["skill_table"])
    fig_season(summary, paths["season"])
    lo, hi = summary["settings"]["edges"]["abs_sla_m"]
    labels = {"low": f"low\n<{lo:.2f} m", "mid": "mid", "high": f"high\n>={hi:.2f} m"}
    _tercile_figure(
        summary["sla"]["all"], labels, "Skill by eddy regime (|SLA| terciles), whole domain",
        "|SLA| tercile", paths["sla"], products,
    )  # fmt: skip
    lo, hi = summary["settings"]["edges"]["bob_surface_salinity"]
    labels = {"low": f"fresh\n<{lo:.1f}", "mid": "mid", "high": f"salty\n>={hi:.1f}"}
    _tercile_figure(
        summary["sss_bay_of_bengal"], labels,
        "Bay of Bengal skill by surface-salinity tercile (barrier-layer proxy)",
        "surface salinity tercile", paths["sss"], products,
    )  # fmt: skip
    return list(paths.values())


def make_r5_report(
    cfg: Config, n_boot: int = 2000, block_length: int | None = None, seed: int = 0
) -> dict:
    """Write ``summary.json``, ``summary.md`` and the figures under ``research/r5/``."""
    out = r5_dir(cfg)
    path = out / SUMS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `oceanembed research r5` first")
    S = load_sums(path)
    summary = build_summary(S, n_boot=n_boot, block_length=block_length, seed=seed)
    write_json(out / "summary.json", summary)
    write_markdown(summary, out / "summary.md", cfg.run_name)
    write_figures(summary, S, out / "figures")
    return summary


__all__ = ["build_summary", "make_r5_report", "write_markdown"]
