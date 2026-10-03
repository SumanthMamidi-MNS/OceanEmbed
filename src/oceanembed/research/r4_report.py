"""R4 analysis: ARMOR3D against Argo and GLORYS with block-bootstrap intervals.

Uses the same estimator as R1/R5: per-day (and per-depth) sums, a moving-block bootstrap over the
test days (replicates shared by every product, depth and basin), paired differences formed inside
each replicate. For Argo the sums are those of the matchup rows of each day; for the grid they are
sums over the cells of the common sample. Writes ``summary.json``, ``summary.md`` and ``figures/``.
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
from oceanembed.eval.evaluate import split_bounds, write_json  # noqa: E402
from oceanembed.eval.metrics import SUM_FIELDS  # noqa: E402
from oceanembed.research import armor3d  # noqa: E402
from oceanembed.research.bootstrap import column_selector, metrics_from_daily  # noqa: E402
from oceanembed.research.common import (  # noqa: E402
    choose_block_length,
    ci_text,
    interval,
    make_counts,
    md_table,
    num,
    paired,
)
from oceanembed.research.physical import tercile_class, tercile_edges  # noqa: E402
from oceanembed.research.r4 import (  # noqa: E402
    GRID_SUMS,
    LABELS,
    REFS,
    REGIONS,
    argo_daily,
    argo_table,
    float_density,
    load_grid_sums,
    r4_dir,
)

log = logging.getLogger(__name__)

POOLED = "pooled_50_200m"
SCORES = ("rmse", "bias", "corr_anom", "skill_vs_clim")
COLORS = {
    "model": "#0072B2",
    "scratch": "#009E73",
    "ridge": "#E69F00",
    "climatology": "#555555",
    "armor3d": "#CC79A7",
    "glorys": "#D55E00",
}
REGION_LABELS = {
    "all": "whole domain",
    "arabian_sea": "Arabian Sea",
    "bay_of_bengal": "Bay of Bengal",
}
# (A, B): RMSE(A) - RMSE(B), negative = A better
PAIRS = [
    ("model", "armor3d"),
    ("scratch", "armor3d"),
    ("glorys", "armor3d"),
    ("armor3d", "ridge"),
    ("armor3d", "climatology"),
    ("model", "glorys"),
    ("model", "ridge"),
]
DENSITY_LABELS = {"low": "few floats nearby", "mid": "some", "high": "many floats nearby"}
F_N, F_SE2 = SUM_FIELDS.index("n"), SUM_FIELDS.index("se2")
DPI = 130
INDEPENDENCE = (
    "ARMOR3D is built from satellite observations and in-situ profiles, Argo among them, so it is "
    "not independent of the floats used here (GLORYS assimilates them too); OceanEmbed sees no "
    "in-situ data at prediction time, but is trained on GLORYS."
)


class _Boot:
    def __init__(self, depths, n_days, mse_series, n_boot, block_length, seed):
        self.depths = np.asarray(depths, dtype=float)
        self.names, self.sel = column_selector(self.depths)
        self.T = n_days
        self.block, self.block_auto = choose_block_length(mse_series, n_days, block_length)
        self.counts = make_counts(n_days, self.block, n_boot, seed)
        self.ones = np.ones((1, n_days))
        self.n_boot, self.seed = n_boot, seed

    def score(self, raw, anom, clim) -> dict:
        """``{score: (point (K,), boot (B, K))}`` from ``(F, T, D)`` sums."""
        pt = metrics_from_daily(raw, anom, clim, self.sel, self.ones)
        bt = metrics_from_daily(raw, anom, clim, self.sel, self.counts)
        return {m: (pt[m][0], bt[m]) for m in (*SCORES, "mse")}


def _columns_block(boot: _Boot, raw, anom, products: list[str], r: int) -> dict:
    """Metrics with intervals per column for every product of one region (``raw`` / ``anom`` are
    ``(R, P, F, T, D)``)."""
    clim = products.index("climatology")
    sc = {p: boot.score(raw[r, i], anom[r, i], raw[r, clim]) for i, p in enumerate(products)}
    n_cols = raw[r, clim, F_N].sum(axis=0) @ boot.sel.T  # samples per column
    out = {
        "columns": boot.names,
        "n": {c: int(n) for c, n in zip(boot.names, n_cols, strict=True)},
        "products": {},
        "paired": {},
    }
    for p in products:
        out["products"][p] = {
            m: {c: interval(sc[p][m][0][k], sc[p][m][1][:, k]) for k, c in enumerate(boot.names)}
            for m in SCORES
        }
    for a, b in PAIRS:
        if a in sc and b in sc:
            out["paired"][f"{a}-{b}"] = {
                m: {
                    c: paired(
                        (sc[a][m][0][k], sc[a][m][1][:, k]), (sc[b][m][0][k], sc[b][m][1][:, k])
                    )
                    for k, c in enumerate(boot.names)
                }
                for m in ("rmse", "corr_anom")
            }
    return out


def _density_block(
    boot: _Boot, cfg: Config, table: pd.DataFrame, days, products, start, end
) -> dict:
    """Does ARMOR3D's advantage grow where more floats are around? Pooled 50-200 m, by terciles of
    the local float density of each profile."""
    prof = table.drop_duplicates("profile_id")[["profile_id", "time", "lat", "lon"]]
    dens = float_density(cfg, prof.reset_index(drop=True), start, end)
    edges = tercile_edges(dens)
    cls = dict(
        zip(prof["profile_id"], tercile_class(dens, edges), strict=True)
    )  # 0 / 1 / 2 per profile
    rows_cls = table["profile_id"].map(cls).to_numpy()
    pooled_k = boot.names.index(POOLED)
    out = {
        "edges": [float(e) for e in edges],
        "terciles": {},
        "median_density": float(np.median(dens)),
    }
    scores = {}
    for ti, t in enumerate(("low", "mid", "high")):
        sub = table[rows_cls == ti]
        d = argo_daily(sub, days, boot.depths, products)
        clim = products.index("climatology")
        sc = {
            p: boot.score(d["raw"][0, i], d["anom"][0, i], d["raw"][0, clim])
            for i, p in enumerate(products)
        }
        scores[t] = sc
        out["terciles"][t] = {
            "n_profiles": int(sub["profile_id"].nunique()),
            "n_matchups": int(len(sub)),
            "products": {
                p: {m: interval(sc[p][m][0][pooled_k], sc[p][m][1][:, pooled_k]) for m in SCORES}
                for p in products
            },
        }
        out["terciles"][t]["paired_rmse"] = {
            f"{a}-{b}": paired(
                (sc[a]["rmse"][0][pooled_k], sc[a]["rmse"][1][:, pooled_k]),
                (sc[b]["rmse"][0][pooled_k], sc[b]["rmse"][1][:, pooled_k]),
            )
            for a, b in (("armor3d", "model"), ("armor3d", "scratch"), ("armor3d", "glorys"))
        }
    contrast = {}
    for other in ("model", "scratch", "glorys"):
        g = {
            t: (
                scores[t]["armor3d"]["rmse"][0][pooled_k] - scores[t][other]["rmse"][0][pooled_k],
                scores[t]["armor3d"]["rmse"][1][:, pooled_k]
                - scores[t][other]["rmse"][1][:, pooled_k],
            )
            for t in ("low", "high")
        }
        contrast[f"armor3d-{other}"] = paired(g["high"], g["low"])
    out["contrast_high_minus_low"] = contrast
    return out


def build_summary(cfg: Config, n_boot=2000, block_length=None, seed=0) -> dict:
    start, end = split_bounds(cfg, "test")
    G = load_grid_sums(r4_dir(cfg) / GRID_SUMS)
    products = [str(p) for p in G["products"]]
    dates = pd.DatetimeIndex([str(d) for d in G["dates"]])
    depths = G["depths"]
    valid_days = G["valid_days"].astype(bool)
    pooled = (depths >= 50) & (depths <= 200)
    raw = G["sums_glorys_raw"]
    mse = [
        raw[0, p, F_SE2][:, pooled].sum(1) / np.maximum(raw[0, p, F_N][:, pooled].sum(1), 1)
        for p in range(len(products))
        if products[p] != "glorys"
    ]
    mse = [s[valid_days] for s in mse]
    boot = _Boot(depths, len(dates), mse, n_boot, block_length, seed)

    table, info = argo_table(cfg, start, end)
    ad = argo_daily(table, dates, depths, products)
    argo = {"counts": info, "regions": {}}
    for r, region in enumerate(REGIONS):
        argo["regions"][region] = _columns_block(boot, ad["raw"], ad["anom"], products, r)
    argo["float_density"] = _density_block(boot, cfg, table, dates, products, start, end)

    grid: dict = {}
    for ref in REFS:
        raw_r, anom_r = G[f"sums_{ref}_raw"], G[f"sums_{ref}_anom"]
        prods = [p for p in products if p != ref]
        idx = [products.index(p) for p in prods]
        grid[ref] = {"products": prods, "regions": {}}
        for r, region in enumerate(REGIONS):
            grid[ref]["regions"][region] = _columns_block(
                boot, raw_r[:, idx], anom_r[:, idx], prods, r
            )
    meta = _dataset_meta(cfg, valid_days, dates)
    return {
        "settings": {
            "test_period": [start, end],
            "n_days": int(len(dates)),
            "n_days_with_armor3d": int(valid_days.sum()),
            "depths": [float(z) for z in depths],
            "columns": boot.names,
            "products": {p: LABELS[p] for p in products},
            "bootstrap": {
                "kind": "moving-block bootstrap over test days (overlapping blocks, no wrap)",
                "n_replicates": n_boot,
                "block_length_days": boot.block,
                "block_length_auto_days": boot.block_auto,
                "ci": "95 % percentile interval",
                "seed": seed,
            },
            "independence_note": INDEPENDENCE,
        },
        "dataset": meta,
        "argo": argo,
        "grid": grid,
    }


def _dataset_meta(cfg: Config, valid_days, dates) -> dict:
    files = sorted(armor3d.armor_dir(cfg).glob("armor3d_??????.nc"))
    versions = set()
    for f in files[:1]:
        import xarray as xr

        with xr.open_dataset(f) as ds:
            versions.add(str(ds.attrs.get("title", "")))
    return {
        "product": armor3d.PRODUCT_ID,
        "dataset_id": armor3d.DATASET_ID,
        "variable": armor3d.VARIABLE,
        "kind": "reprocessed (MY), daily mean, 1/8 degree, 50 levels, 1993-01-01 .. 2024-12-31",
        "version": "202511",
        "title": next(iter(versions), ""),
        "near_real_time_switch_in_2024": False,
        "raw_files": len(files),
        "raw_bytes": int(armor3d.raw_size_bytes(cfg)),
        "first_day": str(dates[valid_days][0].date()) if valid_days.any() else None,
        "last_day": str(dates[valid_days][-1].date()) if valid_days.any() else None,
        "regrid": "vertical linear to the 15 standard depths, then 2x2 block mean (>= 50 % valid); "
        "no temporal interpolation",
    }


# ----------------------------------------------------------------------------------------
# markdown
# ----------------------------------------------------------------------------------------
def _cell(block: dict, p: str, metric: str, col: str, nd=3) -> str:
    return ci_text(block["products"][p][metric][col], nd)


def write_markdown(summary: dict, path: Path, run_name: str) -> None:
    st = summary["settings"]
    ds = summary["dataset"]
    products = list(st["products"])
    L = [
        f"# R4 external benchmark: ARMOR3D - run `{run_name}`",
        "",
        f"ARMOR3D `{ds['dataset_id']}` (version {ds['version']}), temperature `{ds['variable']}`; "
        f"{ds['raw_files']} monthly files, {ds['raw_bytes'] / 1e6:.0f} MB; days with a field: "
        f"{st['n_days_with_armor3d']} of {st['n_days']} ({ds['first_day']} .. {ds['last_day']}). "
        f"Intervals: 95 % moving-block bootstrap over days "
        f"(block {st['bootstrap']['block_length_days']} d, "
        f"{st['bootstrap']['n_replicates']} replicates).",
        "",
        f"**Independence:** {st['independence_note']}",
        "",
        "## Against Argo (identical profile levels for every product)",
        "",
    ]
    c = summary["argo"]["counts"]
    L += [
        f"Matchups kept: {c['matchups_with_armor3d']:,} of {c['matchups_total']:,} "
        f"({c['profiles_with_armor3d']:,} of {c['profiles_total']:,} profiles); "
        "the rest have no ARMOR3D "
        "value in that cell / level.",
        "",
    ]
    for region in REGIONS:
        blk = summary["argo"]["regions"][region]
        L += [f"### {REGION_LABELS[region]}, pooled 50-200 m and all depths", ""]
        rows = []
        for p in products:
            rows.append(
                [
                    LABELS[p],
                    _cell(blk, p, "rmse", POOLED, 2),
                    num(blk["products"][p]["bias"][POOLED]["point"], 2),
                    num(blk["products"][p]["corr_anom"][POOLED]["point"], 2),
                    num(blk["products"][p]["skill_vs_clim"][POOLED]["point"], 2),
                    _cell(blk, p, "rmse", "overall", 2),
                ]
            )
        L += md_table(
            [
                "Product",
                "RMSE 50-200 m",
                "bias",
                "anomaly corr.",
                "skill vs clim.",
                "RMSE all depths",
            ],
            rows,
        )
        L += [f"n = {blk['n'][POOLED]:,} (50-200 m), {blk['n']['overall']:,} (all depths).", ""]
    blk = summary["argo"]["regions"]["all"]
    cols = [c for c in blk["columns"] if not c.startswith(("pooled", "overall"))]
    L += ["### RMSE by depth (whole domain, degC)", ""]
    rows = [
        [LABELS[p], *(num(blk["products"][p]["rmse"][c]["point"], 2) for c in cols)]
        for p in products
    ]
    L += md_table(["Product", *(f"{c} m" for c in cols)], rows) + [""]
    L += ["### Bias by depth (product minus Argo, whole domain, degC)", ""]
    rows = [
        [LABELS[p], *(num(blk["products"][p]["bias"][c]["point"], 2) for c in cols)]
        for p in products
    ]
    L += md_table(["Product", *(f"{c} m" for c in cols)], rows) + [""]
    L += ["### Paired RMSE differences, pooled 50-200 m (A - B; negative = A better)", ""]
    rows = []
    for key in blk["paired"]:
        a, b = key.split("-")
        rows.append(
            [f"{LABELS[a]} - {LABELS[b]}"]
            + [
                ci_text(summary["argo"]["regions"][r]["paired"][key]["rmse"][POOLED], 3)
                for r in REGIONS
            ]
        )
    L += md_table(["A - B", *(REGION_LABELS[r] for r in REGIONS)], rows) + [""]

    d = summary["argo"]["float_density"]
    L += [
        "### Does ARMOR3D benefit from nearby floats?",
        "",
        "Local float density = other Argo profiles within +-3 days and +-2 degrees of the "
        "matchup profile "
        f"(tercile edges {d['edges'][0]:.1f} and {d['edges'][1]:.1f}; "
        f"median {d['median_density']:.1f}). "
        "Pooled 50-200 m RMSE (degC).",
        "",
    ]
    rows = []
    for t in ("low", "mid", "high"):
        b = d["terciles"][t]
        rows.append(
            [DENSITY_LABELS[t], f"{b['n_profiles']:,}"]
            + [ci_text(b["products"][p]["rmse"], 2) for p in products]
        )
    L += md_table(["Density tercile", "profiles", *(LABELS[p] for p in products)], rows) + [""]
    rows = [
        [f"{LABELS[k.split('-')[0]]} - {LABELS[k.split('-')[1]]}", ci_text(v, 3)]
        for k, v in d["contrast_high_minus_low"].items()
    ]
    L += [
        "Difference-in-differences: (RMSE gap, high-density tercile) minus (gap, low one).",
        "",
    ]
    L += md_table(["Gap", "high - low"], rows) + [""]

    for ref in REFS:
        g = summary["grid"][ref]
        L += [
            f"## On the grid against {LABELS[ref]} (cells with mask, GLORYS and ARMOR3D defined)",
            "",
        ]
        prods = g["products"]
        for region in REGIONS:
            blk = g["regions"][region]
            L += [f"### {REGION_LABELS[region]}", ""]
            rows = [
                [
                    LABELS[p],
                    _cell(blk, p, "rmse", POOLED, 3),
                    num(blk["products"][p]["bias"][POOLED]["point"], 3),
                    num(blk["products"][p]["corr_anom"][POOLED]["point"], 3),
                    num(blk["products"][p]["skill_vs_clim"][POOLED]["point"], 3),
                ]
                for p in prods
            ]
            L += md_table(
                ["Product", "RMSE 50-200 m", "bias", "anomaly corr.", "skill vs clim."], rows
            ) + [""]
        blk = g["regions"]["all"]
        cols = [c for c in blk["columns"] if not c.startswith(("pooled", "overall"))]
        rows = [
            [LABELS[p], *(num(blk["products"][p]["rmse"][c]["point"], 2) for c in cols)]
            for p in prods
        ]
        L += ["RMSE by depth (whole domain, degC)", ""]
        L += md_table(["Product", *(f"{c} m" for c in cols)], rows) + [""]
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


def _by_depth(blocks: dict, products, depths, metric, xlabel, title, path: Path, zero_line=False):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5), sharey=True)
    cols = [f"{z:g}" for z in depths]
    for ax, region in zip(axes, REGIONS, strict=True):
        blk = blocks[region]
        for p in products:
            cells = [blk["products"][p][metric][c] for c in cols]
            pt = np.array([np.nan if c["point"] is None else c["point"] for c in cells], float)
            lo = np.array([np.nan if c["point"] is None else c["ci_lo"] for c in cells], float)
            hi = np.array([np.nan if c["point"] is None else c["ci_hi"] for c in cells], float)
            style = {"ls": "--"} if p == "climatology" else {}
            ax.plot(
                pt,
                depths,
                color=COLORS[p],
                lw=2.2 if p == "armor3d" else 1.6,
                label=LABELS[p],
                **style,
            )
            if p in ("armor3d", "model"):
                ax.fill_betweenx(depths, lo, hi, color=COLORS[p], alpha=0.15, lw=0)
        if zero_line:
            ax.axvline(0, color="k", lw=0.8)
        _depth_axis(ax, depths)
        ax.set_title(REGION_LABELS[region])
        ax.set_xlabel(xlabel)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_example(G: dict, path: Path) -> bool:
    keys = ("armor3d", "glorys", "model")
    if not all(f"example_{k}" in G for k in keys):
        return False
    lon, lat = G["lon"], G["lat"]
    ext = (lon[0], lon[-1], lat[0], lat[-1])
    a, g, m = (G[f"example_{k}"] for k in keys)
    fig, axes = plt.subplots(1, 5, figsize=(21, 4), sharey=True)
    vals = np.concatenate([x[np.isfinite(x)] for x in (a, g, m)])
    lo, hi = np.percentile(vals, [2, 98])
    panels = [
        (a, "ARMOR3D", "turbo", lo, hi),
        (g, "GLORYS", "turbo", lo, hi),
        (m, LABELS["model"], "turbo", lo, hi),
        (a - g, "ARMOR3D - GLORYS", "RdBu_r", -4, 4),
        (m - g, f"{LABELS['model']} - GLORYS", "RdBu_r", -4, 4),
    ]
    for ax, (arr, title, cmap, vmin, vmax) in zip(axes, panels, strict=True):
        im = ax.imshow(
            arr, origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto"
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("longitude (E)")
        fig.colorbar(im, ax=ax, shrink=0.85, label="degC")
    axes[0].set_ylabel("latitude (N)")
    fig.suptitle(
        f"Temperature at {float(G['example_depth']):g} m on {G['example_date']}", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return True


def fig_density(summary: dict, path: Path) -> None:
    d = summary["argo"]["float_density"]
    products = ["armor3d", "model", "scratch", "glorys", "ridge"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = 0.8 / len(products)
    for k, p in enumerate(products):
        cells = [d["terciles"][t]["products"][p]["rmse"] for t in ("low", "mid", "high")]
        pt = np.array([c["point"] for c in cells], float)
        lo = np.array([c["ci_lo"] for c in cells], float)
        hi = np.array([c["ci_hi"] for c in cells], float)
        ax.bar(
            np.arange(3) + (k - (len(products) - 1) / 2) * w, pt, w,
            color=COLORS[p], label=LABELS[p],
            yerr=[pt - lo, hi - pt], error_kw={"lw": 0.8, "capsize": 1.5},
        )  # fmt: skip
    ax.set_xticks(range(3))
    ax.set_xticklabels([DENSITY_LABELS[t] for t in ("low", "mid", "high")])
    ax.set_ylabel("pooled 50-200 m RMSE vs Argo (degC)")
    ax.set_title("Error by local Argo density (+-3 d, +-2 deg)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def write_figures(summary: dict, G: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    depths = summary["settings"]["depths"]
    allp = list(summary["settings"]["products"])
    paths = []
    p = out_dir / "rmse_by_depth_argo.png"
    _by_depth(
        summary["argo"]["regions"], allp, depths, "rmse", "RMSE vs Argo (degC)",
        "Error against Argo profiles, identical samples (bands: 95 % block-bootstrap CI)", p,
    )  # fmt: skip
    paths.append(p)
    p = out_dir / "bias_by_depth_argo.png"
    _by_depth(
        summary["argo"]["regions"], allp, depths, "bias", "bias vs Argo (degC, product minus Argo)",
        "Bias against Argo profiles", p, zero_line=True,
    )  # fmt: skip
    paths.append(p)
    for ref in REFS:
        g = summary["grid"][ref]
        p = out_dir / f"rmse_by_depth_grid_vs_{ref}.png"
        _by_depth(
            g["regions"], g["products"], depths, "rmse", f"RMSE vs {LABELS[ref]} (degC)",
            f"Error on the grid against {LABELS[ref]} (cells defined in GLORYS and ARMOR3D)", p,
        )  # fmt: skip
        paths.append(p)
    p = out_dir / "example_100m.png"
    if fig_example(G, p):
        paths.append(p)
    p = out_dir / "argo_density.png"
    fig_density(summary, p)
    paths.append(p)
    return paths


def make_r4_report(cfg: Config, n_boot: int = 2000, block_length: int | None = None, seed: int = 0):
    out = r4_dir(cfg)
    if not (out / GRID_SUMS).exists():
        raise FileNotFoundError(f"{out / GRID_SUMS} not found; run `oceanembed research r4` first")
    summary = build_summary(cfg, n_boot=n_boot, block_length=block_length, seed=seed)
    write_json(out / "summary.json", summary)
    write_markdown(summary, out / "summary.md", cfg.run_name)
    G = load_grid_sums(out / GRID_SUMS)
    write_figures(summary, G, out / "figures")
    return summary


__all__ = ["build_summary", "make_r4_report", "write_markdown"]
