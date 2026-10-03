"""Figures (matplotlib, Agg) and ``report.md`` from the metrics files of a finished run.

One colour per method across every figure (Okabe-Ito colour-blind-safe palette); perceptually
uniform colormaps (``viridis`` for fields and RMSE, ``RdBu_r`` centred on zero for bias and
differences). The depth axis uses a square-root scale so the thermocline is legible, increases
downward and shows the standard depth levels as ticks.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from oceanembed.config import Config  # noqa: E402
from oceanembed.data.harmonize import open_harmonized  # noqa: E402
from oceanembed.eval.evaluate import method_label, metrics_dir  # noqa: E402
from oceanembed.infer.predict import product_path  # noqa: E402

log = logging.getLogger(__name__)

# Okabe-Ito palette
METHOD_COLORS = {
    "model": "#0072B2",  # blue
    "model_scratch": "#009E73",  # bluish green
    "ridge": "#E69F00",  # orange
    "climatology": "#555555",  # grey
    "glorys": "#CC79A7",  # reddish purple
}
_EXTRA_COLORS = ["#D55E00", "#56B4E9", "#F0E442"]
METHOD_STYLES = {"climatology": "--", "glorys": "-."}
FIELD_CMAP = "viridis"
DIVERGING_CMAP = "RdBu_r"
MAP_DEPTHS = (50.0, 100.0, 200.0, 500.0)
DPI = 130


_extra_assigned: dict[str, str] = {}


def color_for(key: str) -> str:
    """Fixed colour per method key (unknown keys get the next free extra colour, once)."""
    if key in METHOD_COLORS:
        return METHOD_COLORS[key]
    if key not in _extra_assigned:
        _extra_assigned[key] = _EXTRA_COLORS[len(_extra_assigned) % len(_EXTRA_COLORS)]
    return _extra_assigned[key]


def _arr(x) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in x], dtype=float)


def _depth_axis(ax, depths, ymax: float | None = None) -> None:
    ax.set_yscale("function", functions=(lambda d: np.sqrt(np.clip(d, 0, None)), lambda s: s**2))
    ymax = float(np.max(depths)) if ymax is None else ymax
    ax.set_ylim(ymax, 0)
    ticks = [t for t in (0, 10, 50, 100, 200, 500, 1000) if t <= ymax]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks])
    ax.set_ylabel("depth (m)")
    ax.grid(alpha=0.3)


def _style(key: str) -> dict:
    return {
        "color": color_for(key),
        "ls": METHOD_STYLES.get(key, "-"),
        "marker": "o",
        "ms": 3.5,
        "lw": 1.6,
    }


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _profile_fig(depths, series, xlabel, title, path, zero_line=False, xlim=None) -> Path:
    fig, ax = plt.subplots(figsize=(4.8, 5.4))
    for key, vals in series.items():
        v = _arr(vals)
        if np.isfinite(v).any():
            ax.plot(v, depths, label=method_label(key), **_style(key))
    if zero_line:
        ax.axvline(0.0, color="k", lw=0.8)
    _depth_axis(ax, depths)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="best")
    return _save(fig, path)


# ----------------------------------------------------------------------------------------
# individual figures
# ----------------------------------------------------------------------------------------
def fig_profiles(gl: dict, fdir: Path, tag: str) -> list[Path]:
    depths = np.array(gl["metadata"]["depths"])
    meth = gl["methods"]
    keys = list(meth)

    def col(name, block="per_depth"):
        return {k: meth[k][block][name] for k in keys}

    sub = f"vs GLORYS, {gl['metadata']['split']} split{tag}"
    out = [
        _profile_fig(
            depths, col("rmse"), "RMSE (degC)", f"RMSE by depth, {sub}", fdir / "rmse_profile.png"
        ),
        _profile_fig(
            depths,
            col("bias"),
            "bias, prediction - reference (degC)",
            f"Bias by depth, {sub}",
            fdir / "bias_profile.png",
            zero_line=True,
        ),
        _profile_fig(
            depths,
            {k: v for k, v in col("corr_anom").items() if k != "climatology"},
            "correlation of anomalies",
            f"Anomaly correlation (climatology removed), {sub}",
            fdir / "anomaly_corr_profile.png",
            xlim=(None, 1.0),
        ),
        _profile_fig(
            depths,
            {k: v for k, v in col("skill_vs_clim").items() if k != "climatology"},
            "skill vs climatology, 1 - MSE / MSE_clim",
            f"Skill score by depth, {sub}",
            fdir / "skill_profile.png",
            zero_line=True,
            xlim=(None, 1.0),
        ),
    ]
    # per-basin RMSE
    basins = list(meth[keys[0]]["per_basin"])
    fig, axes = plt.subplots(1, len(basins), figsize=(4.6 * len(basins), 5.2), sharey=True)
    for ax, b in zip(np.atleast_1d(axes), basins, strict=True):
        for k in keys:
            ax.plot(
                _arr(meth[k]["per_basin"][b]["per_depth"]["rmse"]),
                depths,
                label=method_label(k),
                **_style(k),
            )
        _depth_axis(ax, depths)
        ax.set_xlabel("RMSE (degC)")
        ax.set_title(b.replace("_", " ").title(), fontsize=10)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    out.append(_save(fig, fdir / "rmse_profile_basins.png"))
    return out


def _map_extent(ds: xr.Dataset):
    dlat = float(np.diff(ds["lat"].values).mean())
    dlon = float(np.diff(ds["lon"].values).mean())
    return (
        float(ds["lon"][0]) - dlon / 2,
        float(ds["lon"][-1]) + dlon / 2,
        float(ds["lat"][0]) - dlat / 2,
        float(ds["lat"][-1]) + dlat / 2,
    )


def _cmap(name: str):
    return plt.get_cmap(name).with_extremes(bad="0.88")


def _nearest_depths(depths: np.ndarray, wanted) -> list[int]:
    idx = []
    for w in wanted:
        i = int(np.argmin(np.abs(depths - w)))
        if i not in idx:
            idx.append(i)
    return idx


def fig_maps(maps: xr.Dataset, fdir: Path) -> list[Path]:
    depths = maps["depth"].values
    sel = _nearest_depths(depths, MAP_DEPTHS)
    ext = _map_extent(maps)
    out = []
    # RMSE: model and climatology rows
    rows = [k for k in ("model", "climatology") if f"rmse_{k}" in maps]
    fig, axes = plt.subplots(
        len(rows),
        len(sel),
        figsize=(3.8 * len(sel), 1.9 * len(rows) + 0.9),
        squeeze=False,
        layout="constrained",
    )
    for c, di in enumerate(sel):
        vmax = float(
            np.nanpercentile(
                np.concatenate([maps[f"rmse_{k}"].values[di].ravel() for k in rows]), 99
            )
        )
        for r, k in enumerate(rows):
            ax = axes[r, c]
            im = ax.imshow(
                maps[f"rmse_{k}"].values[di],
                origin="lower",
                extent=ext,
                cmap=_cmap(FIELD_CMAP),
                vmin=0,
                vmax=vmax,
                aspect="auto",
            )
            if r == 0:
                ax.set_title(f"{depths[di]:.0f} m", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{k}\nlat")
            if r == len(rows) - 1:
                ax.set_xlabel("lon")
        fig.colorbar(im, ax=axes[:, c], shrink=0.85, pad=0.02, label="RMSE (degC)")
    fig.suptitle("RMSE maps vs GLORYS (same colour scale per depth)", fontsize=11)
    out.append(_save(fig, fdir / "rmse_maps.png"))
    # model bias maps, diverging around zero
    fig, axes = plt.subplots(
        1, len(sel), figsize=(3.8 * len(sel), 2.6), squeeze=False, layout="constrained"
    )
    for c, di in enumerate(sel):
        b = maps["bias_model"].values[di]
        lim = max(float(np.nanpercentile(np.abs(b), 99)), 1e-6)
        ax = axes[0, c]
        im = ax.imshow(
            b, origin="lower", extent=ext, cmap=_cmap(DIVERGING_CMAP), vmin=-lim, vmax=lim
        )
        ax.set_title(f"{depths[di]:.0f} m", fontsize=10)
        ax.set_xlabel("lon")
        if c == 0:
            ax.set_ylabel("lat")
        fig.colorbar(im, ax=ax, shrink=0.8, label="bias (degC)")
    fig.suptitle("Model bias maps (prediction - GLORYS)", fontsize=11)
    out.append(_save(fig, fdir / "bias_maps.png"))
    return out


def fig_daily_rmse(gl: dict, fdir: Path) -> Path:
    d = gl["daily_rmse"]
    depths = np.array(d["depth"])
    sel = _nearest_depths(depths, MAP_DEPTHS)
    dates = pd.to_datetime(d["dates"])
    fig, axes = plt.subplots(
        len(sel), 1, figsize=(8, 1.9 * len(sel) + 0.9), sharex=True, layout="constrained"
    )
    for ax, di in zip(np.atleast_1d(axes), sel, strict=True):
        for k, series in d["rmse"].items():
            ax.plot(dates, _arr(np.array(series, dtype=object)[:, di]), label=method_label(k),
                    color=color_for(k), ls=METHOD_STYLES.get(k, "-"), lw=1.3)  # fmt: skip
        ax.set_ylabel(f"{depths[di]:.0f} m\nRMSE (degC)", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Daily domain-mean RMSE vs GLORYS", fontsize=10)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, ncol=4, loc="outside lower center")
    return _save(fig, fdir / "daily_rmse.png")


def _read_day(cfg: Config, date: pd.Timestamp):
    """(predicted, target) ``(D, H, W)`` DataArrays for one day, or None if unavailable."""
    path = product_path(cfg, date.strftime("%Y%m"))
    if not path.exists():
        return None
    with xr.open_dataset(path) as nc:
        if date not in pd.DatetimeIndex(nc["time"].values):
            return None
        pred = nc["temperature"].sel(time=date).load()
    z = open_harmonized(cfg)
    if date not in pd.DatetimeIndex(z["time"].values):
        return None
    tgt = z["temp"].sel(time=date).load()
    return pred, tgt


def fig_example_day(cfg: Config, date: pd.Timestamp, fdir: Path, depth: float = 100.0):
    got = _read_day(cfg, date)
    if got is None:
        return None
    pred, tgt = got
    di = int(np.argmin(np.abs(pred["depth"].values - depth)))
    p, t = pred.isel(depth=di).values, tgt.isel(depth=di).values
    ext = _map_extent(pred)
    both = np.concatenate([p[np.isfinite(p)], t[np.isfinite(t)]])
    vmin, vmax = np.percentile(both, [1, 99]) if both.size else (0, 1)
    diff = p - t
    lim = max(float(np.nanpercentile(np.abs(diff), 99)), 1e-6) if np.isfinite(diff).any() else 1
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.0), layout="constrained")
    for ax, f, title in zip(
        axes[:2], (p, t), ("OceanEmbed prediction", "GLORYS target"), strict=True
    ):
        im = ax.imshow(f, origin="lower", extent=ext, cmap=_cmap(FIELD_CMAP), vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("lon")
    fig.colorbar(im, ax=axes[:2], shrink=0.85, pad=0.02, label="temperature (degC)")
    im = axes[2].imshow(
        diff, origin="lower", extent=ext, cmap=_cmap(DIVERGING_CMAP), vmin=-lim, vmax=lim
    )
    axes[2].set_title("difference (prediction - target)", fontsize=10)
    axes[2].set_xlabel("lon")
    fig.colorbar(im, ax=axes[2], shrink=0.85, pad=0.02, label="degC")
    axes[0].set_ylabel("lat")
    fig.suptitle(f"{pred['depth'].values[di]:.0f} m, {date.date()}", fontsize=11)
    return _save(fig, fdir / "example_day_100m.png")


def fig_section(cfg: Config, date: pd.Timestamp, fdir: Path, lat: float = 15.0):
    got = _read_day(cfg, date)
    if got is None:
        return None
    pred, tgt = got
    lats = pred["lat"].values
    if not (lats[0] <= lat <= lats[-1]):
        lat = float(np.mean([lats[0], lats[-1]]))
    li = int(np.argmin(np.abs(lats - lat)))
    p = pred.isel(lat=li).values  # (D, W)
    t = tgt.isel(lat=li).values
    lon, depth = pred["lon"].values, pred["depth"].values
    both = np.concatenate([p[np.isfinite(p)], t[np.isfinite(t)]])
    vmin, vmax = np.percentile(both, [1, 99]) if both.size else (0, 1)
    diff = p - t
    lim = max(float(np.nanpercentile(np.abs(diff), 99)), 1e-6) if np.isfinite(diff).any() else 1
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True, layout="constrained")
    for ax, f, title in zip(
        axes[:2], (p, t), ("OceanEmbed prediction", "GLORYS target"), strict=True
    ):
        im = ax.pcolormesh(
            lon, depth, f, shading="nearest", cmap=_cmap(FIELD_CMAP), vmin=vmin, vmax=vmax
        )
        ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=axes[:2], shrink=0.85, pad=0.02, label="temperature (degC)")
    im = axes[2].pcolormesh(
        lon, depth, diff, shading="nearest", cmap=_cmap(DIVERGING_CMAP), vmin=-lim, vmax=lim
    )
    axes[2].set_title("difference (prediction - target)", fontsize=10)
    fig.colorbar(im, ax=axes[2], shrink=0.85, pad=0.02, label="degC")
    for ax in axes:
        _depth_axis(ax, depth)
        ax.set_xlabel("lon")
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.suptitle(f"Vertical section along {lats[li]:.2f} N, {date.date()}", fontsize=11)
    return _save(fig, fdir / "section_lat.png")


def fig_argo(argo: dict, matchups: pd.DataFrame, fdir: Path) -> Path | None:
    if len(matchups) == 0:
        return None
    depths = np.array(argo["metadata"]["depths"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.2), layout="constrained")
    sub = matchups.sample(min(len(matchups), 20000), random_state=0)
    sc = a1.scatter(
        sub["obs"], sub["model"], c=sub["depth"], s=4, alpha=0.6, cmap=FIELD_CMAP,
        norm=matplotlib.colors.PowerNorm(0.5, vmin=0, vmax=float(sub["depth"].max())),
        rasterized=True,
    )  # fmt: skip
    lo = float(min(sub["obs"].min(), sub["model"].min()))
    hi = float(max(sub["obs"].max(), sub["model"].max()))
    a1.plot([lo, hi], [lo, hi], "k-", lw=0.8)
    a1.set_xlabel("Argo temperature (degC)")
    a1.set_ylabel("OceanEmbed prediction (degC)")
    a1.set_title(f"Argo matchups (n={len(matchups)})", fontsize=10)
    a1.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=a1, shrink=0.8, label="depth (m)")
    for k, v in argo["methods"].items():
        a2.plot(_arr(v["per_depth"]["rmse"]), depths, label=method_label(k), **_style(k))
    _depth_axis(a2, depths)
    a2.set_xlabel("RMSE vs Argo (degC)")
    a2.set_title("RMSE vs Argo by depth", fontsize=10)
    a2.legend(fontsize=8)
    return _save(fig, fdir / "argo_validation.png")


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return pd.DataFrame(rows)


def fig_training(logs_dir: Path, fdir: Path) -> Path | None:
    pre = logs_dir / "pretrain.jsonl"
    recon = sorted(logs_dir.glob("train*.jsonl"))
    if not pre.exists() and not recon:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    if pre.exists():
        d = _read_jsonl(pre)
        axes[0].plot(d["epoch"], d["train_loss"], color="#0072B2", label="train")
        axes[0].plot(d["epoch"], d["val_loss"], color="#D55E00", label="validation")
        if "val_meanfill" in d:
            axes[0].axhline(
                float(d["val_meanfill"].iloc[0]), color="0.4", ls="--", label="mean-fill"
            )
        axes[0].legend(fontsize=8)
    axes[0].set_title("Masked-surface pretraining (MSE)", fontsize=10)
    for p in recon:
        name = "model" if p.stem == "train" else f"model_{p.stem.removeprefix('train_')}"
        d = _read_jsonl(p)
        c, ls = color_for(name), "-"
        axes[1].plot(d["epoch"], d["train_loss"], color=c, ls="--", lw=1.2)
        axes[1].plot(d["epoch"], d["val_loss"], color=c, ls=ls, label=method_label(name))
        axes[2].plot(d["epoch"], d["val_rmse"], color=c, marker="o", ms=3, label=method_label(name))
    axes[1].set_title("Reconstruction loss (dashed: train, solid: val)", fontsize=10)
    axes[2].set_title("Validation RMSE (degC)", fontsize=10)
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    if recon:
        axes[1].legend(fontsize=8)
    return _save(fig, fdir / "training_curves.png")


# ----------------------------------------------------------------------------------------
# markdown
# ----------------------------------------------------------------------------------------
def _f(v, nd=3) -> str:
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _depth_table(block_by_method: dict, depths, field: str, nd=2) -> str:
    keys = list(block_by_method)
    rows = []
    for i, z in enumerate(depths):
        rows.append(
            [f"{z:.0f}"] + [_f(block_by_method[k]["per_depth"][field][i], nd) for k in keys]
        )
    return md_table(["depth (m)", *[method_label(k) for k in keys]], rows)


def _banner(source: str) -> str:
    if source != "synthetic":
        return ""
    return (
        "> **SYNTHETIC DATA - pipeline demonstration only.** Every number and figure below comes "
        "from an analytic synthetic ocean generated by this repository, not from satellite "
        "observations, GLORYS or Argo. They show that the pipeline runs end to end; they are "
        "**not** scientific skill and must not be quoted as such.\n"
    )


def write_markdown(cfg, gl, argo, figures: dict[str, Path | None], run_dir: Path) -> Path:
    md = gl["metadata"]
    source = md["data_source"]
    depths = md["depths"]
    rel = {k: p.relative_to(run_dir).as_posix() for k, p in figures.items() if p is not None}
    lines = [f"# OceanEmbed report - run `{cfg.run_name}`", ""]
    lines += [_banner(source)]
    lines += [
        "## 1. Setup",
        "",
        md_table(
            ["item", "value"],
            [
                ["data source", f"**{source}**"],
                ["domain", f"{cfg.grid.lat_min}-{cfg.grid.lat_max} N, {cfg.grid.lon_min}-"
                           f"{cfg.grid.lon_max} E at {cfg.grid.resolution} deg, daily"],
                ["depth levels (m)", ", ".join(f"{z:.0f}" for z in depths)],
                ["period", f"{cfg.time.start} to {cfg.time.end}"],
                ["train / val / test", f"{cfg.split.train.start}..{cfg.split.train.end} / "
                                       f"{cfg.split.val.start}..{cfg.split.val.end} / "
                                       f"{cfg.split.test.start}..{cfg.split.test.end}"],
                ["evaluated split", f"{md['split']} ({md['start']} to {md['end']}, "
                                    f"{md['n_days']} days)"],
                ["target", md["reference"]],
                ["climatology", f"harmonic fit (mean + annual + semi-annual) on the train split "
                                f"{md['climatology']['train_start']}..{md['climatology']['train_end']}"],
            ],
        ),  # fmt: skip
        "",
        "Methods compared:",
        "",
        md_table(
            ["key", "method", "details"],
            [
                [
                    f"`{k}`",
                    v["label"],
                    (
                        f"checkpoint `{v['checkpoint']}`, epoch {v.get('epoch')}, "
                        f"val RMSE {_f(v.get('val_rmse'))} degC"
                        if v["kind"] == "model"
                        else v.get("checkpoint") or v.get("description", "")
                    ),
                ]
                for k, v in md["methods"].items()
            ],
        ),
        "",
        "Metric conventions: bias = prediction - reference; `corr_raw` is the correlation of "
        "the temperatures themselves (inflated by seasonal, spatial and vertical gradients); "
        "`corr_anom` is the correlation of anomalies with the climatology removed from both "
        "fields (the meaningful day-to-day skill); skill = 1 - MSE / MSE_climatology.",
        "",
        "## 2. Results against the GLORYS target",
        "",
        "### RMSE (degC) by depth",
        "",
        _depth_table(gl["methods"], depths, "rmse"),
        "",
        "### Anomaly correlation by depth",
        "",
        _depth_table({k: v for k, v in gl["methods"].items() if k != "climatology"},
                     depths, "corr_anom"),  # fmt: skip
        "",
        "### Skill vs climatology by depth",
        "",
        _depth_table({k: v for k, v in gl["methods"].items() if k != "climatology"},
                     depths, "skill_vs_clim"),  # fmt: skip
        "",
        "### Model detail by depth",
        "",
        md_table(
            ["depth (m)", "n", "RMSE", "bias", "MAE", "corr_raw", "corr_anom", "skill"],
            [
                [f"{z:.0f}"]
                + [str(int(gl["methods"]["model"]["per_depth"]["n"][i]))]
                + [
                    _f(gl["methods"]["model"]["per_depth"][f][i])
                    for f in ("rmse", "bias", "mae", "corr_raw", "corr_anom", "skill_vs_clim")
                ]
                for i, z in enumerate(depths)
            ],
        ),
        "",
        "### Summary by basin and pooled depth range",
        "",
        md_table(
            ["method", "region", "RMSE all depths", "RMSE 50-200 m", "skill 50-200 m",
             "corr_anom 50-200 m"],
            [
                [method_label(k), region.replace("_", " ")]
                + [
                    _f(blk["overall"]["rmse"]),
                    _f(blk["pooled_50_200m"]["rmse"]),
                    _f(blk["pooled_50_200m"]["skill_vs_clim"]),
                    _f(blk["pooled_50_200m"]["corr_anom"]),
                ]
                for k, m in gl["methods"].items()
                for region, blk in [("all", m), *m["per_basin"].items()]
            ],
        ),
        "",
    ]  # fmt: skip
    if argo is not None:
        am = argo["metadata"]
        lines += [
            "## 3. Validation against Argo profiles",
            "",
            _banner(source),
            f"{am['n_profiles_used']} profiles ({am['n_matchups']} profile-depth matchups) of "
            f"{am['n_profiles_loaded']} loaded; dropped: "
            + ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in am["dropped_profiles"].items())
            + ".",
            "",
            f"Interpolation rule: {am['interpolation_rule']}.",
            "",
            f"**Independence caveat.** {am['independence_note']}",
            "",
            "### RMSE vs Argo (degC) by depth",
            "",
            _depth_table(argo["methods"], depths, "rmse"),
            "",
            "### Bias vs Argo (method - Argo, degC) by depth",
            "",
            _depth_table(argo["methods"], depths, "bias"),
            "",
            "### Matchups per depth",
            "",
            md_table(
                ["depth (m)", "matchups (same sample for every method)"],
                [
                    [f"{z:.0f}", str(argo["methods"]["model"]["per_depth"]["n"][i])]
                    for i, z in enumerate(depths)
                ],
            ),  # fmt: skip
            "",
        ]
        if "gridded_argo" in argo:
            g = argo["gridded_argo"]
            lines += [f"INCOIS gridded ARGO: compared on {g.get('n_times', 0)} time(s); see "
                      "`metrics/metrics_argo.json` (`gridded_argo`).", ""]  # fmt: skip
    lines += ["## 4. Figures", ""]
    captions = {
        "rmse": "RMSE by depth (all methods)",
        "bias": "Bias by depth",
        "anom": "Anomaly correlation by depth",
        "skill": "Skill vs climatology by depth",
        "basins": "RMSE by depth in the Arabian Sea and Bay of Bengal boxes",
        "rmse_maps": "RMSE maps at selected depths (model vs climatology)",
        "bias_maps": "Model bias maps",
        "daily": "Daily domain-mean RMSE time series",
        "day": "Example day at 100 m: prediction, target, difference",
        "section": "Vertical section: prediction, target, difference",
        "argo": "Argo scatter (observed vs model, coloured by depth) and RMSE by depth",
        "training": "Training curves",
    }
    for k, cap in captions.items():
        if k in rel:
            lines += [f"**{cap}**", "", f"![{cap}]({rel[k]})", ""]
    lines += [
        "## 5. Limitations",
        "",
        "- The target is a reanalysis (GLORYS), not observations: the model learns to reproduce a "
        "model product, including its errors. GLORYS assimilates Argo, so the Argo comparison is "
        "independent of the model inputs but not of the training target.",
        "- Daily fields are regridded to 0.25 deg; mesoscale structure below that scale is not "
        "represented. Argo collocation uses the containing cell and the same day only.",
        "- Skill below the thermocline is dominated by the climatology (small variability); "
        "`corr_raw` values are inflated by seasonal and vertical gradients - prefer `corr_anom` "
        "and the skill score.",
        "- One training seed and one temporal split; no uncertainty estimates.",
    ]
    if source == "synthetic":
        lines += [
            "- **Synthetic data**: the surface fields, the GLORYS-like target and the Argo-like "
            "profiles are drawn from one analytic world with a simple surface-subsurface "
            "relation, so skill here says nothing about the real ocean.",
        ]
    path = run_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_report(cfg: Config, example_date: str | None = None) -> Path:
    run_dir = cfg.outputs_dir
    mdir = metrics_dir(cfg)
    gl_path = mdir / "metrics_glorys.json"
    if not gl_path.exists():
        raise FileNotFoundError(f"{gl_path} not found; run `oceanembed evaluate` first")
    gl = json.loads(gl_path.read_text(encoding="utf-8"))
    argo_path = mdir / "metrics_argo.json"
    argo = json.loads(argo_path.read_text(encoding="utf-8")) if argo_path.exists() else None
    fdir = run_dir / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    tag = " [SYNTHETIC]" if gl["metadata"]["data_source"] == "synthetic" else ""

    figures: dict[str, Path | None] = {}
    prof = fig_profiles(gl, fdir, tag)
    figures.update(rmse=prof[0], bias=prof[1], anom=prof[2], skill=prof[3], basins=prof[4])
    maps_path = mdir / "maps_glorys.nc"
    if maps_path.exists():
        with xr.open_dataset(maps_path) as maps:
            maps = maps.load()
        m1, m2 = fig_maps(maps, fdir)
        figures.update(rmse_maps=m1, bias_maps=m2)
    figures["daily"] = fig_daily_rmse(gl, fdir)
    if example_date is None:
        d = pd.DatetimeIndex(pd.to_datetime(gl["daily_rmse"]["dates"]))
        date = d[len(d) // 2]
    else:
        date = pd.Timestamp(example_date)
    figures["day"] = fig_example_day(cfg, date, fdir)
    figures["section"] = fig_section(cfg, date, fdir)
    if argo is not None and (mdir / "argo_matchups.parquet").exists():
        figures["argo"] = fig_argo(argo, pd.read_parquet(mdir / "argo_matchups.parquet"), fdir)
    figures["training"] = fig_training(cfg.logs_dir, fdir)
    return write_markdown(cfg, gl, argo, figures, run_dir)
