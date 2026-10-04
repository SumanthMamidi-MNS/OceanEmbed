"""Input-shift check: the released model on near-real-time inputs vs on the reprocessed inputs.

The released model was trained on the reprocessed SST (OSTIA REP) and sea level anomaly (DUACS MY)
products; the live mode feeds it the near-real-time (NRT) versions. For an overlap period in which
all of them exist (and the GLORYS target too) this module

1. downloads the reprocessed and the NRT inputs and the GLORYS target for the overlap days (own
   folders ``data/raw_nrt/overlap_reprocessed``, ``overlap_nrt``; the target is stored harmonised
   under ``data/processed/live_shift_glorys``),
2. harmonises both input sets with the batch rules and runs the released model on each,
3. reports how the inputs differ (bias, RMSE per variable and basin), how the reconstructions
   differ by depth, and each one's RMSE against GLORYS with a paired moving-block bootstrap of the
   change (the research helpers of :mod:`oceanembed.research.bootstrap`).

The NRT products of the overlap days are the NRT archive *as it is now*, i.e. after the revisions of
the first days: this measures the product difference, not the first-publication error; the latter is
bounded separately by the revision statistics of ``live update``.
"""

from __future__ import annotations

import logging
import tempfile
import time as _time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from oceanembed.config import Config
from oceanembed.data.providers.base import log_transfer
from oceanembed.grid import build_grid
from oceanembed.live import nrt
from oceanembed.live.harmonise import harmonise_files, write_store
from oceanembed.live.nrt import LiveError, day_files
from oceanembed.live.reconstruct import load_weights, reconstruct, surface_dataset
from oceanembed.live.state import live_paths, now_utc, stamp, write_json_atomic
from oceanembed.research.bootstrap import block_counts, percentile_ci
from oceanembed.research.common import choose_block_length

log = logging.getLogger(__name__)

POOLED = (50.0, 200.0)
REGIONS = ("all", "arabian_sea", "bay_of_bengal")
REGION_LABELS = {
    "all": "whole domain",
    "arabian_sea": "Arabian Sea",
    "bay_of_bengal": "Bay of Bengal",
}
INPUTS = ("sst", "sla")
SUMS_FILE = "sums.npz"


# ----------------------------------------------------------------------------------------
# data of the overlap
# ----------------------------------------------------------------------------------------
def overlap_days(cfg: Config, probes: dict[str, tuple[date, date]]) -> list[date]:
    """Days of the configured overlap period inside the time axis of every product in ``probes``."""
    sh = cfg.live.input_shift
    lo = max([sh.start, *(a for a, _ in probes.values())])
    hi = min([sh.end, *(b for _, b in probes.values())])
    return [lo + timedelta(days=i) for i in range((hi - lo).days + 1)] if hi >= lo else []


def target_dir(cfg: Config) -> Path:
    return cfg.processed_root / "live_shift_glorys"


def target_file(cfg: Config, day) -> Path:
    return target_dir(cfg) / f"glorys_{pd.Timestamp(day):%Y%m%d}.npz"


def fetch_target(cfg: Config, days: list[date]) -> float:
    """GLORYS temperature of ``days`` harmonised to the 15 depths and the grid (one small file per
    day). Returns the MB downloaded."""
    live = cfg.live
    prov = nrt.CmemsProvider(cfg)
    ds_id = live.reprocessed["temp"]
    nrt.check_credentials()
    mb = 0.0
    for first, last in nrt.runs_of_days(days, 15):
        kw = prov.subset_kwargs("temp", first, last) | {"dataset_id": ds_id}
        t0 = _time.time()
        with tempfile.TemporaryDirectory(
            dir=nrt._tmp_parent(cfg.raw_root), prefix="glorys_"
        ) as tmp:
            part = Path(tmp) / "range.nc"
            nrt._subset_with_retries(cfg, kw, part)
            size = part.stat().st_size
            rng = pd.date_range(first, last, freq="D")
            arr = harmonise_files(cfg, "temp", [part], rng)["temp"]
        target_dir(cfg).mkdir(parents=True, exist_ok=True)
        for i, day in enumerate(rng):
            if not np.isfinite(arr[i]).any():
                continue
            p = target_file(cfg, day)
            tmp_f = p.with_name(p.name + ".tmp")
            with open(tmp_f, "wb") as fh:
                np.savez_compressed(fh, temp=arr[i].astype(np.float32))
            tmp_f.replace(p)
        mb += size / 1e6
        log_transfer(
            cfg, product="temp", dataset=ds_id, period=f"{first}..{last}",
            route="cmems_subset_overlap", bytes_written=size, bytes_transferred=size,
            transfer_basis="file_size", seconds=_time.time() - t0,
        )  # fmt: skip
    return mb


def _ensure_inputs(cfg: Config, days: list[date], download: bool) -> dict:
    """Make sure both input families have a file for every overlap day; returns the folders and
    the MB downloaded."""
    live = cfg.live
    mb = 0.0
    folders = {}
    for kind, ids in (
        ("reprocessed", live.reprocessed),
        ("nrt", {p: nrt.product_dataset(cfg, p) for p in INPUTS}),
    ):
        root = cfg.raw_root / f"overlap_{kind}"
        folders[kind] = root
        for p in INPUTS:
            need = [d for d in days if d not in day_files(root, p)]
            if need and download:
                mb += (
                    nrt.fetch_days(cfg, p, need, dataset_id=ids[p], root=root).bytes_transferred
                    / 1e6
                )
    if download:
        need = [d for d in days if not target_file(cfg, d).exists()]
        if need:
            mb += fetch_target(cfg, need)
    return {"folders": folders, "mb": mb}


def _attrs_of(files: dict[date, Path]) -> dict:
    if not files:
        return {}
    with xr.open_dataset(files[min(files)]) as ds:
        var = next(v for v in ds.data_vars if ds[v].ndim >= 2)
        return {
            "title": ds.attrs.get("title"),
            "product_version": ds.attrs.get("product_version"),
            "variable": var,
            "units": ds[var].attrs.get("units"),
            "long_name": ds[var].attrs.get("long_name"),
        }


# ----------------------------------------------------------------------------------------
# per-day sums
# ----------------------------------------------------------------------------------------
def region_masks(cfg: Config) -> dict[str, np.ndarray]:
    grid = build_grid(cfg)
    out = {"all": np.ones(grid.shape, bool)}
    out.update(grid.basin_masks())
    return {r: out[r] for r in REGIONS}


def recon_sums(rep, nrt_, target, mask3, regions) -> tuple[np.ndarray, np.ndarray]:
    """Sums of one day over each region and depth.

    Returns ``dd (R, D, 3)`` -- pairs where both reconstructions are valid: ``n, sum(nrt - rep),
    sum((nrt - rep)^2)`` -- and ``tt (R, D, 5)`` -- pairs where the target is valid too: ``n,
    sum e_rep, sum e_rep^2, sum e_nrt, sum e_nrt^2`` with ``e = reconstruction - target``."""
    d = nrt_.astype(np.float64) - rep
    ok = mask3 & np.isfinite(d)
    okt = ok & np.isfinite(target)
    er = np.where(okt, rep - target, 0.0)
    en = np.where(okt, nrt_ - target, 0.0)
    d0 = np.where(ok, d, 0.0)
    dd = np.zeros((len(regions), mask3.shape[0], 3))
    tt = np.zeros((len(regions), mask3.shape[0], 5))
    for r, m in enumerate(regions.values()):
        a, b = ok & m, okt & m
        dd[r, :, 0] = a.sum(axis=(1, 2))
        dd[r, :, 1] = np.where(a, d0, 0).sum(axis=(1, 2))
        dd[r, :, 2] = np.where(a, d0**2, 0).sum(axis=(1, 2))
        tt[r, :, 0] = b.sum(axis=(1, 2))
        tt[r, :, 1] = np.where(b, er, 0).sum(axis=(1, 2))
        tt[r, :, 2] = np.where(b, er**2, 0).sum(axis=(1, 2))
        tt[r, :, 3] = np.where(b, en, 0).sum(axis=(1, 2))
        tt[r, :, 4] = np.where(b, en**2, 0).sum(axis=(1, 2))
    return dd, tt


def input_sums(rep: np.ndarray, nrt_: np.ndarray, mask2, regions) -> np.ndarray:
    """``(R, 4)``: ``n, sum(nrt - rep), sum((nrt - rep)^2), sum(rep)`` over valid ocean points."""
    d = nrt_.astype(np.float64) - rep
    ok = mask2 & np.isfinite(d)
    out = np.zeros((len(regions), 4))
    for r, m in enumerate(regions.values()):
        a = ok & m
        out[r] = [a.sum(), d[a].sum(), (d[a] ** 2).sum(), rep[a].sum()]
    return out


def compute_sums(cfg: Config, days: list[date], folders: dict, device: str | None) -> dict:
    """Harmonise both input families, run the model on both and accumulate the per-day sums."""
    w = load_weights(cfg, device)
    grid = build_grid(cfg)
    regions = region_masks(cfg)
    wdays = pd.DatetimeIndex([pd.Timestamp(d) for d in days])
    stores = {}
    surf = {}
    for kind in ("reprocessed", "nrt"):
        arrays: dict[str, np.ndarray] = {}
        for p in INPUTS:
            files = day_files(folders[kind], p)
            arrays.update(harmonise_files(cfg, p, [files[d] for d in days if d in files], wdays))
        store = cfg.processed_root / f"live_shift_{kind}.zarr"
        write_store(store, cfg, wdays, arrays, w.mask, title=f"input-shift store ({kind})")
        stores[kind], surf[kind] = store, arrays
    ds = {k: surface_dataset(cfg, w, wdays[0], wdays[-1], s) for k, s in stores.items()}
    n, nd = len(days), grid.n_depth
    dd = np.zeros((n, len(regions), nd, 3))
    tt = np.zeros((n, len(regions), nd, 5))
    inp = np.zeros((n, len(regions), len(INPUTS), 4))
    map_sum = np.zeros((len(INPUTS), *grid.shape))
    map_n = np.zeros((len(INPUTS), *grid.shape))
    have_target = np.zeros(n, bool)
    t0 = _time.time()
    for a in range(0, n, 8):
        chunk = wdays[a : a + 8]
        rec = {k: reconstruct(ds[k], w, chunk) for k in ds}
        for i, day in enumerate(chunk, start=a):
            tp = target_file(cfg, day)
            target = np.load(tp)["temp"] if tp.exists() else np.full((nd, *grid.shape), np.nan)
            have_target[i] = tp.exists()
            dd[i], tt[i] = recon_sums(
                rec["reprocessed"][day], rec["nrt"][day], target, w.mask, regions
            )
            for v, name in enumerate(INPUTS):
                r_, n_ = surf["reprocessed"][name][i], surf["nrt"][name][i]
                inp[i, :, v] = input_sums(r_, n_, w.mask[0], regions)
                ok = w.mask[0] & np.isfinite(r_) & np.isfinite(n_)
                map_sum[v] += np.where(ok, n_ - r_, 0.0)
                map_n[v] += ok
        log.info("input shift: %d / %d days", min(a + 8, n), n)
    secs = _time.time() - t0
    return {
        "dd": dd, "tt": tt, "inp": inp, "map_sum": map_sum, "map_n": map_n,
        "have_target": have_target, "days": np.array([str(d) for d in days]),
        "depth": grid.depth, "seconds": secs, "device": str(w.device),
    }  # fmt: skip


# ----------------------------------------------------------------------------------------
# statistics (pure; unit tested on constructed fields)
# ----------------------------------------------------------------------------------------
def depth_selector(depths: np.ndarray) -> np.ndarray:
    d = np.asarray(depths, float)
    return ((d >= POOLED[0]) & (d <= POOLED[1])).astype(float)


def rmse_from(sum_e2: np.ndarray, n: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(sum_e2 / n)


def vs_target(tt: np.ndarray, weights: np.ndarray, sel: np.ndarray) -> dict[str, np.ndarray]:
    """RMSE of both reconstructions against the target for one region, from per-day sums
    ``tt (T, D, 5)``, replicate weights ``(B, T)`` and a 0/1 depth selector ``(K, D)``;
    every value is ``(B, K)``."""
    per_day = np.einsum("bt,tdf->bdf", weights, tt)
    cols = np.einsum("bdf,kd->bkf", per_day, sel)
    n = cols[..., 0]
    return {
        "rep": rmse_from(cols[..., 2], n),
        "nrt": rmse_from(cols[..., 4], n),
        "bias_rep": cols[..., 1] / np.where(n > 0, n, np.nan),
        "bias_nrt": cols[..., 3] / np.where(n > 0, n, np.nan),
        "n": n,
    }


def summarise_shift(sums: dict, n_boot: int, seed: int, block_length: int | None = None) -> dict:
    """The whole report as a dict from the per-day sums."""
    dd, tt, inp = sums["dd"], sums["tt"], sums["inp"]
    depth = np.asarray(sums["depth"], float)
    days = list(sums["days"])
    nd = len(depth)
    pooled = depth_selector(depth)
    sel = np.vstack([np.eye(nd), pooled[None]])  # one row per depth, then pooled
    labels = [f"{z:g}" for z in depth] + ["pooled_50_200m"]
    use = np.asarray(sums["have_target"], bool)
    t_idx = np.flatnonzero(use)

    out: dict = {"n_days": len(days), "first_day": days[0], "last_day": days[-1],
                 "n_days_with_target": int(use.sum()), "n_boot": n_boot}  # fmt: skip
    # inputs
    out["inputs"] = {}
    for v, name in enumerate(INPUTS):
        out["inputs"][name] = {}
        for r, region in enumerate(REGIONS):
            s = inp[:, r, v].sum(axis=0)
            n = s[0]
            out["inputs"][name][region] = {
                "n": int(n),
                "bias": float(s[1] / n) if n else None,
                "rmse": float(np.sqrt(s[2] / n)) if n else None,
                "mean_reprocessed": float(s[3] / n) if n else None,
            }
    # difference of the reconstructions, by depth
    out["recon_difference"] = {}
    for r, region in enumerate(REGIONS):
        s = dd[:, r].sum(axis=0)  # (D, 3)
        sp = (s * pooled[:, None]).sum(axis=0)
        out["recon_difference"][region] = {
            "per_depth": {
                f"{z:g}": {
                    "bias": float(s[i, 1] / s[i, 0]) if s[i, 0] else None,
                    "rmse": float(np.sqrt(s[i, 2] / s[i, 0])) if s[i, 0] else None,
                }
                for i, z in enumerate(depth)
            },
            "pooled_50_200m": {
                "bias": float(sp[1] / sp[0]) if sp[0] else None,
                "rmse": float(np.sqrt(sp[2] / sp[0])) if sp[0] else None,
            },
        }
    # against the target, with the paired bootstrap
    out["vs_glorys"] = {}
    if len(t_idx) >= 5:
        tgt = tt[t_idx]
        series = [
            rmse_from(tgt[:, 0, :, 2].sum(1) , tgt[:, 0, :, 0].sum(1)) ** 2,
            rmse_from(tgt[:, 0, :, 4].sum(1), tgt[:, 0, :, 0].sum(1)) ** 2,
        ]  # fmt: skip
        bl, auto = choose_block_length(series, len(t_idx), block_length)
        counts = block_counts(len(t_idx), bl, n_boot, seed)
        one = np.ones((1, len(t_idx)))
        out["block_length_days"] = bl
        for r, region in enumerate(REGIONS):
            pt = vs_target(tgt[:, r], one, sel)
            bt = vs_target(tgt[:, r], counts, sel)
            cells = {}
            for k, lab in enumerate(labels):
                dist = bt["nrt"][:, k] - bt["rep"][:, k]
                lo, hi = percentile_ci(dist[:, None])
                d_pt = float(pt["nrt"][0, k] - pt["rep"][0, k])
                cells[lab] = {
                    "n": int(pt["n"][0, k]),
                    "rmse_reprocessed": _cell(pt["rep"][0, k], bt["rep"][:, k]),
                    "rmse_nrt": _cell(pt["nrt"][0, k], bt["nrt"][:, k]),
                    "bias_reprocessed": float(pt["bias_rep"][0, k]),
                    "bias_nrt": float(pt["bias_nrt"][0, k]),
                    "rmse_change": {
                        "point": d_pt,
                        "ci_lo": float(lo[0]),
                        "ci_hi": float(hi[0]),
                        "excludes_zero": bool(lo[0] > 0 or hi[0] < 0),
                    },
                }
            out["vs_glorys"][region] = cells
    out["headline"] = _headline(out)
    out["headline_text"] = headline_text(out)
    return out


def _cell(point: float, boot: np.ndarray) -> dict:
    lo, hi = percentile_ci(boot[:, None])
    return {"point": float(point), "ci_lo": float(lo[0]), "ci_hi": float(hi[0])}


def _headline(out: dict) -> dict:
    h: dict = {"period": [out["first_day"], out["last_day"]], "n_days": out["n_days"]}
    for name in INPUTS:
        a = out["inputs"][name]["all"]
        h[f"{name}_bias"], h[f"{name}_rmse"] = a["bias"], a["rmse"]
    h["recon_difference_pooled_50_200m"] = out["recon_difference"]["all"]["pooled_50_200m"]
    v = out["vs_glorys"]
    if v:
        for region in REGIONS:
            c = v[region]["pooled_50_200m"]
            h[f"vs_glorys_{region}"] = {
                "rmse_reprocessed": c["rmse_reprocessed"]["point"],
                "rmse_nrt": c["rmse_nrt"]["point"],
                "change": c["rmse_change"],
            }
        c = v["all"]["pooled_50_200m"]["rmse_change"]
        h["measurably_worse"] = bool(c["excludes_zero"] and c["point"] > 0)
        h["change_point"] = c["point"]
    return h


def headline_text(out: dict) -> str:
    h = out["headline"]
    rd = h["recon_difference_pooled_50_200m"]
    s = (
        f"input shift {h['period'][0]} .. {h['period'][1]} ({h['n_days']} days): "
        f"NRT - reprocessed SST bias {h['sst_bias']:+.3f} C, RMSE {h['sst_rmse']:.3f} C; "
        f"sea level anomaly bias {h['sla_bias'] * 100:+.2f} cm, RMSE {h['sla_rmse'] * 100:.2f} cm; "
        f"reconstruction difference 50-200 m RMSE {rd['rmse']:.3f} C"
    )
    c = out["vs_glorys"].get("all", {}).get("pooled_50_200m")
    if c:
        ch = c["rmse_change"]
        verdict = (
            "NRT is measurably worse"
            if ch["excludes_zero"] and ch["point"] > 0
            else "NRT is measurably better"
            if ch["excludes_zero"]
            else "no measurable difference"
        )
        s += (
            f"; RMSE vs GLORYS 50-200 m {c['rmse_reprocessed']['point']:.3f} (reprocessed) vs "
            f"{c['rmse_nrt']['point']:.3f} (NRT), change {ch['point']:+.3f} "
            f"[{ch['ci_lo']:+.3f}, {ch['ci_hi']:+.3f}] C: {verdict}"
        )
    return s


# ----------------------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------------------
def _ver(v) -> str:
    v = str(v)
    return v if v.startswith("v") else f"v{v}"


def write_markdown(path: Path, out: dict, meta: dict) -> None:
    L = ["# Live nowcast input shift: near-real-time vs reprocessed inputs", ""]
    L += [
        f"Overlap period {out['first_day']} .. {out['last_day']} ({out['n_days']} days, "
        f"{out['n_days_with_target']} with the GLORYS target). The same released model "
        "(`models/final`) is run on the reprocessed inputs it was trained on and on the "
        "near-real-time "
        "(NRT) inputs of the same days. The NRT files are the archive as it is now (after the "
        "revisions of the first days), so this is the product difference, not the "
        "first-publication "
        "error.",
        "",
        "## Products",
        "",
        "| Input | Reprocessed (training) | NRT (live) |",
        "|---|---|---|",
    ]
    for p in INPUTS:
        a, b = meta["products"][p]["reprocessed"], meta["products"][p]["nrt"]
        L.append(
            f"| {p} | `{a['dataset']}` {_ver(a.get('product_version'))} ({a.get('units')}) | "
            f"`{b['dataset']}` {_ver(b.get('product_version'))} ({b.get('units')}) |"
        )
    L += [
        "",
        "## Inputs, NRT minus reprocessed (on the 0.25 deg grid, ocean points)",
        "",
        "| Variable | Region | Bias | RMSE | Mean (reprocessed) |",
        "|---|---|---|---|---|",
    ]
    for p, unit, f in (("sst", "C", 1.0), ("sla", "cm", 100.0)):
        for r in REGIONS:
            a = out["inputs"][p][r]
            L.append(
                f"| {p} ({unit}) | {REGION_LABELS[r]} | {a['bias'] * f:+.3f} | "
                f"{a['rmse'] * f:.3f} | "
                f"{a['mean_reprocessed'] * f:.3f} |"
            )
    L += ["", "## Reconstructions, NRT-driven minus reprocessing-driven (degC)", "",
          "| Depth (m) | " + " | ".join(f"{REGION_LABELS[r]} bias / RMSE" for r in REGIONS) + " |",
          "|---|---|---|---|"]  # fmt: skip
    keys = list(out["recon_difference"]["all"]["per_depth"]) + ["pooled_50_200m"]
    for k in keys:
        cells = []
        for r in REGIONS:
            d = out["recon_difference"][r]
            c = d["pooled_50_200m"] if k == "pooled_50_200m" else d["per_depth"][k]
            cells.append(f"{c['bias']:+.3f} / {c['rmse']:.3f}" if c["rmse"] is not None else "-")
        L.append(
            f"| {'pooled 50-200' if k == 'pooled_50_200m' else k} | " + " | ".join(cells) + " |"
        )
    if out["vs_glorys"]:
        L += [
            "",
            f"## Error against GLORYS (block bootstrap, {out['block_length_days']}-day blocks, "
            f"{out['n_boot']} replicates, 95 % intervals; change = NRT - reprocessed, "
            "positive = NRT worse)",
            "",
            "| Region | Depth | RMSE reprocessed | RMSE NRT | Change | Excludes 0 |",
            "|---|---|---|---|---|---|",
        ]
        for r in REGIONS:
            for k, c in out["vs_glorys"][r].items():
                if k not in ("pooled_50_200m", "0", "100", "300", "1000"):
                    continue
                ch = c["rmse_change"]
                L.append(
                    f"| {REGION_LABELS[r]} | {'pooled 50-200' if k == 'pooled_50_200m' else k} | "
                    f"{c['rmse_reprocessed']['point']:.3f} [{c['rmse_reprocessed']['ci_lo']:.3f}, "
                    f"{c['rmse_reprocessed']['ci_hi']:.3f}] | {c['rmse_nrt']['point']:.3f} "
                    f"[{c['rmse_nrt']['ci_lo']:.3f}, {c['rmse_nrt']['ci_hi']:.3f}] | "
                    f"{ch['point']:+.3f} [{ch['ci_lo']:+.3f}, {ch['ci_hi']:+.3f}] | "
                    f"{'yes' if ch['excludes_zero'] else 'no'} |"
                )
    L += ["", "## Headline", "", out["headline_text"], "",
          "Figures: `figures/rmse_by_depth.png`, `figures/input_difference.png`."]  # fmt: skip
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def make_figures(outdir: Path, out: dict, sums: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = outdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    depth = np.asarray(sums["depth"], float)
    # 1. error against GLORYS by depth, both input sets, per region
    if out["vs_glorys"]:
        fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True)
        for ax, r in zip(axes, REGIONS, strict=True):
            c = out["vs_glorys"][r]
            rep = [c[f"{z:g}"]["rmse_reprocessed"]["point"] for z in depth]
            nr = [c[f"{z:g}"]["rmse_nrt"]["point"] for z in depth]
            ax.plot(rep, depth, "o-", label="reprocessed inputs", color="#1f6fb4")
            ax.plot(nr, depth, "s--", label="near-real-time inputs", color="#d9622b")
            ax.set_title(REGION_LABELS[r])
            ax.set_xlabel("RMSE vs GLORYS (degC)")
            ax.grid(alpha=0.3)
        axes[0].set_ylim(1050, -20)
        axes[0].set_ylabel("depth (m)")
        axes[0].legend(frameon=False)
        fig.suptitle(f"Released model, {out['first_day']} .. {out['last_day']}")
        fig.tight_layout()
        fig.savefig(fig_dir / "rmse_by_depth.png", dpi=140)
        plt.close(fig)
    # 2. mean difference maps of the two inputs
    grid_note = sums["map_sum"] / np.where(sums["map_n"] > 0, sums["map_n"], np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, v, unit, f in zip(axes, range(2), ("degC", "cm"), (1.0, 100.0), strict=True):
        m = grid_note[v] * f
        lim = float(np.nanpercentile(np.abs(m), 98)) or 1.0
        im = ax.imshow(m, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto",
                       extent=(45, 105, 5, 30))  # fmt: skip
        ax.set_title(f"{INPUTS[v].upper()}: mean NRT - reprocessed ({unit})")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(fig_dir / "input_difference.png", dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------------------
def run_input_shift(
    cfg: Config, *, download: bool = True, recompute: bool = False, device: str | None = None
) -> dict:
    if cfg.live is None:
        raise LiveError("this config has no `live` section (use configs/live.yaml)")
    paths = live_paths(cfg)
    outdir = paths.shift
    outdir.mkdir(parents=True, exist_ok=True)
    sh = cfg.live.input_shift
    sums_path = outdir / SUMS_FILE
    if sums_path.exists() and not recompute:
        z = np.load(sums_path, allow_pickle=False)
        sums = {k: z[k] for k in z.files}
        meta = _read_meta(outdir)
    else:
        probes = {}
        for kind, ids in (
            ("reprocessed", cfg.live.reprocessed),
            ("nrt", {p: nrt.product_dataset(cfg, p) for p in INPUTS}),
        ):
            for p in INPUTS:
                a = nrt.describe_dataset(ids[p])
                probes[f"{kind}_{p}"] = (a["first"], a["last"])
        a = nrt.describe_dataset(cfg.live.reprocessed["temp"])
        probes["target"] = (a["first"], a["last"])
        days = overlap_days(cfg, probes)
        if len(days) < 10:
            raise LiveError(
                f"the overlap of all products inside {sh.start} .. {sh.end} is only "
                f"{len(days)} day(s)"
            )
        got = _ensure_inputs(cfg, days, download)
        folders = got["folders"]
        days = [
            d for d in days
            if all(d in day_files(folders[k], p) for k in folders for p in INPUTS)
        ]  # fmt: skip
        if len(days) < 10:
            raise LiveError("not enough overlap days on disk; run with downloads enabled")
        sums = compute_sums(cfg, days, folders, device)
        np.savez_compressed(sums_path, **sums)
        meta = {
            "products": {
                p: {
                    "reprocessed": {
                        "dataset": cfg.live.reprocessed[p],
                        **_attrs_of(day_files(folders["reprocessed"], p)),
                    },
                    "nrt": {
                        "dataset": nrt.product_dataset(cfg, p),
                        **_attrs_of(day_files(folders["nrt"], p)),
                    },
                }
                for p in INPUTS
            },
            "target": cfg.live.reprocessed["temp"],
            "download_mb": round(got["mb"], 1),
            "seconds_model": round(float(sums["seconds"]), 1),
            "device": str(sums["device"]),
        }
        write_json_atomic(outdir / "meta.json", meta)
    out = summarise_shift(sums, sh.n_boot, sh.seed)
    out["meta"] = meta
    out["updated"] = stamp(now_utc())
    out["out_dir"] = str(outdir)
    write_json_atomic(outdir / "summary.json", out)
    write_markdown(outdir / "summary.md", out, meta)
    make_figures(outdir, out, sums)
    return out


def _read_meta(outdir: Path) -> dict:
    import json

    p = outdir / "meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"products": {}}
