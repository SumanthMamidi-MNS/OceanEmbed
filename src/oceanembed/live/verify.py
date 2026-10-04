"""Running verification of the live reconstruction (part of ``oceanembed live update``).

Two references, scored per day on the days of the live window and kept as a rolling record
(``outputs/live/checks/verification/``):

* **Argo profiles** published so far (argopy / ERDDAP, delayed by days; good-QC flags only),
  collocated by the rule of the evaluated runs (cell containing the profile, same UTC day, the
  vertical rule of ``eval.argo_validation``). Independent of the model's inputs. Scored for the
  reconstruction and for the climatology in three depth bands.
* **The Copernicus operational global analysis** (``GLOBAL_ANALYSISFORECAST_PHY_001_024``, daily
  temperature), on the grid, the same day. A model analysis, not an observation, and not the
  GLORYS reanalysis the model was trained on. Fetched only for the newest ``rolling_days`` days
  and only when the download stays below ``analysis_max_mb``; otherwise skipped with the reason
  recorded.

Daily sums (n, sum of errors, sum of squared errors) are the stored quantity, so any rolling
window is one sum; a day is recomputed on every update while it is in the window (it may have
been revised), older days keep their last row.
"""

from __future__ import annotations

import logging
import tempfile
import time as _time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from oceanembed.config import Config
from oceanembed.data.providers.argo import fetch_argo_box
from oceanembed.data.providers.base import log_transfer
from oceanembed.data.providers.cmems import CmemsProvider
from oceanembed.eval.argo_validation import basin_of, collocate, profile_table
from oceanembed.grid import build_grid
from oceanembed.live import nrt
from oceanembed.live.harmonise import harmonise_files
from oceanembed.live.reconstruct import Weights, read_predictions
from oceanembed.live.state import live_paths, load_days, load_state, stamp, write_json_atomic

log = logging.getLogger(__name__)

BANDS: dict[str, tuple[float, float]] = {
    "0_30": (0.0, 30.0),
    "50_200": (50.0, 200.0),
    "300_1000": (300.0, 1000.0),
}
BAND_LABELS = {"0_30": "0-30 m", "50_200": "50-200 m", "300_1000": "300-1000 m"}
ARGO_REFETCH_DAYS = 15
DAILY_FILE = "daily.parquet"
ARRIVAL_FILE = "argo_arrival.parquet"
DAILY_COLUMNS = [
    "date", "ref", "band", "n", "n_profiles",
    "sum_e_model", "sum_e2_model", "sum_e_clim", "sum_e2_clim",
]  # fmt: skip


# ----------------------------------------------------------------------------------------
# pure scoring helpers (unit tested)
# ----------------------------------------------------------------------------------------
def band_masks(depths: np.ndarray) -> dict[str, np.ndarray]:
    d = np.asarray(depths, dtype=float)
    return {k: (d >= lo) & (d <= hi) for k, (lo, hi) in BANDS.items()}


def daily_rows(
    table: pd.DataFrame, ref: str, bands: dict[str, tuple[float, float]] | None = None
) -> pd.DataFrame:
    """Daily sums per depth band from matchups ``(date, profile_id, depth, obs, model, clim)``;
    errors are ``method - reference``. Days / bands without a valid pair get no row."""
    bands = bands or BANDS
    out = []
    if len(table) == 0:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in DAILY_COLUMNS})
    t = table.copy()
    t["date"] = pd.to_datetime(t["date"]).dt.normalize()
    t["e_model"] = t["model"] - t["obs"]
    t["e_clim"] = t["clim"] - t["obs"]
    for day, g in t.groupby("date"):
        n_prof = int(g["profile_id"].nunique())
        for band, (lo, hi) in bands.items():
            s = g[(g["depth"] >= lo) & (g["depth"] <= hi)]
            if len(s) == 0:
                continue
            out.append(
                {
                    "date": day,
                    "ref": ref,
                    "band": band,
                    "n": int(len(s)),
                    "n_profiles": n_prof,
                    "sum_e_model": float(s["e_model"].sum()),
                    "sum_e2_model": float((s["e_model"] ** 2).sum()),
                    "sum_e_clim": float(s["e_clim"].sum()),
                    "sum_e2_clim": float((s["e_clim"] ** 2).sum()),
                }
            )
    return pd.DataFrame(out, columns=DAILY_COLUMNS)


def metrics_from_rows(rows: pd.DataFrame) -> dict:
    """``{n, model_rmse, model_bias, clim_rmse, clim_bias}`` of summed rows (None if n = 0)."""
    n = float(rows["n"].sum()) if len(rows) else 0.0
    if n <= 0:
        return {
            "n": 0,
            "model_rmse": None,
            "model_bias": None,
            "clim_rmse": None,
            "clim_bias": None,
        }
    return {
        "n": int(n),
        "model_rmse": float(np.sqrt(rows["sum_e2_model"].sum() / n)),
        "model_bias": float(rows["sum_e_model"].sum() / n),
        "clim_rmse": float(np.sqrt(rows["sum_e2_clim"].sum() / n)),
        "clim_bias": float(rows["sum_e_clim"].sum() / n),
    }


def rolling_series(daily: pd.DataFrame, ref: str, rolling_days: int, dates=None) -> list[dict]:
    """Per day (``dates``, default every day that has a row for ``ref``): the day's own metrics per
    band, and the metrics over the trailing ``rolling_days`` calendar days (days without matchups
    simply add nothing)."""
    d = daily[daily["ref"] == ref]
    if len(d) == 0:
        return []
    out = []
    for day in sorted(d["date"].unique()) if dates is None else sorted(dates):
        day = pd.Timestamp(day)
        lo = day - pd.Timedelta(days=rolling_days - 1)
        win = d[(d["date"] >= lo) & (d["date"] <= day)]
        today = d[d["date"] == day]
        rec: dict = {"date": str(day.date()), "n_profiles": None, "bands": {}}
        if today["n_profiles"].notna().any():
            rec["n_profiles"] = int(today["n_profiles"].max())
        for band in BANDS:
            rec["bands"][band] = {
                "day": metrics_from_rows(today[today["band"] == band]),
                "rolling": metrics_from_rows(win[win["band"] == band])
                | {"n_days": int(win[win["band"] == band]["date"].nunique())},
            }
        out.append(rec)
    return out


def arrival_by_age(arrival: pd.DataFrame, max_age: int = 20) -> list[dict]:
    """How the number of profiles of a day grows with the age of the day at each update: for each
    age (days between the update and the profile day) the median, minimum and maximum count over
    all updates and days observed so far."""
    if len(arrival) == 0:
        return []
    a = arrival.copy()
    a["update_date"] = pd.to_datetime(a["update_ts"]).dt.tz_localize(None).dt.normalize()
    a["age_days"] = (a["update_date"] - pd.to_datetime(a["date"]).dt.normalize()).dt.days
    out = []
    for age, g in a[(a["age_days"] >= 0) & (a["age_days"] <= max_age)].groupby("age_days"):
        out.append(
            {
                "age_days": int(age),
                "n_obs": int(len(g)),
                "median": float(g["n_profiles"].median()),
                "min": int(g["n_profiles"].min()),
                "max": int(g["n_profiles"].max()),
            }
        )
    return out


# ----------------------------------------------------------------------------------------
# Argo
# ----------------------------------------------------------------------------------------
def fetch_argo_window(cfg: Config, first, last) -> pd.DataFrame:
    """Tidy good-QC Argo profiles of the domain for ``[first, last]`` (network; mocked in tests)."""
    g, a = cfg.grid, cfg.argo
    return fetch_argo_box(
        a.source, (g.lon_min, g.lon_max), (g.lat_min, g.lat_max), first, last, a.qc_flags,
        a.max_depth_m, a.box_deg,
    )  # fmt: skip


def argo_raw_file(cfg: Config) -> Path:
    return cfg.raw_root / "argo" / "argo_live.parquet"


def update_argo_table(cfg: Config, w_start, w_end, now: datetime) -> pd.DataFrame:
    """Profiles of the window: everything on a first call, afterwards the newest
    ``ARGO_REFETCH_DAYS`` days are fetched again (real-time profiles keep arriving) and older days
    kept; days that left the window are dropped."""
    path = argo_raw_file(cfg)
    old = pd.read_parquet(path) if path.exists() else None
    first = pd.Timestamp(w_start)
    if old is not None and len(old):
        first = max(first, pd.Timestamp(w_end) - pd.Timedelta(days=ARGO_REFETCH_DAYS))
    t0 = _time.time()
    new = fetch_argo_window(cfg, first.date(), pd.Timestamp(w_end).date())
    if old is not None and len(old):
        keep = pd.to_datetime(old["time"]) < first
        new = pd.concat([old[keep.to_numpy()], new], ignore_index=True)
    if len(new):
        new = new.drop_duplicates(["profile_id", "pres"])
        t = pd.to_datetime(new["time"])
        new = new[(t >= pd.Timestamp(w_start)).to_numpy()].reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(path, index=False)
    log_transfer(
        cfg, product="argo", dataset=f"argopy:{cfg.argo.source}",
        period=f"{first.date()}..{pd.Timestamp(w_end).date()}", route="argopy_nrt",
        bytes_written=path.stat().st_size, bytes_transferred=None, transfer_basis="unknown",
        seconds=_time.time() - t0,
    )  # fmt: skip
    return new


def argo_matchups(
    cfg: Config,
    profiles: pd.DataFrame,
    recon: dict[pd.Timestamp, np.ndarray],
    clim_fn,
    mask: np.ndarray,
    days: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict]:
    """Collocate ``profiles`` with the reconstruction of the days in ``recon`` and the climatology
    ``clim_fn(day) -> (D, H, W)``. Returns the matchup table (one row per profile and depth where
    observation, mask, reconstruction and climatology are all finite) and counts."""
    grid = build_grid(cfg)
    depths = grid.depth
    prof, obs = profile_table(profiles, depths)
    counts = {"n_profiles": int(len(prof)), "n_collocated": 0, "n_matchups": 0}
    if len(prof) == 0:
        return pd.DataFrame(columns=["date", "profile_id", "depth", "obs", "model", "clim"]), counts
    day_idx, j, k, status = collocate(prof["time"], prof["lat"], prof["lon"], grid, days, mask[0])
    ok = np.flatnonzero(status == "ok")
    counts["n_collocated"] = int(len(ok))
    rows = []
    for di in np.unique(day_idx[ok]):
        day = days[int(di)]
        if day not in recon:
            continue
        sel = ok[day_idx[ok] == di]
        model = recon[day][:, j[sel], k[sel]].T
        clim = clim_fn(day)[:, j[sel], k[sel]].T
        valid = np.isfinite(obs[sel]) & mask[:, j[sel], k[sel]].T & np.isfinite(model)
        valid &= np.isfinite(clim)
        pi, dj = np.nonzero(valid)
        rows.append(
            pd.DataFrame(
                {
                    "date": day,
                    "profile_id": prof["profile_id"].to_numpy()[sel][pi],
                    "depth": depths[dj].astype(np.float32),
                    "obs": obs[sel][pi, dj].astype(np.float32),
                    "model": model[pi, dj].astype(np.float32),
                    "clim": clim[pi, dj].astype(np.float32),
                    "basin": basin_of(j[sel], k[sel], grid)[pi],
                }
            )
        )
    table = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["date", "profile_id", "depth", "obs", "model", "clim", "basin"])
    )
    counts["n_matchups"] = int(len(table))
    return table, counts


# ----------------------------------------------------------------------------------------
# operational analysis
# ----------------------------------------------------------------------------------------
def analysis_dir(cfg: Config) -> Path:
    return cfg.processed_root / "live_analysis"


def analysis_file(cfg: Config, day) -> Path:
    return analysis_dir(cfg) / f"analysis_{pd.Timestamp(day):%Y%m%d}.npz"


def analysis_size_mb(cfg: Config, first, last) -> float:
    """Server-side estimate of the download for ``[first, last]`` (dry run; replaced in tests)."""
    import copernicusmarine

    v = cfg.live.verification
    kw = CmemsProvider(cfg).subset_kwargs("temp", first, last) | {
        "dataset_id": v.analysis_dataset,
        "variables": [v.analysis_variable],
    }
    r = copernicusmarine.subset(dry_run=True, **kw)
    return float(r.file_size)


def fetch_analysis(cfg: Config, days: list) -> tuple[int, float]:
    """Fetch and harmonise (the GLORYS rules) the analysis for ``days``; one small file per day on
    the 15 standard depths. Returns (days written, MB downloaded)."""
    v = cfg.live.verification
    prov = CmemsProvider(cfg)
    grid_days = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in days))
    n = 0
    mb = 0.0
    for first, last in nrt.runs_of_days([d.date() for d in grid_days], cfg.live.request_days):
        kw = prov.subset_kwargs("temp", first, last) | {
            "dataset_id": v.analysis_dataset,
            "variables": [v.analysis_variable],
        }
        t0 = _time.time()
        with tempfile.TemporaryDirectory(dir=_tmp(cfg), prefix="analysis_") as tmp:
            part = Path(tmp) / "range.nc"
            nrt._subset_with_retries(cfg, kw, part)
            size = part.stat().st_size
            rng = pd.date_range(first, last, freq="D")
            arrays = harmonise_files(
                cfg, "temp", [part], rng, variables={"temp": v.analysis_variable}
            )["temp"]
        for i, day in enumerate(rng):
            if not np.isfinite(arrays[i]).any():
                continue
            p = analysis_file(cfg, day)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp_f = p.with_name(p.name + ".tmp")
            with open(tmp_f, "wb") as fh:
                np.savez_compressed(fh, temp=arrays[i].astype(np.float32))
            tmp_f.replace(p)
            n += 1
        mb += size / 1e6
        log_transfer(
            cfg, product="analysis", dataset=v.analysis_dataset, period=f"{first}..{last}",
            route="cmems_subset_nrt", bytes_written=size, bytes_transferred=size,
            transfer_basis="file_size", seconds=_time.time() - t0,
        )  # fmt: skip
    return n, mb


def _tmp(cfg: Config) -> Path:
    p = cfg.raw_root / "_tmp"
    p.mkdir(parents=True, exist_ok=True)
    return p


def analysis_matchups(
    cfg: Config, recon: dict, clim_fn, mask: np.ndarray, days: list[pd.Timestamp]
) -> pd.DataFrame:
    """Daily sums against the analysis on the grid (same day), as ``daily_rows`` rows."""
    grid = build_grid(cfg)
    bm = band_masks(grid.depth)
    rows = []
    for day in days:
        p = analysis_file(cfg, day)
        if not p.exists() or day not in recon:
            continue
        with np.load(p) as z:
            ana = z["temp"]
        model, clim = recon[day], clim_fn(day)
        valid = mask & np.isfinite(ana) & np.isfinite(model) & np.isfinite(clim)
        for band, sel in bm.items():
            v = valid[sel]
            if not v.any():
                continue
            em = (model[sel] - ana[sel])[v].astype(np.float64)
            ec = (clim[sel] - ana[sel])[v].astype(np.float64)
            rows.append(
                {
                    "date": day,
                    "ref": "analysis",
                    "band": band,
                    "n": int(v.sum()),
                    "n_profiles": np.nan,
                    "sum_e_model": float(em.sum()),
                    "sum_e2_model": float((em**2).sum()),
                    "sum_e_clim": float(ec.sum()),
                    "sum_e2_clim": float((ec**2).sum()),
                }
            )
    return pd.DataFrame(rows, columns=DAILY_COLUMNS)


# ----------------------------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------------------------
def _merge_daily(old: pd.DataFrame, new: pd.DataFrame, ref: str, recomputed: set) -> pd.DataFrame:
    """Rows of ``new`` replace the old rows of ``ref`` on the recomputed days; other days stay."""
    if len(old):
        drop = (old["ref"] == ref) & old["date"].isin(recomputed)
        old = old[~drop]
    parts = [p for p in (old, new) if len(p)]
    if not parts:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in DAILY_COLUMNS})
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out.sort_values(["ref", "date", "band"]).reset_index(drop=True)[DAILY_COLUMNS]


def run_verification(cfg: Config, w: Weights, *, now: datetime, progress: bool = True) -> dict:
    live = cfg.live
    assert live is not None
    v = live.verification
    paths = live_paths(cfg)
    outdir = paths.verification
    outdir.mkdir(parents=True, exist_ok=True)
    state = load_state(cfg)
    table = load_days(cfg)
    if not len(table):
        return {"skipped": "no reconstructed days yet"}
    w_start, w_end = pd.Timestamp(state["window"]["start"]), pd.Timestamp(state["window"]["end"])
    days = pd.DatetimeIndex(table.index)
    days_all = pd.date_range(w_start, w_end, freq="D")
    recon = read_predictions(paths.predictions, days)
    stats = w.stats
    clim_cache: dict = {}

    def clim_fn(day):
        if day not in clim_cache:
            clim_cache[day] = stats.climatology(np.array([day.to_datetime64()]))[0]
        return clim_cache[day]

    daily_path = outdir / DAILY_FILE
    daily = (
        pd.read_parquet(daily_path) if daily_path.exists() else pd.DataFrame(columns=DAILY_COLUMNS)
    )
    if len(daily):
        daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()

    # Argo
    argo_info: dict = {}
    profiles = update_argo_table(cfg, w_start, w_end, now)
    profiles = (
        profiles[(pd.to_datetime(profiles["time"]) < w_end + pd.Timedelta(days=1)).to_numpy()]
        if len(profiles)
        else profiles
    )
    t_prof = (
        pd.to_datetime(profiles["time"]).dt.normalize()
        if len(profiles)
        else pd.Series([], dtype="datetime64[ns]")
    )
    per_day = (
        profiles.assign(_d=t_prof).groupby("_d")["profile_id"].nunique()
        if len(profiles)
        else pd.Series(dtype=int)
    )
    arrival_path = outdir / ARRIVAL_FILE
    arrival = (
        pd.read_parquet(arrival_path)
        if arrival_path.exists()
        else pd.DataFrame(columns=["update_ts", "date", "n_profiles"])
    )
    recent = [d for d in days_all if d >= w_end - pd.Timedelta(days=ARGO_REFETCH_DAYS)]
    add = pd.DataFrame(
        {
            "update_ts": stamp(now),
            "date": recent,
            "n_profiles": [int(per_day.get(d, 0)) for d in recent],
        }
    )
    arrival = pd.concat([arrival, add], ignore_index=True)
    arrival.to_parquet(arrival_path, index=False)
    mt, counts = argo_matchups(cfg, profiles, recon, clim_fn, w.mask, days_all)
    argo_info.update(counts)
    mt.to_parquet(outdir / "argo_matchups.parquet", index=False)
    rows_argo = daily_rows(mt, "argo")
    scored_days = set(pd.DatetimeIndex(days))
    daily = _merge_daily(daily, rows_argo, "argo", scored_days)

    # operational analysis
    ana_info: dict = {"reference": v.analysis_dataset}
    try:
        ana_days = [d for d in days if d > w_end - pd.Timedelta(days=v.rolling_days)]
        need = [d for d in ana_days if not analysis_file(cfg, d).exists()]
        if need:
            est = analysis_size_mb(cfg, min(need).date(), max(need).date())
            ana_info["download_estimate_mb"] = round(est, 1)
            if est > v.analysis_max_mb:
                raise nrt.LiveError(
                    f"the download for {len(need)} day(s) is estimated at {est:.0f} MB "
                    f"(> {v.analysis_max_mb:.0f} MB limit)"
                )
            n, mb = fetch_analysis(cfg, need)
            ana_info["fetched_days"], ana_info["fetched_mb"] = n, round(mb, 1)
        rows_ana = analysis_matchups(cfg, recon, clim_fn, w.mask, ana_days)
        daily = _merge_daily(daily, rows_ana, "analysis", set(ana_days))
        ana_info["available"] = bool(len(rows_ana))
        if not len(rows_ana):
            ana_info["reason"] = "the analysis has no data for these days yet"
        _prune_analysis(cfg, w_end - pd.Timedelta(days=v.rolling_days))
    except Exception as e:  # noqa: BLE001 - the analysis is optional; record why it is missing
        ana_info["available"] = False
        ana_info["reason"] = f"{type(e).__name__}: {e}"
        log.warning("analysis comparison skipped: %s", ana_info["reason"])
    daily.to_parquet(daily_path, index=False)

    out = _verification_json(cfg, state, daily, arrival, argo_info, ana_info, now)
    write_json_atomic(outdir / "verification.json", out)
    return out


def _prune_analysis(cfg: Config, before: pd.Timestamp) -> None:
    folder = analysis_dir(cfg)
    if not folder.is_dir():
        return
    for f in folder.glob("analysis_????????.npz"):
        try:
            d = datetime.strptime(f.stem.split("_")[-1], "%Y%m%d")
        except ValueError:
            continue
        if pd.Timestamp(d) <= before:
            f.unlink(missing_ok=True)


def _verification_json(cfg, state, daily, arrival, argo_info, ana_info, now) -> dict:
    v = cfg.live.verification
    win = state.get("window") or {}
    window_days = pd.date_range(win["start"], win["end"], freq="D") if win else pd.DatetimeIndex([])
    argo_dates = set(daily[daily["ref"] == "argo"]["date"]) | set(window_days)
    ana_dates = set(daily[daily["ref"] == "analysis"]["date"])
    if len(window_days) and ana_dates:
        ana_dates |= set(window_days[window_days >= min(ana_dates)])
    argo_series = rolling_series(daily, "argo", v.rolling_days, argo_dates)
    ana_series = rolling_series(daily, "analysis", v.rolling_days, ana_dates)

    def latest(series):
        return series[-1] if series else None

    out = {
        "updated": stamp(now),
        "window": state.get("window"),
        "rolling_days": v.rolling_days,
        "bands": {k: {"label": BAND_LABELS[k], "depth_range_m": list(BANDS[k])} for k in BANDS},
        "argo": {
            **argo_info,
            "reference": "Argo profiles (argopy / ERDDAP), good-QC flags, same grid cell and day",
            "note": (
                "Argo is independent of the model's inputs; near-real-time profiles arrive with "
                "a delay and have had only real-time quality control."
            ),
            "daily": argo_series,
            "latest": latest(argo_series),
            "arrival_by_age": arrival_by_age(arrival),
        },
        "analysis": {
            **ana_info,
            "note": (
                "a model analysis (Mercator operational system), not an observation, and not "
                "the GLORYS reanalysis the model was trained on"
            ),
            "daily": ana_series,
            "latest": latest(ana_series),
        },
    }
    out["headline"] = _headline(out)
    return out


def _headline(out: dict) -> dict:
    """The rolling numbers of the newest day, per band and reference."""
    h: dict = {}
    for ref in ("argo", "analysis"):
        last = (out.get(ref) or {}).get("latest")
        if not last:
            h[ref] = None
            continue
        h[ref] = {"date": last["date"], "n_profiles": last.get("n_profiles")} | {
            band: {
                "rolling_n": b["rolling"]["n"],
                "model_rmse": b["rolling"]["model_rmse"],
                "clim_rmse": b["rolling"]["clim_rmse"],
                "model_bias": b["rolling"]["model_bias"],
            }
            for band, b in last["bands"].items()
        }
    return h


__all__ = [
    "BANDS",
    "analysis_matchups",
    "argo_matchups",
    "arrival_by_age",
    "daily_rows",
    "metrics_from_rows",
    "rolling_series",
    "run_verification",
]
