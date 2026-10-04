"""``oceanembed live update`` and ``live status``: the rolling near-real-time nowcast.

Availability rule
    A day is *available* when **every** input the model uses (SST, sea level anomaly) has a
    non-empty file for it. The two products are published with different delays, so the newest
    day with data in one product may wait for the other (it is reported as *pending*). The window
    is the ``window_days`` days ending at the newest available day.

Revision policy
    Near-real-time maps are corrected for a few days after they first appear. On every update the
    newest ``revision_days`` reconstructed days are fetched again; if a day's inputs changed it is
    reconstructed again and the table records how far the inputs and the reconstruction have moved
    from what was first published (``first_published/``). The table keeps the first-publication
    age of each input; revision sizes are also summarised by the age at which the day was checked.

Safety
    Every file is written to a temp name and renamed; the harmonised store is rebuilt from the raw
    day files of the window on every update, and a day whose reconstruction is missing is
    reconstructed again, so an interrupted update is simply continued by the next one. Pruning
    touches only the live folders (``data/raw_nrt``, ``data/processed/live*``, ``outputs/live``).
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from oceanembed.config import Config
from oceanembed.grid import build_grid
from oceanembed.live import nrt
from oceanembed.live.harmonise import harmonise_files, write_store
from oceanembed.live.nrt import InputAvailability, LiveError, day_files
from oceanembed.live.reconstruct import (
    load_weights,
    prediction_days,
    reconstruct,
    surface_dataset,
    write_days,
)
from oceanembed.live.state import (
    HISTORY_LEN,
    REVISION_LOG_LEN,
    UpdateLock,
    live_paths,
    load_days,
    load_state,
    now_utc,
    save_days,
    save_state,
    stamp,
    write_json_atomic,
)
from oceanembed.runmeta import RUN_META_FILE, build_run_meta

log = logging.getLogger(__name__)

POOLED = (50.0, 200.0)


# ----------------------------------------------------------------------------------------
# small pure helpers (unit tested)
# ----------------------------------------------------------------------------------------
def window_for(last_common: date, previous_end: date | None, window_days: int) -> tuple[date, date]:
    """``(first, last)`` of the window: ``window_days`` days ending at the newest day published for
    all inputs, but never earlier than the end already reached (a product falling back does not
    shrink the window)."""
    end = last_common if previous_end is None else max(last_common, previous_end)
    return end - timedelta(days=window_days - 1), end


def revision_set(table_days: list[date], last_common: date, revision_days: int) -> list[date]:
    """Already reconstructed days that are fetched again: the newest ``revision_days`` days up to
    the newest day published for all inputs."""
    if revision_days <= 0:
        return []
    lo = last_common - timedelta(days=revision_days - 1)
    return sorted(d for d in table_days if lo <= d <= last_common)


def inputs_digest(fields: dict[str, np.ndarray]) -> str:
    """Fingerprint of one day's harmonised inputs (changes when a product is revised)."""
    h = hashlib.sha1(usedforsecurity=False)
    for k in sorted(fields):
        h.update(k.encode())
        h.update(np.ascontiguousarray(fields[k], dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))) if ok.any() else float("nan")


def revision_size(
    now_in: dict[str, np.ndarray],
    first_in: dict[str, np.ndarray],
    now_rec: np.ndarray | None,
    first_rec: np.ndarray | None,
    depths: np.ndarray,
) -> dict[str, float]:
    """How far a day has moved from its first-published version: RMSE of each input over the grid,
    RMSE of the reconstruction pooled over 50-200 m, and its largest absolute change."""
    out = {f"rev_{k}_rmse": _rmse(now_in[k], first_in[k]) for k in ("sst", "sla")}
    out["rev_recon_rmse_50_200"] = float("nan")
    out["rev_recon_maxabs"] = float("nan")
    if now_rec is not None and first_rec is not None:
        sel = (depths >= POOLED[0]) & (depths <= POOLED[1])
        out["rev_recon_rmse_50_200"] = _rmse(now_rec[sel], first_rec[sel])
        d = np.abs(now_rec.astype(np.float64) - first_rec.astype(np.float64))
        out["rev_recon_maxabs"] = float(np.nanmax(d)) if np.isfinite(d).any() else float("nan")
    return out


def summarise_revisions(log_rows: list[dict], max_age: int = 14) -> dict:
    """Revision statistics by the age (days) at which a day was checked: how many checks, how many
    found a changed day, and the size of the move from the first-published version."""
    by_age: dict[int, list[dict]] = {}
    for r in log_rows:
        by_age.setdefault(int(r["age_days"]), []).append(r)
    rows = []
    for age in sorted(by_age):
        if age > max_age:
            continue
        g = by_age[age]
        sst = np.array([x.get("sst_rmse", np.nan) for x in g], dtype=float)
        sla = np.array([x.get("sla_rmse", np.nan) for x in g], dtype=float)
        rec = np.array([x.get("recon_rmse_50_200", np.nan) for x in g], dtype=float)
        rows.append(
            {
                "age_days": age,
                "n_checks": len(g),
                "n_changed_since_first": int(sum(1 for x in g if x.get("changed_since_first"))),
                "sst_rmse_mean": _nanmean(sst),
                "sla_rmse_mean": _nanmean(sla),
                "recon_rmse_50_200_mean": _nanmean(rec),
                "recon_rmse_50_200_max": float(np.nanmax(rec)) if np.isfinite(rec).any() else None,
            }
        )
    ever = {}
    for r in log_rows:
        ever[r["date"]] = ever.get(r["date"], False) or bool(r.get("changed_since_first"))
    return {
        "n_days_checked": len(ever),
        "n_days_revised": int(sum(ever.values())),
        "by_age": rows,
    }


def _nanmean(x: np.ndarray):
    return float(np.nanmean(x)) if np.isfinite(x).any() else None


# ----------------------------------------------------------------------------------------
# first-published snapshots
# ----------------------------------------------------------------------------------------
def snapshot_path(cfg: Config, day: date | pd.Timestamp) -> Path:
    return live_paths(cfg).first / f"first_{pd.Timestamp(day):%Y%m%d}.npz"


def save_snapshot(cfg: Config, day, fields: dict[str, np.ndarray], recon: np.ndarray) -> None:
    p = snapshot_path(cfg, day)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh, sst=fields["sst"], sla=fields["sla"], recon=recon.astype(np.float16)
        )
    tmp.replace(p)


def load_snapshot(cfg: Config, day) -> tuple[dict[str, np.ndarray], np.ndarray] | None:
    p = snapshot_path(cfg, day)
    if not p.exists():
        return None
    with np.load(p) as z:
        return {"sst": z["sst"], "sla": z["sla"]}, z["recon"].astype(np.float32)


# ----------------------------------------------------------------------------------------
# the update
# ----------------------------------------------------------------------------------------
def _as_ts(d) -> pd.Timestamp:
    return pd.Timestamp(d).normalize()


def _day(d) -> date:
    return pd.Timestamp(d).date()


def _product_version(path: Path) -> str | None:
    import xarray as xr

    try:
        with xr.open_dataset(path) as ds:
            v = ds.attrs.get("product_version")
            return None if v is None else str(v)
    except (OSError, ValueError):
        return None


def run_update(
    cfg: Config,
    *,
    device: str | None = None,
    verify: bool | None = None,
    now: datetime | None = None,
    progress: bool = True,
) -> dict:
    """Bring the rolling live run up to date; returns a summary dict (printed by the CLI)."""
    if cfg.live is None:
        raise LiveError("this config has no `live` section (use configs/live.yaml)")
    paths = live_paths(cfg)
    with UpdateLock(paths.lock):
        return _run_update(cfg, device, verify, now or now_utc(), progress)


def _run_update(cfg: Config, device, verify, now: datetime, progress: bool) -> dict:
    live = cfg.live
    assert live is not None
    t0 = time.time()
    today = now.date()
    paths = live_paths(cfg)
    products = nrt.required_products(cfg)
    shutil.rmtree(paths.raw / "_tmp", ignore_errors=True)
    state = load_state(cfg)
    table = load_days(cfg)
    grid = build_grid(cfg)

    # 1. what the catalogue publishes (read only)
    avail: dict[str, InputAvailability] = {p: nrt.probe(cfg, p) for p in products}
    last_common = min(a.last for a in avail.values())
    prev_end = _day(state["window"]["end"]) if state.get("window") else None
    w_start, w_end = window_for(last_common, prev_end, live.window_days)
    window = [w_start + timedelta(days=i) for i in range((w_end - w_start).days + 1)]
    table_days = [_day(d) for d in table.index]
    rev_days = [d for d in revision_set(table_days, last_common, live.revision_days) if d in window]

    # 2. fetch what is new, and look again at the newest reconstructed days
    fetched: dict[str, nrt.FetchResult] = {}
    bytes_dl = 0
    for p in products:
        a = avail[p]
        have = day_files(paths.raw, p)
        need = [d for d in window if a.first <= d <= a.last and d not in have]
        res_new = nrt.fetch_days(cfg, p, need)
        res_rev = nrt.fetch_days(cfg, p, [d for d in rev_days if d in have], revision=True)
        res_new.changed |= res_rev.changed
        res_new.bytes_transferred += res_rev.bytes_transferred
        res_new.requests += res_rev.requests
        fetched[p] = res_new
        bytes_dl += res_new.bytes_transferred
    files = {p: day_files(paths.raw, p) for p in products}
    available = sorted(set.intersection(*(set(files[p]) for p in products)) & set(window))
    if not available:
        raise LiveError("no day has data for every input yet; nothing to reconstruct")
    w_end = max(available)
    w_start = w_end - timedelta(days=live.window_days - 1)
    window = [w_start + timedelta(days=i) for i in range((w_end - w_start).days + 1)]
    available = [d for d in available if d >= w_start]
    upto = max(a.last for a in avail.values())
    pending = [
        w_start + timedelta(days=i)
        for i in range((upto - w_start).days + 1)
        if (w_start + timedelta(days=i)) not in set(available)
    ]

    # 3. harmonise the whole window (cheap) into the live store
    wdays = pd.DatetimeIndex([_as_ts(d) for d in window])
    w = load_weights(cfg, device)
    surface: dict[str, np.ndarray] = {}
    for p in products:
        arrays = harmonise_files(cfg, p, [files[p][d] for d in available if d in files[p]], wdays)
        surface.update(arrays)
    write_store(paths.store, cfg, wdays, surface, w.mask, title="OceanEmbed live window")
    fields = {
        d: {k: surface[k][i] for k in ("sst", "sla")}
        for i, d in enumerate(window)
        if d in set(available)
    }
    digest = {d: inputs_digest(f) for d, f in fields.items()}

    # 4. which days to reconstruct
    pred_have = {_day(t) for t in prediction_days(paths.predictions)}
    new_days = [d for d in available if d not in set(table_days)]
    checked = [d for d in rev_days if d in digest and d in set(table_days)]
    changed = [d for d in checked if digest[d] != table.loc[_as_ts(d), "inputs_digest"]]
    missing_pred = [d for d in available if d not in pred_have]
    todo = sorted(set(new_days) | set(changed) | set(missing_pred))

    # 5. reconstruct and merge into the rolling monthly files
    t_rec = time.time()
    temps: dict = {}
    ds = surface_dataset(cfg, w, wdays[0], wdays[-1], paths.store)
    if todo:
        temps = reconstruct(ds, w, pd.DatetimeIndex([_as_ts(d) for d in todo]))
    sec_per_day = (time.time() - t_rec) / len(todo) if todo else None
    input_info = {p: f"{avail[p].dataset} (version {avail[p].version})" for p in products}
    write_days(cfg, paths.predictions, ds, w, temps, (wdays[0], wdays[-1]), input_info)

    # 6. provenance table, first-published snapshots, revision log
    ts = stamp(now)
    rev_log = list(state.get("revision_log", []))
    depths = grid.depth
    rows = {d: table.loc[_as_ts(d)].to_dict() for d in table_days if d in window}
    for d in todo:
        a = temps[_as_ts(d)]
        snap = load_snapshot(cfg, d) if d in rows else None
        if d not in rows or snap is None:
            save_snapshot(cfg, d, fields[d], a)
            rows[d] = _new_row(cfg, d, avail, files, products, today, ts, digest[d], w, a)
        else:
            sizes = revision_size(fields[d], snap[0], a, snap[1], depths)
            rows[d] = {
                **rows[d],
                "last_update_ts": ts,
                "n_revisions": int(rows[d].get("n_revisions") or 0) + (d in changed),
                "inputs_digest": digest[d],
                **sizes,
            }
            rows[d]["revised"] = bool(rows[d]["n_revisions"])
    for d in checked:
        r = rows[d]
        r["n_checks"] = int(r.get("n_checks") or 0) + 1
        r["last_update_ts"] = ts if d in changed else r.get("last_update_ts")
        rev_log.append(
            {
                "date": str(d),
                "checked_at": ts,
                "age_days": (today - d).days,
                "changed": d in changed,
                "changed_since_first": bool(r.get("n_revisions")),
                "sst_rmse": _num(r.get("rev_sst_rmse")),
                "sla_rmse": _num(r.get("rev_sla_rmse")),
                "recon_rmse_50_200": _num(r.get("rev_recon_rmse_50_200")),
            }
        )
    new_table = pd.DataFrame(list(rows.values())) if rows else table.iloc[0:0]
    if len(new_table):
        new_table["date"] = pd.to_datetime(new_table["date"]).dt.normalize()
        new_table = new_table.set_index("date", drop=False).sort_index()
    save_days(cfg, new_table)

    # 7. prune what fell out of the window (live folders only)
    pruned = _prune(cfg, paths, w_start, w_end, products)
    rev_log = [r for r in rev_log if _day(r["date"]) >= w_start][-REVISION_LOG_LEN:]

    # 8. state and run contract
    last_day = max(available)
    ages = {
        p: {
            **avail[p].to_json(),
            "latest_data_date": str(max(d for d in files[p])),
            "age_days": (today - max(d for d in files[p])).days,
            "delay_days_catalogue": (today - avail[p].last).days,
        }
        for p in products
    }
    elapsed = time.time() - t0
    entry = {
        "ts": ts,
        "n_new_days": len(new_days),
        "n_revision_checks": len(checked),
        "n_revised_days": len(changed),
        "n_reconstructed": len(todo),
        "seconds": round(elapsed, 1),
        "seconds_per_day": None if sec_per_day is None else round(sec_per_day, 3),
        "device": str(w.device),
        "input_delay_days": {p: (today - avail[p].last).days for p in products},
        "bytes_downloaded": int(bytes_dl),
        "requests": int(sum(f.requests for f in fetched.values())),
        "pruned": pruned,
    }
    state.update(
        {
            "run": cfg.run_name,
            "last_update": ts,
            "last_checked": ts,
            "window": {
                "start": str(w_start),
                "end": str(w_end),
                "n_days": len(window),
                "window_days": live.window_days,
            },
            "last_day": str(last_day),
            "n_reconstructed_days": len(new_table),
            "inputs": ages,
            "pending": [str(d) for d in pending],
            "revision_policy": {
                "revision_days": live.revision_days,
                "rule": "the newest revision_days reconstructed days are fetched again on every "
                "update; a day whose inputs changed is reconstructed again",
            },
            "revision_log": rev_log,
            "revision_stats": summarise_revisions(rev_log),
            "model": {"checkpoint": w.checkpoint, "weights": str(w.folder)},
            "history": [*state.get("history", []), entry][-HISTORY_LEN:],
        }
    )
    save_state(cfg, state)
    summary = {
        **entry,
        "window": state["window"],
        "last_day": str(last_day),
        "inputs": ages,
        "pending": state["pending"],
        "missing_inputs": {p: sorted(map(str, fetched[p].missing)) for p in products},
    }
    if todo or not (paths.run / RUN_META_FILE).exists():
        write_json_atomic(paths.run / RUN_META_FILE, _live_meta(cfg, state))

    # 9. running verification
    if verify is None:
        verify = live.verification.enabled
    if verify:
        from oceanembed.live.verify import run_verification

        try:
            summary["verification"] = run_verification(cfg, w, now=now, progress=progress)
        except Exception as e:  # noqa: BLE001 - a verification problem must not undo the update
            log.warning("verification failed: %s: %s", type(e).__name__, e)
            summary["verification"] = {"error": f"{type(e).__name__}: {e}"}
    summary["seconds_total"] = round(time.time() - t0, 1)
    return summary


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _new_row(cfg, d, avail, files, products, today, ts, digest, w, recon) -> dict:
    row = {
        "date": _as_ts(d),
        "first_update_ts": ts,
        "last_update_ts": ts,
        "n_checks": 0,
        "n_revisions": 0,
        "revised": False,
        "rev_sst_rmse": 0.0,
        "rev_sla_rmse": 0.0,
        "rev_recon_rmse_50_200": 0.0,
        "rev_recon_maxabs": 0.0,
        "inputs_digest": digest,
        "checkpoint": w.checkpoint,
        "device": str(w.device),
    }
    for p in products:
        row[f"{p}_dataset"] = avail[p].dataset
        row[f"{p}_version"] = avail[p].version
        row[f"{p}_product_version"] = _product_version(files[p][d])
        row[f"{p}_age_days"] = (today - d).days
    return row


def _prune(cfg: Config, paths, w_start: date, w_end: date, products: list[str]) -> dict:
    """Delete live files of days outside the window. Only the live folders are touched."""
    n_raw = n_first = 0
    for p in products:
        for d, f in day_files(paths.raw, p).items():
            if d < w_start:
                f.unlink(missing_ok=True)
                n_raw += 1
    if paths.first.is_dir():
        for f in paths.first.glob("first_????????.npz"):
            try:
                d = datetime.strptime(f.stem.split("_")[-1], "%Y%m%d").date()
            except ValueError:
                continue
            if d < w_start:
                f.unlink(missing_ok=True)
                n_first += 1
    return {"raw_files": n_raw, "first_published": n_first}


def _live_meta(cfg: Config, state: dict) -> dict:
    meta = build_run_meta(cfg)
    w = state["window"]
    meta["time"] = {"start": w["start"], "end": w["end"]}
    live = cfg.live
    assert live is not None
    meta["paths"]["stats"] = str(Path(live.weights) / "stats.nc")
    meta["live"] = {
        "nrt": True,
        "window": w,
        "last_day": state.get("last_day"),
        "weights": str(live.weights),
        "inputs": {p: v.get("dataset") for p, v in state.get("inputs", {}).items()},
        "note": "nowcast of the same day from near-real-time inputs, not a forecast",
    }
    return meta


# ----------------------------------------------------------------------------------------
# status
# ----------------------------------------------------------------------------------------
def status(cfg: Config, now: datetime | None = None, check: bool = False) -> dict:
    """What the live run holds: window, newest day, age of every input and pending days. Reads local
    files only; with ``check`` it also asks the catalogue what is published now (read only)."""
    now = now or now_utc()
    today = now.date()
    state = load_state(cfg)
    table = load_days(cfg)
    out: dict = {
        "run": cfg.run_name,
        "initialised": bool(state),
        "last_update": state.get("last_update"),
        "window": state.get("window"),
        "last_day": state.get("last_day"),
        "n_reconstructed_days": int(len(table)),
        "inputs": {},
        "pending": state.get("pending", []),
        "checked_catalogue": False,
    }
    if not state:
        return out
    for p, info in state.get("inputs", {}).items():
        latest = _day(info["latest_data_date"])
        out["inputs"][p] = {
            "dataset": info.get("dataset"),
            "version": info.get("version"),
            "latest_data_date": str(latest),
            "age_days": (today - latest).days,
            "catalogue_last_at_update": info.get("last"),
        }
    if check:
        products = nrt.required_products(cfg)
        avail = {p: nrt.probe(cfg, p) for p in products}
        last_common = min(a.last for a in avail.values())
        have = {_day(d) for d in table.index}
        w_end = _day(state["window"]["end"])
        out["checked_catalogue"] = True
        out["catalogue_last"] = {p: str(a.last) for p, a in avail.items()}
        out["pending"] = [
            str(w_end + timedelta(days=i))
            for i in range(1, (max(a.last for a in avail.values()) - w_end).days + 1)
        ]
        out["new_days_available"] = max(0, (last_common - w_end).days)
        out["unreconstructed_in_window"] = [
            str(d) for d in pd.date_range(state["window"]["start"], w_end).date if d not in have
        ]
    return out
