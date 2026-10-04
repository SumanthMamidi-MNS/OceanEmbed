"""Folders, state file and provenance table of the live run (``outputs/live/``).

Layout (everything is rebuilt or pruned by ``oceanembed live update``; nothing here is ever read by
the evaluated runs)::

    outputs/live/
      run_meta.json            run contract (data_source "real", live block)
      predictions/oceanembed_T_<YYYYMM>.nc    rolling window, one file per month
      live_days.parquet        provenance of every reconstructed day in the window
      live_state.json          window, per-input availability, update history, revision log
      first_published/         inputs + reconstruction of each day as first published
      checks/input_shift/      the overlap check (``live input-shift``)
      checks/verification/     running verification against Argo and the operational analysis
      logs/                    wrapper-script logs
    data/raw_nrt/<product>/<product>_<YYYYMMDD>.nc    raw NRT days (+ _download_log.jsonl)
    data/processed/live.zarr   harmonised store of the window
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from oceanembed.config import Config

STATE_FILE = "live_state.json"
DAYS_FILE = "live_days.parquet"
LOCK_FILE = ".update.lock"
LOCK_STALE_S = 3 * 3600
STATE_VERSION = 1
HISTORY_LEN = 60
REVISION_LOG_LEN = 3000

DAY_COLUMNS = [
    "date", "sst_dataset", "sst_version", "sst_product_version", "sst_age_days",
    "sla_dataset", "sla_version", "sla_product_version", "sla_age_days",
    "first_update_ts", "last_update_ts", "n_checks", "n_revisions", "revised",
    "rev_sst_rmse", "rev_sla_rmse", "rev_recon_rmse_50_200", "rev_recon_maxabs",
    "inputs_digest", "checkpoint", "device",
]  # fmt: skip


@dataclass(frozen=True)
class LivePaths:
    run: Path
    predictions: Path
    state: Path
    days: Path
    first: Path
    checks: Path
    logs: Path
    store: Path
    raw: Path

    @property
    def shift(self) -> Path:
        return self.checks / "input_shift"

    @property
    def verification(self) -> Path:
        return self.checks / "verification"

    @property
    def lock(self) -> Path:
        return self.run / LOCK_FILE


def live_paths(cfg: Config) -> LivePaths:
    run = cfg.outputs_dir
    return LivePaths(
        run=run,
        predictions=run / "predictions",
        state=run / STATE_FILE,
        days=run / DAYS_FILE,
        first=run / "first_published",
        checks=run / "checks",
        logs=run / "logs",
        store=cfg.zarr_path,
        raw=cfg.raw_root,
    )


def now_utc() -> datetime:
    return datetime.now(UTC)


def stamp(t: datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.Timestamp | datetime):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_jsonable), encoding="utf-8")
    tmp.replace(path)


def load_state(cfg: Config) -> dict:
    p = live_paths(cfg).state
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(cfg: Config, state: dict) -> None:
    state = {**state, "schema_version": STATE_VERSION}
    write_json_atomic(live_paths(cfg).state, state)


def load_days(cfg: Config) -> pd.DataFrame:
    """The provenance table (one row per reconstructed day of the window), indexed by ``date``."""
    p = live_paths(cfg).days
    if not p.exists():
        df = pd.DataFrame({c: pd.Series(dtype="object") for c in DAY_COLUMNS})
    else:
        df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.set_index("date", drop=False).sort_index()


def save_days(cfg: Config, df: pd.DataFrame) -> None:
    p = live_paths(cfg).days
    p.parent.mkdir(parents=True, exist_ok=True)
    out = df.reset_index(drop=True).copy()
    for c in DAY_COLUMNS:
        if c not in out:
            out[c] = None
    tmp = p.with_name(p.name + ".tmp")
    out[DAY_COLUMNS].sort_values("date").to_parquet(tmp, index=False)
    tmp.replace(p)


class LockedError(RuntimeError):
    """Another ``live update`` is running (or died holding the lock)."""


class UpdateLock:
    """A lock file so that a scheduled and a manual update never run at once. A lock older than
    three hours is taken over (an update takes minutes)."""

    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            age = time.time() - self.path.stat().st_mtime
            if age < LOCK_STALE_S:
                raise LockedError(
                    f"another live update is running (lock {self.path}, {age / 60:.0f} min old); "
                    "if none is, delete that file"
                )
        self.path.write_text(f"pid {os.getpid()} {stamp(now_utc())}\n", encoding="utf-8")
        return self

    def __exit__(self, *exc):
        try:
            self.path.unlink()
        except OSError:
            pass
        return False


def is_live_meta(meta: dict) -> bool:
    return bool(meta.get("live"))
