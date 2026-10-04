"""Released weights -> reconstruction of chosen days, and the rolling monthly prediction files.

The model, statistics (normalisation, climatology) and ocean mask come from the released-weights
folder (``live.weights``, ``models/final``), exactly as in ``oceanembed predict --weights``. The
reconstruction of a day depends on that day's inputs only, so any subset of the days of a store can
be reconstructed and the monthly files of the rolling run are merged day by day.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from oceanembed.config import Config
from oceanembed.data.dataset import SurfaceOnlyDataset, make_surface_dataset
from oceanembed.data.stats import Stats, load_mask
from oceanembed.infer.predict import (
    PRODUCT_FILE,
    WEIGHTS_STATS_FILE,
    _month_dataset,
    check_inputs,
    load_recon_model,
    model_predictor,
    predict_batch,
    write_month,
)
from oceanembed.live.nrt import LiveError
from oceanembed.runmeta import data_source
from oceanembed.train.utils import get_device

log = logging.getLogger(__name__)


@dataclass
class Weights:
    model: torch.nn.Module
    predictor: object
    stats: Stats
    mask: np.ndarray
    device: torch.device
    checkpoint: str
    ckpt_meta: dict
    folder: Path


def weights_folder(cfg: Config) -> Path:
    if cfg.live is None:
        raise LiveError("this config has no `live` section (use configs/live.yaml)")
    return Path(cfg.live.weights)


def load_weights(cfg: Config, device: str | None = None) -> Weights:
    folder = weights_folder(cfg)
    ckpt = folder / "recon.pt"
    stats_file = folder / WEIGHTS_STATS_FILE
    for f in (ckpt, stats_file):
        if not f.exists():
            raise LiveError(f"{f} not found; the live run needs the released weights folder")
    dev = get_device(device)
    model, ckpt_cfg = load_recon_model(ckpt, dev)
    check_inputs(cfg, ckpt_cfg, ckpt)
    mask = load_mask(stats_file)
    if mask is None:
        raise LiveError(f"{stats_file} carries no ocean mask")
    return Weights(
        model=model,
        predictor=model_predictor(model, amp=cfg.train.amp),
        stats=Stats.load(stats_file),
        mask=mask,
        device=dev,
        checkpoint=ckpt.name,
        ckpt_meta=model.ckpt_meta,
        folder=folder,
    )


def surface_dataset(cfg: Config, w: Weights, start, end, store: Path | None = None):
    """Inputs-only dataset over ``[start, end]`` of the live store (or another store)."""
    if store is not None:
        ds = SurfaceOnlyDataset(store, w.stats, start, end)
        ds.mask = np.asarray(w.mask, dtype=bool)
        return ds.set_input_groups(cfg.model.input_groups)
    return make_surface_dataset(cfg, start, end, stats=w.stats, mask=w.mask)


def reconstruct(
    ds: SurfaceOnlyDataset, w: Weights, days: pd.DatetimeIndex | None = None, batch_size: int = 8
) -> dict[pd.Timestamp, np.ndarray]:
    """Temperature ``(15, H, W)`` degC per day (NaN off the ocean mask) for ``days`` (default: every
    day of the dataset). Days are predicted in runs of consecutive days of the store."""
    dates = ds.dates()
    want = np.arange(len(dates)) if days is None else dates.get_indexer(pd.DatetimeIndex(days))
    want = np.sort(want[want >= 0])
    out: dict[pd.Timestamp, np.ndarray] = {}
    i = 0
    while i < len(want):
        j = i
        while j + 1 < len(want) and want[j + 1] == want[j] + 1 and j + 1 - i < batch_size:
            j += 1
        a, b = int(want[i]), int(want[j]) + 1
        temp = predict_batch(w.predictor, ds.batch(a, b), ds, w.device)
        for k in range(b - a):
            out[dates[a + k]] = temp[k]
        i = j + 1
    return out


# ----------------------------------------------------------------------------------------
# rolling monthly files
# ----------------------------------------------------------------------------------------
def product_file(pred_dir: Path, month: pd.Period) -> Path:
    return pred_dir / PRODUCT_FILE.format(ym=month.strftime("%Y%m"))


def prediction_days(pred_dir: Path) -> pd.DatetimeIndex:
    """Days held by the monthly files of the rolling run."""
    days = []
    for f in sorted(pred_dir.glob(PRODUCT_FILE.format(ym="??????"))):
        try:
            with xr.open_dataset(f) as ds:
                days.extend(pd.DatetimeIndex(ds["time"].values))
        except (OSError, ValueError, KeyError):
            continue
    return pd.DatetimeIndex(sorted(set(days)))


def read_predictions(pred_dir: Path, days: pd.DatetimeIndex) -> dict[pd.Timestamp, np.ndarray]:
    """Temperature ``(15, H, W)`` of the given days from the monthly files (days not held are
    left out)."""
    want = set(pd.DatetimeIndex(days))
    out: dict[pd.Timestamp, np.ndarray] = {}
    for f in sorted(pred_dir.glob(PRODUCT_FILE.format(ym="??????"))):
        try:
            with xr.open_dataset(f) as ds:
                times = pd.DatetimeIndex(ds["time"].values)
                hit = [i for i, t in enumerate(times) if t in want]
                if hit:
                    arr = ds["temperature"].isel(time=hit).values
                    for i, a in zip(hit, arr, strict=True):
                        out[times[i]] = a.astype(np.float32)
        except (OSError, ValueError, KeyError):
            continue
    return out


def _live_attrs(ds: xr.Dataset, cfg: Config, inputs: dict) -> xr.Dataset:
    ds.attrs.update(
        title="OceanEmbed live nowcast: subsurface ocean temperature from near-real-time inputs",
        comment=(
            "same-day reconstruction (nowcast, not a forecast) from near-real-time sea surface "
            "temperature and sea level anomaly with the released final model; reanalysis-trained "
            "estimate, not an observation; days are revised for a few days after first "
            "publication of the inputs"
        ),
        live_nrt="true",
        input_datasets=", ".join(f"{k}: {v}" for k, v in inputs.items()),
        method="model",
    )
    return ds


def write_days(
    cfg: Config,
    pred_dir: Path,
    ds: SurfaceOnlyDataset,
    w: Weights,
    temps: dict[pd.Timestamp, np.ndarray],
    keep: tuple[pd.Timestamp, pd.Timestamp],
    inputs: dict,
) -> list[Path]:
    """Merge ``temps`` into the monthly files (replacing days that are already there), dropping
    every day outside ``keep`` (first, last). Files are written to a temp name and renamed."""
    meta = {
        "data_source": data_source(cfg),
        "checkpoint": w.checkpoint,
        "method": "model",
        **w.ckpt_meta,
    }
    months = {pd.Period(t, "M") for t in temps}
    existing = {pd.Period(m, "M") for m in _months_on_disk(pred_dir)}
    written = []
    for m in sorted(months | existing):
        path = product_file(pred_dir, m)
        have: dict[pd.Timestamp, np.ndarray] = {}
        if path.exists():
            try:
                with xr.open_dataset(path) as old:
                    arr = old["temperature"].values
                    for t, a in zip(pd.DatetimeIndex(old["time"].values), arr, strict=True):
                        have[t] = a
            except (OSError, ValueError, KeyError):
                have = {}
        for t, a in temps.items():
            if pd.Period(t, "M") == m:
                have[t] = a
        have = {t: a for t, a in have.items() if keep[0] <= t <= keep[1]}
        if not have:
            path.unlink(missing_ok=True)
            continue
        if not any(pd.Period(t, "M") == m for t in temps) and len(have) == _count_days(path):
            continue  # untouched month: leave the file alone
        dates = pd.DatetimeIndex(sorted(have))
        arr = np.stack([have[t] for t in dates]).astype(np.float32)
        month_ds = _live_attrs(_month_dataset(cfg, arr, dates, ds, meta), cfg, inputs)
        part = path.with_name(path.name + ".part")
        write_month(part, month_ds)
        _replace_file(part, path)
        written.append(path)
    return written


def _replace_file(part: Path, path: Path, tries: int = 6) -> None:
    """Rename ``part`` over ``path``. On Windows a reader that holds the old file open (the API
    serving the live run) makes this fail for a moment: wait and try again."""
    for i in range(tries):
        try:
            part.replace(path)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(1.0 + i)


def _months_on_disk(pred_dir: Path) -> list[pd.Period]:
    out = []
    for f in pred_dir.glob(PRODUCT_FILE.format(ym="??????")):
        ym = f.stem.split("_")[-1]
        try:
            out.append(pd.Period(f"{ym[:4]}-{ym[4:]}", "M"))
        except ValueError:
            continue
    return out


def _count_days(path: Path) -> int:
    try:
        with xr.open_dataset(path) as ds:
            return int(ds.sizes["time"])
    except (OSError, ValueError, KeyError):
        return -1
