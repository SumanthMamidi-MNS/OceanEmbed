"""Evaluate the model, the baselines and tagged ablations against the harmonised GLORYS target.

Streaming: days are processed in small batches; per method two :class:`MetricAccumulator` objects
(raw temperature and anomaly with the climatology removed) are updated, so the evaluated period is
never held in memory. Outputs under ``outputs/<run>/metrics/``:

* ``metrics_glorys.json`` -- per method: ``overall``, ``pooled_50_200m``, ``per_depth`` (column
  oriented), ``per_basin[basin]`` with the same two blocks, plus ``daily_rmse`` (domain-mean RMSE
  per day and depth for every method; stored in the JSON because it is only ``n_days x 15`` floats;
  the same block also holds the daily domain-mean ``bias``, the daily *spatial* anomaly correlation
  ``corr_anom`` and the thermocline-pooled daily ``pooled_rmse`` series) and a ``metadata`` block
  (split dates, n days, data source, depths, units, method provenance).
* ``maps_glorys.nc`` -- ``(depth, lat, lon)`` RMSE / bias / anomaly-correlation maps per method,
  plus raw-correlation and skill-vs-climatology maps for every method.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

from oceanembed.config import Config
from oceanembed.data.dataset import SurfaceOnlyDataset, make_surface_dataset
from oceanembed.data.harmonize import open_harmonized
from oceanembed.eval.metrics import (
    SUM_FIELDS,
    MetricAccumulator,
    metrics_from_sums,
    skill_score,
    sums_from_arrays,
)
from oceanembed.grid import BASINS, build_grid
from oceanembed.infer.predict import (
    Predictor,
    load_recon_model,
    model_predictor,
    predict_batch,
)
from oceanembed.models.baselines import RIDGE_FILE, ClimatologyBaseline, RidgeBaseline
from oceanembed.runmeta import data_source
from oceanembed.train.utils import get_device

log = logging.getLogger(__name__)

POOLED_RANGE = (50.0, 200.0)  # thermocline range pooled in addition to the all-depth summary
SYNTHETIC_NOTE = (
    "SYNTHETIC DATA: the numbers only demonstrate that the pipeline works; they are not "
    "scientific skill."
)


# ----------------------------------------------------------------------------------------
# methods
# ----------------------------------------------------------------------------------------
@dataclass
class Methods:
    """Ordered predictors (key -> predictor) with labels and provenance."""

    predictors: dict[str, Predictor] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    info: dict[str, dict] = field(default_factory=dict)

    def add(self, key: str, predictor: Predictor, label: str, info: dict) -> None:
        self.predictors[key], self.labels[key], self.info[key] = predictor, label, info


def method_label(key: str) -> str:
    fixed = {
        "model": "OceanEmbed (pretrained encoder)",
        "model_scratch": "OceanEmbed (no pretraining)",
        "ridge": "Ridge regression",
        "climatology": "Climatology",
        "glorys": "GLORYS reanalysis",
    }
    if key in fixed:
        return fixed[key]
    return f"OceanEmbed ({key.removeprefix('model_')})"


def load_methods(cfg: Config, device) -> Methods:
    """model (``recon.pt``, required), tagged ablations (``recon_<tag>.pt``), ridge (if fitted)
    and climatology, in that order."""
    ck = cfg.checkpoints_dir
    main = ck / "recon.pt"
    if not main.exists():
        raise FileNotFoundError(f"{main} not found; run `oceanembed train` first")
    m = Methods()

    def add_model(key: str, path: Path) -> None:
        model, _ = load_recon_model(path, device)
        m.add(
            key,
            model_predictor(model, amp=cfg.train.amp),
            method_label(key),
            {"kind": "model", "checkpoint": path.name, **model.ckpt_meta},
        )

    add_model("model", main)
    for p in sorted(ck.glob("recon_*.pt")):
        add_model(f"model_{p.stem.removeprefix('recon_')}", p)
    ridge_path = ck / RIDGE_FILE
    if ridge_path.exists():
        ridge = RidgeBaseline.load(ridge_path)
        m.add("ridge", ridge, method_label("ridge"), {"kind": "baseline", "checkpoint": RIDGE_FILE})
    else:
        log.warning("%s missing: the ridge baseline is skipped", ridge_path)
    m.add(
        "climatology",
        ClimatologyBaseline(cfg.model.n_depths),
        method_label("climatology"),
        {"kind": "baseline", "description": "harmonic climatology fitted on the train split"},
    )
    return m


# ----------------------------------------------------------------------------------------
# helpers shared with argo_validation
# ----------------------------------------------------------------------------------------
def split_bounds(cfg: Config, split: str) -> tuple[str, str]:
    r = cfg.split.get(split)
    return str(r.start), str(r.end)


def read_target(zds: xr.Dataset, ds: SurfaceOnlyDataset, a: int, b: int) -> np.ndarray:
    """Harmonised target ``(b-a, D, H, W)`` float32 for split items ``a .. b-1``."""
    lo, hi = int(ds.indices[a]), int(ds.indices[b - 1]) + 1
    return zds["temp"].isel(time=slice(lo, hi)).values.astype(np.float32)


def json_clean(obj):
    """Recursively convert numpy types and NaN/inf to JSON-safe Python values (NaN -> null)."""
    if isinstance(obj, dict):
        return {str(k): json_clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [json_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_clean(obj.tolist())
    if isinstance(obj, np.generic):
        return json_clean(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_clean(obj), indent=1), encoding="utf-8")


def metrics_dir(cfg: Config) -> Path:
    return cfg.outputs_dir / "metrics"


# ----------------------------------------------------------------------------------------
# summaries
# ----------------------------------------------------------------------------------------
def _summary(raw, anom, clim, region=None, depth_idx=None) -> dict:
    r = raw.metrics_total(region, depth_idx)
    a = anom.metrics_total(region, depth_idx)
    c = clim.metrics_total(region, depth_idx)
    return {
        "n": int(r["n"]),
        "rmse": r["rmse"],
        "bias": r["bias"],
        "mae": r["mae"],
        "corr_raw": r["corr"],
        "corr_anom": a["corr"],
        "skill_vs_clim": float(skill_score(r["mse"], c["mse"])),
    }


def _per_depth(raw, anom, clim, depths, region=None) -> dict:
    r = raw.metrics_by_depth(region)
    a = anom.metrics_by_depth(region)
    c = clim.metrics_by_depth(region)
    return {
        "depth": [float(d) for d in depths],
        "n": r["n"].astype(int),
        "rmse": r["rmse"],
        "bias": r["bias"],
        "mae": r["mae"],
        "corr_raw": r["corr"],
        "corr_anom": a["corr"],
        "skill_vs_clim": skill_score(r["mse"], c["mse"]),
    }


def _block(raw, anom, clim, depths, region=None) -> dict:
    d = np.asarray(depths)
    pooled = np.flatnonzero((d >= POOLED_RANGE[0]) & (d <= POOLED_RANGE[1]))
    return {
        "overall": _summary(raw, anom, clim, region),
        "pooled_50_200m": _summary(raw, anom, clim, region, pooled),
        "per_depth": _per_depth(raw, anom, clim, depths, region),
    }


def _maps(accs: dict[str, tuple[MetricAccumulator, MetricAccumulator]], grid) -> xr.Dataset:
    coords = {"depth": grid.depth, "lat": grid.lat, "lon": grid.lon}
    dims = ("depth", "lat", "lon")
    data = {}
    clim_mse = accs["climatology"][0].metrics_map()["mse"]
    for key, (raw, anom) in accs.items():
        mr = raw.metrics_map()
        ma = anom.metrics_map()
        data[f"rmse_{key}"] = (dims, mr["rmse"].astype(np.float32), {"units": "degC"})
        data[f"bias_{key}"] = (
            dims,
            mr["bias"].astype(np.float32),
            {"units": "degC", "comment": "prediction minus reference"},
        )
        if key != "climatology":
            data[f"corr_anom_{key}"] = (
                dims,
                ma["corr"].astype(np.float32),
                {"comment": "temporal correlation of anomalies (climatology removed)"},
            )
        data[f"corr_raw_{key}"] = (dims, mr["corr"].astype(np.float32), {})
        data[f"skill_vs_clim_{key}"] = (
            dims,
            skill_score(mr["mse"], clim_mse).astype(np.float32),
            {"comment": "1 - MSE_method / MSE_climatology"},
        )
    data["n_valid"] = (dims, accs["model"][0].per_point()["n"].astype(np.int32), {})
    return xr.Dataset(data, coords=coords)


def _daily_block(daily: dict, dates, depths) -> dict:
    """The ``daily_rmse`` JSON block from the per-day, per-depth sums.

    ``rmse`` / ``bias`` / ``corr_anom`` are ``[day][depth]`` per method; ``corr_anom`` is the
    spatial (pattern) correlation of the anomalies over the grid points of that day and depth, not
    the temporal correlation of ``per_depth``. ``pooled_rmse`` is ``[day]`` over the
    ``POOLED_RANGE`` depths pooled together, as are ``pooled_bias`` (method minus reference) and
    ``pooled_corr_anom`` (spatial anomaly correlation over the grid points of all pooled depths).
    """
    d = np.asarray(depths)
    pooled = np.flatnonzero((d >= POOLED_RANGE[0]) & (d <= POOLED_RANGE[1]))
    out: dict = {
        "dates": [str(t.date()) for t in dates],
        "depth": [float(z) for z in depths],
        "pooled_range_m": list(POOLED_RANGE),
        "rmse": {},
        "bias": {},
        "corr_anom": {},
        "pooled_rmse": {},
        "pooled_bias": {},
        "pooled_corr_anom": {},
    }
    for key, sums in daily.items():
        raw = metrics_from_sums(sums["raw"])
        anom = metrics_from_sums(sums["anom"])
        out["rmse"][key] = raw["rmse"]
        out["bias"][key] = np.round(raw["bias"], 5)
        out["corr_anom"][key] = np.round(anom["corr"], 5)
        pooled_sums = {f: sums["raw"][f][:, pooled].sum(axis=1) for f in SUM_FIELDS}
        pooled_raw = metrics_from_sums(pooled_sums)
        out["pooled_rmse"][key] = np.round(pooled_raw["rmse"], 5)
        out["pooled_bias"][key] = np.round(pooled_raw["bias"], 5)
        pooled_anom = {f: sums["anom"][f][:, pooled].sum(axis=1) for f in SUM_FIELDS}
        out["pooled_corr_anom"][key] = np.round(metrics_from_sums(pooled_anom)["corr"], 5)
    return out


# ----------------------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------------------
def evaluate_split(
    cfg: Config,
    split: str = "test",
    device: str | None = None,
    batch_size: int = 8,
    progress: bool = True,
) -> dict:
    """Evaluate every available method on ``split``; write the metrics JSON and the maps file."""
    dev = get_device(device)
    start, end = split_bounds(cfg, split)
    ds = make_surface_dataset(cfg, start, end)
    zds = open_harmonized(cfg)
    grid = build_grid(cfg)
    methods = load_methods(cfg, dev)
    shape = (len(ds.depth), *ds.shape)
    accs = {k: (MetricAccumulator(shape), MetricAccumulator(shape)) for k in methods.predictors}
    n_days = len(ds)
    # per day and depth sums over the whole domain, raw temperature and anomaly, for every method
    daily = {
        k: {kind: {f: np.zeros((n_days, shape[0])) for f in SUM_FIELDS} for kind in ("raw", "anom")}
        for k in methods.predictors
    }

    dates = ds.dates()
    for a in tqdm(range(0, n_days, batch_size), desc=f"evaluate {split}", disable=not progress):
        b = min(a + batch_size, n_days)
        batch = ds.batch(a, b)
        ref = read_target(zds, ds, a, b)
        clim = ds.stats.climatology(dates[a:b].values)
        valid = ds.mask[None] & np.isfinite(ref)
        for key, predictor in methods.predictors.items():
            temp = predict_batch(predictor, batch, ds, dev)
            raw_acc, anom_acc = accs[key]
            raw_acc.update(temp, ref, valid)
            anom_acc.update(temp - clim, ref - clim, valid)
            for kind, (x, y) in {"raw": (temp, ref), "anom": (temp - clim, ref - clim)}.items():
                part = sums_from_arrays(x, y, valid, axis=(2, 3))
                for f in SUM_FIELDS:
                    daily[key][kind][f][a:b] = part[f]

    clim_raw = accs["climatology"][0]
    basin_masks = grid.basin_masks()
    results = {}
    for key, (raw, anom) in accs.items():
        entry = _block(raw, anom, clim_raw, grid.depth)
        entry["per_basin"] = {
            name: _block(raw, anom, clim_raw, grid.depth, region)
            for name, region in basin_masks.items()
        }
        results[key] = entry

    synthetic = data_source(cfg) == "synthetic"
    out = {
        "metadata": {
            "data_source": data_source(cfg),
            "run_name": cfg.run_name,
            "split": split,
            "start": start,
            "end": end,
            "n_days": n_days,
            "depths": [float(d) for d in grid.depth],
            "units": "degC",
            "reference": "GLORYS reanalysis, harmonised to the 0.25 degree grid",
            "conventions": {
                "bias": "prediction minus reference",
                "corr_raw": "correlation of temperature (inflated by seasonal / spatial / "
                "vertical gradients)",
                "corr_anom": "correlation of anomalies, climatology removed from prediction "
                "and reference",
                "skill_vs_clim": "1 - MSE_method / MSE_climatology",
            },
            "pooled_depth_range_m": list(POOLED_RANGE),
            "basins": {k: list(v) for k, v in BASINS.items()},
            "climatology": {
                "train_start": ds.stats.train_start,
                "train_end": ds.stats.train_end,
                "n_harmonic_terms": ds.stats.n_terms,
            },
            "methods": {
                k: {"label": methods.labels[k], **methods.info[k]} for k in methods.predictors
            },
            "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "note": SYNTHETIC_NOTE if synthetic else None,
        },
        "methods": results,
        "daily_rmse": _daily_block(daily, dates, grid.depth),
    }
    mdir = metrics_dir(cfg)
    write_json(mdir / "metrics_glorys.json", out)

    maps = _maps(accs, grid)
    maps.attrs.update(
        title="OceanEmbed verification maps vs GLORYS",
        data_source=data_source(cfg),
        split=split,
        start=start,
        end=end,
        n_days=n_days,
    )
    enc = {v: {"zlib": True, "complevel": 4} for v in maps.data_vars}
    maps.to_netcdf(mdir / "maps_glorys.nc", encoding=enc)
    return out
