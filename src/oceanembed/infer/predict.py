"""Checkpoint loading, batch prediction helpers and the CF-1.8 NetCDF prediction product.

A *predictor* is any callable ``x (B,12,H,W) tensor -> standardised anomaly (B,15,H,W) tensor``
(the reconstruction model, the ridge baseline, the climatology baseline). :func:`predict_batch`
turns a predictor's output into temperature in degC with NaN outside the ocean mask, and
:func:`predict_to_netcdf` writes the monthly ``oceanembed_T_<YYYYMM>.nc`` product.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from tqdm import tqdm

from oceanembed.config import Config
from oceanembed.data.dataset import OceanDataset, make_surface_dataset
from oceanembed.models.recon import ReconModel
from oceanembed.runmeta import data_source

log = logging.getLogger(__name__)

Predictor = Callable[[torch.Tensor], torch.Tensor]


def load_recon_model(path: str | Path, device=None) -> tuple[ReconModel, Config]:
    """Rebuild a :class:`ReconModel` (eval mode) from a ``recon*.pt`` checkpoint.

    The returned model carries ``ckpt_meta`` (path, epoch, pretrained, tag, val_rmse) so products
    and metrics can record their provenance.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config.model_validate(ckpt["config"])
    model = ReconModel(cfg.model)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.ckpt_meta = {
        "path": str(path),
        "epoch": ckpt.get("epoch"),
        "pretrained": ckpt.get("pretrained"),
        "tag": ckpt.get("tag"),
        "val_rmse": (ckpt.get("metrics") or {}).get("val_rmse"),
    }
    if device is not None:
        model.to(device)
    return model, cfg


def check_inputs(cfg: Config, ckpt_cfg: Config, path: str | Path) -> None:
    """A checkpoint must be used with the input groups it was trained with."""
    if list(ckpt_cfg.model.input_groups) != list(cfg.model.input_groups):
        raise ValueError(
            f"{path} was trained with input groups {list(ckpt_cfg.model.input_groups)}, but the "
            f"config selects {list(cfg.model.input_groups)}"
        )


def model_predictor(model: ReconModel, amp: bool = True) -> Predictor:
    """Wrap a model as a no-grad predictor returning float32 anomalies."""

    @torch.no_grad()
    def fn(x: torch.Tensor) -> torch.Tensor:
        model.eval()
        dev = next(model.parameters()).device
        with torch.autocast(dev.type, dtype=torch.float16, enabled=amp and dev.type == "cuda"):
            out = model(x.to(dev))
        return out.float()

    return fn


def predict_batch(predictor: Predictor, batch: dict, dataset: OceanDataset, device) -> np.ndarray:
    """Temperature ``(B, 15, H, W)`` degC (float32) with NaN where the static ocean mask is False.

    ``batch`` is a collated dataset batch (needs ``x`` and ``t``).
    """
    anom = predictor(batch["x"].to(device)).detach().cpu()
    temp = dataset.denormalize(anom, batch["t"])
    return np.where(dataset.mask[None], temp, np.nan).astype(np.float32)


# ----------------------------------------------------------------------------------------
# CF-1.8 NetCDF prediction product
# ----------------------------------------------------------------------------------------
FILL_VALUE = np.float32(9.96921e36)  # netCDF default float fill
PRODUCT_FILE = "oceanembed_T_{ym}.nc"


WEIGHTS_STATS_FILE = "stats.nc"  # statistics + ocean mask inside a released-weights folder
RIDGE_PRODUCT_DIR = "ridge"  # predictions/ridge/: the ridge baseline's product (reserved name)
MLP_PRODUCT_DIR = "mlp"  # predictions/mlp/: the per-pixel MLP baseline's product (reserved name)


def predictions_dir(
    cfg: Config,
    tag: str | None = None,
    ridge: bool = False,
    mlp: bool = False,
    base: Path | None = None,
) -> Path:
    """``outputs/<run>/predictions`` (main model), ``.../predictions/<tag>`` (ablation),
    ``.../predictions/ridge`` or ``.../predictions/mlp`` (baselines). ``base`` replaces
    ``outputs/<run>/predictions`` (prediction from released weights writes elsewhere)."""
    base = cfg.outputs_dir / "predictions" if base is None else Path(base)
    if ridge:
        return base / RIDGE_PRODUCT_DIR
    if mlp:
        return base / MLP_PRODUCT_DIR
    return base / tag if tag else base


def product_path(
    cfg: Config,
    ym: str,
    tag: str | None = None,
    ridge: bool = False,
    mlp: bool = False,
    base: Path | None = None,
) -> Path:
    return predictions_dir(cfg, tag, ridge, mlp, base) / PRODUCT_FILE.format(ym=ym)


def expected_product_files(
    cfg: Config, start, end, tag: str | None = None, ridge: bool = False, mlp: bool = False
) -> list[Path]:
    months = pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M")
    return [product_path(cfg, m.strftime("%Y%m"), tag, ridge, mlp) for m in months]


def _month_dataset(
    cfg: Config, temp: np.ndarray, dates: pd.DatetimeIndex, ds: OceanDataset, meta: dict
) -> xr.Dataset:
    synthetic = meta["data_source"] == "synthetic"
    method = meta.get("method", "model")
    long_name = {
        "ridge": "sea water potential temperature (ridge regression baseline)",
        "mlp": "sea water potential temperature (per-pixel MLP baseline)",
    }.get(method, "sea water potential temperature (OceanEmbed reconstruction)")
    out = xr.Dataset(
        {
            "temperature": (
                ("time", "depth", "lat", "lon"),
                temp.astype(np.float32),
                {
                    "long_name": long_name,
                    "standard_name": "sea_water_potential_temperature",
                    "units": "degree_Celsius",
                    "cell_methods": "time: mean",
                    "comment": "NaN (_FillValue) outside the ocean / below the sea-floor mask",
                },
            )
        },
        coords={
            "time": ("time", dates.values.astype("datetime64[ns]")),
            "depth": ("depth", ds.depth.astype(np.float64)),
            "lat": ("lat", ds.lat.astype(np.float64)),
            "lon": ("lon", ds.lon.astype(np.float64)),
        },
    )
    out["time"].attrs.update(standard_name="time", long_name="time", axis="T")
    out["depth"].attrs.update(
        standard_name="depth", long_name="depth below sea surface", units="m",
        positive="down", axis="Z",
    )  # fmt: skip
    out["lat"].attrs.update(
        standard_name="latitude", long_name="latitude", units="degrees_north", axis="Y"
    )
    out["lon"].attrs.update(
        standard_name="longitude", long_name="longitude", units="degrees_east", axis="X"
    )
    comment = "daily, 0.25 degree grid; reanalysis-trained estimate, not an observation"
    if synthetic:
        comment += "; SYNTHETIC DATA - demonstrates the pipeline only, not real ocean state"
    out.attrs.update(
        Conventions="CF-1.8",
        title={
            "ridge": "Ridge-regression baseline of subsurface ocean temperature",
            "mlp": "Per-pixel MLP baseline of subsurface ocean temperature",
        }.get(method, "OceanEmbed reconstructed subsurface ocean temperature"),
        institution="OceanEmbed proof-of-concept (research prototype)",
        source={
            "ridge": "Pixel-wise ridge regression from surface satellite fields (SST, SSS, SLA, "
            "surface currents, surface winds, day of year, lat / lon); trained on GLORYS "
            "reanalysis",
            "mlp": "Per-pixel multilayer perceptron on the same features as the ridge baseline "
            "(no spatial context); trained on GLORYS reanalysis",
        }.get(
            method,
            "Deep-learning reconstruction from surface satellite fields (SST, SSS, SLA, "
            "surface currents, surface winds) via a learned embedding; trained on GLORYS "
            "reanalysis",
        ),
        method=method,
        input_variables=",".join(cfg.model.input_variables),
        history=f"{datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')} created by oceanembed predict",
        references="OceanEmbed: subsurface temperature reconstructed from surface satellite fields",
        comment=comment,
        data_source=meta["data_source"],
        run_name=cfg.run_name,
        model_checkpoint=meta["checkpoint"],
        model_checkpoint_epoch=-1 if meta["epoch"] is None else int(meta["epoch"]),
        model_pretrained_encoder=str(meta["pretrained"]),
        training_period=f"{cfg.split.train.start} .. {cfg.split.train.end}",
        grid_resolution_deg=cfg.grid.resolution,
        config=json.dumps(cfg.model_dump(mode="json"), default=str),
    )
    return out


def write_month(path: Path, month_ds: xr.Dataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _, nd, h, w = month_ds["temperature"].shape
    enc = {
        "temperature": {
            "dtype": "float32",
            "_FillValue": FILL_VALUE,
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "chunksizes": (1, nd, h, w),
        },
        "time": {"units": "days since 1970-01-01 00:00:00", "dtype": "int32"},
        "depth": {"_FillValue": None},
        "lat": {"_FillValue": None},
        "lon": {"_FillValue": None},
    }
    month_ds.to_netcdf(path, encoding=enc, format="NETCDF4")


@torch.no_grad()
def predict_to_netcdf(
    cfg: Config,
    start,
    end,
    tag: str | None = None,
    device: str | None = None,
    batch_size: int = 8,
    progress: bool = True,
    ridge: bool = False,
    mlp: bool = False,
    weights: str | Path | None = None,
    out_dir: str | Path | None = None,
) -> list[Path]:
    """Write ``oceanembed_T_<YYYYMM>.nc`` for every month touched by ``[start, end]``.

    ``ridge=True`` / ``mlp=True`` predict with the fitted baseline instead of the network and write
    to ``predictions/ridge/`` / ``predictions/mlp/`` (``tag`` must then be ``None``).

    Only the surface inputs (from the harmonised store) and the saved statistics are needed -- the
    target temperature is never read, so any day present in the store works. Partial months
    contain only the requested days (a later run for the same month overwrites the file).

    ``weights`` is a folder of released model files (``recon.pt``, ``ridge.joblib``, ``mlp.pt``,
    ``stats.nc``: see ``oceanembed export-weights``): the model, the statistics and the ocean mask
    then come from it instead of ``outputs/<run>/checkpoints`` and ``data/processed``, and the
    products go to ``out_dir`` (default ``outputs/<run>/predictions_from_weights``).
    """
    from oceanembed.data.stats import Stats, load_mask
    from oceanembed.models.baselines import RIDGE_FILE, RidgeBaseline
    from oceanembed.models.pixel_mlp import MLP_FILE, load_pixel_mlp
    from oceanembed.train.train import recon_ckpt_name
    from oceanembed.train.utils import get_device

    if sum([ridge, mlp, tag is not None]) > 1:
        raise ValueError("ridge, mlp and a tag are mutually exclusive")
    dev = get_device(device)
    ck_dir = Path(weights) if weights is not None else cfg.checkpoints_dir
    if weights is not None and not ck_dir.is_dir():
        raise FileNotFoundError(f"weights folder {ck_dir} not found")
    method = "ridge" if ridge else "mlp" if mlp else "model"
    if ridge:
        ckpt = ck_dir / RIDGE_FILE
        if not ckpt.exists():
            raise FileNotFoundError(f"{ckpt} not found; run `oceanembed baseline` first")
        predictor: Predictor = RidgeBaseline.load(ckpt)
        ckpt_meta = {"epoch": None, "pretrained": "n/a"}
    elif mlp:
        ckpt = ck_dir / MLP_FILE
        if not ckpt.exists():
            raise FileNotFoundError(f"{ckpt} not found; run `oceanembed train-mlp` first")
        net, info = load_pixel_mlp(ckpt, dev)
        predictor = model_predictor(net, amp=False)
        ckpt_meta = {"epoch": info.get("epoch"), "pretrained": "n/a"}
    else:
        ckpt = ck_dir / recon_ckpt_name(tag)
        if not ckpt.exists():
            raise FileNotFoundError(f"{ckpt} not found; run `oceanembed train` first")
        model, ckpt_cfg = load_recon_model(ckpt, dev)
        check_inputs(cfg, ckpt_cfg, ckpt)
        predictor = model_predictor(model, amp=cfg.train.amp)
        ckpt_meta = model.ckpt_meta
    stats = mask = None
    if weights is not None:
        stats_file = ck_dir / WEIGHTS_STATS_FILE
        if not stats_file.exists():
            raise FileNotFoundError(f"{stats_file} not found in the weights folder")
        stats, mask = Stats.load(stats_file), load_mask(stats_file)
    ds = make_surface_dataset(cfg, start, end, stats=stats, mask=mask)
    base = None
    if weights is not None:
        base = Path(out_dir) if out_dir else cfg.outputs_dir / "predictions_from_weights"
    elif out_dir is not None:
        base = Path(out_dir)
    meta = {"data_source": data_source(cfg), "checkpoint": ckpt.name, "method": method, **ckpt_meta}
    dates = ds.dates()
    months = dates.to_period("M")
    written: list[Path] = []
    for m in tqdm(months.unique(), desc="predict", disable=not progress):
        sel = np.flatnonzero(months == m)
        chunks = []
        for a in range(int(sel[0]), int(sel[-1]) + 1, batch_size):
            b = min(a + batch_size, int(sel[-1]) + 1)
            chunks.append(predict_batch(predictor, ds.batch(a, b), ds, dev))
        month_ds = _month_dataset(cfg, np.concatenate(chunks), dates[sel], ds, meta)
        path = product_path(cfg, m.strftime("%Y%m"), tag, ridge, mlp, base)
        write_month(path, month_ds)
        written.append(path)
    log.info(
        "wrote %d prediction file(s) under %s",
        len(written),
        predictions_dir(cfg, tag, ridge, mlp, base),
    )
    return written
