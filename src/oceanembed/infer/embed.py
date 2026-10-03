"""Export encoder embedding maps to ``outputs/<run>/embeddings/embeddings.zarr``.

Dims ``(time, emb, y, x)``; each embedding cell covers 4x4 canonical-grid pixels and its
``lat(y)`` / ``lon(x)`` coordinates are the centres of those blocks.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

from oceanembed.config import Config
from oceanembed.data.dataset import make_dataset
from oceanembed.models.encoder import OceanEncoder
from oceanembed.models.recon import load_encoder_state
from oceanembed.train.pretrain import PRETRAIN_CKPT
from oceanembed.train.train import recon_ckpt_name
from oceanembed.train.utils import get_device, make_loader

DOWNSAMPLE = 4  # encoder stride


def embedding_path(cfg: Config) -> Path:
    return cfg.outputs_dir / "embeddings" / "embeddings.zarr"


def default_checkpoint(cfg: Config) -> Path:
    """Fine-tuned encoder of the final model if it exists, else the pretrained encoder."""
    recon = cfg.checkpoints_dir / recon_ckpt_name(None)
    return recon if recon.exists() else cfg.checkpoints_dir / PRETRAIN_CKPT


def cell_centres(coord: np.ndarray, stride: int = DOWNSAMPLE) -> np.ndarray:
    """Centres of consecutive ``stride``-pixel blocks of a 1-D cell-centre coordinate."""
    return coord.reshape(-1, stride).mean(axis=1)


@torch.no_grad()
def export_embeddings(
    cfg: Config,
    split: str = "test",
    checkpoint: str | Path | None = None,
    device: str | None = None,
    batch_size: int = 8,
    progress: bool = True,
) -> Path:
    dev = get_device(device)
    ckpt = Path(checkpoint) if checkpoint else default_checkpoint(cfg)
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found; run `oceanembed pretrain` / `train` first")
    encoder = OceanEncoder(cfg.model)
    encoder.load_state_dict(load_encoder_state(ckpt))
    encoder.to(dev).eval()

    ds = make_dataset(cfg, split, preload=True)
    loader = make_loader(ds, batch_size, False)
    out = embedding_path(cfg)
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    amp = cfg.train.amp and dev.type == "cuda"
    lat = cell_centres(ds.lat)
    lon = cell_centres(ds.lon)
    first = True
    for batch in tqdm(loader, desc=f"embed {split}", disable=not progress):
        x = batch["x"].to(dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
            emb = encoder(x).emb
        emb = emb.float().cpu().numpy()
        times = ds.time[ds.indices[batch["index"].numpy()]]
        piece = xr.Dataset(
            {"embedding": (("time", "emb", "y", "x"), emb)},
            coords={
                "time": times,
                "lat": ("y", lat),
                "lon": ("x", lon),
            },
        )
        if first:
            piece.attrs.update(
                title="OceanEmbed satellite embeddings",
                split=split,
                checkpoint=str(ckpt),
                stride=DOWNSAMPLE,
            )
            piece["embedding"].encoding["chunks"] = (1, *emb.shape[1:])
            piece.to_zarr(out, mode="w", consolidated=False)
            first = False
        else:
            piece.to_zarr(out, append_dim="time", consolidated=False)
    return out
