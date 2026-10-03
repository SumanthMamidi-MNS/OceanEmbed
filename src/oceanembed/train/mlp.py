"""Training of the per-pixel MLP baseline on a bounded random sample of train points."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from oceanembed.config import Config
from oceanembed.data.dataset import make_dataset
from oceanembed.models.baselines import sample_points
from oceanembed.models.pixel_mlp import PixelMLP
from oceanembed.train.losses import masked_mse
from oceanembed.train.utils import JsonlLogger, cosine_warmup, get_device, set_seed

VAL_SAMPLE_SEED = 1234  # the validation sample is the same for every training seed


@torch.no_grad()
def _val_rmse(model, feats, y, valid, anom_std) -> tuple[float, list[float]]:
    """Pooled and per-depth RMSE in degC on a sample (anomaly error times the per-depth std)."""
    model.eval()
    err = torch.where(valid, (model.forward_features(feats) - y) * anom_std[None], 0.0).double()
    se = (err**2).sum(dim=0)
    cnt = valid.double().sum(dim=0)
    per_depth = torch.sqrt(se / cnt.clamp(min=1)).cpu().tolist()
    return float(torch.sqrt(se.sum() / cnt.sum().clamp(min=1))), per_depth


def run_train_mlp(
    cfg: Config,
    seed: int = 0,
    device: str | None = None,
    progress: bool = True,
    *,
    ckpt_path: Path,
    log_path: Path | None = None,
    datasets: tuple | None = None,
) -> dict:
    """Fit the MLP on ``mlp.max_points`` random train samples (all 7 surface values observed, loss
    over the depths valid at each sample); early stopping on the RMSE in degC of a fixed random
    validation sample. The seed controls the initialisation, the training sample and the batch
    order."""
    mc = cfg.mlp
    set_seed(seed)
    dev = get_device(device)
    train_ds, val_ds = datasets or (
        make_dataset(cfg, "train", preload=True),
        make_dataset(cfg, "val", preload=True),
    )
    anom_std = torch.from_numpy(train_ds.stats.anom_std.astype(np.float32)).to(dev)
    tf, ty, tv = sample_points(train_ds, mc.max_points, seed)
    vf, vy, vv = sample_points(val_ds, mc.val_points, VAL_SAMPLE_SEED)
    tf, ty, tv = (torch.from_numpy(a).to(dev) for a in (tf, ty, tv))
    vf, vy, vv = (torch.from_numpy(a).to(dev) for a in (vf, vy, vv))

    model = PixelMLP(mc.hidden, mc.layers, n_depths=cfg.model.n_depths).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=mc.lr, weight_decay=mc.weight_decay)
    n = len(tf)
    bs = min(mc.batch_size, n)
    steps = max(n // bs, 1)
    sched = cosine_warmup(opt, mc.epochs * steps, steps)
    gen = torch.Generator().manual_seed(seed)

    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(log_path or ckpt_path.with_suffix(".jsonl"))
    best: dict | None = None
    bad = 0
    for epoch in range(1, mc.epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, generator=gen).to(dev)
        total = 0.0
        for i in range(steps):
            idx = perm[i * bs : (i + 1) * bs]
            loss = masked_mse(model.forward_features(tf[idx]), ty[idx], tv[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss)
        rmse, per_depth = _val_rmse(model, vf, vy, vv, anom_std)
        rec = {
            "epoch": epoch,
            "train_loss": total / steps,
            "val_rmse": rmse,
            "val_rmse_per_depth": per_depth,
            "epoch_seconds": time.time() - t0,
        }
        logger.log(rec)
        if progress:
            print(
                f"mlp epoch {epoch}: train {rec['train_loss']:.4f} val RMSE {rmse:.3f} degC",
                flush=True,
            )
        if best is None or rmse < best["val_rmse"]:
            best = {"epoch": epoch, "val_rmse": rmse, "train_loss": rec["train_loss"]}
            bad = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "spec": model.spec,
                    "config": cfg.model_dump(mode="json"),
                    "epoch": epoch,
                    "metrics": best,
                    "seed": seed,
                },
                ckpt_path,
            )
        else:
            bad += 1
            if bad >= mc.patience:
                break
    assert best is not None
    best.update(
        checkpoint=str(ckpt_path),
        epochs_run=epoch,
        seed=seed,
        n_params=sum(p.numel() for p in model.parameters()),
        n_samples=int(n),
    )
    return best
