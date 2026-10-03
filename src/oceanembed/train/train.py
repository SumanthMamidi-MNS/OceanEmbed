"""Supervised reconstruction training (pretrained encoder fine-tuned at a reduced LR)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from oceanembed.config import Config
from oceanembed.data.dataset import make_dataset
from oceanembed.models.recon import ReconModel
from oceanembed.train.losses import reconstruction_loss
from oceanembed.train.pretrain import PRETRAIN_CKPT
from oceanembed.train.utils import (
    JsonlLogger,
    cosine_warmup,
    get_device,
    make_loader,
    random_crop,
    set_seed,
    to_device,
    use_progress_bars,
)


def recon_ckpt_name(tag: str | None) -> str:
    return f"recon_{tag}.pt" if tag else "recon.pt"


@torch.no_grad()
def validate(model: ReconModel, loader, anom_std: torch.Tensor, device, amp: bool, vw: float):
    """Val loss plus RMSE in degC (overall, pooled over valid points, and per depth).

    ``temp_pred - temp_true = (y_pred - y_true) * anom_std[depth]`` -- the climatology cancels --
    so the degC error needs no date handling here.
    """
    model.eval()
    n_depth = anom_std.numel()
    se = torch.zeros(n_depth, dtype=torch.float64, device=device)
    cnt = torch.zeros(n_depth, dtype=torch.float64, device=device)
    loss_sum, n_batches = 0.0, 0
    for batch in loader:
        b = to_device(batch, device)
        with torch.autocast(
            device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"
        ):
            pred = model(b["x"])
        pred = pred.float()
        loss, _, _ = reconstruction_loss(pred, b["y"], b["mask"], vw)
        loss_sum += float(loss)
        n_batches += 1
        err = torch.where(b["mask"], (pred - b["y"]) * anom_std[None, :, None, None], 0.0)
        se += (err.double() ** 2).sum(dim=(0, 2, 3))
        cnt += b["mask"].double().sum(dim=(0, 2, 3))
    per_depth = torch.sqrt(se / cnt.clamp(min=1)).cpu().tolist()
    overall = float(torch.sqrt(se.sum() / cnt.sum().clamp(min=1)))
    return {
        "val_loss": loss_sum / max(n_batches, 1),
        "val_rmse": overall,
        "val_rmse_per_depth": per_depth,
    }


def run_train(
    cfg: Config,
    pretrained: bool = True,
    tag: str | None = None,
    device: str | None = None,
    progress: bool = True,
    *,
    seed: int | None = None,
    arch: str | None = None,
    ckpt_path: Path | None = None,
    log_path: Path | None = None,
    pretrain_path: Path | None = None,
    datasets: tuple | None = None,
) -> dict:
    """Supervised training of the reconstruction model.

    ``seed`` overrides ``train.seed``; ``arch`` overrides ``model.arch`` (``"unet"`` needs
    ``pretrained=False``); ``ckpt_path`` / ``log_path`` / ``pretrain_path`` redirect the output
    checkpoint, the JSONL log and the pretrained-encoder source; ``datasets`` is an optional
    ``(train, val)`` pair of preloaded datasets to reuse.
    """
    tc = cfg.train
    if arch is not None and arch != cfg.model.arch:
        cfg = cfg.model_copy(update={"model": cfg.model.model_copy(update={"arch": arch})})
    if pretrained and cfg.model.arch != "transformer":
        raise ValueError("a pretrained encoder exists only for the transformer architecture")
    if not pretrained and not tag and ckpt_path is None:
        tag = "scratch"  # never overwrite the pretrained model's recon.pt
    seed = tc.seed if seed is None else seed
    set_seed(seed)
    dev = get_device(device)
    amp = tc.amp and dev.type == "cuda"

    train_ds, val_ds = datasets or (
        make_dataset(cfg, "train", preload=True),
        make_dataset(cfg, "val", preload=True),
    )
    train_loader = make_loader(train_ds, tc.batch_size, True, tc.num_workers, seed=seed)
    val_loader = make_loader(val_ds, tc.batch_size, False)
    anom_std = torch.from_numpy(train_ds.stats.anom_std.astype(np.float32)).to(dev)

    model = ReconModel(cfg.model)
    if pretrained:
        path = Path(pretrain_path) if pretrain_path else cfg.checkpoints_dir / PRETRAIN_CKPT
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `oceanembed pretrain` first")
        model.load_pretrained_encoder(path)
    model.to(dev)
    # fine-tuning a pretrained encoder uses a reduced LR; from scratch everything gets the full LR
    scale = tc.encoder_lr_scale if pretrained else 1.0
    opt = torch.optim.AdamW(model.param_groups(tc.lr, scale, tc.weight_decay))
    steps = len(train_loader)
    sched = cosine_warmup(opt, tc.epochs * steps, round(tc.warmup_epochs * steps))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    ckpt_path = Path(ckpt_path) if ckpt_path else cfg.checkpoints_dir / recon_ckpt_name(tag)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(
        log_path or cfg.logs_dir / (f"train_{tag}.jsonl" if tag else "train.jsonl")
    )
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    crop_rng = np.random.default_rng(seed)
    best: dict | None = None
    bad_epochs = 0
    for epoch in range(1, tc.epochs + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for batch in tqdm(
            train_loader, desc=f"train {epoch}/{tc.epochs}", disable=not use_progress_bars(progress)
        ):
            b = random_crop(to_device(batch, dev), tc.crop, crop_rng)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                pred = model(b["x"])
            loss, _, _ = reconstruction_loss(pred, b["y"], b["mask"], tc.vertical_grad_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += float(loss)
            n += 1
        train_time = time.time() - t0
        metrics = validate(model, val_loader, anom_std, dev, amp, tc.vertical_grad_weight)
        rec = {
            "epoch": epoch,
            "train_loss": total / max(n, 1),
            **metrics,
            "lr_decoder": opt.param_groups[-1]["lr"],
            "epoch_seconds": time.time() - t0,
            "train_seconds": train_time,
            "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else 0.0,
        }
        logger.log(rec)
        if best is None or metrics["val_rmse"] < best["val_rmse"]:
            best = {"epoch": epoch, **metrics, "train_loss": rec["train_loss"]}
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg.model_dump(mode="json"),
                    "epoch": epoch,
                    "metrics": best,
                    "pretrained": pretrained,
                    "tag": tag,
                },
                ckpt_path,
            )
        else:
            bad_epochs += 1
        if progress:
            print(
                f"epoch {epoch}: train {rec['train_loss']:.4f} val {metrics['val_loss']:.4f} "
                f"val RMSE {metrics['val_rmse']:.3f} degC {rec['epoch_seconds']:.0f}s",
                flush=True,
            )
        if bad_epochs >= tc.patience:
            if progress:
                print(
                    f"early stopping after epoch {epoch} (best epoch {best['epoch']})", flush=True
                )
            break
    assert best is not None
    best["checkpoint"] = str(ckpt_path)
    best["peak_gpu_mb"] = rec["peak_gpu_mb"]
    best["epochs_run"] = epoch
    best["epoch_seconds"] = rec["epoch_seconds"]
    best["seed"] = seed
    best["n_params"] = sum(p.numel() for p in model.parameters())
    best["n_params_encoder"] = sum(p.numel() for p in model.encoder.parameters())
    return best
