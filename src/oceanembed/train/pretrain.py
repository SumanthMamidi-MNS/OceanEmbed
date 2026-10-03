"""Self-supervised embedding pretraining by masked surface reconstruction."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from oceanembed.config import SURFACE_VARS, Config
from oceanembed.data.dataset import make_dataset
from oceanembed.models.mae import MaskedAutoencoder
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

PRETRAIN_CKPT = "pretrain.pt"


def _param_groups(model: torch.nn.Module, wd: float) -> list[dict]:
    decay = [p for p in model.parameters() if p.ndim >= 2]
    plain = [p for p in model.parameters() if p.ndim < 2]
    return [{"params": decay, "weight_decay": wd}, {"params": plain, "weight_decay": 0.0}]


@torch.no_grad()
def validate(model: MaskedAutoencoder, loader, device, amp: bool, seed: int = 1234) -> dict:
    """Masked-reconstruction MSE vs the mean-fill baseline (predict 0), per channel and pooled.

    A fixed generator makes the hidden blocks identical every epoch.
    """
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    n = len(SURFACE_VARS)
    sq = torch.zeros(n, device=device, dtype=torch.float64)
    mf = torch.zeros(n, device=device, dtype=torch.float64)
    cnt = torch.zeros(n, device=device, dtype=torch.float64)
    for batch in loader:
        b = to_device(batch, device)
        with torch.autocast(
            device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"
        ):
            out = model(b["x"], b["sv"], generator=gen)
        sq += out["sq_err"].double()
        mf += out["sq_meanfill"].double()
        cnt += out["count"].double()
    c = cnt.clamp(min=1)
    per_mse = (sq / c).cpu().tolist()
    per_mf = (mf / c).cpu().tolist()
    return {
        "val_loss": float(sq.sum() / cnt.sum().clamp(min=1)),
        "val_meanfill": float(mf.sum() / cnt.sum().clamp(min=1)),
        "val_mse_per_channel": dict(zip(SURFACE_VARS, per_mse, strict=True)),
        "val_meanfill_per_channel": dict(zip(SURFACE_VARS, per_mf, strict=True)),
    }


def run_pretrain(
    cfg: Config,
    device: str | None = None,
    progress: bool = True,
    *,
    seed: int | None = None,
    ckpt_path: Path | None = None,
    log_path: Path | None = None,
    datasets: tuple | None = None,
) -> dict:
    """Masked-surface pretraining.

    ``seed`` overrides ``pretrain.seed``; ``ckpt_path`` / ``log_path`` redirect the checkpoint and
    the JSONL log (default: the run's ``checkpoints/pretrain.pt`` and ``logs/pretrain.jsonl``);
    ``datasets`` is an optional ``(train, val)`` pair of preloaded datasets to reuse.
    """
    pc = cfg.pretrain
    seed = pc.seed if seed is None else seed
    set_seed(seed)
    dev = get_device(device)
    amp = pc.amp and dev.type == "cuda"
    train_ds, val_ds = datasets or (
        make_dataset(cfg, "train", preload=True),
        make_dataset(cfg, "val", preload=True),
    )
    train_loader = make_loader(train_ds, pc.batch_size, True, seed=seed)
    val_loader = make_loader(val_ds, pc.batch_size, False)

    model = MaskedAutoencoder(cfg.model, pc).to(dev)
    opt = torch.optim.AdamW(_param_groups(model, pc.weight_decay), lr=pc.lr)
    steps_per_epoch = len(train_loader)
    sched = cosine_warmup(
        opt, pc.epochs * steps_per_epoch, round(pc.warmup_epochs * steps_per_epoch)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    ckpt_path = Path(ckpt_path) if ckpt_path else cfg.checkpoints_dir / PRETRAIN_CKPT
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(log_path or cfg.logs_dir / "pretrain.jsonl")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    crop_rng = np.random.default_rng(seed)
    best: dict | None = None
    for epoch in range(1, pc.epochs + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for batch in tqdm(
            train_loader,
            desc=f"pretrain {epoch}/{pc.epochs}",
            disable=not use_progress_bars(progress),
        ):
            b = random_crop(to_device(batch, dev), pc.crop, crop_rng)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                out = model(b["x"], b["sv"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(out["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), pc.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += float(out["loss"])
            n += 1
        train_time = time.time() - t0
        metrics = validate(model, val_loader, dev, amp)
        rec = {
            "epoch": epoch,
            "train_loss": total / max(n, 1),
            **metrics,
            "lr": opt.param_groups[0]["lr"],
            "epoch_seconds": time.time() - t0,
            "train_seconds": train_time,
            "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else 0.0,
        }
        logger.log(rec)
        if best is None or metrics["val_loss"] < best["val_loss"]:
            best = {"epoch": epoch, **metrics, "train_loss": rec["train_loss"]}
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg.model_dump(mode="json"),
                    "epoch": epoch,
                    "metrics": best,
                },
                ckpt_path,
            )
        if progress:
            print(
                f"epoch {epoch}: train {rec['train_loss']:.4f} val {metrics['val_loss']:.4f} "
                f"(mean-fill {metrics['val_meanfill']:.4f}) {rec['epoch_seconds']:.0f}s",
                flush=True,
            )
    assert best is not None
    best["checkpoint"] = str(ckpt_path)
    best["peak_gpu_mb"] = rec["peak_gpu_mb"]
    best["seed"] = seed
    best["n_params"] = sum(p.numel() for p in model.parameters())
    return best
