#!/usr/bin/env python3
"""
UNETR training script for BraTS 2021 (EXP-03, EXP-05, ABL-04/05/06).

Usage:
    # Debug run (20 epochs, fold 0)
    python scripts/train.py --config configs/exp03_unetr.yaml --fold 0 \
        --num_epochs 20 --output_dir experiments/exp03_unetr/fold_0_debug

    # Full training
    python scripts/train.py --config configs/exp03_unetr.yaml --fold 1 \
        --output_dir experiments/exp03_unetr/fold_1

Outputs (all inside --output_dir):
    checkpoints/best_model.pth     — best val WT-Dice checkpoint
    checkpoints/last_model.pth     — last epoch checkpoint
    metrics.csv                    — per-epoch train loss + val Dice (WT/TC/ET)
    results.json                   — final metrics summary
    train.log                      — full training log
"""

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from monai.inferers import sliding_window_inference
from torch.cuda.amp import GradScaler, autocast

# Ensure repo root is on PYTHONPATH when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.unetr import get_unetr_model
from utils.augmentation import get_augmentation
from utils.dataloader import get_dataloader, get_train_dataset, get_val_dataset, ABLATION_SPLITS_FILE
from utils.losses import get_loss
from utils.metrics import BraTSMetrics


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(output_dir / "train.log")
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# ── CSV writer ────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "epoch", "train_loss",
    "val_dice_wt", "val_dice_tc", "val_dice_et",
    "val_dice_wt_std", "val_dice_tc_std", "val_dice_et_std",
    "epoch_time_s",
]


def init_csv(path: Path):
    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()


def append_csv(path: Path, row: dict):
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writerow(row)


# ── Scheduler ─────────────────────────────────────────────────────────────────

def get_scheduler(optimizer, cfg: dict, num_epochs: int):
    warmup = cfg.get("warmup_epochs", 5)
    sched_name = cfg.get("scheduler", "cosine_warmup")
    if sched_name == "cosine_warmup":
        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(1, num_epochs - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    raise ValueError(f"Unknown scheduler: {sched_name}")


# ── Main training function ────────────────────────────────────────────────────

def train(args):
    # Load config
    cfg = yaml.safe_load(Path(args.config).read_text())
    tr = cfg["training"]
    val_cfg = cfg["validation"]
    es_cfg = cfg["early_stopping"]
    ckpt_cfg = cfg["checkpointing"]
    log_cfg = cfg["logging"]

    # Override epochs if provided via CLI
    num_epochs = args.num_epochs if args.num_epochs is not None else tr["num_epochs"]

    output_dir = Path(args.output_dir)
    # Checkpoints can be redirected to a larger disk via --checkpoint_dir
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info(f"Config: {args.config}  Fold: {args.fold}  Epochs: {num_epochs}")
    logger.info(f"Output: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Save effective config
    effective_cfg = dict(cfg)
    effective_cfg["_runtime"] = {"fold": args.fold, "num_epochs": num_epochs,
                                 "output_dir": str(output_dir)}
    (output_dir / "config.yaml").write_text(yaml.dump(effective_cfg))

    # ── Model ────────────────────────────────────────────────────────────────
    model_cfg = cfg["model"]
    model = get_unetr_model(
        img_size=tuple(model_cfg["img_size"]),
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        dropout_rate=model_cfg["dropout_rate"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params / 1e6:.1f}M")

    # ── Data ─────────────────────────────────────────────────────────────────
    patch_size = tuple(tr["patch_size"])
    augmentation = get_augmentation(tr["augmentation"], patch_size)

    data_cfg = cfg["data"]
    splits_file = Path(args.splits_file) if args.splits_file else None
    train_ds = get_train_dataset(
        fold=args.fold,
        augmentation_transforms=augmentation,
        cache_rate=data_cfg["cache_rate"],
        num_workers=data_cfg["num_workers"],
        splits_file=splits_file,
    )
    val_ds = get_val_dataset(
        fold=args.fold,
        cache_rate=data_cfg["cache_rate"],
        num_workers=data_cfg["num_workers"],
        splits_file=splits_file,
    )

    train_loader = get_dataloader(
        train_ds, batch_size=tr["batch_size"], shuffle=True,
        num_workers=data_cfg["num_workers"],
    )
    val_loader = get_dataloader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=data_cfg["num_workers"],
    )

    logger.info(f"Train cases: {len(train_ds)}  Val cases: {len(val_ds)}")

    # ── Loss, optimiser, scheduler ───────────────────────────────────────────
    criterion = get_loss(include_background=False).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tr["lr"],
        weight_decay=tr["weight_decay"],
    )
    scheduler = get_scheduler(optimizer, tr, num_epochs)
    scaler = GradScaler(enabled=tr["mixed_precision"])

    grad_accum = tr.get("gradient_accumulation", 1)

    # ── Metrics & CSV ────────────────────────────────────────────────────────
    brats_metrics = BraTSMetrics()
    csv_path = output_dir / log_cfg["csv_file"]
    init_csv(csv_path)

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_dice_wt = -1.0
    es_counter = 0
    val_interval = val_cfg["interval"]
    sw_cfg = val_cfg["sliding_window"]

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            # RandCropByPosNegLabeld returns a list of dicts when num_samples > 1;
            # when num_samples=1 it returns a list of length 1.  Collation from
            # DataLoader already stacks them to (B, …) tensors.
            imgs = batch["image"].to(device)    # (B, 4, H, W, D)
            lbls = batch["label"].to(device)    # (B, 1, H, W, D), int64

            with autocast(enabled=tr["mixed_precision"]):
                logits = model(imgs)             # (B, 4, H, W, D)
                loss = criterion(logits, lbls) / grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum
            num_batches += 1

            if (step + 1) % log_cfg["log_interval"] == 0:
                logger.info(
                    f"Ep {epoch}/{num_epochs}  step {step+1}/{len(train_loader)}  "
                    f"loss={total_loss/num_batches:.4f}  "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        scheduler.step()
        mean_train_loss = total_loss / max(num_batches, 1)

        # ── Validation ───────────────────────────────────────────────────────
        val_metrics: dict = {}
        if epoch % val_interval == 0:
            # Free cached CUDA memory before validation
            torch.cuda.empty_cache()
            
            model.eval()
            brats_metrics.reset()

            with torch.no_grad():
                for val_batch in val_loader:
                    val_imgs = val_batch["image"].to(device)  # (1, 4, H, W, D)
                    val_lbls = val_batch["label"].to(device)  # (1, 1, H, W, D)

                    val_logits = sliding_window_inference(
                        inputs=val_imgs,
                        roi_size=sw_cfg["roi_size"],
                        sw_batch_size=sw_cfg["sw_batch_size"],
                        predictor=model,
                        overlap=sw_cfg["overlap"],
                        mode=sw_cfg["mode"],
                    )
                    # Argmax predictions (1, H, W, D)
                    pred_argmax = torch.argmax(val_logits, dim=1)
                    gt = val_lbls.squeeze(1).long()

                    brats_metrics.update(pred_argmax, gt, compute_hd95=False)

            val_metrics = brats_metrics.compute()
            val_dice_wt = val_metrics.get("dice_wt", 0.0)

            logger.info(
                f"Ep {epoch}/{num_epochs}  val WT={val_dice_wt:.4f}  "
                f"TC={val_metrics.get('dice_tc', 0.):.4f}  "
                f"ET={val_metrics.get('dice_et', 0.):.4f}  "
                f"loss={mean_train_loss:.4f}"
            )

            # ── Checkpoint: best model ────────────────────────────────────────
            if ckpt_cfg["save_best"] and val_dice_wt > best_val_dice_wt:
                best_val_dice_wt = val_dice_wt
                torch.save(
                    {"epoch": epoch, "model_state_dict": model.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict(),
                     "val_dice_wt": val_dice_wt, "val_metrics": val_metrics},
                    ckpt_dir / "best_model.pth",
                )
                logger.info(f"  ↑ New best WT Dice: {best_val_dice_wt:.4f}  (saved)")
                es_counter = 0
            else:
                es_counter += 1

            # ── Early stopping ────────────────────────────────────────────────
            if es_cfg["enabled"] and es_counter >= es_cfg["patience"]:
                logger.info(
                    f"Early stopping: no improvement for {es_cfg['patience']} epochs."
                )
                break
        else:
            val_dice_wt = 0.0  # not computed this epoch

        # ── Checkpoint: last model ────────────────────────────────────────────
        if ckpt_cfg["save_last"]:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "scheduler_state_dict": scheduler.state_dict(),
                 "val_dice_wt": val_dice_wt},
                ckpt_dir / "last_model.pth",
            )

        # ── CSV row ───────────────────────────────────────────────────────────
        epoch_time = time.time() - epoch_start
        append_csv(csv_path, {
            "epoch": epoch,
            "train_loss": round(mean_train_loss, 6),
            "val_dice_wt": round(val_metrics.get("dice_wt", 0.), 6),
            "val_dice_tc": round(val_metrics.get("dice_tc", 0.), 6),
            "val_dice_et": round(val_metrics.get("dice_et", 0.), 6),
            "val_dice_wt_std": round(val_metrics.get("dice_wt_std", 0.), 6),
            "val_dice_tc_std": round(val_metrics.get("dice_tc_std", 0.), 6),
            "val_dice_et_std": round(val_metrics.get("dice_et_std", 0.), 6),
            "epoch_time_s": round(epoch_time, 1),
        })

    # ── Final fold_info.json ──────────────────────────────────────────────────
    fold_info = {
        "fold": args.fold,
        "experiment": cfg["experiment"]["name"],
        "num_epochs_run": epoch,
        "best_val_dice_wt": round(best_val_dice_wt, 6),
        "config": args.config,
        "output_dir": str(output_dir),
        "checkpoint_dir": str(ckpt_dir),
    }
    (output_dir / "fold_info.json").write_text(
        json.dumps(fold_info, indent=4)
    )
    logger.info(f"Done. Best WT Dice: {best_val_dice_wt:.4f}")
    logger.info(f"fold_info saved to {output_dir / 'fold_info.json'}")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train UNETR on BraTS 2021")
    p.add_argument("--config", required=True,
                   help="Path to YAML config file")
    p.add_argument("--fold", type=int, required=True,
                   help="Cross-validation fold index (0-4)")
    p.add_argument("--num_epochs", type=int, default=None,
                   help="Override num_epochs from config (e.g. 20 for debug)")
    p.add_argument("--output_dir", required=True,
                   help="Directory to save logs, metrics.csv, results.json")
    p.add_argument("--checkpoint_dir", default=None,
                   help="Directory to save checkpoints (default: output_dir/checkpoints). "
                        "Redirect to /data to avoid filling /home.")
    p.add_argument("--splits_file", default=None,
                   help="Path to splits JSON (default: splits_final.json). "
                        "Use splits_ablation.json for RQ2 ablation runs.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
