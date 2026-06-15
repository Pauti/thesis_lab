#!/usr/bin/env python3
"""
Multi-task UNETR training for BraTS 2021 (EXP-05, RQ3).

Segmentation + auxiliary ET-volume ordinal classification (CORAL). The
segmentation path, validation and checkpointing are identical to scripts/train.py
(and thus to the ABL-07 UNETR baseline); the only additions are the auxiliary
head, the joint loss, and per-fold ordinal-target derivation. Validation reports
the same segmentation Dice as the baseline so EXP-05 is directly comparable to
ABL-07.

Usage:
    python scripts/train_multitask.py --config configs/exp05_unetr_mtl.yaml \
        --fold 0 --num_epochs 100 \
        --splits_file /data/.../splits_ablation.json \
        --output_dir experiments/exp05_unetr_mtl/fold_0

Auxiliary target: ET-volume tertile (low/mid/high). Per-fold thresholds are
computed from the fold's training cases (whole-volume ET counts) and applied
on-the-fly to cropped-patch ET counts (label == 3). Justification: the whole
tumour fits inside a 128^3 patch in ~96 % of BraTS 2021 cases, so tumour-centred
patch counts are consistent with the volume-level thresholds.
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
import yaml
from monai.inferers import sliding_window_inference
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.multitask import get_multitask_unetr_model
from utils.augmentation import get_augmentation
from utils.dataloader import get_dataloader, get_train_dataset, get_val_dataset
from utils.losses import get_loss
from utils.metrics import BraTSMetrics
from utils.ordinal import (
    coral_loss,
    compute_fold_thresholds,
    et_count_to_bin,
    label_to_levels,
    logits_to_rank,
    patch_et_counts,
)


# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("train_mtl")
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


# ── CSV ──────────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "epoch", "train_loss", "train_seg_loss", "train_aux_loss", "train_aux_acc",
    "val_dice_wt", "val_dice_tc", "val_dice_et",
    "val_dice_wt_std", "val_dice_tc_std", "val_dice_et_std",
    "epoch_time_s",
]


def init_csv(path: Path):
    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()


def append_csv(path: Path, row: dict):
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore").writerow(row)


# ── Scheduler (identical to baseline) ────────────────────────────────────────

def get_scheduler(optimizer, cfg: dict, num_epochs: int):
    warmup = cfg.get("warmup_epochs", 5)
    if cfg.get("scheduler", "cosine_warmup") == "cosine_warmup":
        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(1, num_epochs - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    raise ValueError(f"Unknown scheduler: {cfg.get('scheduler')}")


# ── Training ─────────────────────────────────────────────────────────────────

def train(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    tr = cfg["training"]
    mt = cfg["multitask"]
    val_cfg = cfg["validation"]
    es_cfg = cfg["early_stopping"]
    ckpt_cfg = cfg["checkpointing"]
    log_cfg = cfg["logging"]

    num_epochs = args.num_epochs if args.num_epochs is not None else tr["num_epochs"]

    output_dir = Path(args.output_dir)
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info(f"Config: {args.config}  Fold: {args.fold}  Epochs: {num_epochs}")
    logger.info(f"Output: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Multi-task hyper-parameters ──────────────────────────────────────────
    K = mt["num_ordinal_classes"]
    lambda_seg = mt["lambda_seg"]
    lambda_aux = mt["lambda_aux"]
    splits_file = args.splits_file or cfg["data"]["splits_file"]
    et_json = mt["et_volume_json"]

    # Per-fold tertile thresholds from THIS fold's training cases only (no leakage)
    t1, t2 = compute_fold_thresholds(args.fold, splits_file, et_json, num_classes=K)
    logger.info(f"Fold {args.fold} ET-volume thresholds (train-only): t1={t1:.0f}  t2={t2:.0f}")

    effective_cfg = dict(cfg)
    effective_cfg["_runtime"] = {
        "fold": args.fold, "num_epochs": num_epochs, "output_dir": str(output_dir),
        "ordinal_thresholds": {"t1": t1, "t2": t2}, "splits_file": str(splits_file),
    }
    (output_dir / "config.yaml").write_text(yaml.dump(effective_cfg))

    # ── Model ────────────────────────────────────────────────────────────────
    model_cfg = cfg["model"]
    model = get_multitask_unetr_model(
        img_size=tuple(model_cfg["img_size"]),
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        dropout_rate=model_cfg["dropout_rate"],
        num_ordinal_classes=K,
        aux_hidden=mt["aux_hidden"],
    ).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # ── Data ─────────────────────────────────────────────────────────────────
    patch_size = tuple(tr["patch_size"])
    augmentation = get_augmentation(tr["augmentation"], patch_size)
    data_cfg = cfg["data"]
    sf = Path(splits_file)

    train_ds = get_train_dataset(
        fold=args.fold, augmentation_transforms=augmentation,
        cache_rate=data_cfg["cache_rate"], num_workers=data_cfg["num_workers"],
        splits_file=sf,
    )
    val_ds = get_val_dataset(
        fold=args.fold, cache_rate=data_cfg["cache_rate"],
        num_workers=data_cfg["num_workers"], splits_file=sf,
    )
    train_loader = get_dataloader(train_ds, batch_size=tr["batch_size"], shuffle=True,
                                  num_workers=data_cfg["num_workers"])
    val_loader = get_dataloader(val_ds, batch_size=1, shuffle=False,
                                num_workers=data_cfg["num_workers"])
    logger.info(f"Train cases: {len(train_ds)}  Val cases: {len(val_ds)}")

    # ── Loss, optimiser, scheduler ───────────────────────────────────────────
    criterion = get_loss(include_background=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr["lr"],
                                  weight_decay=tr["weight_decay"])
    scheduler = get_scheduler(optimizer, tr, num_epochs)
    scaler = GradScaler(enabled=tr["mixed_precision"])
    grad_accum = tr.get("gradient_accumulation", 1)

    brats_metrics = BraTSMetrics()
    csv_path = output_dir / log_cfg["csv_file"]
    init_csv(csv_path)

    best_val_dice_wt = -1.0
    es_counter = 0
    val_interval = val_cfg["interval"]
    sw_cfg = val_cfg["sliding_window"]

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        tot_loss = tot_seg = tot_aux = 0.0
        aux_correct = aux_total = 0
        num_batches = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            imgs = batch["image"].to(device)    # (B, 4, H, W, D)
            lbls = batch["label"].to(device)    # (B, 1, H, W, D), int64

            # Auxiliary target: ET voxels in THIS crop -> ordinal bin
            et_counts = patch_et_counts(lbls)               # (B,)
            ord_labels = et_count_to_bin(et_counts, t1, t2)  # (B,) in {0..K-1}
            levels = label_to_levels(ord_labels, K)          # (B, K-1)

            with autocast(enabled=tr["mixed_precision"]):
                seg_logits, aux_logits = model(imgs, return_aux=True)
                seg_loss = criterion(seg_logits, lbls)
                aux_loss = coral_loss(aux_logits.float(), levels)
                loss = (lambda_seg * seg_loss + lambda_aux * aux_loss) / grad_accum

            scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            tot_loss += loss.item() * grad_accum
            tot_seg += seg_loss.item()
            tot_aux += aux_loss.item()
            with torch.no_grad():
                preds = logits_to_rank(aux_logits.float())
                aux_correct += (preds == ord_labels).sum().item()
                aux_total += ord_labels.numel()
            num_batches += 1

            if (step + 1) % log_cfg["log_interval"] == 0:
                logger.info(
                    f"Ep {epoch}/{num_epochs}  step {step+1}/{len(train_loader)}  "
                    f"loss={tot_loss/num_batches:.4f}  seg={tot_seg/num_batches:.4f}  "
                    f"aux={tot_aux/num_batches:.4f}  aux_acc={aux_correct/max(aux_total,1):.3f}  "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        scheduler.step()
        mean_loss = tot_loss / max(num_batches, 1)
        mean_seg = tot_seg / max(num_batches, 1)
        mean_aux = tot_aux / max(num_batches, 1)
        mean_aux_acc = aux_correct / max(aux_total, 1)

        # ── Validation (segmentation only — protocol identical to ABL-07) ────
        val_metrics: dict = {}
        if epoch % val_interval == 0:
            torch.cuda.empty_cache()
            model.eval()
            brats_metrics.reset()
            with torch.no_grad():
                for val_batch in val_loader:
                    val_imgs = val_batch["image"].to(device)
                    val_lbls = val_batch["label"].to(device)
                    # return_aux defaults to False -> sliding_window gets one tensor
                    val_logits = sliding_window_inference(
                        inputs=val_imgs, roi_size=sw_cfg["roi_size"],
                        sw_batch_size=sw_cfg["sw_batch_size"], predictor=model,
                        overlap=sw_cfg["overlap"], mode=sw_cfg["mode"],
                    )
                    pred_argmax = torch.argmax(val_logits, dim=1)
                    gt = val_lbls.squeeze(1).long()
                    brats_metrics.update(pred_argmax, gt, compute_hd95=False)

            val_metrics = brats_metrics.compute()
            val_dice_wt = val_metrics.get("dice_wt", 0.0)
            logger.info(
                f"Ep {epoch}/{num_epochs}  val WT={val_dice_wt:.4f}  "
                f"TC={val_metrics.get('dice_tc', 0.):.4f}  "
                f"ET={val_metrics.get('dice_et', 0.):.4f}  "
                f"seg_loss={mean_seg:.4f}  aux_acc={mean_aux_acc:.3f}"
            )

            if ckpt_cfg["save_best"] and val_dice_wt > best_val_dice_wt:
                best_val_dice_wt = val_dice_wt
                torch.save(
                    {"epoch": epoch, "model_state_dict": model.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict(),
                     "val_dice_wt": val_dice_wt, "val_metrics": val_metrics,
                     "ordinal_thresholds": {"t1": t1, "t2": t2}},
                    ckpt_dir / "best_model.pth",
                )
                logger.info(f"  ↑ New best WT Dice: {best_val_dice_wt:.4f}  (saved)")
                es_counter = 0
            else:
                es_counter += 1

            if es_cfg["enabled"] and es_counter >= es_cfg["patience"]:
                logger.info(f"Early stopping: no improvement for {es_cfg['patience']} epochs.")
                break
        else:
            val_dice_wt = 0.0

        if ckpt_cfg["save_last"]:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "scheduler_state_dict": scheduler.state_dict(),
                 "val_dice_wt": val_dice_wt},
                ckpt_dir / "last_model.pth",
            )

        append_csv(csv_path, {
            "epoch": epoch,
            "train_loss": round(mean_loss, 6),
            "train_seg_loss": round(mean_seg, 6),
            "train_aux_loss": round(mean_aux, 6),
            "train_aux_acc": round(mean_aux_acc, 6),
            "val_dice_wt": round(val_metrics.get("dice_wt", 0.), 6),
            "val_dice_tc": round(val_metrics.get("dice_tc", 0.), 6),
            "val_dice_et": round(val_metrics.get("dice_et", 0.), 6),
            "val_dice_wt_std": round(val_metrics.get("dice_wt_std", 0.), 6),
            "val_dice_tc_std": round(val_metrics.get("dice_tc_std", 0.), 6),
            "val_dice_et_std": round(val_metrics.get("dice_et_std", 0.), 6),
            "epoch_time_s": round(time.time() - epoch_start, 1),
        })

    fold_info = {
        "fold": args.fold, "experiment": cfg["experiment"]["name"],
        "num_epochs_run": epoch, "best_val_dice_wt": round(best_val_dice_wt, 6),
        "ordinal_thresholds": {"t1": t1, "t2": t2},
        "lambda_seg": lambda_seg, "lambda_aux": lambda_aux,
        "config": args.config, "output_dir": str(output_dir),
        "checkpoint_dir": str(ckpt_dir),
    }
    (output_dir / "fold_info.json").write_text(json.dumps(fold_info, indent=4))
    logger.info(f"Done. Best WT Dice: {best_val_dice_wt:.4f}")


def parse_args():
    p = argparse.ArgumentParser(description="Train multi-task UNETR on BraTS 2021 (RQ3)")
    p.add_argument("--config", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--num_epochs", type=int, default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--checkpoint_dir", default=None)
    p.add_argument("--splits_file", default=None,
                   help="Defaults to data.splits_file in the config (splits_ablation.json).")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
