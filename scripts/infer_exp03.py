#!/usr/bin/env python3
"""
Post-hoc inference for EXP-03 UNETR on BraTS 2021 validation sets.

For each fold: loads best_model.pth, runs sliding-window inference over the
full validation volumes, and computes volumetric Dice (WT/TC/ET) per case —
identical methodology to EXP-02 for a fair comparison.

Output:
  experiments/exp03_unetr/analysis/fold_X_brats_dice.csv   (per case, per fold)
  experiments/exp03_unetr/analysis/all_folds_brats_dice.csv (all cases combined)

Usage:
    python scripts/infer_exp03.py [--folds 0 1 2 3 4] [--device cuda]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monai.inferers import sliding_window_inference

from models.unetr import get_unetr_model
from utils.dataloader import get_dataloader, get_val_dataset
from utils.hd95 import REGIONS, score_case, summarise

# ---------- paths ----------
REPO = Path(__file__).resolve().parent.parent
EXP_DIR = REPO / "experiments" / "exp03_unetr"
CKPT_BASE = Path("/data/paul_daten/experiments/exp03_unetr")
CONFIG_PATH = REPO / "configs" / "exp03_unetr.yaml"
OUT_DIR = EXP_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_FOLDS = 5


# ====================================================================
# Volumetric Dice helpers (identical to EXP-02 post-hoc analysis)
# ====================================================================
def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    intersection = int(np.sum(pred_mask & gt_mask))
    total = int(np.sum(pred_mask)) + int(np.sum(gt_mask))
    if total == 0:
        return 1.0
    return 2.0 * intersection / total


def compute_brats_regions(pred: np.ndarray, gt: np.ndarray) -> dict:
    """BraTS region Dice. Labels: 0=bg, 1=NCR, 2=ED, 3=ET."""
    return {
        "dice_wt": dice_score(pred >= 1, gt >= 1),
        "dice_tc": dice_score(np.isin(pred, [1, 3]), np.isin(gt, [1, 3])),
        "dice_et": dice_score(pred == 3, gt == 3),
    }


# ====================================================================
# Main inference loop
# ====================================================================
def infer_fold(fold: int, model, device: torch.device, sw_cfg: dict) -> pd.DataFrame:
    """Run inference on all validation cases for one fold. Returns per-case DataFrame."""
    val_ds = get_val_dataset(fold=fold, cache_rate=0.0, num_workers=2)
    val_loader = get_dataloader(val_ds, batch_size=1, shuffle=False,
                                num_workers=2, pin_memory=False)

    print(f"  Fold {fold}: {len(val_ds)} validation cases")
    results = []
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            case_id = batch["case_id"][0]
            imgs = batch["image"].to(device)    # (1, 4, H, W, D)
            lbls = batch["label"].to(device)    # (1, 1, H, W, D)

            t0 = time.time()
            logits = sliding_window_inference(
                inputs=imgs,
                roi_size=sw_cfg["roi_size"],
                sw_batch_size=sw_cfg["sw_batch_size"],
                predictor=model,
                overlap=sw_cfg["overlap"],
                mode=sw_cfg["mode"],
            )
            elapsed = time.time() - t0

            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int32)
            gt = lbls.squeeze().cpu().numpy().astype(np.int32)

            # score_case yields Dice + HD95 per region (shared utils/hd95 code path)
            scores = score_case(pred, gt)
            scores["case_id"] = case_id
            scores["infer_time_s"] = round(elapsed, 2)
            results.append(scores)

            print(f"    [{i+1:3d}/{len(val_ds)}] {case_id}  "
                  f"WT={scores['dice_wt']:.4f}  TC={scores['dice_tc']:.4f}  "
                  f"ET={scores['dice_et']:.4f}  ({elapsed:.1f}s)")

    return pd.DataFrame(results)


def run_inference(folds: list[int], device: torch.device):
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    model_cfg = cfg["model"]
    sw_cfg = cfg["validation"]["sliding_window"]

    all_fold_dfs = {}

    for fold in folds:
        ckpt_path = CKPT_BASE / f"fold_{fold}" / "best_model.pth"
        if not ckpt_path.exists():
            print(f"WARNING: checkpoint not found for fold {fold}: {ckpt_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Fold {fold}  –  checkpoint: {ckpt_path}")
        print(f"{'='*60}")

        # Load model
        model = get_unetr_model(
            img_size=tuple(model_cfg["img_size"]),
            in_channels=model_cfg["in_channels"],
            out_channels=model_cfg["out_channels"],
            dropout_rate=model_cfg["dropout_rate"],
        ).to(device)

        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        saved_epoch = ckpt.get("epoch", "?")
        saved_dice = ckpt.get("val_dice_wt", "?")
        print(f"  Loaded best checkpoint (epoch={saved_epoch}, val_dice_wt={saved_dice:.4f})")

        # Run inference
        df_fold = infer_fold(fold, model, device, sw_cfg)
        all_fold_dfs[fold] = df_fold

        # Save per-fold CSV
        out_path = OUT_DIR / f"fold_{fold}_brats_dice_nifti.csv"
        df_fold.to_csv(out_path, index=False)
        print(f"\n  Fold {fold} results saved: {out_path}")

        # Print fold summary
        for region in ["dice_wt", "dice_tc", "dice_et"]:
            vals = df_fold[region].values
            print(f"  {region.upper():10s}: {vals.mean():.4f} ± {vals.std():.4f}  "
                  f"(median={np.median(vals):.4f}, min={vals.min():.4f}, max={vals.max():.4f})")

        # Free GPU memory before next fold
        del model
        torch.cuda.empty_cache()

    if not all_fold_dfs:
        print("ERROR: no folds processed.")
        return

    # Aggregate across all folds (Dice + HD95 in one CSV)
    df_all = pd.concat(all_fold_dfs.values(), ignore_index=True)
    df_all.to_csv(OUT_DIR / "all_folds_brats_dice_nifti.csv", index=False)

    # HD95 summary (NaN-aware, shared utils/hd95 convention) — RQ1 CV protocol
    hd95_summary = {
        "experiment": "exp03_unetr",
        "eval_protocol": "5fold_cv",
        "n_cases": len(df_all),
        "hd95_convention": "NaN (excluded) when region absent in pred or GT; mm == voxel (1mm isotropic)",
        "overall": summarise(df_all),
    }
    with open(OUT_DIR / "summary_hd95.json", "w") as f:
        json.dump(hd95_summary, f, indent=2)

    print(f"\n{'='*60}")
    print("AGGREGATE ACROSS ALL FOLDS (volumetric NIfTI Dice + HD95)")
    print(f"{'='*60}")
    s = hd95_summary["overall"]
    for region in REGIONS:
        rl = region.lower()
        print(f"  {region:3s}: Dice={s[f'mean_dice_{rl}']:.4f}  "
              f"HD95={s[f'mean_hd95_{rl}']:.2f} mm (n={s[f'n_hd95_{rl}']})")

    print("\nPer-fold mean Dice (CV summary):")
    for fold, df in sorted(all_fold_dfs.items()):
        wt = df["dice_wt"].mean()
        tc = df["dice_tc"].mean()
        et = df["dice_et"].mean()
        print(f"  Fold {fold}: WT={wt:.4f}  TC={tc:.4f}  ET={et:.4f}")

    # CV mean ± std across fold means
    print("\nCV mean ± std (fold-level means):")
    for region in ["dice_wt", "dice_tc", "dice_et"]:
        fold_means = np.array([all_fold_dfs[f][region].mean() for f in sorted(all_fold_dfs)])
        print(f"  {region.upper():10s}: {fold_means.mean():.4f} ± {fold_means.std():.4f}")

    # Comparison with EXP-02
    exp02_csv = REPO / "experiments" / "exp02_nnunet" / "analysis" / "all_folds_brats_dice.csv"
    if exp02_csv.exists():
        df02 = pd.read_csv(exp02_csv)
        print(f"\n{'='*60}")
        print("DIRECT COMPARISON (both post-hoc volumetric NIfTI Dice)")
        print(f"{'='*60}")
        print(f"{'Model':<20} {'WT':>8} {'TC':>8} {'ET':>8}")
        print("-" * 50)
        print(f"{'EXP-03 UNETR':<20} "
              f"{df_all['dice_wt'].mean():>8.4f} "
              f"{df_all['dice_tc'].mean():>8.4f} "
              f"{df_all['dice_et'].mean():>8.4f}")
        print(f"{'EXP-02 nnU-Net':<20} "
              f"{df02['dice_wt'].mean():>8.4f} "
              f"{df02['dice_tc'].mean():>8.4f} "
              f"{df02['dice_et'].mean():>8.4f}")
        delta_wt = df_all['dice_wt'].mean() - df02['dice_wt'].mean()
        delta_tc = df_all['dice_tc'].mean() - df02['dice_tc'].mean()
        delta_et = df_all['dice_et'].mean() - df02['dice_et'].mean()
        print(f"{'Delta (UNETR - nnUNet)':<20} "
              f"{delta_wt:>+8.4f} {delta_tc:>+8.4f} {delta_et:>+8.4f}")

    print(f"\nAll results saved to: {OUT_DIR}/")


# ====================================================================
# CLI
# ====================================================================
def main():
    parser = argparse.ArgumentParser(description="EXP-03 UNETR post-hoc inference")
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(NUM_FOLDS)),
                        help="Folds to run inference on (default: 0 1 2 3 4)")
    parser.add_argument("--device", default="cuda",
                        help="Device: cuda or cpu (default: cuda)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nRunning post-hoc inference for EXP-03 UNETR")
    print(f"Folds: {args.folds}")
    print(f"Checkpoints: {CKPT_BASE}/fold_X/best_model.pth")
    print(f"Output:      {OUT_DIR}/")

    run_inference(args.folds, device)


if __name__ == "__main__":
    main()
