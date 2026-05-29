#!/usr/bin/env python3
"""
Inference for UNETR Default Augmentation on BraTS 2021 held-out test set (125 cases).

Loads best_model.pth from each fold, runs sliding-window inference, and computes
volumetric Dice (WT/TC/ET) per case. Ensemble prediction uses majority voting
across 3 folds.

Output:
  experiments/ablation/abl_unetr_default/test_predictions/fold_X_results.csv
  experiments/ablation/abl_unetr_default/test_predictions/ensemble_results.csv
  experiments/ablation/abl_unetr_default/test_predictions/summary.json

Usage:
    python scripts/infer_ablation_unetr.py
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
)

from models.unetr import get_unetr_model

# ========== Paths ==========
REPO = Path(__file__).resolve().parent.parent
DATA_DIR = Path("/data/paul_daten/nnunet/nnUNet_raw/Dataset001_BraTS")
TEST_SET_JSON = Path("/data/paul_daten/nnunet/nnUNet_preprocessed/Dataset001_BraTS/test_set.json")
CONFIG_PATH = REPO / "configs" / "exp03_unetr.yaml"

VALID_TIERS = ("disabled", "reduced", "default", "extended")

# Tier-dependent globals — set in main() based on --tier
EXP_DIR = CKPT_BASE = OUT_DIR = None


# ========== Data Transforms ==========
def get_test_transforms():
    """Same preprocessing as training (without augmentation)."""
    return Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=False, image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ])


def get_test_dataset():
    """Load test set (125 held-out cases)."""
    with open(TEST_SET_JSON) as f:
        test_set = json.load(f)
    
    case_ids = test_set["case_ids"]
    print(f"Loaded {len(case_ids)} test cases from {TEST_SET_JSON}")
    
    data_dicts = []
    for case_id in case_ids:
        data_dicts.append({
            "case_id": case_id,
            "image": [
                str(DATA_DIR / "imagesTr" / f"{case_id}_0000.nii.gz"),
                str(DATA_DIR / "imagesTr" / f"{case_id}_0001.nii.gz"),
                str(DATA_DIR / "imagesTr" / f"{case_id}_0002.nii.gz"),
                str(DATA_DIR / "imagesTr" / f"{case_id}_0003.nii.gz"),
            ],
            "label": str(DATA_DIR / "labelsTr" / f"{case_id}.nii.gz"),
        })
    
    return Dataset(data=data_dicts, transform=get_test_transforms())


# ========== Metrics ==========
def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Volumetric Dice coefficient."""
    intersection = int(np.sum(pred_mask & gt_mask))
    total = int(np.sum(pred_mask)) + int(np.sum(gt_mask))
    if total == 0:
        return 1.0
    return 2.0 * intersection / total


def compute_brats_dice(pred: np.ndarray, gt: np.ndarray) -> dict:
    """BraTS region Dice. Labels: 0=bg, 1=NCR, 2=ED, 3=ET."""
    return {
        "dice_wt": dice_score(pred >= 1, gt >= 1),
        "dice_tc": dice_score(np.isin(pred, [1, 3]), np.isin(gt, [1, 3])),
        "dice_et": dice_score(pred == 3, gt == 3),
    }


# ========== Inference ==========
def infer_fold(fold: int, device: str = "cuda"):
    """Run inference for one fold on test set."""
    print(f"\n{'='*70}")
    print(f"Fold {fold} Inference")
    print(f"{'='*70}")
    
    # Load config
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    
    model_cfg = cfg["model"]
    sw_cfg = cfg["validation"]["sliding_window"]
    
    # Load model
    checkpoint_path = CKPT_BASE / f"fold_{fold}" / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    model = get_unetr_model(
        img_size=tuple(model_cfg["img_size"]),
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        dropout_rate=model_cfg["dropout_rate"],
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded (epoch {checkpoint.get('epoch', 'N/A')})")
    
    # Load test dataset
    test_ds = get_test_dataset()
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
    
    print(f"Test set: {len(test_ds)} cases")
    print(f"Starting inference...")
    
    results = []
    predictions_dict = {}  # For ensemble
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Fold {fold}"):
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
            
            scores = compute_brats_dice(pred, gt)
            scores["case_id"] = case_id
            scores["infer_time_s"] = round(elapsed, 2)
            results.append(scores)
            
            # Store prediction for ensemble
            predictions_dict[case_id] = pred
    
    # Save results
    df = pd.DataFrame(results)
    output_path = OUT_DIR / f"fold_{fold}_results.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\nFold {fold} Results:")
    print(f"  Mean WT Dice: {df['dice_wt'].mean():.4f} ± {df['dice_wt'].std():.4f}")
    print(f"  Mean TC Dice: {df['dice_tc'].mean():.4f} ± {df['dice_tc'].std():.4f}")
    print(f"  Mean ET Dice: {df['dice_et'].mean():.4f} ± {df['dice_et'].std():.4f}")
    print(f"  Mean inference time: {df['infer_time_s'].mean():.2f}s")
    print(f"  Saved: {output_path}")
    
    return df, predictions_dict


def ensemble_predictions(all_preds: list):
    """Ensemble 3 folds using majority voting."""
    print(f"\n{'='*70}")
    print(f"Ensemble Prediction (Majority Voting)")
    print(f"{'='*70}")
    
    # Get common case IDs
    case_ids = set(all_preds[0].keys())
    for preds in all_preds[1:]:
        case_ids &= set(preds.keys())
    
    case_ids = sorted(list(case_ids))
    print(f"Ensembling {len(case_ids)} cases")
    
    # Load ground truth
    test_ds = get_test_dataset()
    gt_dict = {}
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
    
    for batch in test_loader:
        case_id = batch["case_id"][0]
        if case_id in case_ids:
            gt_dict[case_id] = batch["label"].squeeze().numpy().astype(np.int32)
    
    # Ensemble
    results = []
    for case_id in tqdm(case_ids, desc="Ensembling"):
        # Stack predictions from all folds
        preds = np.stack([all_preds[i][case_id] for i in range(len(all_preds))])
        
        # Majority voting
        ensemble_pred = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=4).argmax(),
            axis=0,
            arr=preds
        ).astype(np.int32)
        
        # Compute Dice
        gt = gt_dict[case_id]
        scores = compute_brats_dice(ensemble_pred, gt)
        scores["case_id"] = case_id
        results.append(scores)
    
    df = pd.DataFrame(results)
    output_path = OUT_DIR / "ensemble_results.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\nEnsemble Results:")
    print(f"  Mean WT Dice: {df['dice_wt'].mean():.4f} ± {df['dice_wt'].std():.4f}")
    print(f"  Mean TC Dice: {df['dice_tc'].mean():.4f} ± {df['dice_tc'].std():.4f}")
    print(f"  Mean ET Dice: {df['dice_et'].mean():.4f} ± {df['dice_et'].std():.4f}")
    print(f"  Saved: {output_path}")
    
    return df


def main():
    global EXP_DIR, CKPT_BASE, OUT_DIR

    parser = argparse.ArgumentParser(description="UNETR ablation test inference")
    parser.add_argument(
        "--tier",
        choices=VALID_TIERS,
        default="default",
        help="Augmentation tier (default: default)",
    )
    args = parser.parse_args()
    tier = args.tier

    EXP_DIR = REPO / "experiments" / "ablation" / f"abl_unetr_{tier}"
    CKPT_BASE = Path(f"/data/paul_daten/experiments/ablation/abl_unetr_{tier}")
    OUT_DIR = EXP_DIR / "test_predictions"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Run inference for each fold
    all_results = []
    all_predictions = []
    
    for fold in [0, 1, 2]:
        df, preds = infer_fold(fold, device)
        all_results.append(df)
        all_predictions.append(preds)
    
    # Ensemble
    ensemble_df = ensemble_predictions(all_predictions)
    
    # Summary
    summary = {
        "model": "unetr",
        "augmentation": tier,
        "test_cases": 125,
        "folds": [0, 1, 2],
        "per_fold_results": [],
        "ensemble_results": {
            "mean_wt_dice": float(ensemble_df["dice_wt"].mean()),
            "std_wt_dice": float(ensemble_df["dice_wt"].std()),
            "mean_tc_dice": float(ensemble_df["dice_tc"].mean()),
            "std_tc_dice": float(ensemble_df["dice_tc"].std()),
            "mean_et_dice": float(ensemble_df["dice_et"].mean()),
            "std_et_dice": float(ensemble_df["dice_et"].std()),
        }
    }
    
    for fold, df in enumerate(all_results):
        summary["per_fold_results"].append({
            "fold": fold,
            "mean_wt_dice": float(df["dice_wt"].mean()),
            "mean_tc_dice": float(df["dice_tc"].mean()),
            "mean_et_dice": float(df["dice_et"].mean()),
        })
    
    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"UNETR {tier.capitalize()} Augmentation - Test Set Summary")
    print(f"{'='*70}")
    print(json.dumps(summary, indent=2))
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
