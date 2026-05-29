#!/usr/bin/env python3
"""
Inference for nnU-Net Default Augmentation on BraTS 2021 held-out test set (125 cases).

Uses nnU-Net's built-in predictor for each fold, then ensembles predictions.

Output:
  experiments/ablation/abl_nnunet_default/test_predictions/fold_X_results.csv
  experiments/ablation/abl_nnunet_default/test_predictions/ensemble_results.csv
  experiments/ablation/abl_nnunet_default/test_predictions/summary.json

Usage:
    python scripts/infer_ablation_nnunet.py
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# nnU-Net imports
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# ========== Paths ==========
REPO = Path(__file__).resolve().parent.parent
DATASET_ID = "001"
CONFIG = "3d_fullres"
NNUNET_RESULTS = Path("/data/paul_daten/nnunet/nnUNet_results")
DATA_DIR = Path("/data/paul_daten/nnunet/nnUNet_raw/Dataset001_BraTS")
TEST_SET_JSON = Path("/data/paul_daten/nnunet/nnUNet_preprocessed/Dataset001_BraTS/test_set.json")
TRAINERS = {
    "disabled": "nnUNetTrainer_NoAug",
    "reduced":  "nnUNetTrainer_LightAug",
    "default":  "nnUNetTrainer_DefaultAug",
    "extended": "nnUNetTrainer_HeavyAug",
}

# Tier-dependent globals — set in main() based on --tier
EXP_DIR = MODEL_DIR = OUT_DIR = TRAINER = None

# Set nnU-Net environment variables
os.environ["nnUNet_raw"] = "/data/paul_daten/nnunet/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "/data/paul_daten/nnunet/nnUNet_preprocessed"
os.environ["nnUNet_results"] = "/data/paul_daten/nnunet/nnUNet_results"


# ========== Metrics ==========
def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Volumetric Dice coefficient."""
    intersection = int(np.sum(pred_mask & gt_mask))
    total = int(np.sum(pred_mask)) + int(np.sum(gt_mask))
    if total == 0:
        return 1.0
    return 2.0 * intersection / total


def compute_brats_dice(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    BraTS region Dice. Labels: 0=bg, 1=NCR, 2=ED, 3=ET.
    
    Note: nnU-Net outputs class labels (0,1,2,3), same as our remapping.
    """
    return {
        "dice_wt": dice_score(pred >= 1, gt >= 1),
        "dice_tc": dice_score(np.isin(pred, [1, 3]), np.isin(gt, [1, 3])),
        "dice_et": dice_score(pred == 3, gt == 3),
    }


def load_nifti(path: Path) -> np.ndarray:
    """Load NIfTI file."""
    import nibabel as nib
    return nib.load(str(path)).get_fdata().astype(np.int32)


# ========== Inference ==========
def infer_fold(fold: int):
    """Run inference for one fold on test set using nnUNetPredictor."""
    print(f"\n{'='*70}")
    print(f"Fold {fold} Inference (nnU-Net)")
    print(f"{'='*70}")
    
    # Check checkpoint
    checkpoint_path = MODEL_DIR / f"fold_{fold}" / "checkpoint_best.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Checkpoint: {checkpoint_path}")
    
    # Load test set
    with open(TEST_SET_JSON) as f:
        test_set = json.load(f)
    
    case_ids = test_set["case_ids"]
    print(f"Test set: {len(case_ids)} cases")
    
    # Create temporary directory for predictions
    pred_dir = OUT_DIR / f"fold_{fold}_preds"
    pred_dir.mkdir(exist_ok=True)
    
    # Initialize predictor
    print("Initializing nnUNetPredictor...")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,  # Disable for speed (can enable for better results)
        perform_everything_on_device=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    
    # Initialize from trained model folder (parent directory contains dataset.json)
    predictor.initialize_from_trained_model_folder(
        str(MODEL_DIR),
        use_folds=(fold,),
        checkpoint_name="checkpoint_best.pth",
    )
    
    print("Starting inference...")
    
    results = []
    predictions_dict = {}
    
    for case_id in tqdm(case_ids, desc=f"Fold {fold}"):
        # Input files (nnU-Net format)
        input_files = [
            [
                str(DATA_DIR / "imagesTr" / f"{case_id}_0000.nii.gz"),
                str(DATA_DIR / "imagesTr" / f"{case_id}_0001.nii.gz"),
                str(DATA_DIR / "imagesTr" / f"{case_id}_0002.nii.gz"),
                str(DATA_DIR / "imagesTr" / f"{case_id}_0003.nii.gz"),
            ]
        ]
        
        # Output file
        output_file = pred_dir / f"{case_id}.nii.gz"
        
        # Predict
        t0 = time.time()
        predictor.predict_from_files(
            list_of_lists_or_source_folder=input_files,
            output_folder_or_list_of_truncated_output_files=[str(output_file)],
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0,
        )
        elapsed = time.time() - t0
        
        # Load prediction and ground truth
        pred = load_nifti(output_file)
        gt_path = DATA_DIR / "labelsTr" / f"{case_id}.nii.gz"
        gt = load_nifti(gt_path)
        
        # Compute Dice
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
        
        # Load ground truth
        gt_path = DATA_DIR / "labelsTr" / f"{case_id}.nii.gz"
        gt = load_nifti(gt_path)
        
        # Compute Dice
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
    global EXP_DIR, TRAINER, MODEL_DIR, OUT_DIR

    parser = argparse.ArgumentParser(description="nnU-Net ablation test inference")
    parser.add_argument(
        "--tier",
        choices=list(TRAINERS),
        default="default",
        help="Augmentation tier (default: default)",
    )
    args = parser.parse_args()
    tier = args.tier

    TRAINER = TRAINERS[tier]
    EXP_DIR = REPO / "experiments" / "ablation" / f"abl_nnunet_{tier}"
    MODEL_DIR = NNUNET_RESULTS / f"Dataset{DATASET_ID}_BraTS" / f"{TRAINER}__nnUNetPlans__{CONFIG}"
    OUT_DIR = EXP_DIR / "test_predictions"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"nnU-Net Test Set Inference: {tier.capitalize()} Augmentation")
    print(f"Model: {MODEL_DIR}")
    
    # Run inference for each fold
    all_results = []
    all_predictions = []
    
    for fold in [0, 1, 2]:
        df, preds = infer_fold(fold)
        all_results.append(df)
        all_predictions.append(preds)
    
    # Ensemble
    ensemble_df = ensemble_predictions(all_predictions)
    
    # Summary
    summary = {
        "model": "nnunet",
        "augmentation": tier,
        "trainer": TRAINER,
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
    print(f"nnU-Net {tier.capitalize()} Augmentation - Test Set Summary")
    print(f"{'='*70}")
    print(json.dumps(summary, indent=2))
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
