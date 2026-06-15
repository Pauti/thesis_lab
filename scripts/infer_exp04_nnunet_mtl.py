#!/usr/bin/env python3
"""
Test-set inference for EXP-04 (nnU-Net + MTL auxiliary ET-volume head).

Uses nnUNetPredictor with nnUNetTrainer_MTL checkpoints for each fold,
then ensembles predictions via majority voting.

Output:
  experiments/exp04_nnunet_mtl/test_predictions/fold_X_results.csv
  experiments/exp04_nnunet_mtl/test_predictions/ensemble_results.csv
  experiments/exp04_nnunet_mtl/test_predictions/summary.json

Usage:
    python scripts/infer_exp04_nnunet_mtl.py
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
DATASET_ID = "001"
CONFIG = "3d_fullres"
TRAINER = "nnUNetTrainer_MTL"

NNUNET_RESULTS = Path("/data/paul_daten/nnunet/nnUNet_results")
DATA_DIR = Path("/data/paul_daten/nnunet/nnUNet_raw/Dataset001_BraTS")
TEST_SET_JSON = Path("/data/paul_daten/nnunet/nnUNet_preprocessed/Dataset001_BraTS/test_set.json")

MODEL_DIR = NNUNET_RESULTS / f"Dataset{DATASET_ID}_BraTS" / f"{TRAINER}__nnUNetPlans__{CONFIG}"
EXP_DIR = REPO / "experiments" / "exp04_nnunet_mtl"
OUT_DIR = EXP_DIR / "test_predictions"

os.environ["nnUNet_raw"] = "/data/paul_daten/nnunet/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "/data/paul_daten/nnunet/nnUNet_preprocessed"
os.environ["nnUNet_results"] = "/data/paul_daten/nnunet/nnUNet_results"


# ── Metrics ────────────────────────────────────────────────────────────────────
def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    intersection = int(np.sum(pred_mask & gt_mask))
    total = int(np.sum(pred_mask)) + int(np.sum(gt_mask))
    if total == 0:
        return 1.0
    return 2.0 * intersection / total


def compute_brats_dice(pred: np.ndarray, gt: np.ndarray) -> dict:
    return {
        "dice_wt": dice_score(pred >= 1, gt >= 1),
        "dice_tc": dice_score(np.isin(pred, [1, 3]), np.isin(gt, [1, 3])),
        "dice_et": dice_score(pred == 3, gt == 3),
    }


def load_nifti(path: Path) -> np.ndarray:
    import nibabel as nib
    return nib.load(str(path)).get_fdata().astype(np.int32)


# ── Inference ──────────────────────────────────────────────────────────────────
def infer_fold(fold: int):
    print(f"\n{'='*70}")
    print(f"Fold {fold} Inference (nnU-Net MTL)")
    print(f"{'='*70}")

    checkpoint_path = MODEL_DIR / f"fold_{fold}" / "checkpoint_best.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"Checkpoint: {checkpoint_path}")

    with open(TEST_SET_JSON) as f:
        test_set = json.load(f)
    case_ids = test_set["case_ids"]
    print(f"Test set: {len(case_ids)} cases")

    pred_dir = OUT_DIR / f"fold_{fold}_preds"
    pred_dir.mkdir(exist_ok=True)

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(MODEL_DIR),
        use_folds=(fold,),
        checkpoint_name="checkpoint_best.pth",
    )

    results = []
    predictions_dict = {}

    for case_id in tqdm(case_ids, desc=f"Fold {fold}"):
        input_files = [[
            str(DATA_DIR / "imagesTr" / f"{case_id}_0000.nii.gz"),
            str(DATA_DIR / "imagesTr" / f"{case_id}_0001.nii.gz"),
            str(DATA_DIR / "imagesTr" / f"{case_id}_0002.nii.gz"),
            str(DATA_DIR / "imagesTr" / f"{case_id}_0003.nii.gz"),
        ]]
        output_file = pred_dir / f"{case_id}.nii.gz"

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

        pred = load_nifti(output_file)
        gt = load_nifti(DATA_DIR / "labelsTr" / f"{case_id}.nii.gz")

        scores = compute_brats_dice(pred, gt)
        scores["case_id"] = case_id
        scores["infer_time_s"] = round(elapsed, 2)
        results.append(scores)
        predictions_dict[case_id] = pred

    df = pd.DataFrame(results)
    output_path = OUT_DIR / f"fold_{fold}_results.csv"
    df.to_csv(output_path, index=False)

    print(f"\nFold {fold} Results:")
    print(f"  WT: {df['dice_wt'].mean():.4f} ± {df['dice_wt'].std():.4f}")
    print(f"  TC: {df['dice_tc'].mean():.4f} ± {df['dice_tc'].std():.4f}")
    print(f"  ET: {df['dice_et'].mean():.4f} ± {df['dice_et'].std():.4f}")
    print(f"  Saved: {output_path}")

    return df, predictions_dict


def ensemble_predictions(all_preds: list):
    print(f"\n{'='*70}")
    print(f"Ensemble (Majority Voting)")
    print(f"{'='*70}")

    case_ids = sorted(set.intersection(*[set(p.keys()) for p in all_preds]))
    print(f"Ensembling {len(case_ids)} cases")

    results = []
    for case_id in tqdm(case_ids, desc="Ensembling"):
        preds = np.stack([p[case_id] for p in all_preds])
        ensemble_pred = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=4).argmax(),
            axis=0,
            arr=preds,
        ).astype(np.int32)

        gt = load_nifti(DATA_DIR / "labelsTr" / f"{case_id}.nii.gz")
        scores = compute_brats_dice(ensemble_pred, gt)
        scores["case_id"] = case_id
        results.append(scores)

    df = pd.DataFrame(results)
    output_path = OUT_DIR / "ensemble_results.csv"
    df.to_csv(output_path, index=False)

    print(f"\nEnsemble Results:")
    print(f"  WT: {df['dice_wt'].mean():.4f} ± {df['dice_wt'].std():.4f}")
    print(f"  TC: {df['dice_tc'].mean():.4f} ± {df['dice_tc'].std():.4f}")
    print(f"  ET: {df['dice_et'].mean():.4f} ± {df['dice_et'].std():.4f}")
    print(f"  Saved: {output_path}")

    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"EXP-04 nnU-Net MTL — Test Set Inference")
    print(f"Trainer:  {TRAINER}")
    print(f"Model:    {MODEL_DIR}")
    print(f"Output:   {OUT_DIR}")

    all_results = []
    all_predictions = []

    for fold in [0, 1, 2]:
        df, preds = infer_fold(fold)
        all_results.append(df)
        all_predictions.append(preds)

    ensemble_df = ensemble_predictions(all_predictions)

    summary = {
        "experiment_id": "exp04_nnunet_mtl",
        "model": "nnunet",
        "trainer": TRAINER,
        "aux_task": "ET-volume ordinal (CORAL)",
        "test_cases": 125,
        "folds": [0, 1, 2],
        "per_fold_results": [
            {
                "fold": i,
                "mean_wt_dice": float(df["dice_wt"].mean()),
                "mean_tc_dice": float(df["dice_tc"].mean()),
                "mean_et_dice": float(df["dice_et"].mean()),
            }
            for i, df in enumerate(all_results)
        ],
        "ensemble_results": {
            "mean_wt_dice": float(ensemble_df["dice_wt"].mean()),
            "std_wt_dice":  float(ensemble_df["dice_wt"].std()),
            "mean_tc_dice": float(ensemble_df["dice_tc"].mean()),
            "std_tc_dice":  float(ensemble_df["dice_tc"].std()),
            "mean_et_dice": float(ensemble_df["dice_et"].mean()),
            "std_et_dice":  float(ensemble_df["dice_et"].std()),
        },
    }

    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"EXP-04 NNUNET MTL — Test Set Summary")
    print(f"{'='*70}")
    print(json.dumps(summary, indent=2))
    print(f"\nFull summary: {summary_path}")


if __name__ == "__main__":
    main()
