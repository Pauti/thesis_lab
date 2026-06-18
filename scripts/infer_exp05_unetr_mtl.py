#!/usr/bin/env python3
"""
Test-set inference for EXP-05 (UNETR + MTL auxiliary ET-volume head).

Loads best_model.pth from each fold (MultiTaskUNETR, return_aux=False at inference),
runs sliding-window inference on 125 held-out test cases, computes BraTS Dice
(WT/TC/ET) per case, and produces per-fold + ensemble results.

Output:
  experiments/exp05_unetr_mtl/test_predictions/fold_X_results.csv
  experiments/exp05_unetr_mtl/test_predictions/ensemble_results.csv
  experiments/exp05_unetr_mtl/test_predictions/summary.json

Usage:
    python scripts/infer_exp05_unetr_mtl.py
"""

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

from models.multitask import get_multitask_unetr_model
from utils.hd95 import score_case, summarise

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
DATA_DIR = Path("/data/paul_daten/nnunet/nnUNet_raw/Dataset001_BraTS")
TEST_SET_JSON = Path("/data/paul_daten/nnunet/nnUNet_preprocessed/Dataset001_BraTS/test_set.json")
CONFIG_PATH = REPO / "configs" / "exp05_unetr_mtl.yaml"
EXP_DIR = REPO / "experiments" / "exp05_unetr_mtl"
CKPT_BASE = Path("/data/paul_daten/experiments/exp05_unetr_mtl")
OUT_DIR = EXP_DIR / "test_predictions"


# ── Transforms ─────────────────────────────────────────────────────────────────
def get_test_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=False, image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ])


def get_test_dataset():
    with open(TEST_SET_JSON) as f:
        test_set = json.load(f)
    case_ids = test_set["case_ids"]
    print(f"Loaded {len(case_ids)} test cases")
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


# ── Per-fold inference ──────────────────────────────────────────────────────────
def infer_fold(fold: int, cfg: dict, device: str):
    print(f"\n{'='*70}\nFold {fold} Inference\n{'='*70}")

    model_cfg = cfg["model"]
    sw_cfg = cfg["validation"]["sliding_window"]

    checkpoint_path = CKPT_BASE / f"fold_{fold}" / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    model = get_multitask_unetr_model(
        img_size=tuple(model_cfg["img_size"]),
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        dropout_rate=model_cfg["dropout_rate"],
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Model loaded (epoch {ckpt.get('epoch', 'N/A')}, "
          f"best val WT={ckpt.get('val_dice_wt', 'N/A'):.4f})")

    test_ds = get_test_dataset()
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    results = []
    hd_rows = []
    predictions_dict = {}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Fold {fold}"):
            case_id = batch["case_id"][0]
            imgs = batch["image"].to(device)
            lbls = batch["label"].to(device)

            t0 = time.time()
            # return_aux=False (default) → single tensor output, sliding_window works unchanged
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
            hd_rows.append({"case_id": case_id, **score_case(pred, gt)})
            predictions_dict[case_id] = pred

    df = pd.DataFrame(results)
    out_path = OUT_DIR / f"fold_{fold}_results.csv"
    df.to_csv(out_path, index=False)

    hd_df = pd.DataFrame(hd_rows)
    hd_df.to_csv(OUT_DIR / f"fold_{fold}_results_hd95.csv", index=False)

    hd = summarise(hd_df)
    print(f"\nFold {fold} Results:")
    print(f"  WT: {df['dice_wt'].mean():.4f} ± {df['dice_wt'].std():.4f}")
    print(f"  TC: {df['dice_tc'].mean():.4f} ± {df['dice_tc'].std():.4f}")
    print(f"  ET: {df['dice_et'].mean():.4f} ± {df['dice_et'].std():.4f}")
    print(f"  HD95 WT={hd['mean_hd95_wt']:.2f} TC={hd['mean_hd95_tc']:.2f} "
          f"ET={hd['mean_hd95_et']:.2f} mm")
    print(f"  Saved: {out_path}")

    return df, predictions_dict, hd_df


# ── Ensemble ───────────────────────────────────────────────────────────────────
def ensemble_predictions(all_preds: list):
    print(f"\n{'='*70}\nEnsemble (Majority Voting)\n{'='*70}")

    case_ids = set(all_preds[0].keys())
    for p in all_preds[1:]:
        case_ids &= set(p.keys())
    case_ids = sorted(case_ids)
    print(f"Ensembling {len(case_ids)} cases")

    test_ds = get_test_dataset()
    gt_dict = {}
    for batch in DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2):
        cid = batch["case_id"][0]
        if cid in case_ids:
            gt_dict[cid] = batch["label"].squeeze().numpy().astype(np.int32)

    results = []
    hd_rows = []
    for case_id in tqdm(case_ids, desc="Ensembling"):
        preds = np.stack([all_preds[i][case_id] for i in range(len(all_preds))])
        ensemble_pred = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=4).argmax(), axis=0, arr=preds
        ).astype(np.int32)
        scores = compute_brats_dice(ensemble_pred, gt_dict[case_id])
        scores["case_id"] = case_id
        results.append(scores)
        hd_rows.append({"case_id": case_id, **score_case(ensemble_pred, gt_dict[case_id])})

    df = pd.DataFrame(results)
    out_path = OUT_DIR / "ensemble_results.csv"
    df.to_csv(out_path, index=False)

    hd_df = pd.DataFrame(hd_rows)
    hd_df.to_csv(OUT_DIR / "ensemble_results_hd95.csv", index=False)

    hd = summarise(hd_df)
    print(f"\nEnsemble Results:")
    print(f"  WT: {df['dice_wt'].mean():.4f} ± {df['dice_wt'].std():.4f}")
    print(f"  TC: {df['dice_tc'].mean():.4f} ± {df['dice_tc'].std():.4f}")
    print(f"  ET: {df['dice_et'].mean():.4f} ± {df['dice_et'].std():.4f}")
    print(f"  HD95 WT={hd['mean_hd95_wt']:.2f} TC={hd['mean_hd95_tc']:.2f} "
          f"ET={hd['mean_hd95_et']:.2f} mm")
    print(f"  Saved: {out_path}")

    return df, hd_df


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checkpoints: {CKPT_BASE}")
    print(f"Output:      {OUT_DIR}")

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    all_results = []
    all_predictions = []
    all_hd = []
    for fold in [0, 1, 2]:
        df, preds, hd_df = infer_fold(fold, cfg, device)
        all_results.append(df)
        all_predictions.append(preds)
        all_hd.append(hd_df)

    ensemble_df, ensemble_hd_df = ensemble_predictions(all_predictions)

    # ── HD95 summary (per-fold mean over folds + ensemble) ──────────────────────
    fold_hd = [summarise(h) for h in all_hd]
    ens_hd = summarise(ensemble_hd_df)
    hd95_summary = {
        "experiment": "exp05_unetr_mtl",
        "test_cases": 125,
        "hd95_convention": "NaN (excluded) when region absent in pred or GT; mm == voxel (1mm isotropic)",
        "per_fold": [
            {"fold": f, **{k: v for k, v in m.items() if "hd95" in k}}
            for f, m in enumerate(fold_hd)
        ],
        "per_fold_hd95_mean_over_folds": {
            f"hd95_{r}": {
                "mean": float(np.mean([m[f"mean_hd95_{r}"] for m in fold_hd])),
                "std": float(np.std([m[f"mean_hd95_{r}"] for m in fold_hd])),
            }
            for r in ("wt", "tc", "et")
        },
        "ensemble": ens_hd,
    }
    with open(OUT_DIR / "summary_hd95.json", "w") as f:
        json.dump(hd95_summary, f, indent=2)
    print(f"HD95 summary saved: {OUT_DIR / 'summary_hd95.json'}")

    summary = {
        "experiment": "exp05_unetr_mtl",
        "model": "MultiTaskUNETR (ViT-B/16 + CORAL aux head)",
        "aux_task": "ET-volume ordinal classification (3 tertiles, lambda=0.3)",
        "baseline_comparison": "abl_unetr_default (ABL-07)",
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
    print(f"EXP-05 UNETR MTL — Test Set Summary")
    print(f"{'='*70}")
    print(json.dumps(summary["ensemble_results"], indent=2))
    print(f"\nFull summary: {summary_path}")


if __name__ == "__main__":
    main()
