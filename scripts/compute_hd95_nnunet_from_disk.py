#!/usr/bin/env python3
"""
Retroactively compute HD95 for the nnU-Net experiments *without* GPU re-inference.

All nnU-Net test-set inference runs persisted their per-fold predictions as NIfTI
(test_predictions/fold_X_preds/<case>.nii.gz). HD95 can therefore be computed
directly from disk, and the 3-fold ensemble can be reconstructed by the same
majority vote used at inference time. No model forward passes required.

Covers (per-fold + reconstructed ensemble):
  - exp04_nnunet_mtl
  - ablation/abl_nnunet_{disabled,reduced,default,extended}

For each fold and the ensemble we (re)compute per-case Dice (WT/TC/ET) as a sanity
check — these must reproduce the existing *_results.csv — and per-case HD95 (95th
percentile Hausdorff, mm == voxels for 1 mm isotropic BraTS).

HD95 convention (matches utils/metrics.py): undefined when a region is absent in
either prediction or ground truth; such cases are recorded as NaN and excluded
from the mean. The number of contributing cases is reported per region.

Output (per experiment, alongside the existing Dice CSVs):
  <exp>/test_predictions/fold_X_results_hd95.csv     (per-case dice + hd95)
  <exp>/test_predictions/ensemble_results_hd95.csv
  <exp>/test_predictions/summary_hd95.json

Usage:
    python scripts/compute_hd95_nnunet_from_disk.py            # all experiments
    python scripts/compute_hd95_nnunet_from_disk.py --exp exp04_nnunet_mtl
"""

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.hd95 import REGIONS, score_case, summarise

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
DATA_DIR = Path("/data/paul_daten/nnunet/nnUNet_raw/Dataset001_BraTS")
GT_DIR = DATA_DIR / "labelsTr"

# experiment key → test_predictions dir
EXPERIMENTS = {
    "exp04_nnunet_mtl": REPO / "experiments/exp04_nnunet_mtl/test_predictions",
    "exp06_nnunet_mtl_global": REPO / "experiments/exp06_nnunet_mtl_global/test_predictions",
    "exp08_nnunet_mtl_v3": REPO / "experiments/exp08_nnunet_mtl_v3/test_predictions",
    "abl_nnunet_disabled": REPO / "experiments/ablation/abl_nnunet_disabled/test_predictions",
    "abl_nnunet_reduced": REPO / "experiments/ablation/abl_nnunet_reduced/test_predictions",
    "abl_nnunet_default": REPO / "experiments/ablation/abl_nnunet_default/test_predictions",
    "abl_nnunet_extended": REPO / "experiments/ablation/abl_nnunet_extended/test_predictions",
}
FOLDS = [0, 1, 2]


def load(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata().astype(np.int32)


def process_experiment(key: str, pred_root: Path) -> dict:
    print(f"\n{'='*74}\n  {key}\n{'='*74}")

    # case list from fold_0 predictions
    fold0_dir = pred_root / "fold_0_preds"
    case_ids = sorted(p.name[:-7] for p in fold0_dir.glob("*.nii.gz"))
    if not case_ids:
        raise FileNotFoundError(f"No predictions in {fold0_dir}")
    print(f"  {len(case_ids)} cases")

    fold_dfs = []
    fold_means = []

    for fold in FOLDS:
        pred_dir = pred_root / f"fold_{fold}_preds"
        rows = []
        for cid in tqdm(case_ids, desc=f"  fold {fold}", leave=False):
            pred = load(pred_dir / f"{cid}.nii.gz")
            gt = load(GT_DIR / f"{cid}.nii.gz")
            row = {"case_id": cid, **score_case(pred, gt)}
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(pred_root / f"fold_{fold}_results_hd95.csv", index=False)
        fold_dfs.append(df)
        m = summarise(df)
        fold_means.append(m)
        print(f"  fold {fold}: "
              f"HD95 WT={m['mean_hd95_wt']:.2f} TC={m['mean_hd95_tc']:.2f} "
              f"ET={m['mean_hd95_et']:.2f} | Dice ET={m['mean_dice_et']:.4f}")

    # ── Ensemble via per-voxel majority vote (identical to inference) ───────────
    # Load predictions and GT lazily per case — no in-memory cache needed.
    ens_rows = []
    for cid in tqdm(case_ids, desc="  ensemble", leave=False):
        preds = [load(pred_root / f"fold_{f}_preds" / f"{cid}.nii.gz") for f in FOLDS]
        gt = load(GT_DIR / f"{cid}.nii.gz")
        stack = np.stack(preds)  # (n_folds, H, W, D)
        # Vectorised majority vote: count votes per label, take argmax.
        counts = np.zeros((4, *stack.shape[1:]), dtype=np.int8)
        for label in range(4):
            counts[label] = np.sum(stack == label, axis=0)
        ens = counts.argmax(axis=0).astype(np.int32)
        ens_rows.append({"case_id": cid, **score_case(ens, gt)})
    ens_df = pd.DataFrame(ens_rows)
    ens_df.to_csv(pred_root / "ensemble_results_hd95.csv", index=False)
    ens = summarise(ens_df)
    print(f"  ENSEMBLE: HD95 WT={ens['mean_hd95_wt']:.2f} "
          f"TC={ens['mean_hd95_tc']:.2f} ET={ens['mean_hd95_et']:.2f}")

    # ── Sanity: reproduced ensemble Dice must match existing summary.json ───────
    sanity = {}
    existing = pred_root / "summary.json"
    if existing.exists():
        prev = json.loads(existing.read_text())
        prev_ens = prev.get("ensemble_results", {})
        for region in REGIONS:
            r = region.lower()
            old = prev_ens.get(f"mean_{r}_dice")
            if old is not None:
                diff = abs(old - ens[f"mean_dice_{r}"])
                sanity[f"dice_{r}_diff"] = diff
                flag = "OK" if diff < 1e-3 else "MISMATCH"
                print(f"  sanity Dice {region}: stored={old:.4f} "
                      f"recomputed={ens[f'mean_dice_{r}']:.4f}  Δ={diff:.2e}  [{flag}]")

    result = {
        "experiment": key,
        "n_cases": len(case_ids),
        "hd95_convention": "NaN (excluded) when region absent in pred or GT; mm == voxel (1mm isotropic)",
        "per_fold": [
            {"fold": f, **{k: v for k, v in m.items() if "hd95" in k}}
            for f, m in zip(FOLDS, fold_means)
        ],
        "per_fold_hd95_mean_over_folds": {
            f"hd95_{r.lower()}": {
                "mean": float(np.mean([m[f"mean_hd95_{r.lower()}"] for m in fold_means])),
                "std": float(np.std([m[f"mean_hd95_{r.lower()}"] for m in fold_means])),
            }
            for r in REGIONS
        },
        "ensemble": ens,
        "sanity_dice_vs_stored": sanity,
    }
    with open(pred_root / "summary_hd95.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser(description="HD95 from saved nnU-Net predictions (no GPU)")
    ap.add_argument("--exp", choices=list(EXPERIMENTS), help="single experiment (default: all)")
    args = ap.parse_args()

    keys = [args.exp] if args.exp else list(EXPERIMENTS)
    all_results = {}
    for key in keys:
        all_results[key] = process_experiment(key, EXPERIMENTS[key])

    print(f"\n{'='*74}\n  HD95 summary (ensemble, mm)\n{'='*74}")
    print(f"  {'experiment':<24}{'WT':>8}{'TC':>8}{'ET':>8}")
    for key, res in all_results.items():
        e = res["ensemble"]
        print(f"  {key:<24}{e['mean_hd95_wt']:>8.2f}"
              f"{e['mean_hd95_tc']:>8.2f}{e['mean_hd95_et']:>8.2f}")


if __name__ == "__main__":
    main()
