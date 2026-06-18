#!/usr/bin/env python3
"""
RQ1 HD95 for EXP-02 (nnU-Net 5-fold CV) from disk — no GPU.

nnU-Net writes each fold's validation predictions to disk at the end of training
(fold_X/validation/<case>.nii.gz). RQ1 uses the 5-fold CV protocol: every case is
predicted exactly once, by the fold for which it is a validation case (no held-out
test set, no ensemble — unlike RQ2/RQ3). HD95 + Dice can therefore be computed
directly from those persisted predictions.

This is the CV-protocol counterpart to compute_hd95_nnunet_from_disk.py (which
handles the held-out test set with 3-fold majority-vote ensembling). Same HD95
convention via utils/hd95: NaN when a region is absent in prediction or GT,
excluded from the mean; mm == voxel for 1 mm isotropic BraTS.

For the UNETR RQ1 counterpart (EXP-03) the predictions were not kept, so it needs
GPU re-inference (scripts/infer_exp03.py, extended for HD95).

Output:
  experiments/exp02_nnunet/analysis/all_folds_hd95.csv   (per case: fold, dice, hd95)
  experiments/exp02_nnunet/analysis/summary_hd95.json

Usage:
    python scripts/compute_hd95_exp02_cv_from_disk.py
"""

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.hd95 import REGIONS, score_case, summarise

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = Path("/data/paul_daten/nnunet/nnUNet_raw/Dataset001_BraTS")
GT_DIR = DATA_DIR / "labelsTr"
VAL_BASE = Path(
    "/data/paul_daten/nnunet/nnUNet_results/Dataset001_BraTS"
    "/nnUNetTrainer__nnUNetPlans__3d_fullres"
)
OUT_DIR = REPO / "experiments/exp02_nnunet/analysis"
FOLDS = [0, 1, 2, 3, 4]


def load(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata().astype(np.int32)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in FOLDS:
        val_dir = VAL_BASE / f"fold_{fold}" / "validation"
        cases = sorted(p.name[:-7] for p in val_dir.glob("*.nii.gz"))
        if not cases:
            raise FileNotFoundError(f"No validation predictions in {val_dir}")
        for cid in tqdm(cases, desc=f"fold {fold}", leave=False):
            pred = load(val_dir / f"{cid}.nii.gz")
            gt = load(GT_DIR / f"{cid}.nii.gz")
            rows.append({"case_id": cid, "fold": fold, **score_case(pred, gt)})
        print(f"  fold {fold}: {len(cases)} cases")

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "all_folds_hd95.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        "experiment": "exp02_nnunet",
        "eval_protocol": "5fold_cv",
        "n_cases": len(df),
        "hd95_convention": "NaN (excluded) when region absent in pred or GT; mm == voxel (1mm isotropic)",
        "overall": summarise(df),
    }
    with open(OUT_DIR / "summary_hd95.json", "w") as f:
        json.dump(summary, f, indent=2)

    s = summary["overall"]
    print(f"\n{'='*60}\n  EXP-02 RQ1 CV HD95 ({len(df)} cases)\n{'='*60}")
    for r in REGIONS:
        rl = r.lower()
        print(f"  {r}: Dice={s[f'mean_dice_{rl}']:.4f}  "
              f"HD95={s[f'mean_hd95_{rl}']:.2f} mm  (n={s[f'n_hd95_{rl}']})")
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {OUT_DIR / 'summary_hd95.json'}")


if __name__ == "__main__":
    main()
