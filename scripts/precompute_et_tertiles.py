#!/usr/bin/env python3
"""
Pre-compute ET-volume tertile labels for MTL v2 (Raw-Images variant).

For MTL v2, the auxiliary ordinal head should learn ET-volume classification
from raw images only — not from the segmentation output (which is circular).
This script pre-computes the tertile labels from ground truth ET volumes,
which can then be used during training without exposing the segmentation itself
to the aux head.

Reads:
  experiments/et_volume_distribution.json   (ET voxel counts per case)
  splits_ablation.json                      (training set per fold)

Writes:
  experiments/et_tertiles_ablation.json     (per-fold tertile labels)

Usage:
    python scripts/precompute_et_tertiles.py
"""

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
EXP_DIR = REPO / "experiments"
ET_VOL_FILE = EXP_DIR / "et_volume_distribution.json"
ABLATION_SPLITS = Path("/data/paul_daten/nnunet/nnUNet_preprocessed/Dataset001_BraTS/splits_ablation.json")


def main():
    # Load ET volumes
    with open(ET_VOL_FILE) as f:
        et_volumes = json.load(f)

    # Load ablation splits
    with open(ABLATION_SPLITS) as f:
        splits = json.load(f)

    result = {}

    for fold_idx, split in enumerate(splits):
        print(f"\nFold {fold_idx}:")
        train_cases = split["train"]

        # Get ET volumes for training cases only
        et_vols = np.array([et_volumes[cid]["et"] for cid in train_cases])

        # Tertile thresholds: split at 33.3% and 66.7% quantiles
        t1, t2 = np.percentile(et_vols, [33.33, 66.67])
        print(f"  {len(train_cases)} training cases")
        print(f"  ET tertile thresholds: t1={t1:.0f}, t2={t2:.0f}")

        # Assign tertile labels to ALL cases (test set included)
        # using fold-specific thresholds
        fold_labels = {}
        for cid in et_volumes:
            et = et_volumes[cid]["et"]
            if et < t1:
                label = 0  # low
            elif et < t2:
                label = 1  # mid
            else:
                label = 2  # high
            fold_labels[cid] = label

        result[fold_idx] = {
            "thresholds": {"t1": float(t1), "t2": float(t2)},
            "labels": fold_labels,
        }

    # Write output
    out_path = EXP_DIR / "et_tertiles_ablation.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
