#!/usr/bin/env python3
"""
Generate held-out test set (10%) and 3-fold CV splits for augmentation ablation (RQ2).

This script:
  1. Loads all 1251 BraTS 2021 case IDs from the nnU-Net raw dataset.
  2. Randomly holds out 10% (125 cases) as a fixed test set.
  3. Splits the remaining 1126 cases into 3 stratified folds for cross-validation.
  4. Saves:
     - splits_ablation.json  (nnU-Net format: list of {"train": [...], "val": [...]})
     - test_set.json         (list of held-out case IDs)

The random seed is fixed (42) for full reproducibility across all experiments.

Usage:
    python scripts/generate_splits.py \
        --data_dir /data/paul_daten/nnunet/nnUNet_raw/Dataset001_BraTS \
        --output_dir /data/paul_daten/nnunet/nnUNet_preprocessed/Dataset001_BraTS

Outputs:
    {output_dir}/splits_ablation.json   — 3-fold CV splits (1126 cases)
    {output_dir}/test_set.json          — 125 held-out test cases
"""

import argparse
import json
from pathlib import Path

import numpy as np


SEED = 42
TEST_FRACTION = 0.1  # 10% → 125 cases from 1251
NUM_FOLDS = 3


def get_case_ids(data_dir: Path) -> list[str]:
    """Extract unique case IDs from labelsTr directory."""
    labels_dir = data_dir / "labelsTr"
    case_ids = sorted(
        p.stem.replace(".nii", "")  # handle .nii.gz → stem gives .nii
        for p in labels_dir.glob("*.nii.gz")
    )
    return case_ids


def generate_splits(case_ids: list[str], seed: int, test_fraction: float, num_folds: int):
    """
    Generate test set and k-fold CV splits.

    Returns:
        test_ids: list of held-out case IDs
        folds: list of dicts {"train": [...], "val": [...]} for each fold
    """
    rng = np.random.default_rng(seed)

    # Shuffle deterministically
    ids = np.array(case_ids)
    rng.shuffle(ids)

    # Hold out test set
    n_test = int(len(ids) * test_fraction)
    test_ids = ids[:n_test].tolist()
    train_pool = ids[n_test:].tolist()

    # 3-fold CV on remaining cases
    n = len(train_pool)
    indices = np.arange(n)
    rng.shuffle(indices)

    fold_sizes = [n // num_folds] * num_folds
    for i in range(n % num_folds):
        fold_sizes[i] += 1

    folds = []
    current = 0
    for fold_idx in range(num_folds):
        val_indices = indices[current:current + fold_sizes[fold_idx]]
        train_indices = np.concatenate([indices[:current], indices[current + fold_sizes[fold_idx]:]])

        folds.append({
            "train": sorted([train_pool[i] for i in train_indices]),
            "val": sorted([train_pool[i] for i in val_indices]),
        })
        current += fold_sizes[fold_idx]

    return sorted(test_ids), folds


def main():
    parser = argparse.ArgumentParser(description="Generate test set + 3-fold CV splits for ablation")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/data/paul_daten/nnunet/nnUNet_raw/Dataset001_BraTS"),
        help="Path to nnU-Net raw dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/data/paul_daten/nnunet/nnUNet_preprocessed/Dataset001_BraTS"),
        help="Output directory for split files",
    )
    args = parser.parse_args()

    # Get all case IDs
    case_ids = get_case_ids(args.data_dir)
    print(f"Found {len(case_ids)} cases in {args.data_dir / 'labelsTr'}")
    assert len(case_ids) == 1251, f"Expected 1251 cases, got {len(case_ids)}"

    # Generate splits
    test_ids, folds = generate_splits(case_ids, SEED, TEST_FRACTION, NUM_FOLDS)

    print(f"\nTest set: {len(test_ids)} cases (held out)")
    print(f"Training pool: {sum(len(f['train']) for f in folds) // NUM_FOLDS + len(folds[0]['val'])} cases")
    for i, fold in enumerate(folds):
        print(f"  Fold {i}: train={len(fold['train'])}, val={len(fold['val'])}")

    # Verify no overlap
    test_set = set(test_ids)
    all_cases = set(case_ids)
    for i, fold in enumerate(folds):
        train_set = set(fold["train"])
        val_set = set(fold["val"])
        assert train_set & val_set == set(), f"Fold {i}: train/val overlap!"
        assert train_set & test_set == set(), f"Fold {i}: train/test overlap!"
        assert val_set & test_set == set(), f"Fold {i}: val/test overlap!"
        assert train_set | val_set | test_set == all_cases, f"Fold {i}: cases missing!"
    print("\n✓ No overlaps between train/val/test across all folds")

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits_path = args.output_dir / "splits_ablation.json"
    with open(splits_path, "w") as f:
        json.dump(folds, f, indent=2)
    print(f"\nSaved: {splits_path}")

    test_path = args.output_dir / "test_set.json"
    with open(test_path, "w") as f:
        json.dump({"seed": SEED, "n_test": len(test_ids), "case_ids": test_ids}, f, indent=2)
    print(f"Saved: {test_path}")


if __name__ == "__main__":
    main()
