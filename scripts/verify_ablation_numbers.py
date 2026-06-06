#!/usr/bin/env python3
"""
Reproducible verification of all numerical claims made in the thesis paragraph
'Early Stopping and Convergence Criteria' (ablation study, 100-epoch cap).

Run with:
    venv/bin/python3 scripts/verify_ablation_numbers.py

All values are derived directly from the raw CSV files.
Exit code 0 = all assertions pass.
"""

import csv
import math
import os
import statistics
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NNUNET_DIR = os.path.join(BASE, "experiments", "exp02_nnunet")
UNETR_DIR  = os.path.join(BASE, "experiments", "exp03_unetr")
FOLDS      = 5


# ---------------------------------------------------------------------------
# Load one column value at a specific epoch from a CSV file.
# ---------------------------------------------------------------------------
def read_epoch(path, epoch, column):
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if int(row["epoch"]) == epoch:
                return float(row[column])
    raise KeyError(f"epoch {epoch} not found in {path}")


def read_last_epoch(path):
    last = None
    with open(path) as fh:
        for row in csv.DictReader(fh):
            last = int(row["epoch"])
    return last


# ---------------------------------------------------------------------------
# Collect per-fold values for key epochs.
# ---------------------------------------------------------------------------
def fold_values(base_dir, fold_count, epoch, column):
    return [
        read_epoch(os.path.join(base_dir, f"fold_{f}", "metrics.csv"), epoch, column)
        for f in range(fold_count)
    ]


# ============================================================
# 1. Training-budget arithmetic at epoch 100
# ============================================================
NNUNET_ITERS_PER_EPOCH  = 250          # fixed random-patch iterations
NNUNET_BATCH            = 2
UNETR_TRAIN_CASES       = 1001         # cases per fold (fold 0: 1000, folds 1-4: 1001; use 1001 as ceiling)
UNETR_BATCH             = 2
UNETR_GRAD_ACCUM        = 4
EPOCH_CAP               = 100

unetr_iters_per_epoch   = math.ceil(UNETR_TRAIN_CASES / UNETR_BATCH)  # = 501

nnunet_samples_100 = EPOCH_CAP * NNUNET_ITERS_PER_EPOCH * NNUNET_BATCH   # 50 000
nnunet_updates_100 = EPOCH_CAP * NNUNET_ITERS_PER_EPOCH                   # 25 000

unetr_samples_100  = EPOCH_CAP * unetr_iters_per_epoch * UNETR_BATCH      # 100 200
unetr_updates_100  = EPOCH_CAP * (unetr_iters_per_epoch // UNETR_GRAD_ACCUM)  # 12 500

assert nnunet_samples_100 == 50_000,  nnunet_samples_100
assert nnunet_updates_100 == 25_000,  nnunet_updates_100
assert unetr_samples_100  == 100_200, unetr_samples_100
assert unetr_updates_100  == 12_500,  unetr_updates_100

print("=== 1. Training budget at epoch 100 ===")
print(f"nnU-Net : {nnunet_samples_100:,} samples   {nnunet_updates_100:,} weight updates")
print(f"UNETR   : {unetr_samples_100:,} samples  {unetr_updates_100:,} weight updates")
print(f"UNETR/nnU-Net sample ratio : {unetr_samples_100/nnunet_samples_100:.2f}x")
print(f"nnU-Net/UNETR update ratio : {nnunet_updates_100/unetr_updates_100:.2f}x")
print()


# ============================================================
# 2. nnU-Net pseudo-Dice WT at epochs 0, 100, 999
# ============================================================
nn_dice_ep0   = fold_values(NNUNET_DIR, FOLDS, epoch=0,   column="pseudo_dice_wt")
nn_dice_ep100 = fold_values(NNUNET_DIR, FOLDS, epoch=100, column="pseudo_dice_wt")
nn_dice_ep999 = fold_values(NNUNET_DIR, FOLDS, epoch=999, column="pseudo_dice_wt")

nn_mean_ep0   = statistics.mean(nn_dice_ep0)
nn_mean_ep100 = statistics.mean(nn_dice_ep100)
nn_mean_ep999 = statistics.mean(nn_dice_ep999)

# % of total learning progress achieved by epoch 100
# (not a ratio of Dice scores; normalized to the full ep0→ep999 range)
nn_progress_100 = (nn_mean_ep100 - nn_mean_ep0) / (nn_mean_ep999 - nn_mean_ep0) * 100

print("=== 2. nnU-Net pseudo-Dice WT ===")
print(f"ep0   : {nn_mean_ep0:.4f}  (per fold: {[round(v,4) for v in nn_dice_ep0]})")
print(f"ep100 : {nn_mean_ep100:.4f}  (per fold: {[round(v,4) for v in nn_dice_ep100]})")
print(f"ep999 : {nn_mean_ep999:.4f}  (per fold: {[round(v,4) for v in nn_dice_ep999]})")
print(f"Progress at ep100 = (ep100 - ep0) / (ep999 - ep0) = {nn_progress_100:.1f}%")
print(f"  NOTE: this is NOT ep100/ep999 = {nn_mean_ep100/nn_mean_ep999*100:.1f}%")
print(f"Gain remaining after ep100 = {nn_mean_ep999 - nn_mean_ep100:.4f} Dice points")
print()

assert 90.5 < nn_progress_100 < 91.8, f"expected ~91.1%, got {nn_progress_100:.1f}%"
assert 0.874 < nn_mean_ep100 < 0.876, nn_mean_ep100
assert 0.895 < nn_mean_ep999 < 0.898, nn_mean_ep999


# ============================================================
# 3. UNETR val-Dice WT at epochs 1, 100, 150
# ============================================================
ut_dice_ep1   = fold_values(UNETR_DIR, FOLDS, epoch=1,   column="val_dice_wt")
ut_dice_ep100 = fold_values(UNETR_DIR, FOLDS, epoch=100, column="val_dice_wt")
ut_dice_ep150 = fold_values(UNETR_DIR, FOLDS, epoch=150, column="val_dice_wt")

ut_mean_ep1   = statistics.mean(ut_dice_ep1)
ut_mean_ep100 = statistics.mean(ut_dice_ep100)
ut_mean_ep150 = statistics.mean(ut_dice_ep150)

ut_progress_100 = (ut_mean_ep100 - ut_mean_ep1) / (ut_mean_ep150 - ut_mean_ep1) * 100

print("=== 3. UNETR val-Dice WT ===")
print(f"ep1   : {ut_mean_ep1:.4f}  (per fold: {[round(v,4) for v in ut_dice_ep1]})")
print(f"ep100 : {ut_mean_ep100:.4f}  (per fold: {[round(v,4) for v in ut_dice_ep100]})")
print(f"ep150 : {ut_mean_ep150:.4f}  (per fold: {[round(v,4) for v in ut_dice_ep150]})")
print(f"Progress at ep100 (ref=ep150) = {ut_progress_100:.1f}%")
print()

assert 98.5 < ut_progress_100 < 99.8, f"expected ~99.2%, got {ut_progress_100:.1f}%"
assert 0.915 < ut_mean_ep100 < 0.918, ut_mean_ep100
assert 0.921 < ut_mean_ep150 < 0.924, ut_mean_ep150


# ============================================================
# 4. nnU-Net train-val loss gap at key epochs
# ============================================================
def gap_at_epoch(ep):
    tl = statistics.mean(fold_values(NNUNET_DIR, FOLDS, epoch=ep, column="train_loss"))
    vl = statistics.mean(fold_values(NNUNET_DIR, FOLDS, epoch=ep, column="val_loss"))
    return tl - vl

gap_100 = gap_at_epoch(100)
gap_200 = gap_at_epoch(200)
gap_500 = gap_at_epoch(500)
gap_999 = gap_at_epoch(999)

print("=== 4. nnU-Net train - val loss gap ===")
print(f"ep100 : {gap_100:+.4f}")
print(f"ep200 : {gap_200:+.4f}")
print(f"ep500 : {gap_500:+.4f}")
print(f"ep999 : {gap_999:+.4f}")
print()

assert -0.014 < gap_100 < -0.012, gap_100
assert -0.004 < gap_200 < 0.000,  gap_200
assert -0.050 < gap_500 < -0.040, gap_500
assert -0.065 < gap_999 < -0.055, gap_999


# ============================================================
# 5. UNETR early-stopping terminal epochs per fold
# ============================================================
ut_last_epochs = [
    read_last_epoch(os.path.join(UNETR_DIR, f"fold_{f}", "metrics.csv"))
    for f in range(FOLDS)
]
ut_ep_min  = min(ut_last_epochs)
ut_ep_max  = max(ut_last_epochs)
ut_ep_mean = statistics.mean(ut_last_epochs)

print("=== 5. UNETR early-stopping terminal epochs ===")
print(f"Per fold : {ut_last_epochs}")
print(f"Range    : {ut_ep_min}–{ut_ep_max}")
print(f"Mean     : {ut_ep_mean:.1f}")
print()

assert ut_ep_min >= 150, ut_ep_min
assert ut_ep_max <= 210, ut_ep_max


# ============================================================
# Summary: values to cite in the paragraph
# ============================================================
print("=" * 60)
print("CITE-READY NUMBERS")
print("=" * 60)
print(f"Budget (ep100): nnU-Net {nnunet_samples_100:,} samples / {nnunet_updates_100:,} updates")
print(f"Budget (ep100): UNETR   {unetr_samples_100:,} samples / {unetr_updates_100:,} updates")
print(f"nnU-Net Dice-WT ep100 / ep999 : {nn_mean_ep100:.3f} / {nn_mean_ep999:.3f}")
print(f"  → gain ep0→ep100 covers {nn_progress_100:.1f}% of full ep0→ep999 range")
print(f"  → remaining gain ep100→ep999 : {nn_mean_ep999-nn_mean_ep100:.3f} Dice pts")
print(f"UNETR Dice-WT ep100 / ep150 (ref) : {ut_mean_ep100:.3f} / {ut_mean_ep150:.3f}")
print(f"  → gain ep1→ep100 covers {ut_progress_100:.1f}% of full ep1→ep150 range")
print(f"nnU-Net train-val gap : ep100={gap_100:+.3f}  ep999={gap_999:+.3f}")
print(f"UNETR early-stop epochs: {ut_last_epochs}  mean={ut_ep_mean:.1f}  range={ut_ep_min}–{ut_ep_max}")

print()
print("All assertions passed.")
sys.exit(0)
