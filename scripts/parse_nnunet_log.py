#!/usr/bin/env python3
"""
Parse nnU-Net training logs into structured CSV format.

nnU-Net logs per epoch: train_loss, val_loss, pseudo_dice[per_class], epoch_time, lr.
This script extracts those into our standardized metrics.csv.

Also derives BraTS tumor region pseudo-Dice from per-class Dice:
  - WT (Whole Tumor)  = union of NCR + ED + ET → mean of all three
  - TC (Tumor Core)   = NCR + ET → mean of NCR and ET
  - ET (Enhancing)    = ET only

Usage:
    python scripts/parse_nnunet_log.py \
        --log_file /data/paul_daten/nnunet/nnUNet_results/Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/training_log_*.txt \
        --output experiments/exp02_nnunet/fold_0/metrics.csv
"""

import argparse
import csv
import glob
import re
from pathlib import Path


def parse_log(log_path):
    """Parse a single nnU-Net training log file into epoch records."""
    epochs = []
    current = {}

    with open(log_path) as f:
        for line in f:
            line = line.strip()

            # New epoch
            m = re.match(r".*Epoch (\d+)", line)
            if m and "Current learning rate" not in line:
                if current and "epoch" in current:
                    epochs.append(current)
                current = {"epoch": int(m.group(1))}
                continue

            # Learning rate
            m = re.match(r".*Current learning rate: ([\d.e\-]+)", line)
            if m:
                current["learning_rate"] = float(m.group(1))
                continue

            # Train loss
            m = re.match(r".*train_loss ([\-\d.]+)", line)
            if m:
                current["train_loss"] = float(m.group(1))
                continue

            # Val loss
            m = re.match(r".*val_loss ([\-\d.]+)", line)
            if m:
                current["val_loss"] = float(m.group(1))
                continue

            # Pseudo dice per class: [NCR, ED, ET]
            m = re.match(r".*Pseudo dice \[.*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+)", line)
            if m:
                ncr = float(m.group(1))
                ed = float(m.group(2))
                et = float(m.group(3))
                current["pseudo_dice_ncr"] = ncr
                current["pseudo_dice_ed"] = ed
                current["pseudo_dice_et"] = et
                # Derive BraTS regions
                current["pseudo_dice_wt"] = (ncr + ed + et) / 3.0  # WT = all regions
                current["pseudo_dice_tc"] = (ncr + et) / 2.0       # TC = NCR + ET
                continue

            # Epoch time
            m = re.match(r".*Epoch time: ([\d.]+) s", line)
            if m:
                current["epoch_time_s"] = float(m.group(1))
                continue

            # EMA pseudo Dice
            m = re.match(r".*[Nn]ew best EMA pseudo Dice: ([\d.]+)", line)
            if m:
                current["ema_pseudo_dice"] = float(m.group(1))
                continue

    # Don't forget the last epoch
    if current and "epoch" in current:
        epochs.append(current)

    return epochs


OUTPUT_FIELDS = [
    "epoch",
    "train_loss",
    "val_loss",
    "pseudo_dice_ncr",
    "pseudo_dice_ed",
    "pseudo_dice_et",
    "pseudo_dice_wt",
    "pseudo_dice_tc",
    "learning_rate",
    "epoch_time_s",
    "ema_pseudo_dice",
]


def write_csv(epochs, output_path):
    """Write parsed epochs to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for ep in epochs:
            writer.writerow(ep)

    print(f"Wrote {len(epochs)} epochs to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Parse nnU-Net training log to CSV")
    parser.add_argument("--log_file", required=True, help="Path to training_log_*.txt (supports glob)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    # Support glob patterns
    files = sorted(glob.glob(args.log_file))
    if not files:
        print(f"No log files found matching: {args.log_file}")
        return

    log_path = files[-1]  # Use the latest log if multiple exist
    print(f"Parsing: {log_path}")

    epochs = parse_log(log_path)
    if not epochs:
        print("No epoch data found in log.")
        return

    write_csv(epochs, args.output)

    # Print summary
    last = epochs[-1]
    print(f"\nSummary ({len(epochs)} epochs):")
    print(f"  Final train_loss: {last.get('train_loss', 'N/A')}")
    print(f"  Final val_loss:   {last.get('val_loss', 'N/A')}")
    print(f"  Final pseudo Dice WT: {last.get('pseudo_dice_wt', 'N/A'):.4f}")
    print(f"  Final pseudo Dice TC: {last.get('pseudo_dice_tc', 'N/A'):.4f}")
    print(f"  Final pseudo Dice ET: {last.get('pseudo_dice_et', 'N/A'):.4f}")
    print(f"  Avg epoch time: {sum(e.get('epoch_time_s', 0) for e in epochs) / len(epochs):.1f}s")


if __name__ == "__main__":
    main()
