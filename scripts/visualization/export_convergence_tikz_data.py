#!/usr/bin/env python3
"""
Export training-loss data for the pgfplots convergence figure.

Writes two CSV files to figures/data/:
  nnunet_loss.csv   -- epoch, mean, lower, upper  (all 5 folds, epochs 0-999)
  unetr_loss.csv    -- epoch, mean, lower, upper  (epochs where all 5 folds present)
"""

import csv
import math
import os
import statistics

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NNUNET_DIR = os.path.join(BASE, "experiments", "exp02_nnunet")
UNETR_DIR  = os.path.join(BASE, "experiments", "exp03_unetr")
OUT_DIR    = os.path.join(BASE, "figures", "data")
os.makedirs(OUT_DIR, exist_ok=True)


def load(base_dir, folds=5):
    data = {}
    for fold in range(folds):
        path = os.path.join(base_dir, f"fold_{fold}", "metrics.csv")
        with open(path) as f:
            for row in csv.DictReader(f):
                ep = int(row["epoch"])
                data.setdefault(ep, []).append(float(row["train_loss"]))
    return data


def write_csv(path, data, min_folds=5):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "mean", "lower", "upper"])
        for ep in sorted(data):
            vals = data[ep]
            if len(vals) < min_folds:
                continue
            m = statistics.mean(vals)
            s = statistics.stdev(vals) if len(vals) > 1 else 0.0
            w.writerow([ep, f"{m:.6f}", f"{m - s:.6f}", f"{m + s:.6f}"])
    print(f"  wrote {path}")


nn_data = load(NNUNET_DIR)
ut_data = load(UNETR_DIR)

write_csv(os.path.join(OUT_DIR, "nnunet_loss.csv"), nn_data, min_folds=5)
write_csv(os.path.join(OUT_DIR, "unetr_loss.csv"),  ut_data, min_folds=5)

# Print key values needed by the .tex file
def at(data, target):
    ep = min(data, key=lambda e: abs(e - target))
    return statistics.mean(data[ep])

nn_at100 = at(nn_data, 100)
ut_at100 = at(ut_data, 100)
ut_max   = max(ep for ep, v in ut_data.items() if len(v) == 5)
nn_ymin  = min(statistics.mean(v) for v in nn_data.values()) - 0.02
nn_ymax  = max(statistics.mean(v) for v in nn_data.values()) + 0.02
ut_ymin  = min(statistics.mean(v) for v in ut_data.values()) - 0.05
ut_ymax  = max(statistics.mean(v) for v in ut_data.values()) + 0.05

print()
print("Key values for convergence_training_loss.tex:")
print(f"  nn_at100  = {nn_at100:.4f}")
print(f"  ut_at100  = {ut_at100:.4f}")
print(f"  ut_max_ep = {ut_max}")
print(f"  nn ymin/ymax: {nn_ymin:.3f} / {nn_ymax:.3f}")
print(f"  ut ymin/ymax: {ut_ymin:.3f} / {ut_ymax:.3f}")
print(f"  nnU-Net iter axis: 0 to {999 * 250}")
print(f"  UNETR   iter axis: 0 to {ut_max * 501}")
