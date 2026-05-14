#!/usr/bin/env python3
"""
Comprehensive analysis of EXP-03 (UNETR 5-fold CV on BraTS 2021).

Generates:
  1. Training loss curves for all 5 folds
  2. Validation Dice (WT/TC/ET) curves per fold (mean ± std bands)
  3. Early stopping analysis (when did each fold stop?)
  4. Fold variance analysis
  5. Comparison with EXP-02 (nnU-Net baseline)
  6. GPU monitoring summary
  7. Training time analysis
  8. Recommendations & next steps

Output goes to: experiments/exp03_unetr/analysis/

Usage:
    python scripts/analyze_exp03.py
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------- paths ----------
REPO = Path(__file__).resolve().parent.parent
EXP_DIR = REPO / "experiments" / "exp03_unetr"
EXP02_DIR = REPO / "experiments" / "exp02_nnunet"
OUT_DIR = EXP_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_FOLDS = 5

sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = sns.color_palette("tab10", NUM_FOLDS)


# ====================================================================
# 1. Load training metrics
# ====================================================================
def load_all_folds():
    """Load metrics.csv for each fold into a dict of DataFrames."""
    dfs = {}
    for fold in range(NUM_FOLDS):
        csv_path = EXP_DIR / f"fold_{fold}" / "metrics.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["fold"] = fold
            dfs[fold] = df
        else:
            print(f"WARNING: {csv_path} not found, skipping fold {fold}")
    return dfs


def load_fold_info():
    """Load fold_info.json for each fold."""
    info = {}
    for fold in range(NUM_FOLDS):
        info_path = EXP_DIR / f"fold_{fold}" / "fold_info.json"
        if info_path.exists():
            with open(info_path) as f:
                info[fold] = json.load(f)
    return info


# ====================================================================
# 2. Training curve plots
# ====================================================================
def plot_training_curves(dfs):
    """Plot loss and Dice curves for all folds."""

    # --- 2a. Training loss curves ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for fold, df in dfs.items():
        ax.plot(df["epoch"], df["train_loss"], color=COLORS[fold],
                alpha=0.8, label=f"Fold {fold} ({len(df)} ep)")
    ax.set_title("EXP-03 UNETR – Training Loss (DiceCE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("DiceCE Loss")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_train_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 01_train_loss_curves.png")

    # --- 2b. Val Dice per region (mean ± std band + individual folds) ---
    regions = {
        "WT (Whole Tumor)": ("val_dice_wt", "val_dice_wt_std"),
        "TC (Tumor Core)": ("val_dice_tc", "val_dice_tc_std"),
        "ET (Enhancing Tumor)": ("val_dice_et", "val_dice_et_std"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (region_name, (col_mean, col_std)) in zip(axes, regions.items()):
        # Align epochs to common axis using the shortest fold
        min_ep = min(df["epoch"].max() for df in dfs.values())
        # Plot individual folds
        for fold in sorted(dfs.keys()):
            df = dfs[fold]
            ax.plot(df["epoch"], df[col_mean], color=COLORS[fold],
                    alpha=0.4, linewidth=0.9, label=f"Fold {fold}")
        # Mean ± std across folds (up to shortest fold)
        arrays = []
        for fold in sorted(dfs.keys()):
            df = dfs[fold]
            arr = df[df["epoch"] <= min_ep][col_mean].values
            arrays.append(arr)
        arrays = np.array(arrays)
        mean = arrays.mean(axis=0)
        std = arrays.std(axis=0)
        epochs_shared = dfs[0][dfs[0]["epoch"] <= min_ep]["epoch"].values
        ax.plot(epochs_shared, mean, color="black", linewidth=2.5, label="Mean", zorder=5)
        ax.fill_between(epochs_shared, mean - std, mean + std, alpha=0.2, color="black")
        ax.set_title(f"Val Dice: {region_name}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Dice Score")
        ax.set_ylim(0.4, 1.0)
        ax.legend(fontsize=8)

    plt.suptitle("EXP-03 UNETR – Validation Dice per Region", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_val_dice_regions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 02_val_dice_regions.png")

    # --- 2c. All three regions on one plot (mean across folds) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    region_specs = [
        ("WT", "val_dice_wt", "tab:blue"),
        ("TC", "val_dice_tc", "tab:orange"),
        ("ET", "val_dice_et", "tab:green"),
    ]
    for label, col, clr in region_specs:
        arrays = np.array([dfs[f][col].values for f in sorted(dfs.keys())
                           if len(dfs[f]) == len(dfs[0])])
        if len(arrays) == 0:
            # fallback: pad with last value
            max_ep = max(len(dfs[f]) for f in dfs)
            padded = []
            for f in sorted(dfs.keys()):
                arr = dfs[f][col].values
                if len(arr) < max_ep:
                    arr = np.concatenate([arr, np.full(max_ep - len(arr), arr[-1])])
                padded.append(arr)
            arrays = np.array(padded)
        mean = arrays.mean(axis=0)
        std = arrays.std(axis=0)
        epochs = np.arange(1, len(mean) + 1)
        ax.plot(epochs, mean, color=clr, linewidth=2, label=label)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.15, color=clr)
    ax.set_title("EXP-03 UNETR – Mean Val Dice (±Std, all 5 folds)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice Score")
    ax.set_ylim(0.4, 1.0)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_mean_val_dice_all_regions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 03_mean_val_dice_all_regions.png")


# ====================================================================
# 3. Early stopping analysis
# ====================================================================
def analyze_early_stopping(dfs, fold_info):
    """Analyze when each fold stopped and why."""

    report = []
    report.append("=" * 70)
    report.append("EARLY STOPPING ANALYSIS")
    report.append("=" * 70)

    max_epochs = 200
    patience = 30

    report.append(f"\nConfig: max_epochs={max_epochs}, patience={patience}, metric=val_dice_wt")
    report.append("")

    stopped_early = []
    for fold in sorted(dfs.keys()):
        df = dfs[fold]
        n_epochs = len(df)
        best_idx = df["val_dice_wt"].idxmax()
        best_epoch = int(df.loc[best_idx, "epoch"])
        best_dice = float(df.loc[best_idx, "val_dice_wt"])
        final_dice = float(df.iloc[-1]["val_dice_wt"])
        info = fold_info.get(fold, {})
        duration_h = info.get("duration_hours", "?")

        if n_epochs < max_epochs:
            trigger_epoch = n_epochs
            idle_epochs = n_epochs - best_epoch
            stopped_early.append(fold)
            status = f"early stop at epoch {trigger_epoch} (idle {idle_epochs} ep)"
        else:
            status = f"completed all {max_epochs} epochs"

        report.append(f"Fold {fold}: {n_epochs} epochs | {status}")
        report.append(f"  Best: epoch {best_epoch}, val_dice_wt={best_dice:.4f}")
        report.append(f"  Final epoch val_dice_wt={final_dice:.4f}")
        report.append(f"  Duration: {duration_h:.2f}h")
        report.append("")

    report.append(f"Folds that stopped early: {stopped_early}")
    if len(stopped_early) == NUM_FOLDS:
        report.append("→ All folds used early stopping. Training converged before 200 epochs.")
    elif len(stopped_early) > 0:
        report.append(f"→ {len(stopped_early)}/{NUM_FOLDS} folds used early stopping.")
    else:
        report.append("→ No fold triggered early stopping. Consider increasing max_epochs.")

    # Best val Dice WT summary
    best_vals = []
    for fold in sorted(dfs.keys()):
        df = dfs[fold]
        best_vals.append(float(df["val_dice_wt"].max()))
    report.append(f"\nBest val Dice WT per fold: {[f'{v:.4f}' for v in best_vals]}")
    report.append(f"Mean: {np.mean(best_vals):.4f} ± {np.std(best_vals):.4f}")

    # --- Plot: Epoch length per fold, best checkpoint marker ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    epoch_counts = [len(dfs[f]) for f in sorted(dfs.keys())]
    bars = ax.bar(range(NUM_FOLDS), epoch_counts,
                  color=[COLORS[f] for f in range(NUM_FOLDS)], alpha=0.8)
    ax.axhline(y=max_epochs, color="red", linestyle="--", alpha=0.6, label=f"max_epochs={max_epochs}")
    ax.set_xticks(range(NUM_FOLDS))
    ax.set_xticklabels([f"Fold {f}" for f in range(NUM_FOLDS)])
    ax.set_title("Epochs Trained per Fold (early stopping)")
    ax.set_ylabel("Epochs")
    ax.set_ylim(0, max_epochs + 20)
    for i, cnt in enumerate(epoch_counts):
        ax.text(i, cnt + 2, str(cnt), ha="center", fontsize=10)
    ax.legend()

    ax = axes[1]
    for fold in sorted(dfs.keys()):
        df = dfs[fold]
        ax.plot(df["epoch"], df["val_dice_wt"], color=COLORS[fold],
                alpha=0.7, linewidth=1.5, label=f"Fold {fold}")
        best_idx = df["val_dice_wt"].idxmax()
        ax.axvline(x=df.loc[best_idx, "epoch"], color=COLORS[fold],
                   linestyle=":", alpha=0.5, linewidth=1)
        ax.scatter(df.loc[best_idx, "epoch"], df.loc[best_idx, "val_dice_wt"],
                   color=COLORS[fold], marker="*", s=120, zorder=5)
    ax.set_title("Val Dice WT – Best Checkpoint Marked (*)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Dice WT")
    ax.legend(fontsize=8)

    plt.suptitle("EXP-03 UNETR – Early Stopping Analysis", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_early_stopping.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 04_early_stopping.png")

    return "\n".join(report)


# ====================================================================
# 4. Fold variance analysis
# ====================================================================
def analyze_fold_variance(dfs):
    """Analyze inter-fold variance at best-checkpoint epoch."""

    report = []
    report.append("\n" + "=" * 70)
    report.append("FOLD VARIANCE ANALYSIS")
    report.append("=" * 70)

    # Best-checkpoint metrics per fold
    metrics_cols = {
        "Dice WT (best)": "val_dice_wt",
        "Dice TC (best)": "val_dice_tc",
        "Dice ET (best)": "val_dice_et",
    }

    best_data = {}
    for fold in sorted(dfs.keys()):
        df = dfs[fold]
        best_idx = df["val_dice_wt"].idxmax()
        best_row = df.loc[best_idx]
        best_data[fold] = {name: float(best_row[col]) for name, col in metrics_cols.items()}
        best_data[fold]["best_epoch"] = int(best_row["epoch"])
        best_data[fold]["n_epochs"] = len(df)

    df_best = pd.DataFrame(best_data).T
    df_best.index.name = "Fold"

    report.append("\nBest-checkpoint metrics per fold:")
    report.append(df_best.to_string(float_format=lambda x: f"{x:.4f}"))
    report.append("")

    report.append("Summary (Mean ± Std) at best checkpoint:")
    for col in ["Dice WT (best)", "Dice TC (best)", "Dice ET (best)"]:
        vals = df_best[col].values.astype(float)
        report.append(f"  {col:20s}: {vals.mean():.4f} ± {vals.std():.4f}  "
                      f"(range: {vals.min():.4f} – {vals.max():.4f})")

    # Also at last epoch
    report.append("\nFinal-epoch metrics per fold:")
    for fold in sorted(dfs.keys()):
        row = dfs[fold].iloc[-1]
        report.append(f"  Fold {fold}: WT={row['val_dice_wt']:.4f}  "
                      f"TC={row['val_dice_tc']:.4f}  ET={row['val_dice_et']:.4f}")

    # SEM for WT
    report.append("\nStandard Error of Mean (SEM) for Dice WT:")
    for n_folds in [3, 5]:
        vals = df_best["Dice WT (best)"].values.astype(float)[:n_folds]
        sem = vals.std(ddof=1) / np.sqrt(n_folds)
        report.append(f"  {n_folds} folds: SEM = {sem:.4f}  (95% CI width: ±{1.96*sem:.4f})")

    # --- Fold comparison plots ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart: WT/TC/ET per fold at best checkpoint
    ax = axes[0]
    x = np.arange(3)
    width = 0.15
    region_cols = ["Dice WT (best)", "Dice TC (best)", "Dice ET (best)"]
    for i, fold in enumerate(sorted(dfs.keys())):
        vals = [float(df_best.loc[fold, col]) for col in region_cols]
        ax.bar(x + (i - 2) * width, vals, width, label=f"Fold {fold}",
               color=COLORS[i], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["WT", "TC", "ET"])
    ax.set_title("Best-Checkpoint Dice per Region per Fold")
    ax.set_ylabel("Val Dice")
    ax.legend(fontsize=8)
    ax.set_ylim(0.7, 1.0)

    # Scatter: best WT vs best epoch number
    ax = axes[1]
    best_epochs = df_best["best_epoch"].values.astype(float)
    best_wt = df_best["Dice WT (best)"].values.astype(float)
    for i, fold in enumerate(sorted(dfs.keys())):
        ax.scatter(best_epochs[i], best_wt[i], color=COLORS[i],
                   s=120, label=f"Fold {fold}", zorder=5)
    ax.set_title("Best Val Dice WT vs Best Checkpoint Epoch")
    ax.set_xlabel("Best Checkpoint Epoch")
    ax.set_ylabel("Val Dice WT")
    ax.legend(fontsize=8)

    plt.suptitle("EXP-03 UNETR – Fold Variance Analysis", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_fold_variance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 05_fold_variance.png")

    return "\n".join(report)


# ====================================================================
# 5. Comparison with EXP-02 (nnU-Net)
# ====================================================================
def compare_with_exp02(dfs):
    """Compare EXP-03 UNETR with EXP-02 nnU-Net results."""

    report = []
    report.append("\n" + "=" * 70)
    report.append("COMPARISON: EXP-03 UNETR vs EXP-02 nnU-Net")
    report.append("=" * 70)

    # EXP-03 best-checkpoint CV means
    exp03_wt = np.array([float(dfs[f]["val_dice_wt"].max()) for f in sorted(dfs.keys())])
    exp03_tc = np.array([float(dfs[f]["val_dice_tc"].max()) for f in sorted(dfs.keys())])
    exp03_et = np.array([float(dfs[f]["val_dice_et"].max()) for f in sorted(dfs.keys())])

    # Try to load EXP-02 NIfTI Dice (per-fold mean from all_folds_brats_dice.csv)
    exp02_csv = EXP02_DIR / "analysis" / "all_folds_brats_dice.csv"
    exp02_available = exp02_csv.exists()

    report.append("\nEXP-03 UNETR (best-checkpoint val Dice, 5-fold CV):")
    report.append(f"  WT: {exp03_wt.mean():.4f} ± {exp03_wt.std():.4f}")
    report.append(f"  TC: {exp03_tc.mean():.4f} ± {exp03_tc.std():.4f}")
    report.append(f"  ET: {exp03_et.mean():.4f} ± {exp03_et.std():.4f}")

    if exp02_available:
        df_exp02 = pd.read_csv(exp02_csv)
        exp02_wt_mean = df_exp02["dice_wt"].mean()
        exp02_tc_mean = df_exp02["dice_tc"].mean()
        exp02_et_mean = df_exp02["dice_et"].mean()
        exp02_wt_std = df_exp02["dice_wt"].std()
        exp02_tc_std = df_exp02["dice_tc"].std()
        exp02_et_std = df_exp02["dice_et"].std()

        report.append("\nEXP-02 nnU-Net (post-hoc NIfTI Dice, 5-fold CV):")
        report.append(f"  WT: {exp02_wt_mean:.4f} ± {exp02_wt_std:.4f}")
        report.append(f"  TC: {exp02_tc_mean:.4f} ± {exp02_tc_std:.4f}")
        report.append(f"  ET: {exp02_et_mean:.4f} ± {exp02_et_std:.4f}")

        report.append("\nDifference (UNETR – nnU-Net):")
        for region, u, n in [("WT", exp03_wt.mean(), exp02_wt_mean),
                              ("TC", exp03_tc.mean(), exp02_tc_mean),
                              ("ET", exp03_et.mean(), exp02_et_mean)]:
            delta = u - n
            direction = "↑ better" if delta > 0 else "↓ worse"
            report.append(f"  {region}: {delta:+.4f} ({direction})")

        note = ("NOTE: EXP-03 uses best-checkpoint val Dice (in-batch std per case), "
                "EXP-02 uses post-hoc volumetric NIfTI Dice. Direct comparison is indicative only.")
        report.append(f"\n{note}")
    else:
        report.append("\n[EXP-02 comparison: all_folds_brats_dice.csv not found – skipping NIfTI comparison]")

    # Training resource comparison
    report.append("\n--- Resource Comparison ---")
    exp03_total_h = sum(
        dfs[f]["epoch_time_s"].sum() / 3600 for f in sorted(dfs.keys())
    )
    exp03_epochs = sum(len(dfs[f]) for f in sorted(dfs.keys()))
    report.append(f"EXP-03 UNETR:  {exp03_total_h:.1f}h total, {exp03_epochs} total epochs across 5 folds")

    # Load exp02 timing from metrics if available
    exp02_times = []
    for fold in range(5):
        csv_path = EXP02_DIR / f"fold_{fold}" / "metrics.csv"
        if csv_path.exists():
            df02 = pd.read_csv(csv_path)
            exp02_times.append(df02["epoch_time_s"].sum())
    if exp02_times:
        exp02_total_h = sum(exp02_times) / 3600
        report.append(f"EXP-02 nnU-Net: {exp02_total_h:.1f}h total, 5000 total epochs across 5 folds")
        ratio = exp02_total_h / exp03_total_h
        report.append(f"nnU-Net was {ratio:.1f}x more expensive in training time")

    # --- Comparison plot ---
    fig, ax = plt.subplots(figsize=(9, 6))

    regions = ["WT", "TC", "ET"]
    x = np.arange(len(regions))
    width = 0.35

    unetr_means = [exp03_wt.mean(), exp03_tc.mean(), exp03_et.mean()]
    unetr_stds = [exp03_wt.std(), exp03_tc.std(), exp03_et.std()]

    bars1 = ax.bar(x - width/2, unetr_means, width, yerr=unetr_stds,
                   label="EXP-03 UNETR", color="tab:blue", alpha=0.8,
                   capsize=5)

    if exp02_available:
        nnunet_means = [exp02_wt_mean, exp02_tc_mean, exp02_et_mean]
        nnunet_stds = [exp02_wt_std, exp02_tc_std, exp02_et_std]
        bars2 = ax.bar(x + width/2, nnunet_means, width, yerr=nnunet_stds,
                       label="EXP-02 nnU-Net", color="tab:orange", alpha=0.8,
                       capsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.set_title("EXP-03 UNETR vs EXP-02 nnU-Net – Dice Score Comparison")
    ax.set_ylabel("Dice Score")
    ax.set_ylim(0.7, 1.0)
    ax.legend()

    # Annotate bars with values
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.003, f"{h:.3f}",
                ha="center", va="bottom", fontsize=9)
    if exp02_available:
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.003, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "06_exp02_vs_exp03_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 06_exp02_vs_exp03_comparison.png")

    return "\n".join(report)


# ====================================================================
# 6. GPU monitoring summary
# ====================================================================
def analyze_gpu_monitoring():
    """Summarize GPU monitoring data."""
    csv_path = EXP_DIR / "gpu_monitor.csv"
    if not csv_path.exists():
        return "\n[SKIPPED] gpu_monitor.csv not found"

    report = []
    report.append("\n" + "=" * 70)
    report.append("GPU MONITORING SUMMARY")
    report.append("=" * 70)

    df = pd.read_csv(csv_path)

    if "gpu_temp_c" in df.columns:
        report.append(f"\nGPU Temperature : mean={df['gpu_temp_c'].mean():.1f}°C, "
                      f"max={df['gpu_temp_c'].max():.1f}°C")
    if "gpu_util_pct" in df.columns:
        report.append(f"GPU Utilization : mean={df['gpu_util_pct'].mean():.1f}%, "
                      f"min={df['gpu_util_pct'].min():.1f}%")
    if "gpu_mem_used_mb" in df.columns:
        report.append(f"GPU Memory      : mean={df['gpu_mem_used_mb'].mean():.0f} MB, "
                      f"max={df['gpu_mem_used_mb'].max():.0f} MB")
    if "gpu_power_w" in df.columns:
        report.append(f"GPU Power       : mean={df['gpu_power_w'].mean():.1f}W, "
                      f"max={df['gpu_power_w'].max():.1f}W")
    if "disk_used_pct" in df.columns:
        report.append(f"Disk Usage      : start={df['disk_used_pct'].iloc[0]:.1f}%, "
                      f"end={df['disk_used_pct'].iloc[-1]:.1f}%")

    report.append(f"\nMonitoring entries: {len(df)}")
    if "timestamp" in df.columns:
        try:
            t0 = pd.to_datetime(df["timestamp"].iloc[0])
            t1 = pd.to_datetime(df["timestamp"].iloc[-1])
            hours = (t1 - t0).total_seconds() / 3600
            report.append(f"Monitoring duration: {hours:.1f} hours")
        except Exception:
            pass

    return "\n".join(report)


# ====================================================================
# 7. Training time analysis
# ====================================================================
def analyze_training_time(dfs, fold_info):
    """Analyze epoch times and total training duration."""
    report = []
    report.append("\n" + "=" * 70)
    report.append("TRAINING TIME ANALYSIS")
    report.append("=" * 70)

    for fold in sorted(dfs.keys()):
        times = dfs[fold]["epoch_time_s"].values
        total_h = times.sum() / 3600
        info = fold_info.get(fold, {})
        wall_h = info.get("duration_hours", "?")
        report.append(f"\nFold {fold}: epochs={len(dfs[fold])}, total={total_h:.2f}h (wall: {wall_h}h), "
                      f"mean={times.mean():.1f}s/epoch, "
                      f"min={times.min():.1f}s, max={times.max():.1f}s")

    all_times = np.concatenate([dfs[f]["epoch_time_s"].values for f in sorted(dfs.keys())])
    total_all = all_times.sum() / 3600
    report.append(f"\nTotal across all 5 folds: {total_all:.1f} hours")
    report.append(f"Mean epoch time        : {all_times.mean():.1f}s ± {all_times.std():.1f}s")
    report.append(f"(Note: each epoch ≈ {all_times.mean()/60:.1f} min — UNETR is slower than nnU-Net per epoch "
                  f"due to ViT self-attention over 3D patches)")

    # --- Epoch time plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for fold in sorted(dfs.keys()):
        df = dfs[fold]
        ax.plot(df["epoch"], df["epoch_time_s"], color=COLORS[fold],
                alpha=0.7, linewidth=1, label=f"Fold {fold}")
    ax.axhline(y=all_times.mean(), color="black", linestyle="--",
               linewidth=1.5, label=f"Overall mean ({all_times.mean():.1f}s)")
    ax.set_title(f"EXP-03 UNETR – Epoch Duration (mean={all_times.mean():.1f}s)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds per Epoch")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "07_epoch_time.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 07_epoch_time.png")

    return "\n".join(report)


# ====================================================================
# 8. Recommendations
# ====================================================================
def generate_recommendations(dfs):
    """Generate data-driven recommendations for thesis & next experiments."""
    report = []
    report.append("\n" + "=" * 70)
    report.append("RECOMMENDATIONS & NEXT STEPS")
    report.append("=" * 70)

    # Convergence / epoch count
    report.append("\n--- Epoch count & convergence ---")
    epochs_per_fold = [len(dfs[f]) for f in sorted(dfs.keys())]
    max_fold = max(range(NUM_FOLDS), key=lambda f: len(dfs[f]))
    if max(epochs_per_fold) < 200:
        report.append(f"  All folds stopped before 200 epochs (max: {max(epochs_per_fold)} epochs).")
        report.append("  → Early stopping worked well. 200-epoch budget was sufficient.")
    else:
        report.append(f"  Fold {max_fold} reached max 200 epochs without early stopping.")
        report.append("  → Consider increasing max_epochs for a longer training budget.")

    # Performance assessment
    wt_vals = np.array([float(dfs[f]["val_dice_wt"].max()) for f in sorted(dfs.keys())])
    tc_vals = np.array([float(dfs[f]["val_dice_tc"].max()) for f in sorted(dfs.keys())])
    et_vals = np.array([float(dfs[f]["val_dice_et"].max()) for f in sorted(dfs.keys())])

    report.append("\n--- Performance assessment ---")
    report.append(f"  Dice WT: {wt_vals.mean():.4f} {'✓ Good (>0.90)' if wt_vals.mean() > 0.90 else '✗ Below 0.90 target'}")
    report.append(f"  Dice TC: {tc_vals.mean():.4f} {'✓ Good (>0.85)' if tc_vals.mean() > 0.85 else '✗ Below 0.85 target'}")
    report.append(f"  Dice ET: {et_vals.mean():.4f} {'✓ Good (>0.78)' if et_vals.mean() > 0.78 else '✗ Below 0.78 target'}")

    # Inter-fold variance
    report.append("\n--- Fold variance ---")
    std_wt = wt_vals.std()
    if std_wt < 0.010:
        report.append(f"  Low inter-fold variance (std={std_wt:.4f}). Results are stable.")
    elif std_wt < 0.020:
        report.append(f"  Moderate inter-fold variance (std={std_wt:.4f}). Acceptable for 5-fold CV.")
    else:
        report.append(f"  High inter-fold variance (std={std_wt:.4f}). "
                      "Check for outlier folds or data imbalance issues.")

    return "\n".join(report)


# ====================================================================
# MAIN
# ====================================================================
def main():
    print("=" * 60)
    print("EXP-03 Analysis: UNETR 5-Fold CV on BraTS 2021")
    print("=" * 60)

    # 1. Load data
    print("\n[1/7] Loading training metrics...")
    dfs = load_all_folds()
    if not dfs:
        print("ERROR: No metrics found. Aborting.")
        sys.exit(1)
    for fold, df in dfs.items():
        print(f"  Fold {fold}: {len(df)} epochs loaded")
    fold_info = load_fold_info()

    # 2. Training curves
    print("\n[2/7] Generating training curve plots...")
    plot_training_curves(dfs)

    # 3. Early stopping analysis
    print("\n[3/7] Analyzing early stopping...")
    early_report = analyze_early_stopping(dfs, fold_info)

    # 4. Fold variance
    print("\n[4/7] Analyzing fold variance...")
    fold_report = analyze_fold_variance(dfs)

    # 5. Comparison with EXP-02
    print("\n[5/7] Comparing with EXP-02 nnU-Net...")
    comparison_report = compare_with_exp02(dfs)

    # 6. GPU monitoring
    print("\n[6/7] Analyzing GPU monitoring data...")
    gpu_report = analyze_gpu_monitoring()

    # 7. Training time
    print("\n[7/7] Analyzing training time...")
    time_report = analyze_training_time(dfs, fold_info)

    # Recommendations
    rec_report = generate_recommendations(dfs)

    # Load metadata for header
    meta_path = EXP_DIR / "experiment_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    full_report = "\n".join([
        "EXP-03 ANALYSIS REPORT",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset : BraTS 2021 (1251 cases, 5-fold CV)",
        f"Model   : UNETR (ViT-B/16 + CNN decoder, Hatamizadeh et al. WACV 2022)",
        f"Config  : {meta.get('config', {}).get('num_epochs', '?')} max epochs, "
        f"early stopping patience={meta.get('config', {}).get('early_stopping_patience', '?')}, "
        f"optimizer={meta.get('config', {}).get('optimizer', '?')}, "
        f"lr={meta.get('config', {}).get('lr', '?')}",
        "",
        early_report,
        fold_report,
        comparison_report,
        gpu_report,
        time_report,
        rec_report,
    ])

    report_path = OUT_DIR / "exp03_analysis_report.txt"
    with open(report_path, "w") as f:
        f.write(full_report)

    print(f"\n{'='*60}")
    print(f"Full report: {report_path}")
    print(f"Plots saved to: {OUT_DIR}/")
    print(f"{'='*60}")
    print("\n" + full_report)


if __name__ == "__main__":
    main()
