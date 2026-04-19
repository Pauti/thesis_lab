#!/usr/bin/env python3
"""
Comprehensive analysis of EXP-02 (nnU-Net 5-fold CV on BraTS 2021).

Generates:
  1. Training curves (loss, Dice per region) for all 5 folds
  2. Convergence analysis (when did metrics plateau?)
  3. Fold variance analysis (were all 5 folds needed?)
  4. Post-hoc BraTS region Dice (WT/TC/ET) on NIfTI validation predictions
  5. Summary report with statistics

Output goes to: experiments/exp02_nnunet/analysis/

Usage:
    python scripts/analyze_exp02.py
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
EXP_DIR = REPO / "experiments" / "exp02_nnunet"
NNUNET_RESULTS = Path("/data/paul_daten/nnunet/nnUNet_results/Dataset001_BraTS/"
                       "nnUNetTrainer__nnUNetPlans__3d_fullres")
NNUNET_RAW = Path("/data/paul_daten/nnunet/nnUNet_raw/Dataset001_BraTS")
OUT_DIR = EXP_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_FOLDS = 5
# nnU-Net label mapping (after convert_to_nnunet remapping: BraTS 4→3)
# Labels in predictions: 0=bg, 1=NCR, 2=ED, 3=ET
# BraTS regions:
#   WT (Whole Tumor)     = NCR ∪ ED ∪ ET  → labels {1,2,3}
#   TC (Tumor Core)      = NCR ∪ ET       → labels {1,3}
#   ET (Enhancing Tumor) = ET             → labels {3}

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


# ====================================================================
# 2. Training curve plots
# ====================================================================
def plot_training_curves(dfs):
    """Plot loss and Dice curves for all folds."""

    # --- 2a. Loss curves (train & val, all folds) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for fold, df in dfs.items():
        axes[0].plot(df["epoch"], df["train_loss"], color=COLORS[fold],
                     alpha=0.7, label=f"Fold {fold}")
        axes[1].plot(df["epoch"], df["val_loss"], color=COLORS[fold],
                     alpha=0.7, label=f"Fold {fold}")

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Dice+CE Loss")
    axes[0].legend(fontsize=9)

    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice+CE Loss")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 01_loss_curves.png")

    # --- 2b. Per-region pseudo-Dice (mean ± std across folds) ---
    regions = {
        "WT (Whole Tumor)": "pseudo_dice_wt",
        "TC (Tumor Core)": "pseudo_dice_tc",
        "ET (Enhancing Tumor)": "pseudo_dice_et",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (region_name, col) in zip(axes, regions.items()):
        all_vals = np.array([dfs[f][col].values for f in sorted(dfs.keys())])
        mean = all_vals.mean(axis=0)
        std = all_vals.std(axis=0)
        epochs = dfs[0]["epoch"].values

        ax.plot(epochs, mean, color="tab:blue", linewidth=2, label="Mean")
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color="tab:blue")
        for fold in sorted(dfs.keys()):
            ax.plot(epochs, dfs[fold][col].values, alpha=0.3, linewidth=0.8,
                    color=COLORS[fold])

        ax.set_title(f"Pseudo-Dice: {region_name}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Pseudo-Dice")
        ax.set_ylim(0.5, 1.0)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_dice_regions_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 02_dice_regions_curves.png")

    # --- 2c. Per-class pseudo-Dice (NCR, ED, ET) mean across folds ---
    classes = {
        "NCR (Necrotic Core)": "pseudo_dice_ncr",
        "ED (Edema)": "pseudo_dice_ed",
        "ET (Enhancing Tumor)": "pseudo_dice_et",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    class_colors = ["tab:red", "tab:green", "tab:blue"]
    for (cls_name, col), clr in zip(classes.items(), class_colors):
        all_vals = np.array([dfs[f][col].values for f in sorted(dfs.keys())])
        mean = all_vals.mean(axis=0)
        std = all_vals.std(axis=0)
        epochs = dfs[0]["epoch"].values
        ax.plot(epochs, mean, color=clr, linewidth=2, label=cls_name)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.15, color=clr)

    ax.set_title("Per-Class Pseudo-Dice (Mean ± Std across 5 folds)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pseudo-Dice")
    ax.set_ylim(0.4, 1.0)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_per_class_dice.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 03_per_class_dice.png")

    # --- 2d. Learning rate schedule ---
    fig, ax = plt.subplots(figsize=(8, 4))
    for fold, df in dfs.items():
        ax.plot(df["epoch"], df["learning_rate"], color=COLORS[fold],
                alpha=0.7, label=f"Fold {fold}")
    ax.set_title("Learning Rate Schedule (Poly LR)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_learning_rate.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 04_learning_rate.png")


# ====================================================================
# 3. Convergence analysis
# ====================================================================
def analyze_convergence(dfs):
    """Determine when metrics plateau and whether 1000 epochs were needed."""

    report = []
    report.append("=" * 70)
    report.append("CONVERGENCE ANALYSIS")
    report.append("=" * 70)

    # Use mean pseudo-Dice WT across folds as primary convergence metric
    all_wt = np.array([dfs[f]["pseudo_dice_wt"].values for f in sorted(dfs.keys())])
    mean_wt = all_wt.mean(axis=0)
    epochs = dfs[0]["epoch"].values

    # Find when 95%, 99%, 99.5% of final performance is reached
    final_val = mean_wt[-1]  # epoch 999
    thresholds = [0.95, 0.99, 0.995]
    convergence_epochs = {}

    for thresh in thresholds:
        target = final_val * thresh
        idx = np.argmax(mean_wt >= target)
        if mean_wt[idx] >= target:
            convergence_epochs[thresh] = int(epochs[idx])
        else:
            convergence_epochs[thresh] = None

    report.append(f"\nFinal mean pseudo-Dice WT (epoch 999): {final_val:.4f}")
    for thresh, ep in convergence_epochs.items():
        pct = thresh * 100
        if ep is not None:
            report.append(f"  {pct:.1f}% of final ({final_val*thresh:.4f}) reached at epoch {ep}")
        else:
            report.append(f"  {pct:.1f}% of final: never reached")

    # Moving-window improvement analysis
    # Check if last 200 epochs contributed meaningfully
    window = 50
    improvements = []
    checkpoints = [200, 300, 400, 500, 600, 700, 800, 900, 999]

    report.append(f"\nMean pseudo-Dice WT at checkpoints:")
    for cp in checkpoints:
        val = mean_wt[cp]
        report.append(f"  Epoch {cp:4d}: {val:.4f}")

    report.append(f"\nImprovement over last N epochs (Dice WT):")
    for start in [200, 300, 500, 750]:
        delta = mean_wt[-1] - mean_wt[start]
        report.append(f"  Epoch {start} → 999: Δ = {delta:+.4f} ({delta/mean_wt[start]*100:+.2f}%)")

    # Rolling improvement: max absolute Dice improvement in any 100-epoch window
    report.append(f"\nRolling 100-epoch improvement (Dice WT):")
    for start in range(0, 900, 100):
        end = start + 100
        delta = mean_wt[end] - mean_wt[start]
        marker = " ← significant" if abs(delta) > 0.005 else ""
        report.append(f"  Epoch {start:4d}-{end:4d}: Δ = {delta:+.5f}{marker}")

    # --- Convergence plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 2a: Full curve with threshold lines
    ax = axes[0, 0]
    ax.plot(epochs, mean_wt, "b-", linewidth=2, label="Mean Dice WT")
    for thresh, ep in convergence_epochs.items():
        if ep is not None:
            ax.axhline(y=final_val * thresh, color="gray", linestyle="--", alpha=0.5)
            ax.axvline(x=ep, color="red", linestyle=":", alpha=0.5)
            ax.annotate(f"{thresh*100:.0f}% @ ep {ep}",
                       xy=(ep, final_val * thresh), fontsize=8,
                       xytext=(ep + 30, final_val * thresh - 0.01))
    ax.set_title("Convergence: Mean Dice WT (threshold markers)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pseudo-Dice WT")
    ax.legend()

    # 2b: Zoomed into last 500 epochs
    ax = axes[0, 1]
    for fold in sorted(dfs.keys()):
        ax.plot(epochs[500:], all_wt[fold, 500:], alpha=0.5, linewidth=1,
                color=COLORS[fold], label=f"Fold {fold}")
    ax.plot(epochs[500:], mean_wt[500:], "k-", linewidth=2.5, label="Mean")
    ax.set_title("Dice WT: Last 500 Epochs (zoomed)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pseudo-Dice WT")
    ax.legend(fontsize=8)

    # 2c: Marginal gain per 100 epochs
    ax = axes[1, 0]
    gains = []
    bins = []
    for start in range(0, 900, 100):
        end = start + 100
        delta = mean_wt[end] - mean_wt[start]
        gains.append(delta)
        bins.append(f"{start}-{end}")
    bars = ax.bar(range(len(gains)), gains, color=["tab:blue" if g > 0.005 else "tab:gray" for g in gains])
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bins, rotation=45, ha="right", fontsize=8)
    ax.set_title("Marginal Dice WT Gain per 100-Epoch Block")
    ax.set_ylabel("Δ Dice WT")
    ax.axhline(y=0.005, color="red", linestyle="--", alpha=0.5, label="significance threshold (0.005)")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend(fontsize=8)

    # 2d: Epoch time distribution
    ax = axes[1, 1]
    all_times = []
    for fold in sorted(dfs.keys()):
        times = dfs[fold]["epoch_time_s"].values
        all_times.extend(times)
        ax.plot(epochs, times, alpha=0.5, linewidth=0.8, color=COLORS[fold],
                label=f"Fold {fold}")
    ax.set_title(f"Epoch Duration (mean={np.mean(all_times):.1f}s)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_convergence_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 05_convergence_analysis.png")

    return "\n".join(report)


# ====================================================================
# 4. Fold variance analysis
# ====================================================================
def analyze_fold_variance(dfs):
    """Analyze inter-fold variance to determine if 5 folds were needed."""

    report = []
    report.append("\n" + "=" * 70)
    report.append("FOLD VARIANCE ANALYSIS")
    report.append("=" * 70)

    # Final epoch metrics per fold
    metrics_cols = {
        "Dice WT": "pseudo_dice_wt",
        "Dice TC": "pseudo_dice_tc",
        "Dice ET": "pseudo_dice_et",
        "NCR": "pseudo_dice_ncr",
        "ED": "pseudo_dice_ed",
        "Val Loss": "val_loss",
        "Train Loss": "train_loss",
    }

    final_data = {}
    for fold in sorted(dfs.keys()):
        row = dfs[fold].iloc[-1]
        final_data[fold] = {name: row[col] for name, col in metrics_cols.items()}

    df_final = pd.DataFrame(final_data).T
    df_final.index.name = "Fold"

    report.append("\nFinal metrics at epoch 999:")
    report.append(df_final.to_string(float_format=lambda x: f"{x:.4f}"))
    report.append("")

    # Mean ± std
    report.append("Summary (Mean ± Std):")
    for col in df_final.columns:
        vals = df_final[col].values
        report.append(f"  {col:12s}: {vals.mean():.4f} ± {vals.std():.4f}  "
                      f"(range: {vals.min():.4f} – {vals.max():.4f})")

    # Standard error of mean (SEM) - shows how much the mean estimate benefits from more folds
    report.append("\nStandard Error of Mean (SEM):")
    for n_folds in [3, 5]:
        vals = df_final["Dice WT"].values[:n_folds]
        sem = vals.std(ddof=1) / np.sqrt(n_folds)
        report.append(f"  {n_folds} folds: SEM = {sem:.4f} (95% CI width: ±{1.96*sem:.4f})")

    # Compare 3-fold vs 5-fold stability
    report.append("\n3-fold vs 5-fold comparison (Dice WT):")
    wt_5 = df_final["Dice WT"].values
    for combo_3 in [[0,1,2], [0,1,3], [0,2,4], [1,3,4]]:
        wt_3 = wt_5[combo_3]
        report.append(f"  Folds {combo_3}: mean={wt_3.mean():.4f} ± {wt_3.std():.4f} "
                      f"(vs 5-fold mean {wt_5.mean():.4f})")

    # --- Fold comparison plots ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Box plot of final Dice per region per fold
    box_data = []
    for fold in sorted(dfs.keys()):
        for region, col in [("WT", "pseudo_dice_wt"), ("TC", "pseudo_dice_tc"), ("ET", "pseudo_dice_et")]:
            box_data.append({"Fold": fold, "Region": region,
                           "Pseudo-Dice": dfs[fold].iloc[-1][col]})
    df_box = pd.DataFrame(box_data)

    ax = axes[0]
    x = np.arange(3)
    width = 0.15
    for i, fold in enumerate(sorted(dfs.keys())):
        subset = df_box[df_box["Fold"] == fold]
        vals = subset["Pseudo-Dice"].values
        ax.bar(x + i * width - 2 * width, vals, width, label=f"Fold {fold}",
               color=COLORS[i], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["WT", "TC", "ET"])
    ax.set_title("Final Dice per Region per Fold")
    ax.set_ylabel("Pseudo-Dice (epoch 999)")
    ax.legend(fontsize=8)
    ax.set_ylim(0.82, 0.96)

    # Dice WT trajectory per fold (overlay)
    ax = axes[1]
    for fold in sorted(dfs.keys()):
        ax.plot(dfs[fold]["epoch"], dfs[fold]["pseudo_dice_wt"],
                color=COLORS[fold], alpha=0.7, label=f"Fold {fold}")
    ax.set_title("Dice WT Trajectory per Fold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pseudo-Dice WT")
    ax.legend(fontsize=8)

    # Fold spread over time (std)
    ax = axes[2]
    all_wt = np.array([dfs[f]["pseudo_dice_wt"].values for f in sorted(dfs.keys())])
    fold_std = all_wt.std(axis=0)
    epochs = dfs[0]["epoch"].values
    ax.plot(epochs, fold_std, "b-", linewidth=1.5)
    ax.set_title("Inter-Fold Std(Dice WT) Over Training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Std (Dice WT)")
    ax.axhline(y=0.01, color="red", linestyle="--", alpha=0.5, label="threshold 0.01")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "06_fold_variance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 06_fold_variance.png")

    return "\n".join(report)


# ====================================================================
# 5. Post-hoc BraTS region Dice on NIfTI validation predictions
# ====================================================================
def compute_brats_dice_nifti():
    """Compute proper volumetric Dice (WT/TC/ET) on saved validation predictions."""
    try:
        import nibabel as nib
    except ImportError:
        return "\n[SKIPPED] nibabel not installed – cannot compute post-hoc Dice on NIfTI"

    report = []
    report.append("\n" + "=" * 70)
    report.append("POST-HOC BRATS REGION DICE (on validation NIfTI predictions)")
    report.append("=" * 70)

    def dice_score(pred_mask, gt_mask):
        """Compute Dice between two binary masks."""
        intersection = np.sum(pred_mask & gt_mask)
        total = np.sum(pred_mask) + np.sum(gt_mask)
        if total == 0:
            return 1.0  # Both empty → perfect
        return 2.0 * intersection / total

    def compute_brats_regions(pred, gt):
        """Compute BraTS region Dice from label volumes.
        Labels: 0=bg, 1=NCR, 2=ED, 3=ET (nnU-Net remapped from BraTS 4→3)
        """
        # WT = NCR ∪ ED ∪ ET = labels {1,2,3}
        wt_pred = (pred >= 1)
        wt_gt = (gt >= 1)
        # TC = NCR ∪ ET = labels {1,3}
        tc_pred = np.isin(pred, [1, 3])
        tc_gt = np.isin(gt, [1, 3])
        # ET = label {3}
        et_pred = (pred == 3)
        et_gt = (gt == 3)

        return {
            "dice_wt": dice_score(wt_pred, wt_gt),
            "dice_tc": dice_score(tc_pred, tc_gt),
            "dice_et": dice_score(et_pred, et_gt),
        }

    all_fold_results = {}

    for fold in range(NUM_FOLDS):
        pred_dir = NNUNET_RESULTS / f"fold_{fold}" / "validation"
        # Ground truth labels are in nnUNet_raw
        gt_dir = NNUNET_RAW / "labelsTr"

        if not pred_dir.exists():
            report.append(f"\n  Fold {fold}: prediction dir not found, skipping")
            continue

        pred_files = sorted(pred_dir.glob("*.nii.gz"))
        fold_scores = []
        n_cases = len(pred_files)

        print(f"  Computing Dice for Fold {fold} ({n_cases} cases)...")

        for pred_path in pred_files:
            case_id = pred_path.name  # e.g., BraTS2021_00000.nii.gz
            gt_path = gt_dir / case_id

            if not gt_path.exists():
                continue

            pred_vol = nib.load(str(pred_path)).get_fdata().astype(np.int32)
            gt_vol = nib.load(str(gt_path)).get_fdata().astype(np.int32)

            scores = compute_brats_regions(pred_vol, gt_vol)
            scores["case_id"] = case_id.replace(".nii.gz", "")
            fold_scores.append(scores)

        if not fold_scores:
            report.append(f"\n  Fold {fold}: no cases matched, skipping")
            continue

        df_fold = pd.DataFrame(fold_scores)
        all_fold_results[fold] = df_fold

        # Save per-case results
        csv_path = OUT_DIR / f"fold_{fold}_brats_dice.csv"
        df_fold.to_csv(csv_path, index=False)

        report.append(f"\nFold {fold} ({len(df_fold)} cases):")
        for region in ["dice_wt", "dice_tc", "dice_et"]:
            vals = df_fold[region].values
            report.append(f"  {region.upper():10s}: {vals.mean():.4f} ± {vals.std():.4f} "
                         f"(median={np.median(vals):.4f}, min={vals.min():.4f}, max={vals.max():.4f})")

    if not all_fold_results:
        return "\n".join(report)

    # Aggregate across folds
    report.append("\n" + "-" * 50)
    report.append("AGGREGATE ACROSS ALL FOLDS:")

    all_scores = pd.concat(all_fold_results.values(), ignore_index=True)
    all_scores.to_csv(OUT_DIR / "all_folds_brats_dice.csv", index=False)

    for region in ["dice_wt", "dice_tc", "dice_et"]:
        vals = all_scores[region].values
        report.append(f"  {region.upper():10s}: {vals.mean():.4f} ± {vals.std():.4f} "
                     f"(median={np.median(vals):.4f})")

    # Per-fold means for cross-validation summary
    report.append("\nPer-fold mean Dice (for CV summary):")
    fold_means = {}
    for fold, df_fold in sorted(all_fold_results.items()):
        fm = {r: df_fold[r].mean() for r in ["dice_wt", "dice_tc", "dice_et"]}
        fold_means[fold] = fm
        report.append(f"  Fold {fold}: WT={fm['dice_wt']:.4f}  TC={fm['dice_tc']:.4f}  ET={fm['dice_et']:.4f}")

    # CV mean ± std
    for region in ["dice_wt", "dice_tc", "dice_et"]:
        fold_vals = np.array([fold_means[f][region] for f in sorted(fold_means)])
        report.append(f"  CV {region.upper():10s}: {fold_vals.mean():.4f} ± {fold_vals.std():.4f}")

    # --- Plots ---
    # Box plots per region
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, region in zip(axes, ["dice_wt", "dice_tc", "dice_et"]):
        data_per_fold = [all_fold_results[f][region].values for f in sorted(all_fold_results)]
        bp = ax.boxplot(data_per_fold, labels=[f"Fold {f}" for f in sorted(all_fold_results)],
                       patch_artist=True)
        for patch, color in zip(bp['boxes'], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(f"BraTS {region.replace('dice_', '').upper()} Dice")
        ax.set_ylabel("Dice Score")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "07_brats_dice_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 07_brats_dice_boxplots.png")

    # Violin plots (all folds combined)
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_data = []
    for region in ["dice_wt", "dice_tc", "dice_et"]:
        for val in all_scores[region].values:
            plot_data.append({"Region": region.replace("dice_", "").upper(),
                            "Dice": val})
    df_plot = pd.DataFrame(plot_data)
    sns.violinplot(data=df_plot, x="Region", y="Dice", ax=ax, inner="box",
                   palette=["tab:blue", "tab:orange", "tab:green"])
    ax.set_title("BraTS Region Dice Distribution (all cases, all folds)")
    ax.set_ylabel("Dice Score")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "08_brats_dice_violin.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 08_brats_dice_violin.png")

    # Worst-case analysis
    report.append("\n" + "-" * 50)
    report.append("WORST CASES (bottom 10 by Dice WT):")
    worst = all_scores.nsmallest(10, "dice_wt")
    for _, row in worst.iterrows():
        report.append(f"  {row['case_id']}: WT={row['dice_wt']:.4f} TC={row['dice_tc']:.4f} ET={row['dice_et']:.4f}")

    report.append("\nBEST CASES (top 10 by Dice WT):")
    best = all_scores.nlargest(10, "dice_wt")
    for _, row in best.iterrows():
        report.append(f"  {row['case_id']}: WT={row['dice_wt']:.4f} TC={row['dice_tc']:.4f} ET={row['dice_et']:.4f}")

    # Cases where ET = 0 in ground truth (empty ET region)
    n_zero_et = (all_scores["dice_et"] == 1.0).sum()
    n_low_et = (all_scores["dice_et"] < 0.3).sum()
    report.append(f"\nCases with perfect ET Dice (both empty): {n_zero_et}")
    report.append(f"Cases with ET Dice < 0.3: {n_low_et}")

    return "\n".join(report), all_fold_results


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
        report.append(f"\nGPU Temperature: mean={df['gpu_temp_c'].mean():.1f}°C, "
                     f"max={df['gpu_temp_c'].max():.1f}°C")
    if "gpu_util_pct" in df.columns:
        report.append(f"GPU Utilization: mean={df['gpu_util_pct'].mean():.1f}%, "
                     f"min={df['gpu_util_pct'].min():.1f}%")
    if "gpu_mem_used_mb" in df.columns:
        report.append(f"GPU Memory: mean={df['gpu_mem_used_mb'].mean():.0f} MB, "
                     f"max={df['gpu_mem_used_mb'].max():.0f} MB")
    if "gpu_power_w" in df.columns:
        report.append(f"GPU Power: mean={df['gpu_power_w'].mean():.1f}W, "
                     f"max={df['gpu_power_w'].max():.1f}W")
    if "disk_used_pct" in df.columns:
        report.append(f"Disk Usage: start={df['disk_used_pct'].iloc[0]:.1f}%, "
                     f"end={df['disk_used_pct'].iloc[-1]:.1f}%")

    # Total training time estimate
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
def analyze_training_time(dfs):
    """Analyze epoch times and total training duration."""
    report = []
    report.append("\n" + "=" * 70)
    report.append("TRAINING TIME ANALYSIS")
    report.append("=" * 70)

    for fold in sorted(dfs.keys()):
        times = dfs[fold]["epoch_time_s"].values
        total_h = times.sum() / 3600
        report.append(f"\nFold {fold}: total={total_h:.2f}h, "
                     f"mean={times.mean():.1f}s/epoch, "
                     f"min={times.min():.1f}s, max={times.max():.1f}s")

    all_times = np.concatenate([dfs[f]["epoch_time_s"].values for f in sorted(dfs.keys())])
    total_all = all_times.sum() / 3600
    report.append(f"\nTotal across all 5 folds: {total_all:.1f} hours")
    report.append(f"Mean epoch time: {all_times.mean():.1f}s ± {all_times.std():.1f}s")

    # Hypothetical: how long would 3-fold (folds 0-2) have taken?
    time_3fold = sum(dfs[f]["epoch_time_s"].sum() for f in [0, 1, 2]) / 3600
    report.append(f"\nHypothetical 3-fold (0,1,2): {time_3fold:.1f}h (saved {total_all - time_3fold:.1f}h)")

    return "\n".join(report)


# ====================================================================
# 8. Recommendation / next steps
# ====================================================================
def generate_recommendations(dfs, brats_results=None):
    """Generate data-driven recommendations for next steps."""
    report = []
    report.append("\n" + "=" * 70)
    report.append("RECOMMENDATIONS & NEXT STEPS")
    report.append("=" * 70)

    # Convergence recommendation
    all_wt = np.array([dfs[f]["pseudo_dice_wt"].values for f in sorted(dfs.keys())])
    mean_wt = all_wt.mean(axis=0)

    delta_last200 = mean_wt[-1] - mean_wt[800]
    delta_last400 = mean_wt[-1] - mean_wt[600]

    report.append("\n--- Epoch count ---")
    if abs(delta_last200) < 0.002:
        report.append(f"  Last 200 epochs (800→999) gained only {delta_last200:+.4f} Dice WT.")
        report.append("  → 800 epochs would have been sufficient (saves ~20% training time).")
    elif abs(delta_last200) < 0.005:
        report.append(f"  Last 200 epochs gained {delta_last200:+.4f} — marginal but measurable.")
        report.append("  → 1000 epochs was reasonable; 800 would be close.")
    else:
        report.append(f"  Last 200 epochs gained {delta_last200:+.4f} — still meaningful.")
        report.append("  → 1000 epochs was justified.")

    # Fold count recommendation
    wt_final = all_wt[:, -1]
    std_5 = wt_final.std()
    std_3 = wt_final[:3].std()

    report.append("\n--- Fold count ---")
    report.append(f"  5-fold Dice WT std: {std_5:.4f}")
    report.append(f"  3-fold (0,1,2) Dice WT std: {std_3:.4f}")
    if std_5 < 0.015:
        report.append("  → Low inter-fold variance. 3 folds would have been adequate for this dataset.")
        report.append("  → For the thesis: use all 5 folds (already trained), but UNETR can use 3 folds.")
    else:
        report.append("  → Moderate variance. 5 folds provided a more reliable estimate.")

    # BraTS performance assessment
    if brats_results:
        all_scores = pd.concat(brats_results.values(), ignore_index=True)
        mean_wt_dice = all_scores["dice_wt"].mean()
        mean_tc_dice = all_scores["dice_tc"].mean()
        mean_et_dice = all_scores["dice_et"].mean()

        report.append("\n--- Performance assessment ---")
        report.append(f"  Dice WT: {mean_wt_dice:.4f} {'✓ Good (>0.90)' if mean_wt_dice > 0.90 else '✗ Below target'}")
        report.append(f"  Dice TC: {mean_tc_dice:.4f} {'✓ Good (>0.85)' if mean_tc_dice > 0.85 else '✗ Below target'}")
        report.append(f"  Dice ET: {mean_et_dice:.4f} {'✓ Good (>0.78)' if mean_et_dice > 0.78 else '✗ Below target'}")


    return "\n".join(report)


# ====================================================================
# MAIN
# ====================================================================
def main():
    print("=" * 60)
    print("EXP-02 Analysis: nnU-Net 5-Fold CV on BraTS 2021")
    print("=" * 60)

    # 1. Load data
    print("\n[1/7] Loading training metrics...")
    dfs = load_all_folds()
    if not dfs:
        print("ERROR: No metrics found. Aborting.")
        sys.exit(1)
    print(f"  Loaded {len(dfs)} folds, {len(dfs[0])} epochs each")

    # 2. Training curves
    print("\n[2/7] Generating training curve plots...")
    plot_training_curves(dfs)

    # 3. Convergence analysis
    print("\n[3/7] Analyzing convergence...")
    convergence_report = analyze_convergence(dfs)

    # 4. Fold variance
    print("\n[4/7] Analyzing fold variance...")
    fold_report = analyze_fold_variance(dfs)

    # 5. Post-hoc BraTS Dice
    print("\n[5/7] Computing post-hoc BraTS region Dice on NIfTI predictions...")
    brats_result = compute_brats_dice_nifti()
    brats_results = None
    if isinstance(brats_result, tuple):
        brats_report, brats_results = brats_result
    else:
        brats_report = brats_result

    # 6. GPU monitoring
    print("\n[6/7] Analyzing GPU monitoring data...")
    gpu_report = analyze_gpu_monitoring()

    # 7. Training time
    print("\n[7/7] Analyzing training time...")
    time_report = analyze_training_time(dfs)

    # Recommendations
    rec_report = generate_recommendations(dfs, brats_results)

    # Write full report
    full_report = "\n".join([
        "EXP-02 ANALYSIS REPORT",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset: BraTS 2021 (1251 cases, 5-fold CV)",
        f"Model: nnU-Net (nnUNetTrainer, 3d_fullres, 1000 epochs)",
        "",
        convergence_report,
        fold_report,
        brats_report,
        gpu_report,
        time_report,
        rec_report,
    ])

    report_path = OUT_DIR / "exp02_analysis_report.txt"
    with open(report_path, "w") as f:
        f.write(full_report)
    print(f"\n{'='*60}")
    print(f"Full report: {report_path}")
    print(f"Plots saved to: {OUT_DIR}/")
    print(f"{'='*60}")

    # Print report to stdout as well
    print("\n" + full_report)


if __name__ == "__main__":
    main()
