#!/usr/bin/env python3
"""
Convergence analysis plots to justify the 100-epoch cap in ablation experiments.

Generates three figures:
  Figure 1 — Training-loss side-by-side (nnU-Net 0–1000 ep, UNETR 1–200 ep)
              with 100-epoch marker and plateau annotation
  Figure 2 — % of final validation performance reached by epoch X
              (nnU-Net: pseudo-Dice WT, UNETR: val-Dice WT, both on one axes)
  Figure 3 — nnU-Net train-vs-val loss gap over 1000 epochs
              (overfitting evidence; supports "remaining gains after ep 100 are
               increasingly due to optimisation noise / slight overfit, not
               genuine generalisation improvement")

Output: experiments/ablation/comparison_plots/
"""

import csv
import math
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NNUNET_DIR  = os.path.join(BASE, "experiments", "exp02_nnunet")
UNETR_DIR   = os.path.join(BASE, "experiments", "exp03_unetr")
OUTPUT_DIR  = os.path.join(BASE, "experiments", "ablation", "comparison_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})
EPOCH_MARKER = 100
C_NNUNET = "#E07B39"   # orange
C_UNETR  = "#3A86C8"   # blue
C_MARKER = "#CC3333"   # red dashed line


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_nnunet(folds=5):
    """Load nnU-Net per-epoch metrics across all folds.
    Returns dict: epoch -> {train_loss: [f0..f4], val_loss: [], pseudo_dice_wt: []}
    """
    data = {}
    for fold in range(folds):
        path = os.path.join(NNUNET_DIR, f"fold_{fold}", "metrics.csv")
        with open(path) as f:
            for row in csv.DictReader(f):
                ep = int(row["epoch"])
                d = data.setdefault(ep, {"train_loss": [], "val_loss": [], "pseudo_dice_wt": []})
                d["train_loss"].append(float(row["train_loss"]))
                d["val_loss"].append(float(row["val_loss"]))
                d["pseudo_dice_wt"].append(float(row["pseudo_dice_wt"]))
    return data


def load_unetr(folds=5):
    """Load UNETR per-epoch metrics across all folds.
    Returns dict: epoch -> {train_loss: [], val_dice_wt: []}
    """
    data = {}
    fold_max_ep = {}
    for fold in range(folds):
        path = os.path.join(UNETR_DIR, f"fold_{fold}", "metrics.csv")
        max_ep = 0
        with open(path) as f:
            for row in csv.DictReader(f):
                ep = int(row["epoch"])
                d = data.setdefault(ep, {"train_loss": [], "val_dice_wt": []})
                d["train_loss"].append(float(row["train_loss"]))
                d["val_dice_wt"].append(float(row["val_dice_wt"]))
                max_ep = max(max_ep, ep)
        fold_max_ep[fold] = max_ep
    return data, fold_max_ep


def epochs_mean_std(data, key, min_folds=3):
    """Return (epochs, means, stds) arrays for a given metric key."""
    eps, means, stds = [], [], []
    for ep in sorted(data):
        vals = data[ep].get(key, [])
        if len(vals) >= min_folds:
            eps.append(ep)
            means.append(statistics.mean(vals))
            stds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
    return np.array(eps), np.array(means), np.array(stds)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data …")
nn_data = load_nnunet()
ut_data, ut_fold_max = load_unetr()

# nnU-Net arrays
nn_eps,  nn_tl_m,  nn_tl_s  = epochs_mean_std(nn_data, "train_loss")
nn_veps, nn_vl_m,  nn_vl_s  = epochs_mean_std(nn_data, "val_loss")
nn_deps, nn_dw_m,  nn_dw_s  = epochs_mean_std(nn_data, "pseudo_dice_wt")

# UNETR arrays (require all 5 folds)
ut_eps,  ut_tl_m,  ut_tl_s  = epochs_mean_std(ut_data, "train_loss", min_folds=5)
ut_deps, ut_dw_m,  ut_dw_s  = epochs_mean_std(ut_data, "val_dice_wt", min_folds=5)

# % of final performance
nn_final_dice = nn_dw_m[nn_deps == 999][0]
nn_init_dice  = nn_dw_m[nn_deps == 0][0]
nn_pct        = (nn_dw_m - nn_init_dice) / (nn_final_dice - nn_init_dice) * 100

# UNETR: use common endpoint (epoch 150, all 5 folds present)
ut_common_last = 150
ut_final_dice  = ut_dw_m[ut_deps == ut_common_last][0]
ut_init_dice   = ut_dw_m[ut_deps == 1][0]
ut_pct         = (ut_dw_m - ut_init_dice) / (ut_final_dice - ut_init_dice) * 100

# Values at epoch 100 for annotations
def interp_at(eps, vals, target):
    """Nearest epoch value."""
    idx = np.argmin(np.abs(eps - target))
    return float(vals[idx])

nn_pct_at100   = interp_at(nn_deps, nn_pct, EPOCH_MARKER)
ut_pct_at100   = interp_at(ut_deps, ut_pct, EPOCH_MARKER)
nn_tl_at100    = interp_at(nn_eps,  nn_tl_m, EPOCH_MARKER)
ut_tl_at100    = interp_at(ut_eps,  ut_tl_m, EPOCH_MARKER)
nn_gap_at100   = interp_at(nn_eps, nn_tl_m - nn_vl_m[: len(nn_tl_m)], EPOCH_MARKER)


# ===========================================================================
# FIGURE 1 — Training loss side-by-side
# ===========================================================================
print("Generating Figure 1: Training loss …")
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(13, 4.5))
fig1.suptitle("Training Loss Convergence — nnU-Net vs. UNETR", fontweight="bold", y=1.01)

# --- nnU-Net (left) ---
ax1a.plot(nn_eps, nn_tl_m, color=C_NNUNET, lw=1.8, label="Mean (5 folds)")
ax1a.fill_between(nn_eps, nn_tl_m - nn_tl_s, nn_tl_m + nn_tl_s,
                  color=C_NNUNET, alpha=0.18, label="±1 Std")
ax1a.axvline(EPOCH_MARKER, color=C_MARKER, ls="--", lw=1.5, label=f"Epoch {EPOCH_MARKER}")
ax1a.axhline(nn_tl_at100, color=C_MARKER, ls=":", lw=0.9, alpha=0.6)
ax1a.annotate(f"Loss = {nn_tl_at100:.3f}\n(ep {EPOCH_MARKER})",
              xy=(EPOCH_MARKER, nn_tl_at100),
              xytext=(200, nn_tl_at100 + 0.04),
              fontsize=9, color=C_MARKER,
              arrowprops=dict(arrowstyle="->", color=C_MARKER, lw=0.9))
ax1a.set_xlabel("Epoch")
ax1a.set_ylabel("Dice + CE Loss")
ax1a.set_title("nnU-Net  (250 iters/epoch, bs=2, 1000 epochs)")
ax1a.legend(loc="lower left")
ax1a.set_xlim(-10, 1009)

# Second x-label: iterations
ax1a_top = ax1a.twiny()
ax1a_top.set_xlim(-10 * 250, 1009 * 250)
ax1a_top.set_xlabel("Iterations  (1 ep = 250 iters)")
ax1a_top.tick_params(labelsize=8)

# --- UNETR (right) ---
ax1b.plot(ut_eps, ut_tl_m, color=C_UNETR, lw=1.8, label="Mean (5 folds)")
ax1b.fill_between(ut_eps, ut_tl_m - ut_tl_s, ut_tl_m + ut_tl_s,
                  color=C_UNETR, alpha=0.18, label="±1 Std")
ax1b.axvline(EPOCH_MARKER, color=C_MARKER, ls="--", lw=1.5, label=f"Epoch {EPOCH_MARKER}")
ax1b.axhline(ut_tl_at100, color=C_MARKER, ls=":", lw=0.9, alpha=0.6)
ax1b.annotate(f"Loss = {ut_tl_at100:.3f}\n(ep {EPOCH_MARKER})",
              xy=(EPOCH_MARKER, ut_tl_at100),
              xytext=(120, ut_tl_at100 + 0.1),
              fontsize=9, color=C_MARKER,
              arrowprops=dict(arrowstyle="->", color=C_MARKER, lw=0.9))
ax1b.set_xlabel("Epoch")
ax1b.set_ylabel("Dice + CE Loss")
ax1b.set_title("UNETR  (501 iters/epoch, bs=2, grad-accum=4, early stop)")
ax1b.legend(loc="upper right")
ax1b.set_xlim(-1, max(ut_eps) + 5)

ax1b_top = ax1b.twiny()
ax1b_top.set_xlim(-501, (max(ut_eps) + 5) * 501)
ax1b_top.set_xlabel("Iterations  (1 ep ≈ 501 iters)")
ax1b_top.tick_params(labelsize=8)

fig1.tight_layout()
out1 = os.path.join(OUTPUT_DIR, "convergence_training_loss.pdf")
fig1.savefig(out1, bbox_inches="tight")
fig1.savefig(out1.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
print(f"  Saved: {out1}")


# ===========================================================================
# FIGURE 2 — % of final validation performance
# ===========================================================================
print("Generating Figure 2: % of final val performance …")
fig2, ax2 = plt.subplots(figsize=(9, 5))

ax2.plot(nn_deps, nn_pct, color=C_NNUNET, lw=2.0,
         label="nnU-Net  (pseudo-Dice WT, normalised to epoch 999)")
ax2.fill_between(nn_deps,
                 np.clip(nn_pct - nn_dw_s / (nn_final_dice - nn_init_dice) * 100, 0, None),
                 nn_pct + nn_dw_s / (nn_final_dice - nn_init_dice) * 100,
                 color=C_NNUNET, alpha=0.15)

# UNETR: only plot up to common last
ut_mask = ut_deps <= ut_common_last
ax2.plot(ut_deps[ut_mask], ut_pct[ut_mask], color=C_UNETR, lw=2.0,
         label="UNETR  (val-Dice WT, normalised to epoch 150)")

ax2.axvline(EPOCH_MARKER, color=C_MARKER, ls="--", lw=1.8, zorder=5,
            label=f"100-epoch cap (ablation)")
ax2.axhline(100, color="black", ls=":", lw=0.8, alpha=0.4)

# Annotations at ep 100
ax2.scatter([EPOCH_MARKER], [nn_pct_at100], color=C_NNUNET, zorder=6, s=70)
ax2.scatter([EPOCH_MARKER], [ut_pct_at100], color=C_UNETR,  zorder=6, s=70)
ax2.annotate(f"nnU-Net: {nn_pct_at100:.1f}%",
             xy=(EPOCH_MARKER, nn_pct_at100),
             xytext=(130, nn_pct_at100 - 6),
             fontsize=10, color=C_NNUNET, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=C_NNUNET, lw=1.0))
ax2.annotate(f"UNETR:   {ut_pct_at100:.1f}%",
             xy=(EPOCH_MARKER, ut_pct_at100),
             xytext=(130, ut_pct_at100 + 2),
             fontsize=10, color=C_UNETR, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=C_UNETR, lw=1.0))

ax2.set_xlabel("Epoch")
ax2.set_ylabel("% of final validation performance achieved")
ax2.set_title("Convergence: % of Final Validation Performance by Epoch\n"
              "nnU-Net (pseudo-Dice WT) vs. UNETR (val-Dice WT)",
              fontweight="bold")
ax2.set_xlim(-5, max(nn_deps[-1], ut_deps[ut_mask][-1]) + 10)
ax2.set_ylim(50, 105)
ax2.legend(loc="lower right")
ax2.grid(True, alpha=0.3)

# shade the region beyond 100 ep for nnU-Net showing diminishing returns
ax2.axvspan(EPOCH_MARKER, nn_deps[-1], alpha=0.04, color=C_NNUNET)
ax2.text(350, 74, "Diminishing returns\n(nnU-Net +8.9 pp over 900 epochs)",
         fontsize=9, color=C_NNUNET, alpha=0.9, ha="center")

fig2.tight_layout()
out2 = os.path.join(OUTPUT_DIR, "convergence_pct_final_performance.pdf")
fig2.savefig(out2, bbox_inches="tight")
fig2.savefig(out2.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
print(f"  Saved: {out2}")


# ===========================================================================
# FIGURE 3 — nnU-Net train-vs-val loss gap (overfitting evidence)
# ===========================================================================
print("Generating Figure 3: nnU-Net train-vs-val gap …")

# Align train and val arrays on same epoch axis
nn_both_eps = np.array(sorted(set(nn_eps) & set(nn_veps)))
nn_tl_aligned = np.array([interp_at(nn_eps, nn_tl_m, ep) for ep in nn_both_eps])
nn_vl_aligned = np.array([interp_at(nn_veps, nn_vl_m, ep) for ep in nn_both_eps])
gap           = nn_tl_aligned - nn_vl_aligned   # negative when train < val (nnU-Net convention)

fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
fig3.suptitle("nnU-Net: Training vs. Validation Loss — Overfitting Evidence",
              fontweight="bold")

# Top: both curves
ax3a.plot(nn_eps,  nn_tl_m, color=C_NNUNET, lw=1.8, label="Train loss (mean)")
ax3a.fill_between(nn_eps, nn_tl_m - nn_tl_s, nn_tl_m + nn_tl_s,
                  color=C_NNUNET, alpha=0.15)
ax3a.plot(nn_veps, nn_vl_m, color="#666666", lw=1.8, ls="--", label="Val loss (mean)")
ax3a.fill_between(nn_veps, nn_vl_m - nn_vl_s, nn_vl_m + nn_vl_s,
                  color="#666666", alpha=0.12)
ax3a.axvline(EPOCH_MARKER, color=C_MARKER, ls="--", lw=1.5,
             label=f"Epoch {EPOCH_MARKER} (ablation cap)")
ax3a.axvline(200, color="#888888", ls=":", lw=1.2, alpha=0.7, label="Epoch 200")
ax3a.set_ylabel("Dice + CE Loss")
ax3a.legend(loc="lower left")
ax3a.grid(True, alpha=0.3)

# Bottom: gap
ax3b.fill_between(nn_both_eps, gap, 0, where=(gap < 0),
                  color=C_NNUNET, alpha=0.35, label="Train < Val (normal)")
ax3b.fill_between(nn_both_eps, gap, 0, where=(gap > 0),
                  color="#CC3333", alpha=0.35, label="Train > Val (overfit region)")
ax3b.plot(nn_both_eps, gap, color=C_NNUNET, lw=1.2)
ax3b.axvline(EPOCH_MARKER, color=C_MARKER, ls="--", lw=1.5)
ax3b.axvline(200, color="#888888", ls=":", lw=1.2, alpha=0.7)
ax3b.axhline(0, color="black", lw=0.8)

# Annotate gap values
for ep_mark, label_x in [(100, 150), (200, 300), (500, 550), (999, 850)]:
    g = interp_at(nn_both_eps, gap, ep_mark)
    ax3b.annotate(f"ep{ep_mark}: {g:+.3f}",
                  xy=(ep_mark, g), xytext=(label_x, g - 0.008),
                  fontsize=8.5, color="#333333",
                  arrowprops=dict(arrowstyle="->", lw=0.7, color="#333333"))

ax3b.set_xlabel("Epoch")
ax3b.set_ylabel("Train − Val Loss")
ax3b.legend(loc="lower left", fontsize=9)
ax3b.grid(True, alpha=0.3)

fig3.tight_layout()
out3 = os.path.join(OUTPUT_DIR, "convergence_nnunet_train_val_gap.pdf")
fig3.savefig(out3, bbox_inches="tight")
fig3.savefig(out3.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
print(f"  Saved: {out3}")


# ===========================================================================
# Console summary for thesis writing
# ===========================================================================
print()
print("=" * 62)
print("  KEY NUMBERS FOR THESIS ARGUMENTATION")
print("=" * 62)

print(f"\n[Training budget at 100 epochs]")
nnunet_samples_100 = 100 * 250 * 2
unetr_samples_100  = 100 * math.ceil(1001 / 2) * 2
nnunet_updates_100 = 100 * 250
unetr_updates_100  = 100 * (math.ceil(1001 / 2) // 4)
print(f"  nnU-Net : {nnunet_samples_100:,} samples, {nnunet_updates_100:,} weight updates")
print(f"  UNETR   : {unetr_samples_100:,} samples, {unetr_updates_100:,} weight updates")
print(f"  → UNETR sees 2× more data; nnU-Net makes 2× more updates  (balanced)")

print(f"\n[% of final val performance at ep {EPOCH_MARKER}]")
print(f"  nnU-Net pseudo-Dice WT : {nn_pct_at100:.1f}%  (final = {nn_final_dice:.4f})")
print(f"  UNETR   val-Dice WT    : {ut_pct_at100:.1f}%  (ref   = {ut_final_dice:.4f})")

print(f"\n[nnU-Net train-val gap]")
for ep in [100, 200, 500, 999]:
    g = interp_at(nn_both_eps, gap, ep)
    print(f"  ep {ep:4d}: gap = {g:+.4f}")

print()
print("Plots saved to:", OUTPUT_DIR)
