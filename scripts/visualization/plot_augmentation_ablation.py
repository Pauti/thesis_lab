#!/usr/bin/env python3
"""
RQ2 - Augmentation Ablation: comparison plots.

Unlike plot_ablation_comparison.py (which only compares nnU-Net vs UNETR at the
default tier), this script analyses the *augmentation effect* across all four
tiers (disabled / reduced / default / extended) for both models, using the REAL
per-case ensemble Dice scores (125 hold-out cases) - no synthetic distributions.

Statistics:
  - Friedman test across the 4 tiers (paired over 125 cases) -> "does tier matter?"
  - Wilcoxon signed-rank of each tier vs. the `default` tier (paired).

Figures written to experiments/ablation/comparison_plots/ (PNG @300dpi + PDF):
  aug_01_tier_means        grouped bars, tier effect per region/model
  aug_02_tier_distributions  real per-case boxplots across tiers
  aug_03_delta_vs_default  effect size vs default with significance stars
  aug_04_summary_table     means +/- SD and Friedman p-values
"""

import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import friedmanchisquare, wilcoxon

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 10

BASE = Path("/home/paul/repositories/thesis_lab/experiments/ablation")
OUT = BASE / "comparison_plots"
OUT.mkdir(exist_ok=True, parents=True)

MODELS = ["nnunet", "unetr"]
MODEL_LABEL = {"nnunet": "nnU-Net", "unetr": "UNETR"}
TIERS = ["disabled", "reduced", "default", "extended"]
TIER_LABEL = {
    "disabled": "Disabled\n(NoAug)",
    "reduced": "Reduced\n(LightAug)",
    "default": "Default\n(DefaultAug)",
    "extended": "Extended\n(HeavyAug)",
}
# sequential palette: light (no aug) -> dark (heavy aug)
TIER_COLOR = {
    "disabled": "#cfe8f3",
    "reduced": "#7fc0dd",
    "default": "#2E86AB",
    "extended": "#16455c",
}
REGIONS = ["dice_wt", "dice_tc", "dice_et"]
REGION_LABEL = {"dice_wt": "WT", "dice_tc": "TC", "dice_et": "ET"}
REGION_FULL = {
    "dice_wt": "Whole Tumor (WT)",
    "dice_tc": "Tumor Core (TC)",
    "dice_et": "Enhancing Tumor (ET)",
}


def load(model, tier):
    """Per-case ensemble Dice, keyed by case_id."""
    path = BASE / f"abl_{model}_{tier}" / "test_predictions" / "ensemble_results.csv"
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            out[row["case_id"]] = {r: float(row[r]) for r in REGIONS}
    return out


DATA = {(m, t): load(m, t) for m in MODELS for t in TIERS}

# paired case list (verify identical across all experiments)
CASES = sorted(DATA[("nnunet", "default")])
for (m, t), d in DATA.items():
    assert sorted(d) == CASES, f"case-id mismatch in {m}/{t}"
N = len(CASES)


def vec(model, tier, region):
    return np.array([DATA[(model, tier)][c][region] for c in CASES])


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


# ---------------------------------------------------------------------------
# Figure 1: tier means (grouped bars) - one panel per model
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for ax, model in zip(axes, MODELS):
    x = np.arange(len(REGIONS))
    width = 0.2
    for i, tier in enumerate(TIERS):
        means = [vec(model, tier, r).mean() for r in REGIONS]
        stds = [vec(model, tier, r).std() for r in REGIONS]
        offset = (i - (len(TIERS) - 1) / 2) * width
        bars = ax.bar(
            x + offset, means, width, yerr=stds, capsize=3,
            label=TIER_LABEL[tier].replace("\n", " "),
            color=TIER_COLOR[tier], alpha=0.95,
            error_kw=dict(alpha=0.4, lw=1),
        )
        for bar, m_ in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, m_ + 0.002,
                    f"{m_:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.set_title(MODEL_LABEL[model], fontweight="bold", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([REGION_FULL[r] for r in REGIONS])
    ax.set_xlabel("Tumor Region", fontweight="bold")
    ax.set_ylim([0.78, 0.98])
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("Ensemble Dice (mean ± SD over 125 cases)", fontweight="bold")
axes[1].legend(title="Augmentation Tier", fontsize=9, title_fontsize=10,
               loc="lower right", framealpha=0.95)
fig.suptitle("RQ2 - Augmentation Tier Effect on Test-Set Performance",
             fontweight="bold", fontsize=14, y=1.00)
plt.tight_layout()
for ext in ("png", "pdf"):
    plt.savefig(OUT / f"aug_01_tier_means.{ext}", dpi=300, bbox_inches="tight")
plt.close()
print("✓ aug_01_tier_means")

# ---------------------------------------------------------------------------
# Figure 2: real per-case distributions (boxplots) - 2 rows x 3 cols
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
for r_idx, region in enumerate(REGIONS):
    for m_idx, model in enumerate(MODELS):
        ax = axes[m_idx, r_idx]
        box_data = [vec(model, t, region) for t in TIERS]
        bp = ax.boxplot(box_data, patch_artist=True, showmeans=True, meanline=True,
                        widths=0.6,
                        medianprops=dict(color="black", lw=1.5),
                        meanprops=dict(color="red", lw=1.5, ls="--"),
                        flierprops=dict(marker="o", ms=2, alpha=0.35))
        for patch, tier in zip(bp["boxes"], TIERS):
            patch.set_facecolor(TIER_COLOR[tier])
            patch.set_alpha(0.85)
        # Friedman p across tiers for this model/region
        _, p = friedmanchisquare(*box_data)
        ax.set_title(f"{MODEL_LABEL[model]} - {REGION_LABEL[region]}   "
                     f"(Friedman p={p:.3f})", fontweight="bold", fontsize=10)
        ax.set_xticks(range(1, len(TIERS) + 1))
        ax.set_xticklabels([TIER_LABEL[t] for t in TIERS], fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        if r_idx == 0:
            ax.set_ylabel(f"{MODEL_LABEL[model]}\nDice", fontweight="bold")
legend_elems = [
    plt.Line2D([0], [0], color="black", lw=1.5, label="Median"),
    plt.Line2D([0], [0], color="red", lw=1.5, ls="--", label="Mean"),
]
fig.legend(handles=legend_elems, loc="upper right", fontsize=9, framealpha=0.95)
fig.suptitle("RQ2 - Per-Case Dice Distributions across Augmentation Tiers "
             "(125 hold-out cases, real ensemble scores)",
             fontweight="bold", fontsize=14, y=1.00)
plt.tight_layout()
for ext in ("png", "pdf"):
    plt.savefig(OUT / f"aug_02_tier_distributions.{ext}", dpi=300, bbox_inches="tight")
plt.close()
print("✓ aug_02_tier_distributions")

# ---------------------------------------------------------------------------
# Figure 3: effect size vs default with significance (Wilcoxon)
# ---------------------------------------------------------------------------
others = [t for t in TIERS if t != "default"]
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for ax, model in zip(axes, MODELS):
    x = np.arange(len(others))
    width = 0.25
    for ri, region in enumerate(REGIONS):
        deltas, ps = [], []
        base = vec(model, "default", region)
        for tier in others:
            a = vec(model, tier, region)
            deltas.append(a.mean() - base.mean())
            ps.append(wilcoxon(a, base).pvalue)
        offset = (ri - 1) * width
        bars = ax.bar(x + offset, deltas, width,
                      label=REGION_LABEL[region], alpha=0.9)
        for bar, d_, p_ in zip(bars, deltas, ps):
            va = "bottom" if d_ >= 0 else "top"
            y = d_ + (0.0006 if d_ >= 0 else -0.0006)
            ax.text(bar.get_x() + bar.get_width() / 2, y,
                    f"{stars(p_)}", ha="center", va=va, fontsize=8,
                    fontweight="bold")
    ax.axhline(0, color="black", lw=1)
    ax.set_title(MODEL_LABEL[model], fontweight="bold", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([TIER_LABEL[t] for t in others])
    ax.set_xlabel("Augmentation Tier (vs. Default)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Region", fontsize=9)
axes[0].set_ylabel("Δ Mean Dice vs. Default tier", fontweight="bold")
fig.suptitle("RQ2 - Augmentation Effect Size relative to Default "
             "(Wilcoxon signed-rank;  * p<0.05  ** p<0.01  *** p<0.001)",
             fontweight="bold", fontsize=13, y=1.00)
plt.tight_layout()
for ext in ("png", "pdf"):
    plt.savefig(OUT / f"aug_03_delta_vs_default.{ext}", dpi=300, bbox_inches="tight")
plt.close()
print("✓ aug_03_delta_vs_default")

# ---------------------------------------------------------------------------
# Figure 4: summary table (means +/- SD, best per region bold, Friedman p)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 6))
ax.axis("off")

header = ["Model", "Tier", "WT Dice", "TC Dice", "ET Dice"]
rows, cell_meta = [header], []
for model in MODELS:
    # best tier per region (for bold highlight)
    best = {r: max(TIERS, key=lambda t: vec(model, t, r).mean()) for r in REGIONS}
    for tier in TIERS:
        row = [MODEL_LABEL[model], TIER_LABEL[tier].replace("\n", " ")]
        meta = []
        for r in REGIONS:
            a = vec(model, tier, r)
            row.append(f"{a.mean():.4f} ± {a.std():.3f}")
            meta.append(tier == best[r])
        rows.append(row)
        cell_meta.append(meta)
    # Friedman p row
    fried = ["", "Friedman p"]
    for r in REGIONS:
        _, p = friedmanchisquare(*[vec(model, t, r) for t in TIERS])
        fried.append(f"p={p:.4f} ({stars(p)})")
    rows.append(fried)
    cell_meta.append([None, None, None])

table = ax.table(cellText=rows, cellLoc="center", loc="center",
                 colWidths=[0.13, 0.17, 0.23, 0.23, 0.23])
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1, 2.0)

# header styling
for c in range(len(header)):
    table[(0, c)].set_facecolor("#4A4A4A")
    table[(0, c)].set_text_props(weight="bold", color="white")

# body styling
row_i = 1
for model in MODELS:
    base_color = "#D6EAF8" if model == "nnunet" else "#F8D7DA"
    for t_i, tier in enumerate(TIERS):
        table[(row_i, 0)].set_facecolor(base_color)
        table[(row_i, 1)].set_facecolor(base_color)
        for c in range(2, 5):
            if cell_meta[row_i - 1][c - 2]:
                table[(row_i, c)].set_text_props(weight="bold", color="darkgreen")
        row_i += 1
    # Friedman row
    for c in range(len(header)):
        table[(row_i, c)].set_facecolor("#EDEDED")
        table[(row_i, c)].set_text_props(style="italic")
    row_i += 1

plt.title("RQ2 - Augmentation Ablation Summary  "
          "(Ensemble Dice, 125 cases; best tier per region in green)",
          fontweight="bold", fontsize=12, pad=18)
for ext in ("png", "pdf"):
    plt.savefig(OUT / f"aug_04_summary_table.{ext}", dpi=300, bbox_inches="tight")
plt.close()
print("✓ aug_04_summary_table")

print(f"\nAll augmentation-ablation plots written to {OUT}")
