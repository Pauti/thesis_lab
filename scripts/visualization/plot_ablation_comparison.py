#!/usr/bin/env python3
"""
Create comparison plots for augmentation ablation study.
Focuses on Test Set performance and augmentation impact.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

# Output directory
output_dir = Path("/home/paul/repositories/thesis_lab/experiments/ablation/comparison_plots")
output_dir.mkdir(exist_ok=True, parents=True)

# Load test set results
with open("/home/paul/repositories/thesis_lab/experiments/ablation/abl_nnunet_default/test_predictions/summary.json") as f:
    nnunet_default = json.load(f)

with open("/home/paul/repositories/thesis_lab/experiments/ablation/abl_unetr_default/test_predictions/summary.json") as f:
    unetr_default = json.load(f)

# Extract ensemble results
nnunet_ens = nnunet_default['ensemble_results']
unetr_ens = unetr_default['ensemble_results']

# ============================================================================
# Plot 1: Test Set Performance Comparison (Bar plot with error bars)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

regions = ['WT', 'TC', 'ET']
x = np.arange(len(regions))
width = 0.35

nnunet_means = [nnunet_ens['mean_wt_dice'], nnunet_ens['mean_tc_dice'], nnunet_ens['mean_et_dice']]
nnunet_stds = [nnunet_ens['std_wt_dice'], nnunet_ens['std_tc_dice'], nnunet_ens['std_et_dice']]

unetr_means = [unetr_ens['mean_wt_dice'], unetr_ens['mean_tc_dice'], unetr_ens['mean_et_dice']]
unetr_stds = [unetr_ens['std_wt_dice'], unetr_ens['std_tc_dice'], unetr_ens['std_et_dice']]

bars1 = ax.bar(x - width/2, nnunet_means, width, yerr=nnunet_stds, 
               label='nnU-Net', alpha=0.8, capsize=5, color='#2E86AB')
bars2 = ax.bar(x + width/2, unetr_means, width, yerr=unetr_stds,
               label='UNETR', alpha=0.8, capsize=5, color='#A23B72')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)

# Add difference annotations
for i, region in enumerate(regions):
    diff = nnunet_means[i] - unetr_means[i]
    pct_diff = (diff / unetr_means[i]) * 100
    y_pos = max(nnunet_means[i], unetr_means[i]) + max(nnunet_stds[i], unetr_stds[i]) + 0.01
    ax.text(x[i], y_pos, f'Δ={diff:+.4f}\n({pct_diff:+.1f}%)', 
            ha='center', fontsize=8, color='darkgreen' if diff > 0 else 'darkred',
            fontweight='bold')

ax.set_ylabel('Dice Score', fontweight='bold')
ax.set_xlabel('Tumor Region', fontweight='bold')
ax.set_title('Test Set Performance: nnU-Net vs UNETR (Default Augmentation)\n125 Hold-Out Cases, 3-Fold Ensemble', 
             fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(['Whole Tumor (WT)', 'Tumor Core (TC)', 'Enhancing Tumor (ET)'])
ax.legend(fontsize=11, framealpha=0.9)
ax.set_ylim([0.75, 1.0])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "01_test_set_comparison.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / '01_test_set_comparison.png'}")
plt.close()

# ============================================================================
# Plot 2: Per-Fold Test Set Performance (Grouped bar plot)
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (region, region_name) in enumerate([('wt', 'Whole Tumor'), ('tc', 'Tumor Core'), ('et', 'Enhancing Tumor')]):
    ax = axes[idx]
    
    folds = [0, 1, 2]
    x = np.arange(len(folds))
    width = 0.35
    
    nnunet_fold_scores = [fold[f'mean_{region}_dice'] for fold in nnunet_default['per_fold_results']]
    unetr_fold_scores = [fold[f'mean_{region}_dice'] for fold in unetr_default['per_fold_results']]
    
    bars1 = ax.bar(x - width/2, nnunet_fold_scores, width, label='nnU-Net', alpha=0.8, color='#2E86AB')
    bars2 = ax.bar(x + width/2, unetr_fold_scores, width, label='UNETR', alpha=0.8, color='#A23B72')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Dice Score', fontweight='bold')
    ax.set_xlabel('Fold', fontweight='bold')
    ax.set_title(f'{region_name} ({region.upper()})', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Fold 0', 'Fold 1', 'Fold 2'])
    ax.legend(fontsize=9)
    ax.set_ylim([0.80, 0.95])
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Per-Fold Test Set Performance (125 Hold-Out Cases)', fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(output_dir / "02_per_fold_test_comparison.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / '02_per_fold_test_comparison.png'}")
plt.close()

# ============================================================================
# Plot 3: Variability Comparison (Box plots)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Prepare data for box plot
data_to_plot = []
labels = []
colors = []

for region, region_name in [('wt', 'WT'), ('tc', 'TC'), ('et', 'ET')]:
    # Get per-case scores (using std as proxy for distribution)
    nnunet_mean = nnunet_ens[f'mean_{region}_dice']
    nnunet_std = nnunet_ens[f'std_{region}_dice']
    unetr_mean = unetr_ens[f'mean_{region}_dice']
    unetr_std = unetr_ens[f'std_{region}_dice']
    
    # Create synthetic distributions (normal approximation)
    np.random.seed(42)
    nnunet_dist = np.random.normal(nnunet_mean, nnunet_std, 125)
    unetr_dist = np.random.normal(unetr_mean, unetr_std, 125)
    
    data_to_plot.extend([nnunet_dist, unetr_dist])
    labels.extend([f'{region_name}\nnnU-Net', f'{region_name}\nUNETR'])
    colors.extend(['#2E86AB', '#A23B72'])

bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                showmeans=True, meanline=True,
                medianprops=dict(color='black', linewidth=2),
                meanprops=dict(color='red', linewidth=2, linestyle='--'))

# Color boxes
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('Dice Score', fontweight='bold')
ax.set_title('Test Set Score Distributions (125 Cases, Approximated from Mean ± SD)\nBlack=Median, Red Dashed=Mean', 
             fontweight='bold', fontsize=12)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0.5, 1.0])

plt.tight_layout()
plt.savefig(output_dir / "03_variability_comparison.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / '03_variability_comparison.png'}")
plt.close()

# ============================================================================
# Plot 4: Summary Statistics Table (as image)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')

# Create table data
table_data = [
    ['Model', 'WT Dice', 'TC Dice', 'ET Dice', 'Mean Dice', 'Training Time\n(per fold)'],
    ['nnU-Net', 
     f"{nnunet_ens['mean_wt_dice']:.4f} ± {nnunet_ens['std_wt_dice']:.3f}",
     f"{nnunet_ens['mean_tc_dice']:.4f} ± {nnunet_ens['std_tc_dice']:.3f}",
     f"{nnunet_ens['mean_et_dice']:.4f} ± {nnunet_ens['std_et_dice']:.3f}",
     f"{np.mean(nnunet_means):.4f}",
     '~1.5h'],
    ['UNETR',
     f"{unetr_ens['mean_wt_dice']:.4f} ± {unetr_ens['std_wt_dice']:.3f}",
     f"{unetr_ens['mean_tc_dice']:.4f} ± {unetr_ens['std_tc_dice']:.3f}",
     f"{unetr_ens['mean_et_dice']:.4f} ± {unetr_ens['std_et_dice']:.3f}",
     f"{np.mean(unetr_means):.4f}",
     '~15.8h'],
    ['Δ (nnU-Net - UNETR)',
     f"{nnunet_ens['mean_wt_dice'] - unetr_ens['mean_wt_dice']:+.4f}",
     f"{nnunet_ens['mean_tc_dice'] - unetr_ens['mean_tc_dice']:+.4f}",
     f"{nnunet_ens['mean_et_dice'] - unetr_ens['mean_et_dice']:+.4f}",
     f"{np.mean(nnunet_means) - np.mean(unetr_means):+.4f}",
     '10.5× faster']
]

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.15, 0.18, 0.18, 0.18, 0.15, 0.16])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for i in range(6):
    table[(0, i)].set_facecolor('#4A4A4A')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style model rows
table[(1, 0)].set_facecolor('#D6EAF8')
table[(2, 0)].set_facecolor('#F8D7DA')
table[(3, 0)].set_facecolor('#FFF3CD')

# Color difference row cells based on sign
for col in range(1, 5):
    cell = table[(3, col)]
    val = float(table_data[3][col].split()[0])
    if val > 0:
        cell.set_facecolor('#C3E6CB')  # Green for positive
        cell.set_text_props(weight='bold', color='darkgreen')
    else:
        cell.set_facecolor('#F5C6CB')  # Red for negative
        cell.set_text_props(weight='bold', color='darkred')

plt.title('Test Set Performance Summary (Default Augmentation, 125 Cases)', 
          fontweight='bold', fontsize=13, pad=20)
plt.savefig(output_dir / "04_summary_table.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / '04_summary_table.png'}")
plt.close()

# ============================================================================
# Plot 5: Radar chart comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

# Data
categories = ['WT', 'TC', 'ET']
N = len(categories)

# Angles for each axis
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

# Values
nnunet_values = nnunet_means + [nnunet_means[0]]
unetr_values = unetr_means + [unetr_means[0]]

# Plot
ax.plot(angles, nnunet_values, 'o-', linewidth=2, label='nnU-Net', color='#2E86AB')
ax.fill(angles, nnunet_values, alpha=0.25, color='#2E86AB')

ax.plot(angles, unetr_values, 'o-', linewidth=2, label='UNETR', color='#A23B72')
ax.fill(angles, unetr_values, alpha=0.25, color='#A23B72')

# Labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
ax.set_ylim(0.80, 0.95)
ax.set_yticks([0.80, 0.85, 0.90, 0.95])
ax.set_yticklabels(['0.80', '0.85', '0.90', '0.95'], fontsize=10)
ax.grid(True)

plt.title('Test Set Performance Radar Chart\nDefault Augmentation (125 Cases)', 
          fontweight='bold', fontsize=13, pad=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / "05_radar_comparison.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / '05_radar_comparison.png'}")
plt.close()

print("\n" + "="*70)
print("All comparison plots created successfully!")
print(f"Output directory: {output_dir}")
print("="*70)
