#!/usr/bin/env python3
"""
plot_dice_distribution.py

Per-case Dice distributions (RQ1, 5-fold CV, n=1251) for all three regions
(WT | TC | ET), nnU-Net vs UNETR. Violin (density) + inner box (median/IQR) +
jittered failure cases below 0.5. Matches the convergence-plot style.

Usage:
    /path/to/venv/bin/python figures/plot_dice_distribution.py
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── fonts (identical to plot_convergence_loss.py) ─────────────────────────────
mpl.rcParams.update({
    'font.family'      : 'serif',
    'font.serif'       : ['STIX Two Text', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset' : 'stix',
    'font.size'        : 12,
    'axes.labelsize'   : 12,
    'xtick.labelsize'  : 11,
    'ytick.labelsize'  : 11,
    'legend.fontsize'  : 11,
    'axes.linewidth'   : 0.8,
    'xtick.direction'  : 'out',
    'ytick.direction'  : 'out',
    'xtick.major.size' : 3.5,
    'ytick.major.size' : 3.5,
    'grid.color'       : 'gray',
    'grid.alpha'       : 0.25,
    'grid.linewidth'   : 0.4,
    'pdf.fonttype'     : 42,
})

C_NN = '#E07B39'   # orange – nnU-Net
C_UT = '#3A86C8'   # blue   – UNETR
C_MK = '#CC3333'   # red    – failure threshold

HERE = Path(__file__).parent
REPO = HERE.parent

_nn = pd.read_csv(REPO / 'experiments/exp02_nnunet/analysis/all_folds_brats_dice.csv')
_ut = pd.read_csv(REPO / 'experiments/exp03_unetr/analysis/all_folds_brats_dice_nifti.csv')

REGIONS = [('wt', 'WT (whole tumor)'), ('tc', 'TC (tumor core)'),
           ('et', 'ET (enhancing tumor)')]
MODELS = [('nnU-Net', _nn, C_NN, 1), ('UNETR', _ut, C_UT, 2)]
FAIL = 0.5

n_cases = _nn['dice_et'].dropna().size

fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.8), sharey=True)
fig.subplots_adjust(top=0.86, bottom=0.12, left=0.075, right=0.985, wspace=0.10)

rng = np.random.default_rng(0)

for ax, (rkey, rlabel) in zip(axes, REGIONS):
    ax.grid(True, axis='y', which='major', zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(FAIL, color=C_MK, linestyle='--', linewidth=1.0, zorder=2)

    data = [m[1][f'dice_{rkey}'].dropna().to_numpy() for m in MODELS]
    pos = [m[3] for m in MODELS]
    cols = [m[2] for m in MODELS]

    vp = ax.violinplot(data, positions=pos, widths=0.8,
                       showmeans=False, showmedians=False, showextrema=False)
    for body, col in zip(vp['bodies'], cols):
        body.set_facecolor(col); body.set_alpha(0.28)
        body.set_edgecolor(col); body.set_linewidth(0.9); body.set_zorder(1)

    bp = ax.boxplot(data, positions=pos, widths=0.16, patch_artist=True,
                    showfliers=False, zorder=3,
                    medianprops=dict(color='black', linewidth=1.2),
                    whiskerprops=dict(color='#444', linewidth=0.9),
                    capprops=dict(color='#444', linewidth=0.9),
                    boxprops=dict(linewidth=0.9))
    for patch, col in zip(bp['boxes'], cols):
        patch.set_facecolor('white'); patch.set_edgecolor(col)

    for d, p, col in zip(data, pos, cols):
        fails = d[d < FAIL]
        jit = rng.uniform(-0.11, 0.11, size=fails.size)
        ax.scatter(np.full(fails.size, p) + jit, fails, s=7,
                   facecolor=col, edgecolor='none', alpha=0.5, zorder=4)
        ax.text(p - 0.30, 0.035, f'{int((d < FAIL).sum())}', fontsize=10,
                color=col, ha='center', va='bottom', zorder=5)

    # header above the frame: region label, then the two medians (no axes title)
    xaxt = ax.get_xaxis_transform()  # x in data, y in axes fraction
    ax.text(1.5, 1.13, rlabel, transform=xaxt, fontsize=12,
            ha='center', va='bottom', clip_on=False)
    for d, p, col in zip(data, pos, cols):
        ax.text(p, 1.02, f'{np.median(d):.3f}', transform=xaxt, fontsize=10.5,
                color=col, ha='center', va='bottom', clip_on=False)

    ax.set_xlim(0.4, 2.6)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(pos)
    ax.set_xticklabels(['nnU-Net', 'UNETR'])

axes[0].set_ylabel('Per-case Dice (DSC)')
axes[0].set_yticks(np.arange(0.0, 1.01, 0.2))

legend = [
    mpatches.Patch(facecolor=C_NN, alpha=0.28, edgecolor=C_NN, label='nnU-Net'),
    mpatches.Patch(facecolor=C_UT, alpha=0.28, edgecolor=C_UT, label='UNETR'),
    mlines.Line2D([], [], color='black', linewidth=1.2, label='Median'),
    mlines.Line2D([], [], color=C_MK, linestyle='--', linewidth=1.0,
                  label='Failure (DSC $<0.5$); count shown'),
]
fig.legend(handles=legend, loc='lower center', ncol=4, framealpha=0.9,
           edgecolor='#bbbbbb', handlelength=1.5, fontsize=10.5,
           columnspacing=1.5, bbox_to_anchor=(0.5, -0.04))

out = HERE / 'rq1_dice_distribution.pdf'
fig.savefig(out, bbox_inches='tight')
print(f'Saved → {out}')
