#!/usr/bin/env python3
"""
plot_rq2_delta_vs_default.py

RQ2 augmentation effect size: Δ mean Dice of each tier vs. the Default tier,
with Holm–Bonferroni-corrected significance markers from the verified JSON.

Usage:
    /path/to/venv/bin/python figures/plot_rq2_delta_vs_default.py
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ── fonts ─────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family'      : 'serif',
    'font.serif'       : ['STIX Two Text', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset' : 'stix',
    'font.size'        : 12,
    'axes.labelsize'   : 12,
    'xtick.labelsize'  : 11,
    'ytick.labelsize'  : 11,
    'axes.linewidth'   : 0.8,
    'xtick.direction'  : 'out',
    'ytick.direction'  : 'out',
    'grid.color'       : 'gray',
    'grid.alpha'       : 0.25,
    'grid.linewidth'   : 0.4,
    'pdf.fonttype'     : 42,
})

C_WT = '#3A86C8'   # blue
C_TC = '#E07B39'   # orange
C_ET = '#2EAD4B'   # green

HERE = Path(__file__).parent
REPO = HERE.parent

# ── load Holm-corrected significance from verified source ─────────────────────
sig_data = json.loads((REPO / 'experiments/significance_rq1_rq2.json').read_text())

# build lookup: (model, tier) -> {region: {delta_mean, holm_sig}}
# delta_sign in the JSON is "positive = tier better than default"
# We use the Hodges-Lehmann as the robust effect-size estimator, and
# the mean delta (tier_mean - default_mean) for the bar height.
SIG = {}
for item in sig_data['rq2']:
    # parse "RQ2  nnunet: extended vs default" -> model="nnunet", tier="extended"
    parts = item['comparison'].split()
    model = parts[1].rstrip(':')
    tier = parts[2]
    SIG[(model, tier)] = {}
    for r in item['regions']:
        reg = r['region']
        SIG[(model, tier)][reg] = {
            'delta': r['a_mean'] - r['b_mean'],
            'holm': r['p_holm_reject'],
            'p': r['p_value'],
        }

MODELS = [('nnunet', 'nnU-Net'), ('unetr', 'UNETR')]
TIERS = ['disabled', 'reduced', 'extended']
TIER_LABELS = ['Disabled', 'Reduced', 'Extended']
REGIONS = [('WT', C_WT), ('TC', C_TC), ('ET', C_ET)]

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.8), sharey=True)
fig.subplots_adjust(top=0.88, bottom=0.18, left=0.10, right=0.98, wspace=0.08)

x = np.arange(len(TIERS))
width = 0.24

for ax, (mkey, mlabel) in zip(axes, MODELS):
    ax.grid(True, axis='y', which='major', zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(0, color='black', lw=0.9, zorder=2)

    for ri, (reg, col) in enumerate(REGIONS):
        deltas = []
        holms = []
        for tier in TIERS:
            info = SIG[(mkey, tier)][reg]
            deltas.append(info['delta'])
            holms.append(info['holm'])

        offset = (ri - 1) * width
        bars = ax.bar(x + offset, deltas, width, color=col, alpha=0.85,
                      label=reg if ax is axes[1] else None, zorder=3)

        for bar, d, holm in zip(bars, deltas, holms):
            va = 'bottom' if d >= 0 else 'top'
            y = d + (0.0003 if d >= 0 else -0.0003)
            if holm:
                ax.text(bar.get_x() + bar.get_width() / 2, y, '*',
                        ha='center', va=va, fontsize=14, color=col, zorder=5)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, y, 'n.s.',
                        ha='center', va=va, fontsize=7, color='#777', zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(TIER_LABELS)
    ax.set_xlabel('Augmentation tier (vs. Default)')
    ax.set_ylim(-0.007, 0.009)
    ax.text(0.5, 1.06, mlabel, transform=ax.transAxes, fontsize=12,
            ha='center', va='bottom')

axes[0].set_ylabel(r'$\Delta$ mean Dice vs. Default')

axes[1].legend(loc='upper right', framealpha=0.9, edgecolor='#bbbbbb',
               handlelength=1.2, fontsize=10)

out = HERE / 'rq2_delta_vs_default.pdf'
fig.savefig(out, bbox_inches='tight')
print(f'Saved → {out}')
