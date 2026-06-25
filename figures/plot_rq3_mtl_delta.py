#!/usr/bin/env python3
"""
plot_rq3_mtl_delta.py

RQ3 multi-task effect size: Δ mean Dice of each MTL variant vs. its single-task
baseline, with Holm-corrected significance markers. Analogous to the RQ2 delta
plot but grouped by architecture × MTL variant.

Usage:
    /path/to/venv/bin/python figures/plot_rq3_mtl_delta.py
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

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

C_WT = '#3A86C8'
C_TC = '#E07B39'
C_ET = '#2EAD4B'

HERE = Path(__file__).parent
REPO = HERE.parent

sig_data = json.loads((REPO / 'experiments/mtl_significance.json').read_text())

# (comparison key, x-label) — skip v3 and v2-vs-v1
COMPARISONS = [
    ('nnU-Net MTL v1 (per-patch) vs baseline', 'v1 (patch)'),
    ('nnU-Net MTL v2 (global) vs baseline',    'v2 (global)'),
    ('UNETR MTL v1 (per-patch) vs baseline',   'v1 (patch)'),
    ('UNETR MTL v2 (global) vs baseline',      'v2 (global)'),
]

# group into panels
PANELS = [
    ('nnU-Net', COMPARISONS[:2]),
    ('UNETR',   COMPARISONS[2:]),
]

REGIONS = [('WT', C_WT), ('TC', C_TC), ('ET', C_ET)]

# index comparisons from JSON
BY_NAME = {c['architecture']: c for c in sig_data['comparisons']}

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.8), sharey=True)
fig.subplots_adjust(top=0.88, bottom=0.18, left=0.10, right=0.98, wspace=0.08)

width = 0.24

for ax, (panel_label, comps) in zip(axes, PANELS):
    ax.grid(True, axis='y', which='major', zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(0, color='black', lw=0.9, zorder=2)

    x = np.arange(len(comps))

    for ri, (reg, col) in enumerate(REGIONS):
        deltas = []
        holms = []
        for comp_key, _ in comps:
            comp = BY_NAME[comp_key]
            r = next(r for r in comp['regions'] if r['region'] == reg)
            deltas.append(r['mtl_mean'] - r['baseline_mean'])
            holms.append(r['p_holm_reject'])

        offset = (ri - 1) * width
        bars = ax.bar(x + offset, deltas, width, color=col, alpha=0.85,
                      label=reg if ax is axes[1] else None, zorder=3)

        for bar, d, holm in zip(bars, deltas, holms):
            va = 'bottom' if d >= 0 else 'top'
            y = d + (0.0003 if d >= 0 else -0.0003)
            if holm:
                ax.text(bar.get_x() + bar.get_width() / 2, y, '*',
                        ha='center', va=va, fontsize=14, color=col,
                        zorder=5, clip_on=True)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, y, 'n.s.',
                        ha='center', va=va, fontsize=8.5, color='#777',
                        zorder=5, clip_on=True)

    ax.set_ylim(-0.010, 0.0015)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in comps])
    ax.set_xlabel('MTL variant (vs. baseline)')
    ax.text(0.5, 1.06, panel_label, transform=ax.transAxes, fontsize=12,
            ha='center', va='bottom')

axes[0].set_ylabel(r'$\Delta$ mean Dice vs. baseline')
axes[1].legend(loc='upper right', framealpha=0.9, edgecolor='#bbbbbb',
               handlelength=1.2, fontsize=10)

out = HERE / 'rq3_mtl_delta.pdf'
fig.savefig(out, bbox_inches='tight')
print(f'Saved → {out}')
