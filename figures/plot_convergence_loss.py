#!/usr/bin/env python3
"""
plot_convergence_loss.py

Training-loss convergence: nnU-Net (top) and UNETR (bottom).
Replaces convergence_training_loss.tex — no LaTeX required.

Usage:
    /path/to/venv/bin/python figures/plot_convergence_loss.py
"""

from pathlib import Path
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

# ── fonts ────────────────────────────────────────────────────────────────────
# STIX gives Computer-Modern-style rendering without a LaTeX installation.
mpl.rcParams.update({
    'font.family'       : 'serif',
    'font.serif'        : ['STIX Two Text', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset'  : 'stix',
    'font.size'         : 10,
    'axes.labelsize'    : 10,
    'xtick.labelsize'   : 9,
    'ytick.labelsize'   : 9,
    'legend.fontsize'   : 9,
    'axes.linewidth'    : 0.8,
    'xtick.direction'   : 'out',
    'ytick.direction'   : 'out',
    'xtick.major.size'  : 3.5,
    'ytick.major.size'  : 3.5,
    'grid.color'        : 'gray',
    'grid.alpha'        : 0.25,
    'grid.linewidth'    : 0.4,
    'pdf.fonttype'      : 42,   # embed fonts as TrueType in PDF
})

C_NN = '#E07B39'   # orange – nnU-Net
C_UT = '#3A86C8'   # blue   – UNETR
C_MK = '#CC3333'   # red    – epoch-100 marker

# ── data ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
REPO = HERE.parent

nn = pd.read_csv(HERE / 'data' / 'nnunet_loss.csv')
# nnU-Net CSV stores −(Dice + CE); the original TikZ adds +1 to make it positive.
nn[['mean', 'lower', 'upper']] += 1

# Build UNETR loss from raw per-fold metrics (all available epochs per fold).
_unetr_folds = [
    pd.read_csv(REPO / 'experiments' / 'exp03_unetr' / f'fold_{i}' / 'metrics.csv')
      [['epoch', 'train_loss']]
      .rename(columns={'train_loss': f'f{i}'})
    for i in range(5)
]
_ut_wide = _unetr_folds[0]
for _df in _unetr_folds[1:]:
    _ut_wide = _ut_wide.merge(_df, on='epoch', how='outer')
_ut_wide = _ut_wide.sort_values('epoch').reset_index(drop=True)

_cols = [f'f{i}' for i in range(5)]
ut = pd.DataFrame({
    'epoch': _ut_wide['epoch'],
    'mean':  _ut_wide[_cols].mean(axis=1),
    'lower': _ut_wide[_cols].mean(axis=1) - _ut_wide[_cols].std(axis=1, ddof=1),
    'upper': _ut_wide[_cols].mean(axis=1) + _ut_wide[_cols].std(axis=1, ddof=1),
})


# ── helpers ───────────────────────────────────────────────────────────────────
def _style(ax: mpl.axes.Axes) -> None:
    ax.grid(True, which='major', zorder=0)
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Dice + CE Loss', fontsize=10)
    ax.tick_params(direction='out', labelsize=9)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)


def _legend_handles(color: str, label_mean: str) -> list:
    return [
        mpl.lines.Line2D([], [], color=color, linewidth=1.6, label=label_mean),
        mpatches.Patch(facecolor=color, alpha=0.4, linewidth=0, label=r'$\pm 1$ Std'),
        mpl.lines.Line2D([], [], color=C_MK, linestyle='--',
                         linewidth=1.2, label='Epoch 100'),
    ]


# ── figure ────────────────────────────────────────────────────────────────────
fig, (ax_nn, ax_ut) = plt.subplots(2, 1, figsize=(5.5, 6.5))
fig.subplots_adjust(hspace=0.55, top=0.91, bottom=0.08, left=0.13, right=0.97)


# ════════════════════════════════════════════════════════════════════════════
# (a) nnU-Net
# ════════════════════════════════════════════════════════════════════════════
_style(ax_nn)
ax_nn.set_xlim(-10, 1009)
ax_nn.set_ylim(0.10, 0.82)
ax_nn.set_yticks([0.2, 0.4, 0.6, 0.8])

ax_nn.fill_between(nn['epoch'], nn['lower'], nn['upper'],
                   color=C_NN, alpha=0.18, linewidth=0, zorder=1)
ax_nn.plot(nn['epoch'], nn['mean'], color=C_NN, linewidth=1.6, zorder=2)
ax_nn.axvline(100, color=C_MK, linestyle='--', linewidth=1.2, zorder=3)

# dotted reference line from epoch 0 to epoch-100 marker
ax_nn.plot([0, 100], [0.2316, 0.2316],
           color=C_MK, linestyle=':', linewidth=0.7, zorder=4)
ax_nn.annotate(
    r'Loss$\,{=}\,0.232$ (ep$\,$100)',
    xy=(102, 0.242), xytext=(120, 0.285),
    fontsize=9, color=C_MK,
    arrowprops=dict(arrowstyle='->', color=C_MK, lw=0.7),
    ha='left',
)

ax_nn.legend(
    handles=_legend_handles(C_NN, 'Mean (5 folds)'),
    loc='upper right', framealpha=0.88, edgecolor='#bbbbbb', handlelength=1.6,
)

# secondary x-axis: iterations on top
ax_nn_top = ax_nn.secondary_xaxis(
    'top',
    functions=(lambda ep: ep * 250, lambda it: it / 250),
)
ax_nn_top.set_xlabel(
    r'Iterations  ($1\,\mathrm{ep} = 250\,\mathrm{iters}$)', fontsize=9, labelpad=4,
)
ax_nn_top.set_xticks([0, 50_000, 100_000, 150_000, 200_000, 250_000])
ax_nn_top.set_xticklabels(
    [r'$0$', r'$50\mathrm{k}$', r'$100\mathrm{k}$',
     r'$150\mathrm{k}$', r'$200\mathrm{k}$', r'$250\mathrm{k}$'],
    fontsize=9,
)
ax_nn_top.tick_params(direction='out', labelsize=9)
ax_nn_top.spines['top'].set_linewidth(0.8)

ax_nn.set_title(
    r'(a)  nnU-Net  ($250\,\mathrm{iters/ep}$,'
    r' $\mathrm{bs}{=}2$, $1000\,\mathrm{epochs}$)',
    fontsize=10, pad=42,
)


# ════════════════════════════════════════════════════════════════════════════
# (b) UNETR  – x-axis extended to 200 epochs
# ════════════════════════════════════════════════════════════════════════════
_style(ax_ut)
ax_ut.set_xlim(-1, 201)
ax_ut.set_ylim(0.18, 2.30)
ax_ut.set_yticks([0.5, 1.0, 1.5, 2.0])

ax_ut.fill_between(ut['epoch'], ut['lower'], ut['upper'],
                   color=C_UT, alpha=0.18, linewidth=0, zorder=1)
ax_ut.plot(ut['epoch'], ut['mean'], color=C_UT, linewidth=1.6, zorder=2)
ax_ut.axvline(100, color=C_MK, linestyle='--', linewidth=1.2, zorder=3)

ax_ut.plot([0, 100], [0.2755, 0.2755],
           color=C_MK, linestyle=':', linewidth=0.7, zorder=4)
ax_ut.annotate(
    r'Loss$\,{=}\,0.276$ (ep$\,$100)',
    xy=(98, 0.282), xytext=(92, 0.44),
    fontsize=9, color=C_MK,
    arrowprops=dict(arrowstyle='->', color=C_MK, lw=0.7),
    ha='right',
)

ax_ut.legend(
    handles=_legend_handles(C_UT, 'Mean (5 folds)'),
    loc='upper right', framealpha=0.88, edgecolor='#bbbbbb', handlelength=1.6,
)

# secondary x-axis: iterations on top (scaled to 200 epochs)
ax_ut_top = ax_ut.secondary_xaxis(
    'top',
    functions=(lambda ep: ep * 501, lambda it: it / 501),
)
ax_ut_top.set_xlabel(
    r'Iterations  ($1\,\mathrm{ep} \approx 501\,\mathrm{iters}$)', fontsize=9, labelpad=4,
)
ax_ut_top.set_xticks([0, 20_000, 40_000, 60_000, 80_000, 100_000])
ax_ut_top.set_xticklabels(
    [r'$0$', r'$20\mathrm{k}$', r'$40\mathrm{k}$',
     r'$60\mathrm{k}$', r'$80\mathrm{k}$', r'$100\mathrm{k}$'],
    fontsize=9,
)
ax_ut_top.tick_params(direction='out', labelsize=9)
ax_ut_top.spines['top'].set_linewidth(0.8)

ax_ut.set_title(
    r'(b)  UNETR  ($501\,\mathrm{iters/ep}$,'
    r' $\mathrm{bs}{=}2$, grad-accum$\,{=}\,4$, early stop)',
    fontsize=10, pad=42,
)

# ── save ──────────────────────────────────────────────────────────────────────
out = HERE / 'convergence_training_loss.pdf'
fig.savefig(out, bbox_inches='tight')
print(f'Saved → {out}')
