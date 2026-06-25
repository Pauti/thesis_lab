#!/usr/bin/env python3
"""
plot_rq1_multiplanar.py

Multi-planar segmentation example for one representative case: axial, sagittal,
and coronal slices through the tumour centroid, shown as ground truth, nnU-Net,
and UNETR (BraTS label overlay on T1ce). 3 rows (planes) x 3 cols (GT/nn/ut).

Usage:
    /path/to/venv/bin/python figures/plot_rq1_multiplanar.py
"""

import sys
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from _rq1_data import (  # noqa: E402
    ED_FILL, NCR_FILL, ET_FILL, brain_bbox_2d, load_nnunet_cv_pred,
    load_t1ce_gt, norm_gray, region_dice, unetr_fold, unetr_predict,
)

mpl.rcParams.update({
    'font.family'     : 'serif',
    'font.serif'      : ['STIX Two Text', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size'       : 12,
    'pdf.fonttype'    : 42,
})

CASE = 'BraTS2021_01305'   # compact ring-enhancing tumor, sharp in all 3 planes


def overlay(ax, gray2d, mask2d):
    ax.imshow(gray2d, cmap='gray', origin='lower', interpolation='nearest')
    h, w = mask2d.shape
    rgba = np.zeros((h, w, 4), np.float32)
    rgba[mask2d == 2] = ED_FILL
    rgba[mask2d == 1] = NCR_FILL
    rgba[mask2d == 3] = ET_FILL
    ax.imshow(rgba, origin='lower', interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def plane_slices(vol3d, plane, idx):
    """Return a 2-D slice oriented superior-up / radiological for display."""
    if plane == 'axial':       # spanned by L(x), P(y); A up, radiological L-R
        return vol3d[:, :, idx].T[::-1, :]
    if plane == 'sagittal':    # spanned by P(y), S(z); S up
        return vol3d[idx, :, :].T
    if plane == 'coronal':     # spanned by L(x), S(z); S up
        return vol3d[:, idx, :].T


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}  case: {CASE}')

    t1ce, gt = load_t1ce_gt(CASE)
    nn_pred, _ = load_nnunet_cv_pred(CASE)
    ut_pred = unetr_predict(CASE, unetr_fold(CASE), device)
    sc_nn, sc_ut = region_dice(nn_pred, gt), region_dice(ut_pred, gt)
    print(f"  nnU-Net WT/TC/ET={sc_nn['wt']:.3f}/{sc_nn['tc']:.3f}/{sc_nn['et']:.3f}  "
          f"UNETR={sc_ut['wt']:.3f}/{sc_ut['tc']:.3f}/{sc_ut['et']:.3f}")

    # tumour centroid → one index per axis
    cx, cy, cz = (int(round(c)) for c in np.argwhere(gt >= 1).mean(axis=0))
    PLANES = [('axial', cz), ('sagittal', cx), ('coronal', cy)]
    VOLS = [('Ground truth', gt), ('nnU-Net', nn_pred), ('UNETR', ut_pred)]

    fig, axes = plt.subplots(3, 3, figsize=(6.3, 6.6))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.07,
                        wspace=0.04, hspace=0.05)

    for r, (plane, idx) in enumerate(PLANES):
        t_sl = plane_slices(t1ce, plane, idx)
        ys, xs = brain_bbox_2d(t_sl)
        gray = norm_gray(t_sl[ys, xs])
        for c, (_, vol) in enumerate(VOLS):
            m_sl = plane_slices(vol, plane, idx)[ys, xs]
            overlay(axes[r, c], gray, m_sl)
        axes[r, 0].text(-0.07, 0.5, plane.capitalize(), transform=axes[r, 0].transAxes,
                        rotation=90, ha='right', va='center', fontsize=12)
    for c, (title, _) in enumerate(VOLS):
        axes[0, c].set_title(title, fontsize=13, pad=5)

    legend = [
        mpatches.Patch(facecolor=ED_FILL[:3], alpha=ED_FILL[3], label='WT (whole tumor)'),
        mpatches.Patch(facecolor=NCR_FILL[:3], alpha=NCR_FILL[3], label='TC (tumor core)'),
        mpatches.Patch(facecolor=ET_FILL[:3], alpha=ET_FILL[3], label='ET (enhancing tumor)'),
    ]
    fig.legend(handles=legend, loc='lower center', ncol=3, fontsize=11,
               framealpha=0.9, edgecolor='#bbbbbb', columnspacing=1.6,
               handlelength=1.4, bbox_to_anchor=(0.54, -0.005))

    out = HERE / 'rq1_multiplanar.pdf'
    fig.savefig(out, bbox_inches='tight', facecolor='white')
    print(f'Saved -> {out}')


if __name__ == '__main__':
    main()
