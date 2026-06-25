#!/usr/bin/env python3
"""
plot_rq1_error_maps.py

Whole-tumour error maps (true positive / over-segmentation / under-segmentation)
for nnU-Net vs UNETR on representative harder cases. Visualises the boundary
behaviour behind the HD95 gap reported in RQ1: green = correct, red = predicted
tumour outside the ground truth (over-segmentation, false positive), blue = ground
-truth tumour missed (under-segmentation, false negative).

Usage:
    /path/to/venv/bin/python figures/plot_rq1_error_maps.py
"""

import sys
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from _rq1_data import (  # noqa: E402
    brain_bbox_2d, load_nnunet_cv_pred, load_t1ce_gt, norm_gray,
    region_dice, unetr_fold, unetr_predict,
)

mpl.rcParams.update({
    'font.family'     : 'serif',
    'font.serif'      : ['STIX Two Text', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size'       : 12,
    'pdf.fonttype'    : 42,
})

TP_COL = (0.20, 0.70, 0.25, 0.50)   # green  – correct
FP_COL = (0.92, 0.15, 0.15, 0.75)   # red    – over-segmentation (false positive)
FN_COL = (0.15, 0.45, 0.95, 0.75)   # blue   – missed (false negative)

# Cases with a large whole-tumour HD95 gap: nnU-Net localised, UNETR producing a
# distant spurious false-positive region (the driver of UNETR's higher HD95). WT
# Dice stays high in all of them, so HD95 is the discriminating metric.
CASES = ['BraTS2021_00533', 'BraTS2021_00567', 'BraTS2021_01288']

REPO = Path(__file__).resolve().parent.parent
_NN_HD = pd.read_csv(REPO / 'experiments/exp02_nnunet/analysis/all_folds_hd95.csv'
                     ).set_index('case_id')['hd95_wt']
_UT_HD = pd.read_csv(REPO / 'experiments/exp03_unetr/analysis/all_folds_brats_dice_nifti.csv'
                     ).set_index('case_id')['hd95_wt']


def wt_error(ax, gray2d, gt_wt, pred_wt):
    ax.imshow(gray2d, cmap='gray', origin='lower', interpolation='nearest')
    h, w = gt_wt.shape
    rgba = np.zeros((h, w, 4), np.float32)
    rgba[gt_wt & pred_wt] = TP_COL
    rgba[pred_wt & ~gt_wt] = FP_COL
    rgba[gt_wt & ~pred_wt] = FN_COL
    ax.imshow(rgba, origin='lower', interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    rows = []
    for cid in CASES:
        t1ce, gt = load_t1ce_gt(cid)
        nn_pred, _ = load_nnunet_cv_pred(cid)
        ut_pred = unetr_predict(cid, unetr_fold(cid), device)
        # slice showing the UNETR spurious prediction (its WT false positives)
        z = int(np.argmax(((ut_pred >= 1) & (gt < 1)).sum(axis=(0, 1))))
        ys, xs = brain_bbox_2d(t1ce[:, :, z])
        gray = norm_gray(t1ce[:, :, z][ys, xs].T[::-1, :])
        gt_wt = (gt[:, :, z][ys, xs].T[::-1, :] >= 1)
        nn_wt = (nn_pred[:, :, z][ys, xs].T[::-1, :] >= 1)
        ut_wt = (ut_pred[:, :, z][ys, xs].T[::-1, :] >= 1)
        h_nn, h_ut = float(_NN_HD[cid]), float(_UT_HD[cid])
        print(f'  {cid}  z={z}  WT HD95 nn={h_nn:.1f} ut={h_ut:.1f}')
        rows.append((cid, gray, gt_wt, nn_wt, ut_wt, h_nn, h_ut))

    # transposed layout: rows = models (nnU-Net / UNETR), cols = cases
    fig, axes = plt.subplots(2, len(CASES), figsize=(2.3 * len(CASES) + 0.4, 5.0))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.91, bottom=0.13,
                        wspace=0.05, hspace=0.04)

    def cell(ax, gray, gt_wt, pred_wt, hd):
        wt_error(ax, gray, gt_wt, pred_wt)
        ax.text(0.5, 0.02, f'HD95 {hd:.1f} mm', transform=ax.transAxes,
                fontsize=10, ha='center', va='bottom', color='#111',
                bbox=dict(facecolor='white', alpha=0.78, edgecolor='none', pad=1.3))

    for c, (cid, gray, gt_wt, nn_wt, ut_wt, h_nn, h_ut) in enumerate(rows):
        cell(axes[0, c], gray, gt_wt, nn_wt, h_nn)
        cell(axes[1, c], gray, gt_wt, ut_wt, h_ut)
        axes[0, c].set_title(cid[-5:], fontsize=12, pad=4)
    for r, name in enumerate(['nnU-Net', 'UNETR']):
        axes[r, 0].text(-0.08, 0.5, name, transform=axes[r, 0].transAxes,
                        rotation=90, ha='right', va='center', fontsize=13)

    legend = [
        mpatches.Patch(facecolor=TP_COL[:3], alpha=TP_COL[3], label='Correct (TP)'),
        mpatches.Patch(facecolor=FP_COL[:3], alpha=FP_COL[3], label='Over-seg. (FP)'),
        mpatches.Patch(facecolor=FN_COL[:3], alpha=FN_COL[3], label='Missed (FN)'),
    ]
    fig.legend(handles=legend, loc='lower center', ncol=3, fontsize=10.5,
               framealpha=0.9, edgecolor='#bbbbbb', columnspacing=1.4,
               handlelength=1.3, bbox_to_anchor=(0.54, -0.005))

    out = HERE / 'rq1_error_maps.pdf'
    fig.savefig(out, bbox_inches='tight', facecolor='white')
    print(f'Saved -> {out}')


if __name__ == '__main__':
    main()
