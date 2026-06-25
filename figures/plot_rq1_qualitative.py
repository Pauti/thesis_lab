#!/usr/bin/env python3
"""
plot_rq1_qualitative.py

RQ1 qualitative segmentation overlays (5-fold CV). 2 rows x 3 cols:
  row 0 = nnU-Net, row 1 = UNETR; cols = worst / median / best ET case.
Each cell: T1ce background + predicted mask (filled WT/TC/ET) + ground-truth
contour. nnU-Net predictions are loaded from the stored CV validation masks;
UNETR predictions are recomputed with the matching validation-fold checkpoint
(single fold, reproducing the RQ1 per-case score).

Usage:
    /path/to/venv/bin/python figures/plot_rq1_qualitative.py
"""

import sys
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import yaml
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged,
    NormalizeIntensityd, Orientationd,
)

HERE = Path(__file__).parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
from models.unetr import get_unetr_model  # noqa: E402

DATA_DIR = Path('/data/paul_daten/nnunet/nnUNet_raw/Dataset001_BraTS')
NN_RESULTS = Path('/data/paul_daten/nnunet/nnUNet_results/Dataset001_BraTS/'
                  'nnUNetTrainer__nnUNetPlans__3d_fullres')
UT_CKPT = Path('/data/paul_daten/experiments/exp03_unetr')

# ── fonts (match plot_convergence_loss.py) ────────────────────────────────────
mpl.rcParams.update({
    'font.family'     : 'serif',
    'font.serif'      : ['STIX Two Text', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size'       : 12,
    'pdf.fonttype'    : 42,
})

# BraTS label colours (green/red/yellow convention per Menze et al.).
# WT = union of all three colours, TC = red + yellow, ET = yellow.
ED_FILL  = (0.30, 0.69, 0.29, 0.55)   # green  – peritumoral edema (label 2)
NCR_FILL = (0.80, 0.17, 0.17, 0.65)   # red    – necrotic / non-enhancing core (label 1)
ET_FILL  = (0.98, 0.85, 0.14, 0.75)   # yellow – enhancing tumor (label 3)

# One shared case per row (so GT | nnU-Net | UNETR are directly comparable).
# (case_id, row label, UNETR validation fold). Cases are chosen jointly over both
# models among those with a substantial ET (>=10k voxels): "Hardest" = both models
# fail, "Median" = both near their median, "Best" = both near-perfect. ET-absent /
# tiny-ET cases (where Dice collapses to 0 trivially) are deliberately avoided.
SHARED_CASES = [
    ('BraTS2021_01572', 'Hardest', 1),   # nn 0.095 / ut 0.127
    ('BraTS2021_01323', 'Median',  1),   # nn 0.928 / ut 0.923
    ('BraTS2021_00156', 'Best',    2),   # nn 0.987 / ut 0.987
]


# ── data helpers ──────────────────────────────────────────────────────────────
def load_t1ce_gt(case_id):
    t1ce = nib.load(str(DATA_DIR / 'imagesTr' / f'{case_id}_0002.nii.gz')).get_fdata().astype(np.float32)
    gt = np.round(nib.load(str(DATA_DIR / 'labelsTr' / f'{case_id}.nii.gz')).get_fdata()).astype(np.int8)
    return t1ce, gt


def load_nnunet_cv_pred(case_id):
    hits = list(NN_RESULTS.glob(f'fold_*/validation/{case_id}.nii.gz'))
    assert hits, f'no stored nnU-Net CV prediction for {case_id}'
    return np.round(nib.load(str(hits[0])).get_fdata()).astype(np.int8)


def unetr_predict(case_id, fold, device):
    cfg = yaml.safe_load((REPO / 'configs/exp03_unetr.yaml').read_text())
    mc, swc = cfg['model'], cfg['validation']['sliding_window']
    tfm = Compose([
        LoadImaged(keys=['image'], ensure_channel_first=False, image_only=False),
        EnsureChannelFirstd(keys=['image']),
        Orientationd(keys=['image'], axcodes='RAS'),
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        EnsureTyped(keys=['image'], dtype=torch.float32),
    ])
    sample = {'image': [str(DATA_DIR / 'imagesTr' / f'{case_id}_000{i}.nii.gz') for i in range(4)]}
    batch = next(iter(DataLoader(Dataset([sample], transform=tfm), batch_size=1, num_workers=0)))
    model = get_unetr_model(img_size=tuple(mc['img_size']), in_channels=mc['in_channels'],
                            out_channels=mc['out_channels'], dropout_rate=mc['dropout_rate'])
    ckpt = torch.load(UT_CKPT / f'fold_{fold}' / 'best_model.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=batch['image'].to(device), roi_size=swc['roi_size'],
            sw_batch_size=swc['sw_batch_size'], predictor=model,
            overlap=swc['overlap'], mode=swc['mode'])
    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int8)
    # Orientationd RAS reorients LPS->RAS (flip axes 0,1); convert back to native LPS
    return np.ascontiguousarray(pred[::-1, ::-1, :])


def _dice(a, b):
    tot = a.sum() + b.sum()
    return 1.0 if tot == 0 else 2.0 * (a & b).sum() / tot


def region_dice(pred, gt):
    """WT/TC/ET Dice for a {0,1,2,3} mask (ET=3, TC={1,3}, WT>=1)."""
    return {'wt': _dice(pred >= 1, gt >= 1),
            'tc': _dice(np.isin(pred, [1, 3]), np.isin(gt, [1, 3])),
            'et': _dice(pred == 3, gt == 3)}


def best_et_slice(gt):
    et = (gt == 3).sum(axis=(0, 1))
    return int(np.argmax(et)) if et.max() > 0 else int(np.argmax((gt >= 1).sum(axis=(0, 1))))


def brain_bbox(t1ce_sl, margin=6):
    ys, xs = np.where(t1ce_sl > 0)
    if ys.size == 0:
        return slice(None), slice(None)
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, t1ce_sl.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, t1ce_sl.shape[1])
    return slice(y0, y1), slice(x0, x1)


def draw_cell(ax, t1ce_sl, mask_sl):
    """T1ce grayscale background + filled WT/TC/ET mask overlay."""
    p1, p99 = np.percentile(t1ce_sl[t1ce_sl > 0], [1, 99]) if t1ce_sl.max() > 0 else (0, 1)
    bg = np.clip((t1ce_sl - p1) / max(p99 - p1, 1e-6), 0, 1)
    ax.imshow(bg.T[::-1, :], cmap='gray', origin='lower', interpolation='nearest')

    h, w = mask_sl.shape
    rgba = np.zeros((h, w, 4), np.float32)
    rgba[mask_sl == 2] = ED_FILL    # edema (green)
    rgba[mask_sl == 1] = NCR_FILL   # necrotic core (red)
    rgba[mask_sl == 3] = ET_FILL    # enhancing tumor (yellow)
    ax.imshow(rgba.transpose(1, 0, 2)[::-1, :], origin='lower',
              interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


# ── build figure ──────────────────────────────────────────────────────────────
def collect(device):
    """Per shared case: (label, case_id, t1ce_sl, gt_sl, nn_sl, ut_sl). Prints the
    full WT/TC/ET Dice of both models (for the figure caption)."""
    rows = []
    for cid, lab, fold in SHARED_CASES:
        t1ce, gt = load_t1ce_gt(cid)
        nn_pred = load_nnunet_cv_pred(cid)
        ut_pred = unetr_predict(cid, fold, device)
        z = best_et_slice(gt); ys, xs = brain_bbox(t1ce[:, :, z])
        nn_s, ut_s = region_dice(nn_pred, gt), region_dice(ut_pred, gt)
        print(f'  {lab:8s} {cid}  z={z}  '
              f"nnU-Net WT/TC/ET={nn_s['wt']:.3f}/{nn_s['tc']:.3f}/{nn_s['et']:.3f}  "
              f"UNETR WT/TC/ET={ut_s['wt']:.3f}/{ut_s['tc']:.3f}/{ut_s['et']:.3f}")
        rows.append((lab, cid, t1ce[ys, xs, z], gt[ys, xs, z],
                     nn_pred[ys, xs, z], ut_pred[ys, xs, z]))
    return rows


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    rows = collect(device)

    # 3 rows (cases) x 3 cols (Ground truth | nnU-Net | UNETR).
    fig, axes = plt.subplots(3, 3, figsize=(6.3, 6.6))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.93, bottom=0.07,
                        wspace=0.04, hspace=0.06)

    col_titles = ['Ground truth', 'nnU-Net', 'UNETR']
    for r, (lab, cid, t1ce_sl, gt_sl, nn_sl, ut_sl) in enumerate(rows):
        draw_cell(axes[r, 0], t1ce_sl, gt_sl)
        draw_cell(axes[r, 1], t1ce_sl, nn_sl)
        draw_cell(axes[r, 2], t1ce_sl, ut_sl)
        axes[r, 0].text(-0.10, 0.5, f'{lab}\n({cid[-5:]})',
                        transform=axes[r, 0].transAxes, rotation=90,
                        ha='right', va='center', fontsize=12, linespacing=1.4)
    for col, ct in enumerate(col_titles):
        axes[0, col].set_title(ct, fontsize=13, pad=5)

    legend = [
        mpatches.Patch(facecolor=ED_FILL[:3], alpha=ED_FILL[3],
                       label='WT (whole tumor)'),
        mpatches.Patch(facecolor=NCR_FILL[:3], alpha=NCR_FILL[3],
                       label='TC (tumor core)'),
        mpatches.Patch(facecolor=ET_FILL[:3], alpha=ET_FILL[3],
                       label='ET (enhancing tumor)'),
    ]
    fig.legend(handles=legend, loc='lower center', ncol=3, fontsize=11,
               framealpha=0.9, edgecolor='#bbbbbb', columnspacing=1.6,
               handlelength=1.4, bbox_to_anchor=(0.54, -0.005))

    out = HERE / 'rq1_qualitative_overlays.pdf'
    fig.savefig(out, bbox_inches='tight', facecolor='white')
    print(f'\nSaved -> {out}')


if __name__ == '__main__':
    main()
