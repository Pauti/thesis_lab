#!/usr/bin/env python3
"""
Shared data loading for the RQ1 qualitative figures.

nnU-Net predictions are loaded from the stored 5-fold CV validation masks;
UNETR predictions are recomputed with the matching validation-fold checkpoint
(exp03 uses the same splits_final.json as nnU-Net, so a case's validation fold
equals the fold directory its nnU-Net prediction sits in). All masks are in the
{0:bg, 1:NCR, 2:ED, 3:ET} convention; T1ce is modality channel _0002.
"""

from pathlib import Path

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

import sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from models.unetr import get_unetr_model  # noqa: E402

DATA_DIR = Path('/data/paul_daten/nnunet/nnUNet_raw/Dataset001_BraTS')
NN_RESULTS = Path('/data/paul_daten/nnunet/nnUNet_results/Dataset001_BraTS/'
                  'nnUNetTrainer__nnUNetPlans__3d_fullres')
UT_CKPT = Path('/data/paul_daten/experiments/exp03_unetr')

# BraTS label colours (green/red/yellow convention per Menze et al.).
ED_FILL  = (0.30, 0.69, 0.29, 0.55)   # green  – peritumoral edema (label 2)
NCR_FILL = (0.80, 0.17, 0.17, 0.65)   # red    – necrotic / non-enhancing core (1)
ET_FILL  = (0.98, 0.85, 0.14, 0.75)   # yellow – enhancing tumor (label 3)


def load_t1ce_gt(case_id):
    t1ce = nib.load(str(DATA_DIR / 'imagesTr' / f'{case_id}_0002.nii.gz')).get_fdata().astype(np.float32)
    gt = np.round(nib.load(str(DATA_DIR / 'labelsTr' / f'{case_id}.nii.gz')).get_fdata()).astype(np.int8)
    return t1ce, gt


def load_nnunet_cv_pred(case_id):
    hits = list(NN_RESULTS.glob(f'fold_*/validation/{case_id}.nii.gz'))
    assert hits, f'no stored nnU-Net CV prediction for {case_id}'
    return np.round(nib.load(str(hits[0])).get_fdata()).astype(np.int8), hits[0]


def unetr_fold(case_id):
    """Validation fold of a case = the fold dir its nnU-Net CV prediction sits in."""
    _, path = load_nnunet_cv_pred(case_id)
    for part in path.parts:
        if part.startswith('fold_'):
            return int(part.split('_')[1])
    raise RuntimeError(f'could not resolve fold for {case_id}')


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


def brain_bbox_2d(t1ce_sl, margin=6):
    ys, xs = np.where(t1ce_sl > 0)
    if ys.size == 0:
        return slice(None), slice(None)
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, t1ce_sl.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, t1ce_sl.shape[1])
    return slice(y0, y1), slice(x0, x1)


def norm_gray(sl):
    p1, p99 = np.percentile(sl[sl > 0], [1, 99]) if sl.max() > 0 else (0, 1)
    return np.clip((sl - p1) / max(p99 - p1, 1e-6), 0, 1)


def region_dice(pred, gt):
    def d(a, b):
        tot = a.sum() + b.sum()
        return 1.0 if tot == 0 else 2.0 * (a & b).sum() / tot
    return {'wt': d(pred >= 1, gt >= 1),
            'tc': d(np.isin(pred, [1, 3]), np.isin(gt, [1, 3])),
            'et': d(pred == 3, gt == 3)}
