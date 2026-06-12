#!/usr/bin/env python3
"""
Generate segmentation comparison figure for presentation slide 13.
Layout: 2 rows × 4 cols — (T1ce | GT | nnU-Net | UNETR) per case.
Cases: BraTS2021_00407 (representative) and BraTS2021_01110 (hard ET).
"""

import sys
import json
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, EnsureChannelFirstd, EnsureTyped,
    LoadImaged, NormalizeIntensityd, Orientationd,
)
from monai.data import Dataset, DataLoader

REPO     = Path(__file__).resolve().parent.parent
DATA_DIR = Path("/data/paul_daten/nnunet/nnUNet_raw/Dataset001_BraTS")
CKPT_DIR = Path("/data/paul_daten/experiments/ablation/abl_unetr_default")
NN_PREDS = REPO / "experiments/ablation/abl_nnunet_default/test_predictions"
OUT      = REPO / "experiments/ablation/comparison_plots/segmentation_example.png"

sys.path.insert(0, str(REPO))
from models.unetr import get_unetr_model

CASES = [
    ("BraTS2021_00022", "Beispiel-Segmentierung"),
]

# WT=hellblau, TC=gelb, ET=rot  (BraTS-Konvention)
WT_COLOR  = np.array([0.40, 0.75, 1.00, 0.55])
TC_COLOR  = np.array([1.00, 0.85, 0.10, 0.70])
ET_COLOR  = np.array([1.00, 0.15, 0.15, 0.85])


def load_unetr_ensemble(case_id: str, device: str) -> np.ndarray:
    with open(REPO / "configs/exp03_unetr.yaml") as f:
        cfg = yaml.safe_load(f)
    mc  = cfg["model"]
    swc = cfg["validation"]["sliding_window"]

    transforms = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=False, image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ])

    sample = {
        "case_id": case_id,
        "image": [str(DATA_DIR / "imagesTr" / f"{case_id}_000{i}.nii.gz") for i in range(4)],
        "label": str(DATA_DIR / "labelsTr" / f"{case_id}.nii.gz"),
    }
    ds     = Dataset(data=[sample], transform=transforms)
    loader = DataLoader(ds, batch_size=1, num_workers=0)
    batch  = next(iter(loader))
    imgs   = batch["image"].to(device)

    fold_preds = []
    for fold in range(3):
        ckpt_path = CKPT_DIR / f"fold_{fold}" / "best_model.pth"
        print(f"  Loading UNETR fold {fold} from {ckpt_path}")
        model = get_unetr_model(
            img_size=tuple(mc["img_size"]),
            in_channels=mc["in_channels"],
            out_channels=mc["out_channels"],
            dropout_rate=mc["dropout_rate"],
        )
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device).eval()

        with torch.no_grad():
            logits = sliding_window_inference(
                inputs=imgs,
                roi_size=swc["roi_size"],
                sw_batch_size=swc["sw_batch_size"],
                predictor=model,
                overlap=swc["overlap"],
                mode=swc["mode"],
            )
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int8)
        fold_preds.append(pred)

    # Majority voting across 3 folds
    stacked = np.stack(fold_preds, axis=0)  # (3, H, W, D)
    ensemble = np.zeros(stacked.shape[1:], dtype=np.int8)
    for label in [1, 2, 3]:
        votes = np.sum(stacked == label, axis=0)
        ensemble[votes >= 2] = label

    # MONAI Orientationd reoriented LPS→RAS (flip axes 0 and 1); convert back to LPS
    ensemble = np.ascontiguousarray(ensemble[::-1, ::-1, :])
    return ensemble


def load_nnunet_prediction(case_id: str) -> np.ndarray:
    import os, tempfile, shutil
    os.environ['nnUNet_raw']          = '/data/paul_daten/nnunet/nnUNet_raw'
    os.environ['nnUNet_preprocessed'] = '/data/paul_daten/nnunet/nnUNet_preprocessed'
    os.environ['nnUNet_results']      = '/data/paul_daten/nnunet/nnUNet_results'
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True,
        use_mirroring=False, device=torch.device('cuda'), verbose=False,
    )
    predictor.initialize_from_trained_model_folder(
        '/data/paul_daten/nnunet/nnUNet_results/Dataset001_BraTS/'
        'nnUNetTrainer_DefaultAug__nnUNetPlans__3d_fullres',
        use_folds=(0, 1, 2),
        checkpoint_name='checkpoint_best.pth',
    )

    in_dir  = tempfile.mkdtemp()
    out_dir = tempfile.mkdtemp()
    try:
        for i in range(4):
            src = DATA_DIR / "imagesTr" / f"{case_id}_000{i}.nii.gz"
            shutil.copy(str(src), f"{in_dir}/{case_id}_000{i}.nii.gz")
        predictor.predict_from_files(
            [[f"{in_dir}/{case_id}_000{i}.nii.gz" for i in range(4)]],
            [f"{out_dir}/{case_id}.nii.gz"],
            save_probabilities=False, overwrite=True,
            num_processes_preprocessing=1, num_processes_segmentation_export=1,
        )
        pred = nib.load(f"{out_dir}/{case_id}.nii.gz").get_fdata().astype(np.int8)
    finally:
        shutil.rmtree(in_dir); shutil.rmtree(out_dir)

    # nnU-Net internally uses label 4 for ET; remap 4→3 to match our convention
    pred[pred == 4] = 3
    return pred


def load_t1ce_gt(case_id: str):
    t1ce_nib = nib.load(str(DATA_DIR / "imagesTr" / f"{case_id}_0002.nii.gz"))
    gt_nib   = nib.load(str(DATA_DIR / "labelsTr" / f"{case_id}.nii.gz"))
    t1ce = t1ce_nib.get_fdata().astype(np.float32)
    gt   = np.round(gt_nib.get_fdata()).astype(np.int8)
    return t1ce, gt


def best_axial_slice(gt: np.ndarray) -> int:
    """Slice with most tumor voxels (axis 2 = axial in RAS)."""
    tumor = (gt >= 1).astype(np.float32)
    return int(np.argmax(tumor.sum(axis=(0, 1))))


def overlay_mask(ax, t1ce_slice, mask, dice_str=None):
    """Draw T1ce background + colored segmentation overlay."""
    p1, p99 = np.percentile(t1ce_slice[t1ce_slice > 0], [1, 99]) if t1ce_slice.max() > 0 else (0, 1)
    t1ce_norm = np.clip((t1ce_slice - p1) / max(p99 - p1, 1e-6), 0, 1)

    ax.imshow(t1ce_norm.T, cmap='gray', origin='lower', interpolation='nearest')

    h, w = t1ce_slice.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)

    rgba[mask >= 1]          = WT_COLOR
    rgba[np.isin(mask,[1,3])] = TC_COLOR
    rgba[mask == 3]          = ET_COLOR

    ax.imshow(rgba.transpose(1, 0, 2), origin='lower', interpolation='nearest')
    if dice_str:
        ax.text(0.5, -0.03, dice_str, transform=ax.transAxes,
                fontsize=7.8, ha='center', va='top', color='#222',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5))
    ax.axis('off')


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    def dice(a, b):
        inter = int(np.sum(a & b)); tot = int(np.sum(a)) + int(np.sum(b))
        return 1.0 if tot == 0 else 2.0 * inter / tot

    def brats_dice(pred, gt):
        return {'dice_wt': dice(pred >= 1, gt >= 1),
                'dice_tc': dice(np.isin(pred,[1,3]), np.isin(gt,[1,3])),
                'dice_et': dice(pred == 3, gt == 3)}

    n_rows = len(CASES)
    fig, axes = plt.subplots(n_rows, 4, figsize=(14, 4.5 * n_rows + 1.0), facecolor='white')
    if n_rows == 1:
        axes = axes[np.newaxis, :]  # keep 2-D indexing
    fig.subplots_adjust(hspace=0.18, wspace=0.04)

    col_titles = ["T1ce Input", "Ground Truth", "nnU-Net", "UNETR"]
    # Draw column headers as figure text above top row
    for col, ct in enumerate(col_titles):
        ax = axes[0, col]
        ax.set_title(ct, fontsize=12, fontweight='bold', pad=6, color='#1a2e4a')

    legend_patches = [
        mpatches.Patch(color=WT_COLOR[:3], alpha=0.85, label='WT (Whole Tumor)'),
        mpatches.Patch(color=TC_COLOR[:3], alpha=0.90, label='TC (Tumor Core)'),
        mpatches.Patch(color=ET_COLOR[:3], alpha=0.90, label='ET (Enhancing Tumor)'),
    ]

    captions = []

    for row, (case_id, desc) in enumerate(CASES):
        print(f"\n=== {case_id} ({desc}) ===")
        t1ce, gt = load_t1ce_gt(case_id)

        z = best_axial_slice(gt)
        print(f"  Best axial slice: {z}")

        print("  Running nnU-Net inference...")
        nn_pred = load_nnunet_prediction(case_id)

        print("  Running UNETR inference...")
        ut_pred = load_unetr_ensemble(case_id, device)

        nn_sc = brats_dice(nn_pred, gt)
        ut_sc = brats_dice(ut_pred, gt)

        t1ce_sl = t1ce[:, :, z]
        gt_sl   = gt[:, :, z]
        nn_sl   = nn_pred[:, :, z]
        ut_sl   = ut_pred[:, :, z]

        # Col 0: T1ce (no mask)
        ax = axes[row, 0]
        p1, p99 = np.percentile(t1ce_sl[t1ce_sl > 0], [1, 99]) if t1ce_sl.max() > 0 else (0, 1)
        t1_norm = np.clip((t1ce_sl - p1) / max(p99 - p1, 1e-6), 0, 1)
        ax.imshow(t1_norm.T, cmap='gray', origin='lower', interpolation='nearest')
        ax.text(0.5, -0.03, f"{case_id} · Slice z={z}", transform=ax.transAxes,
                fontsize=7.8, ha='center', va='top', color='#444',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5))
        ax.axis('off')

        # Col 1: GT
        overlay_mask(axes[row, 1], t1ce_sl, gt_sl)

        # Col 2: nnU-Net
        nn_str = f"WT {nn_sc['dice_wt']:.3f} · TC {nn_sc['dice_tc']:.3f} · ET {nn_sc['dice_et']:.3f}"
        overlay_mask(axes[row, 2], t1ce_sl, nn_sl, dice_str=nn_str)

        # Col 3: UNETR
        ut_str = f"WT {ut_sc['dice_wt']:.3f} · TC {ut_sc['dice_tc']:.3f} · ET {ut_sc['dice_et']:.3f}"
        overlay_mask(axes[row, 3], t1ce_sl, ut_sl, dice_str=ut_str)

        captions.append(
            f"{case_id} ({desc}) | T1ce Axialschnitt z={z} | "
            f"WT: nnU-Net {nn_sc['dice_wt']:.3f} vs UNETR {ut_sc['dice_wt']:.3f} | "
            f"ET: nnU-Net {nn_sc['dice_et']:.3f} vs UNETR {ut_sc['dice_et']:.3f}"
        )

    # Row labels
    for row, (_, desc) in enumerate(CASES):
        axes[row, 0].annotate(
            desc, xy=(0, 0.5), xycoords='axes fraction',
            xytext=(-8, 0), textcoords='offset points',
            ha='right', va='center', fontsize=8.5, color='#444',
            rotation=90,
        )

    fig.legend(handles=legend_patches, loc='lower center', ncol=3,
               fontsize=9.5, framealpha=0.8, bbox_to_anchor=(0.5, -0.02))

    fig.savefig(str(OUT), dpi=180, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {OUT}")
    print("\nCaption lines:")
    for c in captions:
        print(f"  {c}")


if __name__ == "__main__":
    main()
