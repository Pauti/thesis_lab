#!/usr/bin/env python3
"""
Generate a simple 2-row figure from one BraTS case:
- Top: raw MRI slice (no preprocessing)
- Bottom: brain-only crop (background removed + cropped bounding box)

No legend and no panel labels, as requested.

Usage:
    python scripts/generate_raw_vs_brain_crop.py
    python scripts/generate_raw_vs_brain_crop.py --patient BraTS2021_00022 --modality flair
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_brats import best_tumor_slice, load

VALID_MODALITIES = {"flair", "t1", "t1ce", "t2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a two-panel image: raw MRI (top) vs brain-only crop (bottom)."
    )
    parser.add_argument(
        "--input_dir",
        default="/data/paul_daten/extracted",
        help="Directory containing BraTS2021_* patient folders.",
    )
    parser.add_argument(
        "--patient",
        default="BraTS2021_00022",
        help="Patient ID to visualize.",
    )
    parser.add_argument(
        "--modality",
        default="flair",
        help="One of: flair, t1, t1ce, t2.",
    )
    parser.add_argument(
        "--out",
        default="samples/raw_vs_brain_crop_00022.png",
        help="Output PNG path relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI.",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=8,
        help="Extra pixels around the brain crop bounding box.",
    )
    return parser.parse_args()


def estimate_background(image: np.ndarray) -> float:
    border = np.concatenate([
        image[0, :],
        image[-1, :],
        image[:, 0],
        image[:, -1],
    ])
    return float(np.median(border))


def make_brain_only_crop(image: np.ndarray, margin: int = 8) -> np.ndarray:
    """Remove background, then crop to the brain bounding box."""
    bg = estimate_background(image)
    # BraTS volumes are skull-stripped; nonzero voxels represent brain region.
    mask = image > 0
    if np.any(mask):
        # Keep only the largest connected component for a stable brain mask.
        lbl, n_lbl = ndimage.label(mask)
        if n_lbl > 1:
            sizes = ndimage.sum(mask, lbl, index=np.arange(1, n_lbl + 1))
            largest = int(np.argmax(sizes)) + 1
            mask = lbl == largest

    if not np.any(mask):
        return image

    # Remove background values so only brain anatomy remains visually.
    brain_only = image.copy()
    brain_only[~mask] = bg

    ys, xs = np.where(mask)
    y0 = max(int(ys.min()) - margin, 0)
    y1 = min(int(ys.max()) + margin + 1, image.shape[0])
    x0 = max(int(xs.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin + 1, image.shape[1])

    return brain_only[y0:y1, x0:x1]


def get_slice(brats_dir: Path, patient_id: str, modality: str) -> np.ndarray:
    folder = brats_dir / patient_id
    if not folder.exists():
        raise FileNotFoundError(f"Patient folder not found: {folder}")

    seg = load(folder / f"{patient_id}_seg.nii.gz")
    z_idx = best_tumor_slice(seg)

    volume = load(folder / f"{patient_id}_{modality}.nii.gz")
    return volume[:, :, z_idx].T


def save_figure(raw_image: np.ndarray, cropped_image: np.ndarray, out_path: Path, dpi: int) -> None:
    nonzero_raw = raw_image[raw_image > 0]
    if nonzero_raw.size > 0:
        vmin = float(np.percentile(nonzero_raw, 1))
        vmax = float(np.percentile(nonzero_raw, 99.5))
    else:
        vmin = float(raw_image.min())
        vmax = float(raw_image.max())

    fig, axes = plt.subplots(2, 1, figsize=(4.2, 8.4))
    fig.patch.set_facecolor("white")

    axes[0].imshow(raw_image, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].axis("off")

    axes[1].imshow(cropped_image, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].axis("off")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, hspace=0.08)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    modality = args.modality.lower()
    if modality not in VALID_MODALITIES:
        raise ValueError(f"Invalid modality '{args.modality}'. Choose from {sorted(VALID_MODALITIES)}")

    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    repo_root = script_dir.parent

    brats_dir = Path(args.input_dir)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path

    raw_img = get_slice(brats_dir, args.patient, modality)
    cropped_img = make_brain_only_crop(raw_img, margin=args.margin)

    save_figure(raw_img, cropped_img, out_path, args.dpi)
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
