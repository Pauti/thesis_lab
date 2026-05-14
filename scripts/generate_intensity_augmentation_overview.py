"""
Generate an intensity-augmentation overview figure from a single BraTS sample.

The output is a single row with panels:
Input, Brightness, Noise, Gamma, Histogram Equalization, Blur.

By default the script uses patient BraTS2021_00022 and the best tumor slice of
FLAIR to match the existing augmentation sample workflow.

Usage:
    python scripts/generate_intensity_augmentation_overview.py
    python scripts/generate_intensity_augmentation_overview.py --patient BraTS2021_00022
    python scripts/generate_intensity_augmentation_overview.py --modality t1ce
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_brats import best_tumor_slice, load, normalize


TITLES = ["Input", "Brightness", "Noise", "Gamma", "Histogram Equalization", "Blur"]
VALID_MODALITIES = {"flair", "t1", "t1ce", "t2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a thesis-style intensity-augmentation overview from one BraTS slice."
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
        default="samples/augmentation_intensity_overview_00022.png",
        help="Output PNG path relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI.",
    )
    return parser.parse_args()


def estimate_background(image: np.ndarray) -> float:
    """Estimate the typical background intensity from image borders."""
    border = np.concatenate([
        image[0, :],
        image[-1, :],
        image[:, 0],
        image[:, -1],
    ])
    return float(np.median(border))


def crop_or_pad(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Center-crop or pad a 2D image to target shape using border background."""
    fill_value = estimate_background(image)
    out = np.full(target_shape, fill_value, dtype=np.float32)

    src_h, src_w = image.shape
    dst_h, dst_w = target_shape
    copy_h = min(src_h, dst_h)
    copy_w = min(src_w, dst_w)

    src_y0 = max((src_h - copy_h) // 2, 0)
    src_x0 = max((src_w - copy_w) // 2, 0)
    dst_y0 = max((dst_h - copy_h) // 2, 0)
    dst_x0 = max((dst_w - copy_w) // 2, 0)

    out[dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = image[
        src_y0:src_y0 + copy_h,
        src_x0:src_x0 + copy_w,
    ]
    return out


def crop_foreground(image: np.ndarray, margin: int = 8) -> np.ndarray:
    """Crop around foreground anatomy and keep a square canvas."""
    background = estimate_background(image)
    mask = image > (background + 0.03)
    if not np.any(mask):
        return image

    ys, xs = np.where(mask)
    y0 = max(int(ys.min()) - margin, 0)
    y1 = min(int(ys.max()) + margin + 1, image.shape[0])
    x0 = max(int(xs.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin + 1, image.shape[1])
    cropped = image[y0:y1, x0:x1]
    side = max(cropped.shape)
    return crop_or_pad(cropped, (side, side))


def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Apply global histogram equalization while preserving background darkness."""
    flat = np.clip(image, 0.0, 1.0).ravel()
    hist, bin_edges = np.histogram(flat, bins=256, range=(0.0, 1.0))
    cdf = hist.cumsum().astype(np.float64)
    if cdf[-1] == 0:
        return image

    cdf = cdf / cdf[-1]
    equalized = np.interp(flat, bin_edges[:-1], cdf).reshape(image.shape).astype(np.float32)

    background = estimate_background(image)
    equalized[image <= background + 0.01] = image[image <= background + 0.01]
    return np.clip(equalized, 0.0, 1.0)


def build_variants(image: np.ndarray) -> list[np.ndarray]:
    """Create deterministic intensity augmentation examples from one input slice."""
    rng = np.random.default_rng(22)

    brightness = np.clip(image * 0.82, 0.0, 1.0)
    noise = np.clip(image + rng.normal(0.0, 0.05, size=image.shape), 0.0, 1.0)
    gamma = np.clip(np.power(image, 1.30), 0.0, 1.0)
    hist_eq = apply_histogram_equalization(image)
    blur = np.clip(gaussian_filter(image, sigma=1.2), 0.0, 1.0)

    return [image, brightness, noise, gamma, hist_eq, blur]


def get_slice(brats_dir: Path, patient_id: str, modality: str) -> np.ndarray:
    folder = brats_dir / patient_id
    if not folder.exists():
        raise FileNotFoundError(f"Patient folder not found: {folder}")

    seg = load(folder / f"{patient_id}_seg.nii.gz")
    z_index = best_tumor_slice(seg)

    volume = normalize(load(folder / f"{patient_id}_{modality}.nii.gz"))
    return crop_foreground(volume[:, :, z_index].T)


def save_figure(images: list[np.ndarray], out_path: Path, dpi: int) -> None:
    n_panels = len(images)
    fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 1.9, 2.3))
    fig.patch.set_facecolor("white")

    for ax, title, img in zip(axes, TITLES, images):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=13, pad=10)
        ax.axis("off")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.06, wspace=0.06)
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

    input_slice = get_slice(brats_dir, args.patient, modality)
    variants = build_variants(input_slice)
    save_figure(variants, out_path, args.dpi)
    print(f"Saved intensity augmentation overview: {out_path}")


if __name__ == "__main__":
    main()
