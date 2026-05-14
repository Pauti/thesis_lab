"""
Generate an augmentation overview figure from a single BraTS sample.

The output mirrors the reference layout used in the thesis: a single row of
panels titled "Input", "Rotation", "Flip", "Translation", "Scale",
"Crop", and "Shear", but without any arrows or annotations.

By default the script uses patient BraTS2021_00022 and the best tumor slice of
the FLAIR modality.

Usage:
    python scripts/generate_augmentation_overview.py
    python scripts/generate_augmentation_overview.py --patient BraTS2021_00022
    python scripts/generate_augmentation_overview.py --modality t1ce
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import affine_transform, rotate, shift, zoom


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_brats import best_tumor_slice, load, normalize


TITLES = ["Input", "Rotation", "Flip", "Translation", "Scale", "Crop", "Shear"]
VALID_MODALITIES = {"flair", "t1", "t1ce", "t2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a thesis-style augmentation overview from one BraTS slice."
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
        default="samples/augmentation_overview_00022.png",
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
    """Estimate the typical background intensity from the image border."""
    border = np.concatenate([
        image[0, :],
        image[-1, :],
        image[:, 0],
        image[:, -1],
    ])
    return float(np.median(border))


def crop_or_pad(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Center-crop or zero-pad a 2D image to the target shape."""
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
    """Crop the slice around the brain region and keep a square canvas."""
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


def apply_shear(image: np.ndarray, background: float, shear_factor: float = 0.18) -> np.ndarray:
    """Apply a horizontal shear while keeping the output image size constant."""
    matrix = np.array([[1.0, shear_factor], [0.0, 1.0]], dtype=np.float32)
    center = (np.array(image.shape, dtype=np.float32) - 1.0) / 2.0
    offset = center - matrix @ center
    sheared = affine_transform(
        image,
        matrix=matrix,
        offset=offset,
        order=1,
        mode="constant",
        cval=background,
    )
    return np.clip(sheared, 0.0, 1.0)


def apply_scaling(
    image: np.ndarray,
    scale_factor: float = 1.28,
    shift_up: int = 18,
    shift_right: int = 18,
) -> np.ndarray:
    """Scale up the slice and shift it toward the upper-right on a black canvas."""
    scaled = zoom(image, zoom=scale_factor, order=1)
    out = np.zeros(image.shape, dtype=np.float32)

    start_y = (image.shape[0] - scaled.shape[0]) // 2 - shift_up
    start_x = (image.shape[1] - scaled.shape[1]) // 2 + shift_right
    end_y = start_y + scaled.shape[0]
    end_x = start_x + scaled.shape[1]

    dst_y0 = max(start_y, 0)
    dst_x0 = max(start_x, 0)
    dst_y1 = min(end_y, image.shape[0])
    dst_x1 = min(end_x, image.shape[1])

    src_y0 = max(-start_y, 0)
    src_x0 = max(-start_x, 0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    src_x1 = src_x0 + (dst_x1 - dst_x0)

    out[dst_y0:dst_y1, dst_x0:dst_x1] = scaled[src_y0:src_y1, src_x0:src_x1]
    return np.clip(out, 0.0, 1.0)


def apply_crop(image: np.ndarray, crop_factor: float = 0.78, top_trim: float = 0.10) -> np.ndarray:
    """Crop with a stronger cut from the top and pad back to the original canvas size."""
    crop_h = max(1, int(image.shape[0] * crop_factor))
    crop_w = max(1, int(image.shape[1] * crop_factor))
    extra_top = int(image.shape[0] * top_trim)
    y0 = min(max((image.shape[0] - crop_h) // 2 + extra_top, 0), image.shape[0] - crop_h)
    x0 = max((image.shape[1] - crop_w) // 2, 0)
    cropped = image[y0:y0 + crop_h, x0:x0 + crop_w]
    return crop_or_pad(cropped, image.shape)


def build_variants(image: np.ndarray) -> list[np.ndarray]:
    """Create deterministic augmentation examples from one input slice."""
    background = estimate_background(image)

    variants = [
        image,
        rotate(image, angle=26, reshape=False, order=1, mode="constant", cval=background),
        np.fliplr(image),
        shift(image, shift=(12, -15), order=1, mode="constant", cval=background),
        apply_scaling(image, scale_factor=1.28, shift_up=18, shift_right=18),
        apply_crop(image, crop_factor=0.78, top_trim=0.10),
        apply_shear(image, background=background, shear_factor=0.18),
    ]

    return [np.clip(variant, 0.0, 1.0) for variant in variants]


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
    fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 1.75, 2.3))
    fig.patch.set_facecolor("white")

    for ax, title, image in zip(axes, TITLES, images):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
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
    print(f"Saved augmentation overview: {out_path}")


if __name__ == "__main__":
    main()