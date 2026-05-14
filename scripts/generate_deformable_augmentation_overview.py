"""
Generate a deformable-augmentation overview figure from a single BraTS sample.

The output is a single row with panels:
Input, Elastic (Coarse), Elastic (Fine), Random Field, Diffeomorphic-like,
Local Expansion.

By default the script uses patient BraTS2021_00022 and the best tumor slice of
FLAIR to match the existing augmentation sample workflow.

Usage:
    python scripts/generate_deformable_augmentation_overview.py
    python scripts/generate_deformable_augmentation_overview.py --patient BraTS2021_00022
    python scripts/generate_deformable_augmentation_overview.py --modality t1ce
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, zoom


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_brats import best_tumor_slice, load, normalize


TITLES = [
    "Input",
    "Elastic",
    "Random Field",
    "Diffeomorphic-like",
    "Local Expansion",
]
VALID_MODALITIES = {"flair", "t1", "t1ce", "t2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a thesis-style deformable-augmentation overview from one BraTS slice."
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
        default="samples/augmentation_deformable_overview_00022.png",
        help="Output PNG path relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=22,
        help="Random seed for deterministic deformation fields.",
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


def crop_or_pad(
    image: np.ndarray,
    target_shape: tuple[int, int],
    fill_value: float | None = None,
) -> np.ndarray:
    """Center-crop or pad a 2D image to target shape."""
    if fill_value is None:
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


def crop_foreground_pair(
    image: np.ndarray,
    label: np.ndarray,
    margin: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop around foreground anatomy and keep image/label aligned on a square canvas."""
    background = estimate_background(image)
    mask = image > (background + 0.03)
    if not np.any(mask):
        return image, label

    ys, xs = np.where(mask)
    y0 = max(int(ys.min()) - margin, 0)
    y1 = min(int(ys.max()) + margin + 1, image.shape[0])
    x0 = max(int(xs.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin + 1, image.shape[1])

    image_c = image[y0:y1, x0:x1]
    label_c = label[y0:y1, x0:x1]
    side = max(image_c.shape)

    return (
        crop_or_pad(image_c, (side, side), fill_value=background),
        crop_or_pad(label_c, (side, side), fill_value=0.0),
    )


def apply_displacement(
    image: np.ndarray,
    disp_y: np.ndarray,
    disp_x: np.ndarray,
    background: float,
) -> np.ndarray:
    """Warp image with dense displacement fields."""
    h, w = image.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    warped = map_coordinates(
        image,
        [yy + disp_y, xx + disp_x],
        order=1,
        mode="constant",
        cval=background,
    )
    return np.clip(warped.astype(np.float32), 0.0, 1.0)


def build_elastic_displacement(
    shape: tuple[int, int],
    rng: np.random.Generator,
    sigma: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create smooth random displacement fields (classic elastic deformation style)."""
    dy = gaussian_filter(rng.standard_normal(shape), sigma=sigma, mode="reflect")
    dx = gaussian_filter(rng.standard_normal(shape), sigma=sigma, mode="reflect")
    max_val = max(np.max(np.abs(dy)), np.max(np.abs(dx)), 1e-6)
    scale = alpha / max_val
    return (dy * scale).astype(np.float32), (dx * scale).astype(np.float32)


def build_bspline_like_displacement(
    shape: tuple[int, int],
    rng: np.random.Generator,
    control_points: int = 6,
    alpha: float = 16.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate B-spline deformation with a coarse control grid and interpolation."""
    cp_y = rng.uniform(-1.0, 1.0, size=(control_points, control_points)).astype(np.float32)
    cp_x = rng.uniform(-1.0, 1.0, size=(control_points, control_points)).astype(np.float32)

    zoom_factors = (
        shape[0] / control_points,
        shape[1] / control_points,
    )
    dy = zoom(cp_y, zoom=zoom_factors, order=3)[:shape[0], :shape[1]]
    dx = zoom(cp_x, zoom=zoom_factors, order=3)[:shape[0], :shape[1]]

    dy = gaussian_filter(dy, sigma=2.0, mode="reflect")
    dx = gaussian_filter(dx, sigma=2.0, mode="reflect")
    max_val = max(np.max(np.abs(dy)), np.max(np.abs(dx)), 1e-6)
    scale = alpha / max_val
    return (dy * scale).astype(np.float32), (dx * scale).astype(np.float32)


def build_local_expansion_displacement(
    shape: tuple[int, int],
    label: np.ndarray,
    strength: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a localized smooth expansion around tumor region center."""
    h, w = shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    tumor_mask = label > 0
    if np.any(tumor_mask):
        ys, xs = np.where(tumor_mask)
        cy = float(np.mean(ys))
        cx = float(np.mean(xs))
        region_scale = max(np.sqrt(float(tumor_mask.sum())) * 1.8, 10.0)
    else:
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        region_scale = min(h, w) * 0.2

    ry = yy - cy
    rx = xx - cx
    r = np.sqrt(ry ** 2 + rx ** 2) + 1e-6
    weight = np.exp(-(r ** 2) / (2.0 * region_scale ** 2))

    dy = strength * weight * (ry / r)
    dx = strength * weight * (rx / r)
    return dy.astype(np.float32), dx.astype(np.float32)


def build_diffeomorphic_like_displacement(
    shape: tuple[int, int],
    rng: np.random.Generator,
    sigma: float = 20.0,
    max_step: float = 0.9,
    steps: int = 14,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a smooth stationary velocity field into a deformation map."""
    v_y = gaussian_filter(rng.standard_normal(shape), sigma=sigma, mode="reflect")
    v_x = gaussian_filter(rng.standard_normal(shape), sigma=sigma, mode="reflect")
    max_vel = max(np.max(np.abs(v_y)), np.max(np.abs(v_x)), 1e-6)
    v_y = (v_y / max_vel) * max_step
    v_x = (v_x / max_vel) * max_step

    h, w = shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    disp_y = np.zeros(shape, dtype=np.float32)
    disp_x = np.zeros(shape, dtype=np.float32)

    for _ in range(steps):
        curr_y = yy + disp_y
        curr_x = xx + disp_x
        step_y = map_coordinates(v_y, [curr_y, curr_x], order=1, mode="nearest")
        step_x = map_coordinates(v_x, [curr_y, curr_x], order=1, mode="nearest")
        disp_y += (step_y / steps).astype(np.float32)
        disp_x += (step_x / steps).astype(np.float32)

    return disp_y, disp_x


def build_variants(image: np.ndarray, label: np.ndarray, seed: int) -> list[np.ndarray]:
    """Create deterministic deformable augmentation examples from one input slice."""
    background = estimate_background(image)
    rng = np.random.default_rng(seed)

    edy, edx = build_elastic_displacement(image.shape, rng, sigma=9.0, alpha=8.0)
    rdy, rdx = build_elastic_displacement(image.shape, rng, sigma=14.0, alpha=11.0)
    ddy, ddx = build_diffeomorphic_like_displacement(image.shape, rng, sigma=20.0, max_step=0.9, steps=14)
    ldy, ldx = build_local_expansion_displacement(image.shape, label, strength=8.0)

    return [
        image,
        apply_displacement(image, edy, edx, background),
        apply_displacement(image, rdy, rdx, background),
        apply_displacement(image, ddy, ddx, background),
        apply_displacement(image, ldy, ldx, background),
    ]


def get_slice(brats_dir: Path, patient_id: str, modality: str) -> tuple[np.ndarray, np.ndarray]:
    folder = brats_dir / patient_id
    if not folder.exists():
        raise FileNotFoundError(f"Patient folder not found: {folder}")

    seg = load(folder / f"{patient_id}_seg.nii.gz")
    z_index = best_tumor_slice(seg)

    volume = normalize(load(folder / f"{patient_id}_{modality}.nii.gz"))
    image_slice = volume[:, :, z_index].T.astype(np.float32)
    seg_slice = seg[:, :, z_index].T.astype(np.float32)
    return crop_foreground_pair(image_slice, seg_slice)


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

    input_slice, seg_slice = get_slice(brats_dir, args.patient, modality)
    variants = build_variants(input_slice, seg_slice, seed=args.seed)
    save_figure(variants, out_path, args.dpi)
    print(f"Saved deformable augmentation overview: {out_path}")


if __name__ == "__main__":
    main()