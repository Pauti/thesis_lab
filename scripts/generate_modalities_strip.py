"""
Generate modalities strips (T1, T1CE, T2, FLAIR) from BraTS NIfTI data.

By default generates one strip per patient for the first --top_n patients,
saved as samples/strips/BraTS2021_XXXXX.png — pick the best one visually,
then copy it to samples/modalities.png and run modalities_figure.py.

Usage:
    # Generate strips for first 20 patients (default)
    python scripts/generate_modalities_strip.py

    # Generate strips for first 50 patients
    python scripts/generate_modalities_strip.py --top_n 50

    # Generate a single strip for one specific patient
    python scripts/generate_modalities_strip.py --patient BraTS2021_00495

    # After picking, copy the best strip and build the thesis figure:
    cp samples/strips/BraTS2021_00495.png samples/modalities.png
    python scripts/modalities_figure.py --layout row
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
import os

# Reuse helpers already defined in visualize_brats.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_brats import load, normalize, best_tumor_slice as best_slice


MODALITIES  = ["t1", "t1ce", "t2", "flair"]
MOD_LABELS  = ["T1", "T1CE", "T2", "FLAIR"]


def tumor_score(seg: np.ndarray) -> int:
    """Total tumor voxels — used to rank patients."""
    return int((seg > 0).sum())


def find_best_patient(brats_dir: Path, top_n: int) -> str:
    """Return the patient ID with the largest tumor volume among the first top_n patients."""
    patients = sorted(p.name for p in brats_dir.iterdir()
                      if p.is_dir() and p.name.startswith("BraTS2021_"))[:top_n]
    if not patients:
        raise FileNotFoundError(f"No BraTS2021_* folders found in {brats_dir}")

    best_pid, best_score = None, -1
    for pid in patients:
        seg_path = brats_dir / pid / f"{pid}_seg.nii.gz"
        if not seg_path.exists():
            continue
        seg = load(seg_path)
        score = tumor_score(seg)
        print(f"  {pid}: {score} tumor voxels")
        if score > best_score:
            best_score, best_pid = score, pid

    print(f"Best patient: {best_pid} ({best_score} voxels)")
    return best_pid


def generate_strip(pid: str, brats_dir: Path, out_path: Path,
                   dpi: int = 300, bg: str = "white"):
    """
    Render one axial slice for each of the four modalities side by side
    with clean white background and bold labels above each panel.
    The output format matches what modalities_figure.py expects:
    a horizontal strip with label text at the top (~30 px).
    """
    folder = brats_dir / pid
    seg    = load(folder / f"{pid}_seg.nii.gz")
    z      = best_slice(seg)
    print(f"Using axial slice z={z} for patient {pid}")

    n = len(MODALITIES)
    fig, axes = plt.subplots(1, n, figsize=(n * 3, 3.5))
    fig.patch.set_facecolor(bg)

    for ax, mod, label in zip(axes, MODALITIES, MOD_LABELS):
        vol  = normalize(load(folder / f"{pid}_{mod}.nii.gz"))
        sl   = vol[:, :, z].T
        ax.imshow(sl, cmap="gray", vmin=0, vmax=1)
        ax.set_title(label, fontsize=13, fontweight="bold", pad=5,
                     color="black" if bg == "white" else "white")
        ax.axis("off")

    fig.tight_layout(pad=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight",
                facecolor=bg)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate BraTS modalities strips for thesis figures.")
    parser.add_argument("--input_dir", default="/data/paul_daten/extracted",
                        help="Directory containing BraTS2021_* patient folders.")
    parser.add_argument("--out_dir", default="samples/strips",
                        help="Output directory for per-patient strip PNGs.")
    parser.add_argument("--patient", default=None,
                        help="Specific patient ID. If set, generates only that one strip.")
    parser.add_argument("--top_n", type=int, default=20,
                        help="Number of patients to generate strips for (default: 20).")
    parser.add_argument("--dpi", type=int, default=150,
                        help="DPI for preview strips (use 300 for final).")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root  = Path(os.path.dirname(script_dir))
    brats_dir  = Path(args.input_dir)
    out_dir    = repo_root / args.out_dir if not os.path.isabs(args.out_dir) else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.patient:
        patients = [args.patient]
    else:
        patients = sorted(p.name for p in brats_dir.iterdir()
                          if p.is_dir() and p.name.startswith("BraTS2021_"))[:args.top_n]
        if not patients:
            raise FileNotFoundError(f"No BraTS2021_* folders found in {brats_dir}")
        print(f"Generating strips for {len(patients)} patients → {out_dir}")

    for pid in patients:
        out_path = out_dir / f"{pid}.png"
        try:
            generate_strip(pid, brats_dir, out_path, dpi=args.dpi)
        except Exception as e:
            print(f"  Skipped {pid}: {e}")

    print(f"\nDone. Browse samples/strips/, pick the best one, then:\n"
          f"  cp samples/strips/<patient_id>.png samples/modalities.png\n"
          f"  python scripts/modalities_figure.py --layout row")


if __name__ == "__main__":
    main()
