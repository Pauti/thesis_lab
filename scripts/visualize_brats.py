"""
Visualize BraTS 2021 MRT-Daten mit Segmentierungsoverlay.

Usage:
    python scripts/visualize_brats.py
    python scripts/visualize_brats.py --input_dir /data/paul_daten/extracted --out_dir samples --n_patients 3
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib

LABEL_COLORS = {
    1: [255,  50,  50],  # NCR — nekrotischer Kern (rot)
    2: [ 50, 200, 100],  # ED  — peritumorales Ödem (grün)
    3: [ 80, 150, 255],  # ET  — enhancing Tumor (blau, orig. Label 4)
}

MODALITIES = ["flair", "t1", "t1ce", "t2"]
MOD_LABELS  = ["FLAIR", "T1", "T1ce (Kontrastmittel)", "T2"]


def load(path: Path) -> np.ndarray:
    return nib.load(path).get_fdata().astype(np.float32)


def normalize(vol: np.ndarray) -> np.ndarray:
    """Z-Score-Normalisierung auf Gehirnvoxel, dann skaliert auf [0, 1]."""
    mask = vol > 0
    if mask.sum() == 0:
        return vol
    m, s = vol[mask].mean(), vol[mask].std() + 1e-8
    out = np.zeros_like(vol)
    out[mask] = np.clip((vol[mask] - m) / s, -3, 3)
    return (out + 3) / 6


def overlay(flair_slice: np.ndarray, seg_slice: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Legt farbige Segmentierungsmaske über FLAIR-Graustufenbild."""
    rgb = np.stack([flair_slice] * 3, axis=-1)
    for label, color in LABEL_COLORS.items():
        mask = seg_slice == label
        for c, v in enumerate(color):
            rgb[:, :, c][mask] = rgb[:, :, c][mask] * (1 - alpha) + (v / 255) * alpha
    return rgb


def best_tumor_slice(seg: np.ndarray) -> int:
    """Gibt den z-Index der Scheibe mit den meisten Tumorvoxeln zurück."""
    return int(np.argmax((seg > 0).sum(axis=(0, 1))))


def plot_patient_grid(patients: list, brats_dir: Path, out_path: Path):
    """
    Figur 1: n_patients × 5 Grid
    Spalten: FLAIR | T1 | T1ce | T2 | FLAIR + Segm.-Overlay
    """
    n = len(patients)
    fig, axes = plt.subplots(n, 5, figsize=(18, 4 * n))
    fig.patch.set_facecolor('#0d1117')

    for row, pid in enumerate(patients):
        folder = brats_dir / pid
        seg = load(folder / f"{pid}_seg.nii.gz")
        z = best_tumor_slice(seg)
        seg_sl = seg[:, :, z].T

        for col, (mod, mlabel) in enumerate(zip(MODALITIES, MOD_LABELS)):
            vol = normalize(load(folder / f"{pid}_{mod}.nii.gz"))
            ax = axes[row, col] if n > 1 else axes[col]
            ax.imshow(vol[:, :, z].T, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            if row == 0:
                ax.set_title(mlabel, color='white', fontsize=11, fontweight='bold', pad=6)
            if col == 0:
                ax.set_ylabel(pid.replace("BraTS2021_", "Patient "),
                              color='#aaaaaa', fontsize=9, labelpad=4)

        # Overlay-Spalte
        flair_vol = normalize(load(folder / f"{pid}_flair.nii.gz"))
        ov = overlay(flair_vol[:, :, z].T, seg_sl)
        ax = axes[row, 4] if n > 1 else axes[4]
        ax.imshow(ov, vmin=0, vmax=1)
        ax.axis('off')
        if row == 0:
            ax.set_title("FLAIR + Segm.\n■ NCR  ■ ED  ■ ET",
                         color='white', fontsize=10, fontweight='bold', pad=6)
        for label, name, color in [(1, 'NCR', '#ff3232'), (2, 'ED', '#32c864'), (3, 'ET', '#5096ff')]:
            if (seg_sl == label).any():
                ax.text(0.02, 0.98 - (label - 1) * 0.08, f"■ {name}",
                        transform=ax.transAxes, color=color, fontsize=8, va='top', fontweight='bold')

    fig.suptitle("BraTS 2021 — Patienten-Übersicht\n"
                 "Spalten: 4 MRT-Modalitäten | Rechts: FLAIR mit Segmentierungsoverlay",
                 color='white', fontsize=13, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close(fig)
    print(f"Gespeichert: {out_path}")


def plot_tumor_slices(pid: str, brats_dir: Path, out_path: Path, n_slices: int = 8):
    """
    Figur 2: 1 Patient, n_slices Scheiben mit den meisten Tumorvoxeln.
    Oben: rohes FLAIR, unten: FLAIR + Segmentierungsoverlay.
    """
    folder = brats_dir / pid
    seg = load(folder / f"{pid}_seg.nii.gz")
    flair_vol = normalize(load(folder / f"{pid}_flair.nii.gz"))

    z_counts = (seg > 0).sum(axis=(0, 1))
    top_z = sorted(np.argsort(z_counts)[-n_slices:])

    fig, axes = plt.subplots(2, n_slices, figsize=(2.5 * n_slices, 6))
    fig.patch.set_facecolor('#0d1117')

    for i, z in enumerate(top_z):
        flair_sl = flair_vol[:, :, z].T
        seg_sl   = seg[:, :, z].T

        ax = axes[0, i]
        ax.imshow(flair_sl, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"z={z}", color='#aaaaaa', fontsize=8)
        ax.axis('off')

        ax = axes[1, i]
        ax.imshow(overlay(flair_sl, seg_sl, alpha=0.5), vmin=0, vmax=1)
        ax.axis('off')

    axes[0, 0].set_ylabel("FLAIR", color='white', fontsize=9)
    axes[1, 0].set_ylabel("+ Tumormaske", color='white', fontsize=9)

    fig.suptitle(f"{pid} — {n_slices} Tumorscheiben  (■ NCR rot  ■ ED grün  ■ ET blau)",
                 color='white', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close(fig)
    print(f"Gespeichert: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="BraTS 2021 Visualisierung")
    parser.add_argument("--input_dir",  default="/data/paul_daten/extracted",
                        help="Verzeichnis mit extrahierten BraTS-Patientenordnern")
    parser.add_argument("--out_dir",    default="samples",
                        help="Ausgabeverzeichnis für PNG-Dateien")
    parser.add_argument("--n_patients", type=int, default=3,
                        help="Anzahl Patienten für die Übersichts-Figur")
    parser.add_argument("--patient",    default="BraTS2021_00000",
                        help="Patient-ID für die Scheiben-Detail-Figur")
    parser.add_argument("--n_slices",   type=int, default=8,
                        help="Anzahl Tumorscheiben in der Detail-Figur")
    args = parser.parse_args()

    brats_dir = Path(args.input_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ersten n_patients alphabetisch sortierte Patienten auswählen
    all_patients = sorted(p.name for p in brats_dir.iterdir()
                          if p.is_dir() and p.name.startswith("BraTS2021_"))
    if not all_patients:
        print(f"Fehler: Keine BraTS2021_*-Ordner in {brats_dir} gefunden.")
        return

    patients = all_patients[: args.n_patients]
    print(f"Patienten: {patients}")

    plot_patient_grid(patients, brats_dir, out_dir / "brats_samples.png")
    plot_tumor_slices(args.patient, brats_dir, out_dir / "brats_tumor_slices.png", args.n_slices)


if __name__ == "__main__":
    main()
