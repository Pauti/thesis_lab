"""
Convert BraTS 2021 data to nnU-Net v2 dataset format.

Reads raw NIfTI files from extracted BraTS patient folders and creates
the nnU-Net expected structure:

    Dataset001_BraTS/
    ├── dataset.json
    ├── imagesTr/
    │   ├── BraTS2021_00000_0000.nii.gz  (FLAIR)
    │   ├── BraTS2021_00000_0001.nii.gz  (T1)
    │   ├── BraTS2021_00000_0002.nii.gz  (T1ce)
    │   └── BraTS2021_00000_0003.nii.gz  (T2)
    └── labelsTr/
        └── BraTS2021_00000.nii.gz       (seg, labels remapped: 4→3)

Reference: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md
"""

import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np


# BraTS modality → nnU-Net channel index
MODALITY_MAP = {
    "flair": "0000",
    "t1": "0001",
    "t1ce": "0002",
    "t2": "0003",
}


def remap_labels(seg_path: Path, output_path: Path) -> None:
    """Load segmentation NIfTI, remap label 4→3, and save.

    nnU-Net requires consecutive integer labels (0, 1, 2, 3, ...).
    BraTS 2021 uses: 0=background, 1=NCR, 2=ED, 4=ET.
    We remap 4→3 so labels become {0, 1, 2, 3}.
    """
    img = nib.load(seg_path)
    data = np.asarray(img.dataobj).copy()
    data[data == 4] = 3
    remapped = nib.Nifti1Image(data, img.affine, img.header)
    nib.save(remapped, output_path)


def find_patient_dirs(input_dir: Path) -> list[Path]:
    """Find all BraTS patient directories, handling nested structures."""
    # Direct children matching BraTS2021_*
    patients = sorted(
        d for d in input_dir.iterdir()
        if d.is_dir() and d.name.startswith("BraTS2021_")
    )
    if patients:
        return patients

    # One level deeper (e.g., extracted ZIP has a wrapper folder)
    for child in input_dir.iterdir():
        if child.is_dir():
            patients = sorted(
                d for d in child.iterdir()
                if d.is_dir() and d.name.startswith("BraTS2021_")
            )
            if patients:
                return patients

    return []


def validate_patient(patient_dir: Path) -> bool:
    """Check that all 4 modalities + segmentation exist."""
    sid = patient_dir.name
    for mod in MODALITY_MAP:
        if not (patient_dir / f"{sid}_{mod}.nii.gz").exists():
            return False
    if not (patient_dir / f"{sid}_seg.nii.gz").exists():
        return False
    return True


def convert_dataset(
    input_dir: Path,
    output_dir: Path,
    dataset_id: int = 1,
) -> None:
    dataset_name = f"Dataset{dataset_id:03d}_BraTS"
    dataset_dir = output_dir / dataset_name
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Find patient directories
    patients = find_patient_dirs(input_dir)
    if not patients:
        raise FileNotFoundError(
            f"No BraTS2021_* patient directories found in {input_dir} "
            f"or its immediate subdirectories."
        )
    print(f"Found {len(patients)} patient directories")

    # Validate and convert
    converted = 0
    skipped = []
    for patient_dir in patients:
        sid = patient_dir.name

        if not validate_patient(patient_dir):
            skipped.append(sid)
            continue

        # Copy modality images with nnU-Net naming
        for mod, channel_id in MODALITY_MAP.items():
            src = patient_dir / f"{sid}_{mod}.nii.gz"
            dst = images_dir / f"{sid}_{channel_id}.nii.gz"
            if not dst.exists():
                shutil.copy2(src, dst)

        # Remap and save segmentation
        seg_src = patient_dir / f"{sid}_seg.nii.gz"
        seg_dst = labels_dir / f"{sid}.nii.gz"
        if not seg_dst.exists():
            remap_labels(seg_src, seg_dst)

        converted += 1
        if converted % 100 == 0:
            print(f"  Converted {converted}/{len(patients)} patients...")

    print(f"Converted {converted} patients, skipped {len(skipped)}")
    if skipped:
        print(f"  Skipped (missing files): {skipped[:10]}{'...' if len(skipped) > 10 else ''}")

    # Write dataset.json
    dataset_json = {
        "channel_names": {
            "0": "FLAIR",
            "1": "T1",
            "2": "T1ce",
            "3": "T2",
        },
        "labels": {
            "background": 0,
            "NCR": 1,
            "ED": 2,
            "ET": 3,
        },
        "numTraining": converted,
        "file_ending": ".nii.gz",
    }

    dataset_json_path = dataset_dir / "dataset.json"
    with open(dataset_json_path, "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\nDataset created at: {dataset_dir}")
    print(f"  imagesTr: {len(list(images_dir.glob('*.nii.gz')))} files")
    print(f"  labelsTr: {len(list(labels_dir.glob('*.nii.gz')))} files")
    print(f"  dataset.json: {dataset_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert BraTS 2021 data to nnU-Net v2 dataset format."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Path to extracted BraTS data (containing BraTS2021_* patient folders).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Path to nnUNet_raw directory.",
    )
    parser.add_argument(
        "--dataset_id",
        type=int,
        default=1,
        help="nnU-Net dataset ID (default: 1 → Dataset001_BraTS).",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    convert_dataset(args.input_dir, args.output_dir, args.dataset_id)
