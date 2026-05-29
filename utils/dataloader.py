"""
BraTS 2021 MONAI DataLoader for UNETR experiments.

Reads raw NIfTI images from nnUNet_raw and uses split files for
reproducible fold splits.

  - splits_final.json: Original 5-fold CV on all 1251 cases (EXP-02/03, RQ1)
  - splits_ablation.json: 3-fold CV on 1126 cases, 125 held out (ablation, RQ2)

Image paths follow nnU-Net multi-channel convention:
  imagesTr/{case_id}_0000.nii.gz  (FLAIR)
  imagesTr/{case_id}_0001.nii.gz  (T1)
  imagesTr/{case_id}_0002.nii.gz  (T1ce)
  imagesTr/{case_id}_0003.nii.gz  (T2)
  labelsTr/{case_id}.nii.gz       (remapped: bg=0, NCR=1, ED=2, ET=3)
"""

import json
from pathlib import Path
from typing import Literal

from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)

SPLITS_FILE = Path(
    "/data/paul_daten/nnunet/nnUNet_preprocessed/Dataset001_BraTS/splits_final.json"
)
ABLATION_SPLITS_FILE = Path(
    "/data/paul_daten/nnunet/nnUNet_preprocessed/Dataset001_BraTS/splits_ablation.json"
)
RAW_DIR = Path("/data/paul_daten/nnunet/nnUNet_raw/Dataset001_BraTS")
MODALITY_SUFFIXES = ["_0000", "_0001", "_0002", "_0003"]


def _build_file_list(case_ids: list[str]) -> list[dict]:
    """Build MONAI-style data dicts for a list of case IDs."""
    records = []
    for cid in case_ids:
        img_paths = [
            str(RAW_DIR / "imagesTr" / f"{cid}{s}.nii.gz")
            for s in MODALITY_SUFFIXES
        ]
        label_path = str(RAW_DIR / "labelsTr" / f"{cid}.nii.gz")
        records.append({"image": img_paths, "label": label_path, "case_id": cid})
    return records


def get_base_transforms() -> Compose:
    """
    Preprocessing transforms applied to both train and val.
    - Load 4-channel image + label
    - Orient to RAS
    - Z-score normalise each modality within brain mask (nonzero voxels)
    - BraTS is already 1mm³ isotropic → no resampling needed
    """
    return Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=False),
        EnsureChannelFirstd(keys=["label"]),          # label: (1,H,W,D)
        # image is loaded as list → MONAI stacks to (4,H,W,D) automatically
        # with ensure_channel_first=False we handle it via EnsureChannelFirstd
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(
            keys=["image"],
            nonzero=True,       # normalise within brain mask only
            channel_wise=True,  # per-modality z-score
        ),
        EnsureTyped(keys=["image", "label"], dtype=("float32", "int64")),
    ])


def get_train_dataset(
    fold: int,
    augmentation_transforms: Compose,
    cache_rate: float = 0.0,
    num_workers: int = 4,
    splits_file: Path | None = None,
) -> Dataset:
    """Return MONAI dataset for training split of the given fold."""
    sf = splits_file or SPLITS_FILE
    splits = json.loads(sf.read_text())
    case_ids = splits[fold]["train"]
    data = _build_file_list(case_ids)

    base_tf = get_base_transforms()
    transforms = Compose(base_tf.transforms + augmentation_transforms.transforms)

    if cache_rate > 0:
        return CacheDataset(data=data, transform=transforms, cache_rate=cache_rate,
                            num_workers=num_workers)
    return Dataset(data=data, transform=transforms)


def get_val_dataset(
    fold: int,
    cache_rate: float = 0.0,
    num_workers: int = 4,
    splits_file: Path | None = None,
) -> Dataset:
    """Return MONAI dataset for validation split of the given fold (no augmentation)."""
    sf = splits_file or SPLITS_FILE
    splits = json.loads(sf.read_text())
    case_ids = splits[fold]["val"]
    data = _build_file_list(case_ids)

    transforms = get_base_transforms()

    if cache_rate > 0:
        return CacheDataset(data=data, transform=transforms, cache_rate=cache_rate,
                            num_workers=num_workers)
    return Dataset(data=data, transform=transforms)


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
