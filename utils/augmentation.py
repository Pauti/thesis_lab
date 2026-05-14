"""
MONAI augmentation transforms for UNETR experiments.

Four tiers (None / Light / Medium / Heavy) matching the nnU-Net ABL tiers:
  - Medium is used for all primary UNETR experiments (EXP-03, EXP-05)
  - None/Light/Heavy are used for augmentation ablation (ABL-04/05/06)

All transforms operate on dict keys ["image", "label"].
Patch extraction (RandCropByPosNegLabeld) is included here because it is
sampling-strategy-dependent and must apply after augmentation.
"""

from monai.transforms import (
    Compose,
    Rand3DElasticd,
    RandAdjustContrastd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
)


def _patch_sampler(patch_size: tuple[int, ...], num_samples: int = 1) -> list:
    """Random patch sampler with 1:1 foreground/background ratio (paper §4.1)."""
    return [
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=1,
            neg=1,
            num_samples=num_samples,
            image_key="image",
            image_threshold=0,
        )
    ]


def get_none_augmentation(patch_size=(128, 128, 128)) -> Compose:
    """ABL-04: No augmentation — patch extraction only."""
    return Compose(_patch_sampler(patch_size))


def get_light_augmentation(patch_size=(128, 128, 128)) -> Compose:
    """ABL-05: Light — spatial transforms only (rotation + flip)."""
    return Compose([
        *_patch_sampler(patch_size),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 2)),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    ])


def get_medium_augmentation(patch_size=(128, 128, 128)) -> Compose:
    """
    Medium augmentation — used for all primary UNETR experiments (EXP-03, EXP-05).
    Spatial + intensity transforms as used in the original UNETR paper (§4.3):
      rotation 90/180/270°, flip (3 axes), scale + shift intensity, affine scaling.
    """
    return Compose([
        *_patch_sampler(patch_size),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 2)),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        RandAffined(
            keys=["image", "label"],
            prob=0.5,
            rotate_range=(0.26, 0.26, 0.26),   # ±15°
            scale_range=(0.1, 0.1, 0.1),
            mode=("bilinear", "nearest"),
        ),
    ])


def get_heavy_augmentation(patch_size=(128, 128, 128)) -> Compose:
    """ABL-06: Heavy — Medium + elastic deformation + Gaussian noise/smoothing."""
    return Compose([
        *_patch_sampler(patch_size),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 2)),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        RandAffined(
            keys=["image", "label"],
            prob=0.5,
            rotate_range=(0.26, 0.26, 0.26),
            scale_range=(0.1, 0.1, 0.1),
            mode=("bilinear", "nearest"),
        ),
        RandGaussianNoised(keys=["image"], prob=0.3, std=0.01),
        RandGaussianSmoothd(keys=["image"], prob=0.3, sigma_x=(0.5, 1.0)),
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.8, 1.2)),
        Rand3DElasticd(
            keys=["image", "label"],
            prob=0.3,
            sigma_range=(5, 7),
            magnitude_range=(50, 150),
            mode=("bilinear", "nearest"),
        ),
    ])


_AUGMENTATION_MAP = {
    "none": get_none_augmentation,
    "light": get_light_augmentation,
    "medium": get_medium_augmentation,
    "heavy": get_heavy_augmentation,
}


def get_augmentation(name: str, patch_size=(128, 128, 128)) -> Compose:
    """Factory: get augmentation by name string."""
    name = name.lower()
    if name not in _AUGMENTATION_MAP:
        raise ValueError(f"Unknown augmentation '{name}'. Choose from {list(_AUGMENTATION_MAP)}")
    return _AUGMENTATION_MAP[name](patch_size)
