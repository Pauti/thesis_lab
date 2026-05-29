"""
Custom nnU-Net Trainer subclasses for augmentation ablation (RQ2).

All ablation trainers inherit from nnUNetTrainer_Ablation, which overrides
do_split() to read from splits_ablation.json (3-fold CV, 1126 cases,
excluding 125 held-out test cases) instead of splits_final.json.

Tiers:
  - nnUNetTrainer_DefaultAug   → ABL-03: Default augmentation (= standard pipeline, ablation splits)
  - nnUNetTrainer_NoAug        → ABL-01: No augmentation
  - nnUNetTrainer_LightAug     → ABL-02: Spatial-only (rotation + mirror)
  - nnUNetTrainer_HeavyAug     → ABL-03: Default + elastic deformation + extra noise

Each subclass overrides only `get_training_transforms`. All other settings
(optimizer, LR schedule, loss, deep supervision, patch size) are inherited
from the default nnUNetTrainer to isolate augmentation as the single variable.

Usage:
    nnUNetv2_train 001 3d_fullres 0 -tr nnUNetTrainer_DefaultAug
    nnUNetv2_train 001 3d_fullres 0 -tr nnUNetTrainer_NoAug
    nnUNetv2_train 001 3d_fullres 0 -tr nnUNetTrainer_LightAug
    nnUNetv2_train 001 3d_fullres 0 -tr nnUNetTrainer_HeavyAug

Reference: Isensee et al. "nnU-Net: a self-configuring method for deep
learning-based biomedical image segmentation." Nature Methods 18 (2021).
"""

from typing import List, Tuple, Union

import numpy as np
from os.path import join, isfile
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import (
    RemoveRandomConnectedComponentFromOneHotEncodingTransform,
)

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class


# ---------------------------------------------------------------------------
# Splits filename for ablation experiments (3-fold, excluding 10% test set)
# ---------------------------------------------------------------------------

ABLATION_SPLITS_FILE = "splits_ablation.json"


# ---------------------------------------------------------------------------
# Base class for ablation trainers: uses splits_ablation.json
# ---------------------------------------------------------------------------

class nnUNetTrainer_Ablation(nnUNetTrainer):
    """
    Base class for all ablation trainers. Overrides do_split() to read from
    splits_ablation.json (3-fold CV on 1126 cases, excluding 125 test cases)
    instead of the default splits_final.json (5-fold CV on all 1251 cases).
    
    Overrides num_epochs to 100 (instead of default 1000) for ablation experiments.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device=None):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100

    def do_split(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        if self.fold == "all":
            case_identifiers = self.dataset_class.get_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, ABLATION_SPLITS_FILE)
            dataset = self.dataset_class(
                self.preprocessed_dataset_folder,
                identifiers=None,
                folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            )

            if not isfile(splits_file):
                raise FileNotFoundError(
                    f"Ablation splits file not found: {splits_file}\n"
                    f"Run: python scripts/generate_splits.py"
                )

            from batchgenerators.utilities.file_and_folder_operations import load_json
            self.print_to_log_file(f"Using ablation splits from: {splits_file}")
            splits = load_json(splits_file)
            self.print_to_log_file(f"The split file contains {len(splits)} splits.")

            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
            else:
                raise ValueError(
                    f"Requested fold {self.fold} but {splits_file} only has "
                    f"{len(splits)} folds (0-{len(splits)-1})."
                )

            self.print_to_log_file(
                f"Fold {self.fold}: train={len(tr_keys)}, val={len(val_keys)} "
                f"(test set excluded via {ABLATION_SPLITS_FILE})"
            )

        tr_keys.sort()
        val_keys.sort()
        return tr_keys, val_keys


# ---------------------------------------------------------------------------
# Debug trainer (unchanged)
# ---------------------------------------------------------------------------

class nnUNetTrainer_20epochs(nnUNetTrainer):
    """Debug trainer: identical to default nnUNetTrainer but limited to 20 epochs."""

    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        if device is None:
            import torch
            device = torch.device("cuda")
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 20


# ---------------------------------------------------------------------------
# Helper: build the common "tail" transforms (label cleanup, deep supervision)
# These are required by every trainer regardless of augmentation.
# ---------------------------------------------------------------------------

def _common_tail_transforms(
    use_mask_for_norm: List[bool] = None,
    is_cascaded: bool = False,
    foreground_labels: Union[Tuple[int, ...], List[int]] = None,
    regions: List[Union[List[int], Tuple[int, ...], int]] = None,
    ignore_label: int = None,
    deep_supervision_scales: Union[List, Tuple, None] = None,
) -> list:
    """Transforms that must appear at the end of every training pipeline."""
    transforms = []
    if use_mask_for_norm is not None and any(use_mask_for_norm):
        transforms.append(MaskImageTransform(
            apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
            channel_idx_in_seg=0,
            set_outside_to=0,
        ))
    transforms.append(RemoveLabelTansform(-1, 0))
    if is_cascaded:
        assert foreground_labels is not None
        transforms.append(MoveSegAsOneHotToDataTransform(
            source_channel_idx=1, all_labels=foreground_labels,
            remove_channel_from_source=True,
        ))
        transforms.append(RandomTransform(
            ApplyRandomBinaryOperatorTransform(
                channel_idx=list(range(-len(foreground_labels), 0)),
                strel_size=(1, 8), p_per_label=0.5,
            ), apply_probability=0.4,
        ))
        transforms.append(RandomTransform(
            RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                channel_idx=list(range(-len(foreground_labels), 0)),
                fill_with_other_class_p=0, dont_do_if_covers_more_than_x_percent=0.15,
                p_per_label=0.5,
            ), apply_probability=0.2,
        ))
    if regions is not None:
        transforms.append(ConvertSegmentationToRegionsTransform(
            regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
            channel_in_seg=0,
        ))
    if deep_supervision_scales is not None:
        transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
    return transforms


# ---------------------------------------------------------------------------
# ABL-01: No Augmentation
# ---------------------------------------------------------------------------

class nnUNetTrainer_NoAug(nnUNetTrainer_Ablation):
    """
    ABL-01: Disable all data augmentation.

    Uses the official nnUNetTrainerNoDA implementation:
    - Training transforms = Validation transforms (no augmentation)
    - Mirroring disabled for training and inference
    - Only label cleanup + deep supervision downsampling
    
    Note: torch.compile disabled for consistency.
    """

    def _do_i_compile(self):
        # Disable torch.compile for consistency
        return False

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        # Use validation transforms for training (no augmentation!)
        # This is the official nnUNet method from nnUNetTrainerNoDA
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        return nnUNetTrainer.get_validation_transforms(
            deep_supervision_scales, is_cascaded, foreground_labels,
            regions, ignore_label
        )

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        # Disable mirroring for training and inference
        # (copied from nnUNetTrainerNoDA)
        rotation_for_DA, do_dummy_2d_data_aug, _, _ = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        initial_patch_size = self.configuration_manager.patch_size
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes


# ---------------------------------------------------------------------------
# ABL-02: Light Augmentation (spatial only)
# ---------------------------------------------------------------------------

class nnUNetTrainer_LightAug(nnUNetTrainer_Ablation):
    """
    ABL-02: Light augmentation — rotation + mirror only.

    Matches the "Light" tier: basic spatial transforms without any
    intensity augmentation. No elastic deformation, no scaling.
    
    Note: torch.compile disabled for consistency with NoAug trainer.
    """

    def _do_i_compile(self):
        # Disable torch.compile for consistency
        return False

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []

        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        # Rotation only (no scaling, no elastic deformation)
        transforms.append(SpatialTransform(
            patch_size_spatial,
            patch_center_dist_from_border=0,
            random_crop=False,
            p_elastic_deform=0,
            p_rotation=0.2,
            rotation=rotation_for_DA,
            p_scaling=0,
            scaling=(1.0, 1.0),
            p_synchronize_scaling_across_axes=1,
            bg_style_seg_sampling=False,
        ))

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        # Mirror (flipping)
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(MirrorTransform(allowed_axes=mirror_axes))

        # No intensity transforms

        transforms.extend(_common_tail_transforms(
            use_mask_for_norm, is_cascaded, foreground_labels,
            regions, ignore_label, deep_supervision_scales,
        ))
        return ComposeTransforms(transforms)


# ---------------------------------------------------------------------------
# ABL-03: Heavy Augmentation (default + elastic + extra noise)
# ---------------------------------------------------------------------------

class nnUNetTrainer_HeavyAug(nnUNetTrainer_Ablation):
    """
    ABL-03: Heavy augmentation — default pipeline + elastic deformation +
    additional Gaussian noise and blur.

    Changes vs default:
      - p_elastic_deform: 0 → 0.2
      - GaussianNoise probability: 0.1 → 0.2
      - GaussianBlur probability: 0.2 → 0.3
      - Added second GaussianNoise with higher std
      
    Note: torch.compile disabled for consistency with other ablation trainers.
    """

    def _do_i_compile(self):
        # Disable torch.compile for consistency
        return False

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []

        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        # Spatial: rotation + scaling + elastic deformation
        # Parameters follow Isensee et al. (2020, BraTS) §2.4:
        # p_rotation/p_scaling 0.2→0.3, scale range (0.85,1.25)→(0.65,1.6),
        # per-axis scaling, elastic deformation p=0.3
        transforms.append(SpatialTransform(
            patch_size_spatial,
            patch_center_dist_from_border=0,
            random_crop=False,
            p_elastic_deform=0.3,
            elastic_deform_scale=(0.05, 0.2),   # ~5-20% of patch size (smooth)
            elastic_deform_magnitude=(0, 6),    # max displacement in voxels
            p_rotation=0.3,
            rotation=rotation_for_DA,
            p_scaling=0.3,
            scaling=(0.65, 1.6),
            p_synchronize_scaling_across_axes=0,  # per-axis scaling
            bg_style_seg_sampling=False,
        ))

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        # Intensity: increased noise probability
        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1), p_per_channel=1,
                synchronize_channels=True,
            ), apply_probability=0.2,  # ← increased from 0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.), synchronize_channels=False,
                synchronize_axes=False, p_per_channel=0.5, benchmark=True,
            ), apply_probability=0.3,  # ← increased from 0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False, p_per_channel=1,
            ), apply_probability=0.3,  # Isensee 2020 BraTS: brightness p=0.3
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True, synchronize_channels=False,
                p_per_channel=1,
            ), apply_probability=0.15,
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1), synchronize_channels=False,
                synchronize_axes=True, ignore_axes=ignore_axes,
                allowed_channels=None, p_per_channel=0.5,
            ), apply_probability=0.25,
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)), p_invert_image=1,
                synchronize_channels=False, p_per_channel=1,
                p_retain_stats=1,
            ), apply_probability=0.1,
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)), p_invert_image=0,
                synchronize_channels=False, p_per_channel=1,
                p_retain_stats=1,
            ), apply_probability=0.3,
        ))

        # Additional heavy noise (not in default)
        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.2), p_per_channel=1,
                synchronize_channels=False,
            ), apply_probability=0.15,  # ← extra noise with higher variance
        ))

        # Mirror
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(MirrorTransform(allowed_axes=mirror_axes))

        transforms.extend(_common_tail_transforms(
            use_mask_for_norm, is_cascaded, foreground_labels,
            regions, ignore_label, deep_supervision_scales,
        ))
        return ComposeTransforms(transforms)


# ---------------------------------------------------------------------------
# ABL-03 Default: Standard nnU-Net augmentation but with ablation splits
# ---------------------------------------------------------------------------

class nnUNetTrainer_DefaultAug(nnUNetTrainer_Ablation):
    """
    ABL-03 (nnU-Net Default tier): Uses the standard nnUNetTrainer augmentation
    pipeline (unchanged) but reads from splits_ablation.json (3-fold CV,
    excluding 10% test set).

    This is needed because the standard nnUNetTrainer reads splits_final.json
    which includes all 1251 cases. For the ablation study we need the same
    data split (1126 training pool) across all tiers.
    
    Note: torch.compile disabled for consistency with other ablation trainers.
    """
    
    def _do_i_compile(self):
        # Disable torch.compile for consistency
        return False
