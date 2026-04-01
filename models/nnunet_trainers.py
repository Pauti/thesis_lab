"""
Custom nnU-Net Trainer subclasses.

For now, only contains a debug trainer with reduced epochs..
"""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_20epochs(nnUNetTrainer):
    """Debug trainer: identical to default nnUNetTrainer but limited to 20 epochs."""

    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        if device is None:
            import torch
            device = torch.device("cuda")
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 20
