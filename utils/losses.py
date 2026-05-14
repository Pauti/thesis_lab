"""
Loss functions for UNETR experiments.

Uses MONAI's DiceCELoss (Dice + Cross-Entropy), matching nnU-Net's default loss
for fair comparison between EXP-02 (nnU-Net) and EXP-03 (UNETR).
"""

from monai.losses import DiceCELoss


def get_loss(include_background: bool = False) -> DiceCELoss:
    """
    Dice + Cross-Entropy loss (equal weighting).

    include_background=False: background class excluded from loss computation,
    matching standard BraTS practice and nnU-Net's DeepSupervisionWrapper behaviour.
    to_onehot_y=True: label tensor is (B,1,H,W,D) with integer class indices.
    softmax=True: applies softmax to model logits before Dice computation.
    """
    return DiceCELoss(
        include_background=include_background,
        to_onehot_y=True,
        softmax=True,
        lambda_dice=1.0,
        lambda_ce=1.0,
    )
