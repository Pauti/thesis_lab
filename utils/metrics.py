"""
BraTS 2021 evaluation metrics for UNETR and nnU-Net experiments.

BraTS region definitions (nnU-Net remapped labels: bg=0, NCR=1, ED=2, ET=3):
  WT (Whole Tumor)     = NCR ∪ ED ∪ ET  → foreground labels {1, 2, 3}
  TC (Tumor Core)      = NCR ∪ ET       → labels {1, 3}
  ET (Enhancing Tumor) = ET             → label {3}

Predictions are one-hot softmax outputs (B, 4, H, W, D) or argmax (B, H, W, D).
Ground truth is integer label map (B, 1, H, W, D) or (B, H, W, D).

Both volumetric Dice (fast, per-epoch) and HD95 (slow, final evaluation) are
provided. HD95 is expensive; only compute it for final validation, not every epoch.
"""

from __future__ import annotations

import numpy as np
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric


def _to_brats_regions(
    pred: torch.Tensor,   # (B, H, W, D) — argmax predictions, integer labels
    gt: torch.Tensor,     # (B, H, W, D) — integer labels
) -> tuple[dict, dict]:
    """
    Extract binary masks for WT, TC, ET from argmax predictions and ground truth.
    Returns dicts keyed by region name with (B, 1, H, W, D) boolean tensors.
    """
    regions_pred = {
        "WT": (pred >= 1).unsqueeze(1).float(),
        "TC": (torch.isin(pred, torch.tensor([1, 3], device=pred.device))).unsqueeze(1).float(),
        "ET": (pred == 3).unsqueeze(1).float(),
    }
    regions_gt = {
        "WT": (gt >= 1).unsqueeze(1).float(),
        "TC": (torch.isin(gt, torch.tensor([1, 3], device=gt.device))).unsqueeze(1).float(),
        "ET": (gt == 3).unsqueeze(1).float(),
    }
    return regions_pred, regions_gt


class BraTSMetrics:
    """
    Computes BraTS region Dice (WT/TC/ET) accumulated over batches.
    Optionally computes HD95 (call compute_hd95=True only for final evaluation).
    """

    def __init__(self):
        self._dice_metric = DiceMetric(
            include_background=True,
            reduction="mean_batch",
            get_not_nans=False,
        )
        self._hd_metric = HausdorffDistanceMetric(
            include_background=True,
            percentile=95,
            reduction="mean_batch",
            get_not_nans=False,
        )
        self._dice_buffers: dict[str, list[float]] = {"WT": [], "TC": [], "ET": []}
        self._hd_buffers: dict[str, list[float]] = {"WT": [], "TC": [], "ET": []}

    def reset(self):
        self._dice_buffers = {"WT": [], "TC": [], "ET": []}
        self._hd_buffers = {"WT": [], "TC": [], "ET": []}

    @torch.no_grad()
    def update(
        self,
        pred_argmax: torch.Tensor,   # (B, H, W, D) integer predictions (CPU or GPU)
        gt: torch.Tensor,            # (B, H, W, D) integer ground truth
        compute_hd95: bool = False,
    ):
        """Accumulate per-case Dice (and optionally HD95) for one batch."""
        pred_argmax = pred_argmax.cpu()
        gt = gt.cpu()

        regions_pred, regions_gt = _to_brats_regions(pred_argmax, gt)

        for region in ["WT", "TC", "ET"]:
            p = regions_pred[region]   # (B, 1, H, W, D)
            g = regions_gt[region]     # (B, 1, H, W, D)

            # Per-case Dice
            for i in range(p.shape[0]):
                pi = p[i:i+1]  # (1, 1, H, W, D)
                gi = g[i:i+1]
                intersection = (pi * gi).sum()
                denom = pi.sum() + gi.sum()
                if denom == 0:
                    dice = 1.0  # both empty → perfect
                else:
                    dice = float(2 * intersection / denom)
                self._dice_buffers[region].append(dice)

            if compute_hd95:
                for i in range(p.shape[0]):
                    pi = p[i:i+1]
                    gi = g[i:i+1]
                    if pi.sum() == 0 or gi.sum() == 0:
                        # Skip HD95 if prediction or GT is empty
                        continue
                    self._hd_metric.reset()
                    self._hd_metric(y_pred=pi, y=gi)
                    hd = float(self._hd_metric.aggregate())
                    self._hd_buffers[region].append(hd)

    def compute(self) -> dict[str, float]:
        """Return mean Dice (and HD95 if computed) across all accumulated cases."""
        result = {}
        for region in ["WT", "TC", "ET"]:
            vals = self._dice_buffers[region]
            if vals:
                result[f"dice_{region.lower()}"] = float(np.mean(vals))
                result[f"dice_{region.lower()}_std"] = float(np.std(vals))
            hd_vals = self._hd_buffers[region]
            if hd_vals:
                result[f"hd95_{region.lower()}"] = float(np.mean(hd_vals))
        return result
