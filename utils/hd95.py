"""
Shared HD95 / Dice scoring for retroactive metric computation.

Single source of truth so the from-disk nnU-Net path and the UNETR re-inference
path compute identical numbers. Label convention matches utils/metrics.py and the
inference scripts: bg=0, NCR=1, ED=2, ET=3.

  WT = {1,2,3}   TC = {1,3}   ET = {3}

HD95 = 95th-percentile symmetric Hausdorff distance (MONAI). BraTS is 1 mm
isotropic, so voxel distance == mm. A region absent in either prediction or
ground truth has undefined HD95 and is returned as NaN; callers exclude NaN
from the mean (np.nanmean) and report the contributing-case count separately.
"""

from __future__ import annotations

import numpy as np
import torch
from monai.metrics import HausdorffDistanceMetric

REGIONS = ["WT", "TC", "ET"]

_hd_metric = HausdorffDistanceMetric(include_background=True, percentile=95)


def region_mask(vol: np.ndarray, region: str) -> np.ndarray:
    if region == "WT":
        return vol >= 1
    if region == "TC":
        return np.isin(vol, [1, 3])
    if region == "ET":
        return vol == 3
    raise ValueError(f"unknown region: {region}")


def dice_from_masks(pred_m: np.ndarray, gt_m: np.ndarray) -> float:
    inter = int(np.sum(pred_m & gt_m))
    total = int(np.sum(pred_m)) + int(np.sum(gt_m))
    return 1.0 if total == 0 else 2.0 * inter / total


def hd95_from_masks(pred_m: np.ndarray, gt_m: np.ndarray) -> float:
    """HD95 for a binary region; NaN if either mask is empty (region absent)."""
    if not pred_m.any() or not gt_m.any():
        return float("nan")
    p = torch.from_numpy(np.ascontiguousarray(pred_m)).unsqueeze(0).unsqueeze(0).float()
    g = torch.from_numpy(np.ascontiguousarray(gt_m)).unsqueeze(0).unsqueeze(0).float()
    _hd_metric.reset()
    _hd_metric(y_pred=p, y=g)
    return float(_hd_metric.aggregate())


def score_case(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Per-case Dice + HD95 for all three BraTS regions."""
    out = {}
    for region in REGIONS:
        pm, gm = region_mask(pred, region), region_mask(gt, region)
        r = region.lower()
        out[f"dice_{r}"] = dice_from_masks(pm, gm)
        out[f"hd95_{r}"] = hd95_from_masks(pm, gm)
    return out


def summarise(df) -> dict:
    """Mean Dice and NaN-aware mean/std HD95 (+ contributing-case count) per region."""
    s = {}
    for region in REGIONS:
        r = region.lower()
        s[f"mean_dice_{r}"] = float(df[f"dice_{r}"].mean())
        hd = df[f"hd95_{r}"]
        s[f"mean_hd95_{r}"] = float(np.nanmean(hd))
        s[f"std_hd95_{r}"] = float(np.nanstd(hd))
        s[f"n_hd95_{r}"] = int(hd.notna().sum())
    return s
