"""
Ordinal classification utilities for the RQ3 multi-task auxiliary head.

Implements the CORAL formulation (Cao, Mirjalili & Raschka 2020, Pattern
Recognition Letters 140:325-331): a K-class ordinal problem is decomposed into
K-1 binary "is the rank > k?" tasks that share a single projection weight and
use K-1 independent bias terms. This guarantees rank-monotonic predicted
probabilities P(y>r_1) >= P(y>r_2) >= ... by construction.

Auxiliary target = ET-volume tertile (low / mid / high). Thresholds are derived
PER FOLD from that fold's training cases only (no leakage), using whole-volume
ET voxel counts. At train time the bin is computed on-the-fly from the cropped
patch label (label == 3), which is consistent with the volume-level thresholds
because the whole tumour fits inside a 128^3 patch in ~96 % of BraTS 2021 cases
(measured 24/25 in a size-stratified sample). Background-centred patches that
contain little ET correctly fall into the "low" bin.

ET label value after nnU-Net remapping: bg=0, NCR=1, ED=2, ET=3.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ET_LABEL = 3  # remapped enhancing-tumour label value


# ── CORAL output layer ──────────────────────────────────────────────────────

class CoralLayer(nn.Module):
    """
    CORAL output layer (Cao et al. 2020): one shared weight vector and K-1
    independent bias terms, producing K-1 logits g(x) + b_k for the binary
    sub-tasks "is the rank greater than k?".
    """

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.coral_weights = nn.Linear(in_features, 1, bias=False)
        # K-1 independent biases, initialised so the thresholds are ordered.
        self.coral_bias = nn.Parameter(
            torch.arange(num_classes - 1, 0, -1, dtype=torch.float32)
            - (num_classes - 1) / 2.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, in_features) -> (B, K-1)
        return self.coral_weights(x) + self.coral_bias


# ── CORAL loss + label/prediction helpers ──────────────────────────────────

def label_to_levels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert integer ordinal labels (B,) in [0, K-1] to extended binary level
    targets (B, K-1): label c -> [1]*c + [0]*(K-1-c).
      c=0 -> [0,0]   c=1 -> [1,0]   c=2 -> [1,1]   (for K=3)
    """
    rng = torch.arange(num_classes - 1, device=labels.device)
    return (labels.unsqueeze(1) > rng.unsqueeze(0)).float()


def coral_loss(logits: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """
    CORAL loss with uniform task weighting (lambda^(k) = 1 for all k, as in all
    experiments of Cao et al. 2020) and mean reduction over the batch.

    Uses the numerically stable identity log(1 - sigmoid(z)) = logsigmoid(z) - z.
    """
    per_task = (
        F.logsigmoid(logits) * levels
        + (F.logsigmoid(logits) - logits) * (1.0 - levels)
    )
    return -torch.sum(per_task, dim=1).mean()


def logits_to_rank(logits: torch.Tensor) -> torch.Tensor:
    """Predicted ordinal rank (B,) in [0, K-1] = number of passed thresholds."""
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)


# ── Auxiliary-target derivation ─────────────────────────────────────────────

def et_count_to_bin(counts: torch.Tensor, t1: float, t2: float) -> torch.Tensor:
    """Bin ET voxel counts (B,) into ordinal classes {0,1,2} via thresholds."""
    return (counts > t1).long() + (counts > t2).long()


def patch_et_counts(labels: torch.Tensor) -> torch.Tensor:
    """
    Count ET voxels (label == 3) per sample in a cropped label batch.
    labels: (B, 1, H, W, D) or (B, H, W, D) int -> returns (B,) float.
    """
    if labels.dim() == 5:
        labels = labels.squeeze(1)
    return (labels == ET_LABEL).flatten(1).sum(dim=1).float()


def compute_fold_thresholds(
    fold: int,
    splits_file: str | Path,
    et_json: str | Path,
    num_classes: int = 3,
) -> tuple[float, float]:
    """
    Per-fold tertile breakpoints from that fold's TRAINING cases only.

    Uses whole-volume ET voxel counts from `et_json`. Returns (t1, t2) so that
    counts <= t1 -> class 0, t1 < counts <= t2 -> class 1, counts > t2 -> 2.
    For num_classes != 3 the (K-1) evenly spaced quantiles are returned, but
    the rest of the pipeline currently assumes K=3.
    """
    splits = json.loads(Path(splits_file).read_text())
    train_ids = splits[fold]["train"]
    et = json.loads(Path(et_json).read_text())
    counts = [et[c]["et"] for c in train_ids if c in et]
    if not counts:
        raise ValueError(
            f"No ET counts found for fold {fold} train cases in {et_json}"
        )
    quantiles = np.linspace(0, 1, num_classes + 1)[1:-1]  # K-1 interior quantiles
    thr = np.quantile(counts, quantiles)
    return float(thr[0]), float(thr[1])
