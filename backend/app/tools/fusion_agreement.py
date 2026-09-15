"""Cross-modal agreement: IoU between an optical-derived mask and a
SAR-derived mask for the same physical phenomenon (water, built-up). This
is the confidence signal -- two physically independent sensors concurring
is a stronger trust signal than a single model's softmax score.
"""
from __future__ import annotations

import numpy as np


def compute_agreement(mask_a: np.ndarray, mask_b: np.ndarray) -> dict:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    iou = float(intersection / union) if union > 0 else 0.0
    agreement_mask = mask_a & mask_b
    disagreement_mask = mask_a ^ mask_b
    return {
        "iou": iou,
        "agreement_fraction": float(agreement_mask.mean()),
        "disagreement_fraction": float(disagreement_mask.mean()),
        "agreement_mask": agreement_mask,
        "disagreement_mask": disagreement_mask,
    }
