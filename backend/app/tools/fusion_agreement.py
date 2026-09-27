"""Cross-sensor agreement: IoU between an optical-derived mask and a
SAR-derived mask for the same physical phenomenon. Optical and SAR fail in
different ways (clouds and shadow vs. speckle, wind roughening and layover),
so two physically independent sensors concurring is a stronger trust
signal than any single model's softmax score. This IoU is what calibration
turns into the confidence score.
"""
from __future__ import annotations

import numpy as np


def compute_agreement(mask_a: np.ndarray, mask_b: np.ndarray) -> dict:
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    iou = float(intersection / union) if union > 0 else 0.0
    dice = float(2 * intersection / (mask_a.sum() + mask_b.sum())) if (mask_a.sum() + mask_b.sum()) > 0 else 0.0
    agreement_mask = mask_a & mask_b
    optical_only = mask_a & ~mask_b
    sar_only = mask_b & ~mask_a
    return {
        "iou": iou,
        "dice": dice,
        "agreement_fraction": float(agreement_mask.mean()),
        "disagreement_fraction": float((optical_only | sar_only).mean()),
        "optical_only_fraction": float(optical_only.mean()),
        "sar_only_fraction": float(sar_only.mean()),
        "agreement_mask": agreement_mask,
        "disagreement_mask": optical_only | sar_only,
        "optical_only_mask": optical_only,
        "sar_only_mask": sar_only,
    }
