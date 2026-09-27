"""Bi-temporal change is arithmetic on two already-computed masks --
T2 AND NOT T1 is gained, T1 AND NOT T2 is lost. No CNN, nothing trained.

A minimum-mapping-unit filter drops isolated changed pixels (residual
misregistration and threshold jitter), so reported hectares reflect
contiguous change rather than noise.
"""
from __future__ import annotations

import numpy as np

from app.tools.threshold import clean_mask

SIGNIFICANCE_RATIO = 1.05


def compute_delta(
    mask_t1: np.ndarray,
    mask_t2: np.ndarray,
    gsd_m: float | None,
    label: str,
    valid: np.ndarray | None = None,
    min_pixels: int = 9,
) -> dict:
    pixel_area_ha = ((gsd_m or 10.0) ** 2) / 10_000.0
    if valid is None:
        valid = np.ones_like(mask_t1, dtype=bool)

    gained = clean_mask(mask_t2 & ~mask_t1 & valid, min_pixels)
    lost = clean_mask(mask_t1 & ~mask_t2 & valid, min_pixels)
    n_valid = max(int(valid.sum()), 1)

    gained_ha = float(gained.sum() * pixel_area_ha)
    lost_ha = float(lost.sum() * pixel_area_ha)
    net_ha = gained_ha - lost_ha
    t1_ha = float((mask_t1 & valid).sum() * pixel_area_ha)
    t2_ha = float((mask_t2 & valid).sum() * pixel_area_ha)

    if gained_ha > lost_ha * SIGNIFICANCE_RATIO and gained_ha > 0:
        direction = "increase"
    elif lost_ha > gained_ha * SIGNIFICANCE_RATIO and lost_ha > 0:
        direction = "decrease"
    else:
        direction = "no significant change"

    return {
        "label": label,
        "direction": direction,
        f"{label}_gained_ha": gained_ha,
        f"{label}_lost_ha": lost_ha,
        f"{label}_net_change_ha": net_ha,
        f"{label}_t1_area_ha": t1_ha,
        f"{label}_t2_area_ha": t2_ha,
        "t1_fraction": float((mask_t1 & valid).sum() / n_valid),
        "t2_fraction": float((mask_t2 & valid).sum() / n_valid),
        "changed_fraction": float((gained | lost).sum() / n_valid),
        "relative_change_percent": float(net_ha / t1_ha * 100) if t1_ha > 0 else 0.0,
        "change_mask": gained | lost,
        "gained_mask": gained,
        "lost_mask": lost,
    }
