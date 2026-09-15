"""Bi-temporal change is arithmetic on two already-computed index masks --
subtract T1 from T2, threshold the difference. No CNN, nothing trained.
"""
from __future__ import annotations

import numpy as np


def compute_delta(mask_t1: np.ndarray, mask_t2: np.ndarray, gsd_m: float | None, label: str) -> dict:
    pixel_area_ha = ((gsd_m or 10.0) ** 2) / 10_000.0

    gained = mask_t2 & ~mask_t1
    lost = mask_t1 & ~mask_t2
    unchanged_present = mask_t1 & mask_t2

    gained_ha = float(gained.sum() * pixel_area_ha)
    lost_ha = float(lost.sum() * pixel_area_ha)
    net_ha = gained_ha - lost_ha

    if gained_ha > lost_ha * 1.05:
        direction = "increase"
    elif lost_ha > gained_ha * 1.05:
        direction = "decrease"
    else:
        direction = "no significant change"

    return {
        "label": label,
        "direction": direction,
        f"{label}_gained_ha": gained_ha,
        f"{label}_lost_ha": lost_ha,
        f"{label}_net_change_ha": net_ha,
        "t1_fraction": float(mask_t1.mean()),
        "t2_fraction": float(mask_t2.mean()),
        "change_mask": gained | lost,
        "gained_mask": gained,
        "lost_mask": lost,
    }
