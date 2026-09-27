"""Residual co-registration check for bi-temporal pairs.

Even when two scenes share a grid, orthorectification differences leave a
residual sub-pixel to few-pixel offset, and every pixel of offset shows up
as a spurious ring of "change" along every edge. Phase correlation
(Guizar-Sicairos et al. 2008) on a structure image estimates that offset;
if it is meaningful it is corrected before any change is measured, and the
estimate is recorded in the ledger either way.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import shift as nd_shift
from skimage.registration import phase_cross_correlation

MIN_CORRECT_PX = 0.5
MAX_CORRECT_FRACTION = 0.1  # larger than 10% of the scene is not residual error


def _structure(img: np.ndarray) -> np.ndarray:
    img = np.nan_to_num(img.astype("float64"))
    gy, gx = np.gradient(img)
    mag = np.hypot(gx, gy)
    std = mag.std()
    return (mag - mag.mean()) / std if std > 0 else mag


def estimate_shift(ref: np.ndarray, moving: np.ndarray) -> tuple[float, float, float]:
    """Returns (dy, dx, error) such that shifting `moving` by (dy, dx) aligns it to `ref`."""
    shift, error, _ = phase_cross_correlation(_structure(ref), _structure(moving), upsample_factor=10)
    return float(shift[0]), float(shift[1]), float(error)


def coregister(ref_band: np.ndarray, moving_stack: np.ndarray) -> tuple[np.ndarray, dict]:
    dy, dx, err = estimate_shift(ref_band, moving_stack[0])
    h, w = ref_band.shape
    magnitude = float(np.hypot(dy, dx))
    info = {"residual_shift_px": [round(dy, 2), round(dx, 2)], "magnitude_px": round(magnitude, 2), "error": round(err, 4)}
    if MIN_CORRECT_PX <= magnitude <= MAX_CORRECT_FRACTION * min(h, w):
        corrected = np.stack([nd_shift(b.astype("float64"), (dy, dx), order=1, mode="nearest") for b in moving_stack])
        info["corrected"] = True
        return corrected.astype(moving_stack.dtype if moving_stack.dtype.kind == "f" else "float64"), info
    info["corrected"] = False
    return moving_stack, info
