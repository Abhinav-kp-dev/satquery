"""Thresholding shared by every index/backscatter tool.

Plain Otsu picks whichever split maximises between-class variance, which on
a multi-class scene is not necessarily the water/land split -- in a scene of
water, crops and a town, NDWI's largest gap can fall between vegetation and
everything else. So Otsu's threshold is clamped to the range the remote
sensing literature treats as physically plausible for that index, and both
the raw and the clamped value are reported.

Otsu's effectiveness metric (eta = between-class variance / total variance)
is also returned: it is 1.0 for a perfectly bimodal histogram and falls
towards 0 when there is no clean split. Calibration uses it as the
"how cleanly did this measurement separate" signal.
"""
from __future__ import annotations

import numpy as np
from skimage.filters import threshold_otsu

MAX_SAMPLE = 400_000


def _sample(values: np.ndarray, rng_seed: int = 0) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size > MAX_SAMPLE:
        rng = np.random.default_rng(rng_seed)
        finite = rng.choice(finite, MAX_SAMPLE, replace=False)
    return finite


def separability(values: np.ndarray, thresh: float) -> float:
    finite = _sample(values)
    if finite.size < 2:
        return 0.0
    total_var = float(finite.var())
    if total_var <= 0:
        return 0.0
    lo, hi = finite[finite <= thresh], finite[finite > thresh]
    if lo.size == 0 or hi.size == 0:
        return 0.0
    w0, w1 = lo.size / finite.size, hi.size / finite.size
    between = w0 * w1 * (float(lo.mean()) - float(hi.mean())) ** 2
    return float(np.clip(between / total_var, 0.0, 1.0))


def constrained_otsu(
    values: np.ndarray,
    valid: np.ndarray | None = None,
    bounds: tuple[float, float] | None = None,
    above: bool = True,
) -> dict:
    """Returns {mask, threshold, otsu_raw, clamped, separability}.

    above=True means the class of interest is values > threshold."""
    data = values if valid is None else np.where(valid, values, np.nan)
    finite = _sample(data)
    if finite.size == 0 or float(finite.min()) == float(finite.max()):
        return {
            "mask": np.zeros(values.shape, dtype=bool),
            "threshold": 0.0,
            "otsu_raw": 0.0,
            "clamped": False,
            "separability": 0.0,
        }
    raw = float(threshold_otsu(finite))
    thresh = raw
    if bounds is not None:
        thresh = float(np.clip(raw, bounds[0], bounds[1]))
    with np.errstate(invalid="ignore"):
        mask = (data > thresh) if above else (data < thresh)
    mask = np.nan_to_num(mask, nan=False).astype(bool)
    return {
        "mask": mask,
        "threshold": thresh,
        "otsu_raw": raw,
        "clamped": thresh != raw,
        "separability": separability(data, thresh),
    }


def clean_mask(mask: np.ndarray, min_pixels: int = 9) -> np.ndarray:
    """Minimum-mapping-unit filter: drop connected blobs smaller than
    `min_pixels`. Removes salt-and-pepper noise that would otherwise inflate
    change areas and grounding region counts."""
    if min_pixels <= 1 or not mask.any():
        return mask
    from scipy import ndimage

    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_pixels
    keep[0] = False
    return keep[labels]
