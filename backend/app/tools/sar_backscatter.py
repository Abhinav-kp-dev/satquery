"""SAR backscatter analysis.

Water is a near-specular reflector: it returns very little energy to the
radar and shows up dark (low VV/HH). Buildings produce double-bounce
returns, which raise co-pol far more than cross-pol: bright co-pol with a
*large* co-/cross-pol gap. Vegetation's volume scattering does the
opposite (relatively strong cross-pol, small gap). Both are standard SAR
interpretation, not learned behaviour.

Speckle is suppressed first with a Lee filter (Lee 1980) -- thresholding
raw single-look speckle produces salt-and-pepper masks and inflated areas.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from app.tools.bands import resolve_sar_bands
from app.tools.threshold import clean_mask, constrained_otsu

EPS = 1e-6
WATER_DB_BOUNDS = (-24.0, -14.0)
BUILT_DB_BOUNDS = (-6.0, 0.0)


def lee_filter(img: np.ndarray, size: int = 5) -> np.ndarray:
    """Classic Lee filter on linear intensity."""
    img = img.astype("float64")
    mean = uniform_filter(img, size)
    sq_mean = uniform_filter(img * img, size)
    var = np.clip(sq_mean - mean * mean, 0, None)
    noise_var = float(np.mean(var))
    weights = var / (var + noise_var + EPS)
    return mean + weights * (img - mean)


def is_db(band: np.ndarray) -> bool:
    finite = band[np.isfinite(band)]
    return finite.size > 0 and float(np.percentile(finite, 5)) < 0


def to_linear(band: np.ndarray) -> np.ndarray:
    band = band.astype("float64")
    if is_db(band):
        return np.power(10.0, band / 10.0)
    return np.clip(band, EPS, None)


def to_db(linear: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.clip(linear, EPS, None))


def compute_sar_backscatter(
    raw: np.ndarray,
    band_count: int,
    gsd_m: float | None,
    band_names: list[str] | None = None,
    speckle_filter: bool = True,
) -> dict:
    co_idx, cross_idx = resolve_sar_bands(band_count, band_names)
    pixel_area_ha = ((gsd_m or 10.0) ** 2) / 10_000.0
    warnings: list[str] = []

    input_scale = "dB" if is_db(raw[co_idx]) else "linear"
    co_lin = to_linear(raw[co_idx])
    valid = np.isfinite(co_lin) & (co_lin > 0)
    if speckle_filter:
        co_lin = lee_filter(np.where(valid, co_lin, np.nanmedian(co_lin[valid]) if valid.any() else 1.0))
    co_db = to_db(co_lin)

    water = constrained_otsu(co_db, valid, WATER_DB_BOUNDS, above=False)
    water_mask = clean_mask(water["mask"] & valid)
    n_valid = max(int(valid.sum()), 1)

    result = {
        "input_scale": input_scale,
        "speckle_filter": "lee_5x5" if speckle_filter else "none",
        "mean_co_pol_db": float(np.mean(co_db[valid])) if valid.any() else 0.0,
        "water_fraction": float(water_mask.sum() / n_valid),
        "water_area_ha": float(water_mask.sum() * pixel_area_ha),
        "threshold_db": water["threshold"],
        "otsu_raw_db": water["otsu_raw"],
        "separability": water["separability"],
        "mask": water_mask,
        "co_db": co_db,
        "warnings": warnings,
    }

    if cross_idx is not None:
        cross_lin = to_linear(raw[cross_idx])
        if speckle_filter:
            cross_lin = lee_filter(np.where(np.isfinite(cross_lin), cross_lin, EPS))
        cross_db = to_db(cross_lin)
        ratio = co_db - cross_db
        land = valid & ~water_mask
        bright = constrained_otsu(co_db, land, BUILT_DB_BOUNDS, above=True)
        med_ratio = float(np.median(ratio[land])) if land.any() else 0.0
        builtup_mask = clean_mask(bright["mask"] & land & (ratio > med_ratio))
        result["mean_co_minus_cross_db"] = float(np.mean(ratio[valid])) if valid.any() else 0.0
        result["built_up_fraction"] = float(builtup_mask.sum() / n_valid)
        result["built_up_area_ha"] = float(builtup_mask.sum() * pixel_area_ha)
        result["built_up_threshold_db"] = bright["threshold"]
        result["built_up_mask"] = builtup_mask
    else:
        warnings.append("Single-polarization SAR input; built-up detection needs co- and cross-pol and was skipped.")

    return result
