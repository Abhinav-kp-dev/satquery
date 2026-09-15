"""SAR backscatter thresholding. Water is a near-specular reflector, so it
returns very little energy to the radar and shows up dark (low VV) in
amplitude/intensity SAR products -- this is standard SAR interpretation,
not a learned behaviour.
"""
from __future__ import annotations

import numpy as np
from skimage.filters import threshold_otsu

from app.tools.bands import resolve_sar_bands

EPS = 1e-6


def _to_db(band: np.ndarray) -> np.ndarray:
    band = band.astype("float64")
    band = np.clip(band, EPS, None)
    return 10 * np.log10(band)


def compute_sar_backscatter(raw: np.ndarray, band_count: int, gsd_m: float | None) -> dict:
    vv_idx, vh_idx = resolve_sar_bands(band_count)
    pixel_area_ha = ((gsd_m or 10.0) ** 2) / 10_000.0

    vv_db = _to_db(raw[vv_idx])
    finite = vv_db[np.isfinite(vv_db)]
    if finite.size == 0 or finite.min() == finite.max():
        water_mask = np.zeros_like(vv_db, dtype=bool)
        thresh = 0.0
    else:
        thresh = float(threshold_otsu(finite))
        water_mask = vv_db < thresh  # low backscatter = water

    result = {
        "mean_vv_db": float(np.mean(vv_db)),
        "water_fraction": float(water_mask.mean()),
        "water_area_ha": float(water_mask.sum() * pixel_area_ha),
        "threshold_db": thresh,
        "mask": water_mask,
        "warnings": [],
    }

    if vh_idx is not None:
        vh_db = _to_db(raw[vh_idx])
        ratio = vv_db - vh_db
        result["mean_vv_minus_vh_db"] = float(np.mean(ratio))
        # High VV+VH with a small VV-VH gap indicates double-bounce reflection,
        # a classic built-up/urban SAR signature.
        builtup_mask = (vv_db > np.percentile(vv_db, 70)) & (np.abs(ratio) < np.percentile(np.abs(ratio), 40))
        result["built_up_fraction"] = float(builtup_mask.mean())
        result["built_up_area_ha"] = float(builtup_mask.sum() * pixel_area_ha)
        result["built_up_mask"] = builtup_mask
    else:
        result["warnings"].append("Single-polarization SAR input; built-up detection needs VV+VH and was skipped.")

    return result
