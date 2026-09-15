"""Pure band arithmetic on optical imagery. No model involved -- these are
the numbers the VLM is handed as text, and the numbers the adjudicator
checks the VLM's claims against.
"""
from __future__ import annotations

import numpy as np
from skimage.filters import threshold_otsu

from app.tools.bands import resolve_optical_bands

EPS = 1e-6


def _safe_index(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype("float64")
    b = b.astype("float64")
    return (a - b) / (a + b + EPS)


def _otsu_mask(index: np.ndarray) -> tuple[np.ndarray, float]:
    finite = index[np.isfinite(index)]
    if finite.size == 0 or finite.min() == finite.max():
        return np.zeros_like(index, dtype=bool), 0.0
    thresh = float(threshold_otsu(finite))
    return index > thresh, thresh


def compute_spectral_indices(raw: np.ndarray, band_count: int, gsd_m: float | None) -> dict:
    """raw: (bands, H, W). Returns NDWI/NDBI/NDVI summaries where computable."""
    slots = resolve_optical_bands(band_count)
    pixel_area_ha = ((gsd_m or 10.0) ** 2) / 10_000.0
    result: dict = {"warnings": list(slots.warnings)}

    if slots.green is not None and slots.nir is not None:
        ndwi = _safe_index(raw[slots.green], raw[slots.nir])
        mask, thresh = _otsu_mask(ndwi)
        result["ndwi"] = {
            "water_fraction": float(mask.mean()),
            "water_area_ha": float(mask.sum() * pixel_area_ha),
            "threshold": thresh,
            "mask": mask,
            "index": ndwi,
        }
    if slots.nir is not None and slots.red is not None:
        ndvi = _safe_index(raw[slots.nir], raw[slots.red])
        mask, thresh = _otsu_mask(ndvi)
        result["ndvi"] = {
            "vegetation_fraction": float(mask.mean()),
            "vegetation_area_ha": float(mask.sum() * pixel_area_ha),
            "threshold": thresh,
            "mask": mask,
            "index": ndvi,
        }
    if slots.swir1 is not None and slots.nir is not None:
        ndbi = _safe_index(raw[slots.swir1], raw[slots.nir])
        mask, thresh = _otsu_mask(ndbi)
        result["ndbi"] = {
            "built_up_fraction": float(mask.mean()),
            "built_up_area_ha": float(mask.sum() * pixel_area_ha),
            "threshold": thresh,
            "mask": mask,
            "index": ndbi,
        }
    if slots.nir is None and slots.green is not None and slots.blue is not None:
        # Visible-only fallback: water is typically darker and less saturated than
        # surrounding land in true-colour RGB. Weaker than NDWI -- flagged as such.
        brightness = (raw[slots.blue].astype("float64") + raw[slots.green].astype("float64")) / 2
        mask, thresh = _otsu_mask(-brightness)
        result["rgb_dark_water_heuristic"] = {
            "water_fraction": float(mask.mean()),
            "water_area_ha": float(mask.sum() * pixel_area_ha),
            "threshold": thresh,
            "mask": mask,
            "index": -brightness,
        }
        result["warnings"].append("Water estimate uses an RGB brightness heuristic, not NDWI -- treat as low-confidence.")

    return result
