"""Converts raw GeoTIFF band arrays into 8-bit composites a browser (and,
if configured, a VLM) can display. The VLM never receives the raw
multi-band array -- only this composite plus the tool-computed numbers.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from app.models.schemas import Modality
from app.tools.bands import resolve_optical_bands, resolve_sar_bands

MASK_COLORS = {
    "water": (56, 132, 255),
    "built_up": (255, 122, 69),
    "vegetation": (86, 199, 118),
    "change": (255, 79, 129),
    "gained": (255, 79, 129),
    "lost": (255, 196, 0),
    "agreement": (167, 139, 250),
}


def _percentile_stretch(band: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    band = band.astype("float64")
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.zeros_like(band, dtype="uint8")
    lo_v, hi_v = np.percentile(finite, [lo, hi])
    if hi_v <= lo_v:
        hi_v = lo_v + 1
    stretched = np.clip((band - lo_v) / (hi_v - lo_v), 0, 1)
    return (stretched * 255).astype("uint8")


def render_composite(raw: np.ndarray, band_count: int, modality: Modality) -> Image.Image:
    if modality == Modality.SAR:
        vv_idx, vh_idx = resolve_sar_bands(band_count)
        vv = np.clip(raw[vv_idx].astype("float64"), 1e-6, None)
        vv_db = 10 * np.log10(vv)
        if vh_idx is not None:
            vh = np.clip(raw[vh_idx].astype("float64"), 1e-6, None)
            vh_db = 10 * np.log10(vh)
            ratio_db = vv_db - vh_db
        else:
            vh_db = vv_db
            ratio_db = np.zeros_like(vv_db)
        rgb = np.stack([_percentile_stretch(vv_db), _percentile_stretch(vh_db), _percentile_stretch(ratio_db)], axis=-1)
        return Image.fromarray(rgb, mode="RGB")

    slots = resolve_optical_bands(band_count)
    if slots.red is not None and slots.green is not None and slots.blue is not None:
        rgb = np.stack(
            [_percentile_stretch(raw[slots.red]), _percentile_stretch(raw[slots.green]), _percentile_stretch(raw[slots.blue])],
            axis=-1,
        )
    else:
        # Single/odd-band fallback: greyscale render of the first band.
        grey = _percentile_stretch(raw[0])
        rgb = np.stack([grey, grey, grey], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def render_overlay(base: Image.Image, masks: dict[str, np.ndarray], alpha: float = 0.45) -> Image.Image:
    """masks: {label: boolean HxW array}. Later entries painted on top."""
    base_rgb = np.array(base.convert("RGB")).astype("float64")
    out = base_rgb.copy()
    for label, mask in masks.items():
        if mask is None or not mask.any():
            continue
        color = np.array(MASK_COLORS.get(label, (255, 255, 255)), dtype="float64")
        out[mask] = out[mask] * (1 - alpha) + color * alpha
    return Image.fromarray(out.astype("uint8"), mode="RGB")
