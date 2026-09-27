"""Converts raw band arrays into 8-bit composites a browser (and, if
configured, a VLM) can display. The VLM never receives the raw multi-band
array -- only this composite plus the tool-computed numbers.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from app.models.schemas import Modality
from app.tools.bands import resolve_optical_bands, resolve_sar_bands
from app.tools.sar_backscatter import to_db, to_linear

MASK_COLORS: dict[str, tuple[int, int, int]] = {
    "water": (56, 132, 255),
    "built_up": (255, 122, 69),
    "vegetation": (86, 199, 118),
    "bare_soil": (214, 170, 92),
    "change": (255, 79, 129),
    "gained": (255, 79, 129),
    "lost": (255, 196, 0),
    "agreement": (167, 139, 250),
    "optical_only": (56, 200, 255),
    "sar_only": (255, 96, 200),
}


def _percentile_stretch(band: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    band = band.astype("float64")
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.zeros(band.shape, dtype="uint8")
    lo_v, hi_v = np.percentile(finite, [lo, hi])
    if hi_v <= lo_v:
        hi_v = lo_v + 1
    stretched = np.clip((np.nan_to_num(band, nan=lo_v) - lo_v) / (hi_v - lo_v), 0, 1)
    return (stretched * 255).astype("uint8")


def render_composite(raw: np.ndarray, band_count: int, modality: Modality, band_names: list[str] | None = None) -> Image.Image:
    if modality == Modality.SAR:
        co_idx, cross_idx = resolve_sar_bands(band_count, band_names)
        co_db = to_db(to_linear(raw[co_idx]))
        if cross_idx is not None:
            cross_db = to_db(to_linear(raw[cross_idx]))
            ratio_db = co_db - cross_db
        else:
            cross_db = co_db
            ratio_db = np.zeros_like(co_db)
        rgb = np.stack([_percentile_stretch(co_db), _percentile_stretch(cross_db), _percentile_stretch(ratio_db)], axis=-1)
        return Image.fromarray(rgb, mode="RGB")

    slots = resolve_optical_bands(band_count, band_names)
    if slots.red is not None and slots.green is not None and slots.blue is not None:
        rgb = np.stack(
            [_percentile_stretch(raw[slots.red]), _percentile_stretch(raw[slots.green]), _percentile_stretch(raw[slots.blue])],
            axis=-1,
        )
    else:
        grey = _percentile_stretch(raw[0])
        rgb = np.stack([grey, grey, grey], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def render_overlay(base: Image.Image, masks: dict[str, np.ndarray], alpha: float = 0.45) -> Image.Image:
    """masks: {label: boolean HxW array}. Later entries painted on top."""
    out = np.array(base.convert("RGB")).astype("float64")
    for label, mask in masks.items():
        if mask is None or not mask.any():
            continue
        color = np.array(MASK_COLORS.get(label, (255, 255, 255)), dtype="float64")
        out[mask] = out[mask] * (1 - alpha) + color * alpha
    return Image.fromarray(out.astype("uint8"), mode="RGB")


def render_layer(mask: np.ndarray, label: str, alpha: int = 150) -> Image.Image:
    """A transparent RGBA PNG of one mask, so the UI can toggle layers."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype="uint8")
    rgba[mask, :3] = MASK_COLORS.get(label, (255, 255, 255))
    rgba[mask, 3] = alpha
    return Image.fromarray(rgba, mode="RGBA")
