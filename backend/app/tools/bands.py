"""Canonical band-slot resolution.

No VLM ever sees these raw arrays. This module is the only place that
decides which channel index is "green" or "NIR" for a given raster, so
every downstream tool works off the same resolved slots instead of
re-guessing band order.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BandSlots:
    blue: int | None = None
    green: int | None = None
    red: int | None = None
    nir: int | None = None
    swir1: int | None = None
    warnings: tuple[str, ...] = ()


def resolve_optical_bands(band_count: int) -> BandSlots:
    """Sentinel-2-style 12/13-band stacks and generic 4-band (B,G,R,NIR)
    multispectral scenes are the two layouts this prototype targets --
    those cover BigEarthNet-style training imagery and typical
    Cartosat-class 4-band products respectively."""
    if band_count >= 8:
        # Sentinel-2 L2A 12/13-band order: B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,(B10),B11,B12
        return BandSlots(blue=1, green=2, red=3, nir=7, swir1=min(10, band_count - 1))
    if band_count == 4:
        return BandSlots(blue=0, green=1, red=2, nir=3)
    if band_count == 3:
        return BandSlots(
            blue=0,
            green=1,
            red=2,
            nir=None,
            warnings=("3-band RGB input has no NIR channel; NDWI/NDVI fall back to a visible-only water heuristic.",),
        )
    return BandSlots(warnings=(f"Unrecognized optical band layout ({band_count} bands); indices unavailable.",))


def resolve_sar_bands(band_count: int) -> tuple[int, int | None]:
    """Returns (vv_index, vh_index). Sentinel-1/RISAT dual-pol products are
    ordered VV, VH by convention; single-pol products carry VV only."""
    if band_count >= 2:
        return 0, 1
    return 0, None
