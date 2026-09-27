"""Canonical band-slot resolution.

No VLM ever sees these raw arrays. This module is the only place that
decides which channel index is "green" or "NIR" for a given raster, so
every downstream tool works off the same resolved slots instead of
re-guessing band order.

Resolution order:
1. Band descriptions stored in the GeoTIFF itself ("B04", "red", "nir",
   "VV" ...). This is the only way to be certain, so it always wins.
2. Known band-count layouts (Sentinel-2 L2A 12/13-band, BigEarthNet
   10-band, generic 4-band B/G/R/NIR, 3-band RGB).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BandSlots:
    blue: int | None = None
    green: int | None = None
    red: int | None = None
    nir: int | None = None
    swir1: int | None = None
    swir2: int | None = None
    layout: str = "unknown"
    warnings: tuple[str, ...] = field(default_factory=tuple)


# Sentinel-2 band id -> canonical slot
_S2_IDS = {
    "B02": "blue", "B2": "blue",
    "B03": "green", "B3": "green",
    "B04": "red", "B4": "red",
    "B08": "nir", "B8": "nir",
    "B11": "swir1",
    "B12": "swir2",
}
_WORDS = {
    "blue": "blue", "green": "green", "red": "red",
    "nir": "nir", "near-infrared": "nir", "near infrared": "nir", "nir1": "nir",
    "swir": "swir1", "swir1": "swir1", "swir16": "swir1", "swir-1": "swir1",
    "swir2": "swir2", "swir22": "swir2", "swir-2": "swir2",
}


def _slot_for_name(name: str) -> str | None:
    n = name.strip().lower()
    if not n:
        return None
    up = name.strip().upper()
    if up in _S2_IDS:
        return _S2_IDS[up]
    return _WORDS.get(n)


def resolve_from_names(names: list[str]) -> BandSlots | None:
    slots: dict[str, int] = {}
    for i, name in enumerate(names):
        if not name:
            continue
        slot = _slot_for_name(name)
        if slot and slot not in slots:
            slots[slot] = i
    if not {"red", "green"} <= slots.keys():
        return None
    return BandSlots(**slots, layout="band_descriptions")


def resolve_optical_bands(band_count: int, names: list[str] | None = None) -> BandSlots:
    if names:
        named = resolve_from_names(names)
        if named is not None:
            return named

    if band_count == 13:
        # Sentinel-2 L1C: B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B10,B11,B12
        return BandSlots(blue=1, green=2, red=3, nir=7, swir1=11, swir2=12, layout="sentinel2_13band")
    if band_count == 12:
        # Sentinel-2 L2A (no B10): B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12
        return BandSlots(blue=1, green=2, red=3, nir=7, swir1=10, swir2=11, layout="sentinel2_12band")
    if band_count == 10:
        # BigEarthNet S2 10-band: B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12
        return BandSlots(blue=0, green=1, red=2, nir=6, swir1=8, swir2=9, layout="bigearthnet_10band")
    if band_count == 4:
        return BandSlots(blue=0, green=1, red=2, nir=3, layout="bgrn_4band")
    if band_count == 3:
        return BandSlots(
            blue=0,
            green=1,
            red=2,
            layout="rgb_3band",
            warnings=("3-band RGB input has no NIR channel; NDWI/NDVI fall back to a visible-only water heuristic.",),
        )
    if band_count >= 5:
        return BandSlots(
            blue=0, green=1, red=2, nir=3, layout=f"assumed_bgrn_{band_count}band",
            warnings=(f"Unrecognised {band_count}-band layout without band descriptions; assuming B,G,R,NIR in the first four bands.",),
        )
    return BandSlots(warnings=(f"Unrecognized optical band layout ({band_count} bands); indices unavailable.",))


def resolve_sar_bands(band_count: int, names: list[str] | None = None) -> tuple[int, int | None]:
    """Returns (co-pol index, cross-pol index). Sentinel-1/RISAT dual-pol
    products are ordered VV, VH by convention; band descriptions override."""
    if names:
        lowered = [n.strip().upper() for n in names]
        co = next((i for i, n in enumerate(lowered) if n in ("VV", "HH")), None)
        cross = next((i for i, n in enumerate(lowered) if n in ("VH", "HV")), None)
        if co is not None:
            return co, cross
    if band_count >= 2:
        return 0, 1
    return 0, None


def looks_like_sar(names: list[str]) -> bool:
    return any(n.strip().upper() in ("VV", "VH", "HH", "HV") for n in names if n)
