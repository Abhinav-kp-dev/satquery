"""Pure band arithmetic on optical imagery. No model involved -- these are
the numbers the VLM is handed as text, and the numbers the adjudicator
checks the VLM's claims against.

Indices:
  NDWI  (McFeeters 1996)  (G - NIR) / (G + NIR)          open water
  MNDWI (Xu 2006)         (G - SWIR1) / (G + SWIR1)      water, robust to built-up
  NDVI                    (NIR - R) / (NIR + R)          vegetation
  NDBI  (Zha 2003)        (SWIR1 - NIR) / (SWIR1 + NIR)  built-up
  BSI   (Rikimaru 2002)   ((SWIR1+R)-(NIR+B))/((SWIR1+R)+(NIR+B))  bare soil / excavation
"""
from __future__ import annotations

import numpy as np

from app.tools.bands import resolve_optical_bands
from app.tools.threshold import clean_mask, constrained_otsu

EPS = 1e-6

# Literature-plausible threshold ranges each index's Otsu split is clamped to.
BOUNDS = {
    "ndwi": (-0.05, 0.3),
    "mndwi": (-0.05, 0.4),
    "ndvi": (0.2, 0.55),
    "ndbi": (-0.05, 0.25),
    "bsi": (0.0, 0.25),
}


def _safe_index(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype("float64")
    b = b.astype("float64")
    return (a - b) / (a + b + EPS)


def _valid_mask(raw: np.ndarray, nodata: float | None) -> np.ndarray:
    valid = np.all(np.isfinite(raw), axis=0)
    if nodata is not None:
        valid &= ~np.all(raw == nodata, axis=0)
    return valid


def _summarise(name: str, label: str, index: np.ndarray, valid: np.ndarray, pixel_area_ha: float) -> dict:
    res = constrained_otsu(index, valid, BOUNDS[name], above=True)
    mask = clean_mask(res["mask"] & valid)
    n_valid = max(int(valid.sum()), 1)
    return {
        f"{label}_fraction": float(mask.sum() / n_valid),
        f"{label}_area_ha": float(mask.sum() * pixel_area_ha),
        "threshold": res["threshold"],
        "otsu_raw": res["otsu_raw"],
        "clamped": res["clamped"],
        "separability": res["separability"],
        "mean_index": float(np.nanmean(np.where(valid, index, np.nan))),
        "mask": mask,
        "index": index,
    }


def _split_built_bare(result: dict, valid: np.ndarray, pixel_area_ha: float) -> None:
    """NDBI and BSI both respond to built-up surfaces *and* bare soil -- a
    well-known confusion. Pixels flagged by either (and by neither water nor
    vegetation index) are assigned by which index dominates: bare soil's
    high red reflectance lifts BSI above NDBI; built-up keeps NDBI above BSI."""
    if "ndbi" not in result:
        return
    exclude = np.zeros(valid.shape, dtype=bool)
    for other in ("mndwi", "ndwi", "ndvi"):
        if other in result:
            exclude |= result[other]["mask"]
    ndbi = result["ndbi"]
    candidate = ndbi["mask"] & ~exclude
    if "bsi" in result:
        bsi = result["bsi"]
        candidate |= bsi["mask"] & ~exclude
        built = clean_mask(candidate & (ndbi["index"] >= bsi["index"]))
        bare = clean_mask(candidate & (bsi["index"] > ndbi["index"]))
        n_valid = max(int(valid.sum()), 1)
        bsi.update(mask=bare, bare_soil_fraction=float(bare.sum() / n_valid), bare_soil_area_ha=float(bare.sum() * pixel_area_ha))
    else:
        built = clean_mask(candidate)
    n_valid = max(int(valid.sum()), 1)
    ndbi.update(mask=built, built_up_fraction=float(built.sum() / n_valid), built_up_area_ha=float(built.sum() * pixel_area_ha))


def compute_spectral_indices(
    raw: np.ndarray,
    band_count: int,
    gsd_m: float | None,
    band_names: list[str] | None = None,
    nodata: float | None = None,
) -> dict:
    """raw: (bands, H, W). Returns per-index summaries where computable."""
    slots = resolve_optical_bands(band_count, band_names)
    pixel_area_ha = ((gsd_m or 10.0) ** 2) / 10_000.0
    valid = _valid_mask(raw, nodata)
    result: dict = {"warnings": list(slots.warnings), "layout": slots.layout, "valid_fraction": float(valid.mean())}

    def band(i: int | None) -> np.ndarray | None:
        return None if i is None else raw[i].astype("float64")

    B, G, R, N, S1 = band(slots.blue), band(slots.green), band(slots.red), band(slots.nir), band(slots.swir1)

    if G is not None and N is not None:
        result["ndwi"] = _summarise("ndwi", "water", _safe_index(G, N), valid, pixel_area_ha)
    if G is not None and S1 is not None:
        result["mndwi"] = _summarise("mndwi", "water", _safe_index(G, S1), valid, pixel_area_ha)
    if N is not None and R is not None:
        result["ndvi"] = _summarise("ndvi", "vegetation", _safe_index(N, R), valid, pixel_area_ha)
    if S1 is not None and N is not None:
        result["ndbi"] = _summarise("ndbi", "built_up", _safe_index(S1, N), valid, pixel_area_ha)
    if S1 is not None and R is not None and N is not None and B is not None:
        bsi_idx = ((S1 + R) - (N + B)) / ((S1 + R) + (N + B) + EPS)
        result["bsi"] = _summarise("bsi", "bare_soil", bsi_idx, valid, pixel_area_ha)
    _split_built_bare(result, valid, pixel_area_ha)

    if N is None and G is not None and B is not None:
        # Visible-only fallback: water is typically darker than surrounding land
        # in true-colour RGB. Weaker than NDWI -- flagged as such.
        brightness = (B + G) / 2
        res = constrained_otsu(-brightness, valid, None, above=True)
        mask = clean_mask(res["mask"] & valid)
        result["rgb_dark_water_heuristic"] = {
            "water_fraction": float(mask.sum() / max(int(valid.sum()), 1)),
            "water_area_ha": float(mask.sum() * pixel_area_ha),
            "threshold": res["threshold"],
            "otsu_raw": res["otsu_raw"],
            "clamped": False,
            "separability": res["separability"],
            "mask": mask,
            "index": -brightness,
        }
        result["warnings"].append("Water estimate uses an RGB brightness heuristic, not NDWI -- treat as low-confidence.")

    return result


def primary_water(res: dict) -> tuple[str, dict] | None:
    """MNDWI is preferred when SWIR is present (it separates water from
    built-up better than NDWI); NDWI otherwise; the RGB heuristic last."""
    for key in ("mndwi", "ndwi", "rgb_dark_water_heuristic"):
        if key in res:
            return key, res[key]
    return None


def primary_for(res: dict, target: str) -> tuple[str, dict] | None:
    if target == "water":
        return primary_water(res)
    key = {"vegetation": "ndvi", "built_up": "ndbi", "bare_soil": "bsi"}.get(target)
    if key and key in res:
        return key, res[key]
    return None
