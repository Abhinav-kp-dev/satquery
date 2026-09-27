"""Visual grounding without a detection head: connected components of a
tool-derived mask become regions with pixel bounding boxes, areas, and --
when the raster is georeferenced -- WGS84 centroids and bounds. The same
regions are exported as GeoJSON.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from app.tools.georef import GeoContext


def extract_regions(
    mask: np.ndarray,
    label: str,
    gsd_m: float | None,
    geo: GeoContext | None = None,
    image_index: int = 0,
    max_regions: int = 8,
    min_pixels: int = 16,
) -> list[dict]:
    if mask is None or not mask.any():
        return []
    pixel_area_ha = ((gsd_m or 10.0) ** 2) / 10_000.0
    h, w = mask.shape
    labels, n = ndimage.label(mask)
    if n == 0:
        return []
    sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    order = np.argsort(sizes)[::-1]
    slices = ndimage.find_objects(labels)
    regions: list[dict] = []
    for rank, idx in enumerate(order[:max_regions]):
        size = int(sizes[idx])
        if size < min_pixels:
            break
        sl = slices[idx]
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        ys, xs = np.nonzero(labels[sl] == idx + 1)
        cy, cx = float(ys.mean() + y0), float(xs.mean() + x0)
        region = {
            "region_id": f"{label}-{rank + 1}",
            "label": label,
            "image_index": image_index,
            "bbox_px": [int(x0), int(y0), int(x1), int(y1)],
            "bbox_norm": [x0 / w, y0 / h, x1 / w, y1 / h],
            "area_ha": round(size * pixel_area_ha, 3),
            "pixel_count": size,
            "centroid_px": [round(cx, 1), round(cy, 1)],
            "centroid_lonlat": None,
            "bbox_lonlat": None,
        }
        if geo is not None and geo.has_crs:
            region["centroid_lonlat"] = [round(v, 6) for v in geo.pixel_to_lonlat(cx + 0.5, cy + 0.5)]
            region["bbox_lonlat"] = [round(v, 6) for v in geo.bbox_to_lonlat(x0, y0, x1, y1)]
        regions.append(region)
    return regions


def regions_to_geojson(regions: list[dict], properties: dict | None = None) -> dict:
    features = []
    for r in regions:
        if not r.get("bbox_lonlat"):
            continue
        w, s, e, n = r["bbox_lonlat"]
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]},
                "properties": {
                    "region_id": r["region_id"],
                    "label": r["label"],
                    "area_ha": r["area_ha"],
                    "centroid_lonlat": r.get("centroid_lonlat"),
                    "ledger_entry_id": r.get("ledger_entry_id"),
                    **(properties or {}),
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}
