"""Georeferencing helpers: map analysis-grid pixels to WGS84 lon/lat."""
from __future__ import annotations

from dataclasses import dataclass

from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import transform as warp_transform
from rasterio.warp import transform_bounds


@dataclass
class GeoContext:
    transform: Affine | None
    crs: CRS | None

    @property
    def has_crs(self) -> bool:
        return self.crs is not None and self.transform is not None

    def pixel_to_lonlat(self, col: float, row: float) -> tuple[float, float]:
        x, y = self.transform * (col, row)
        if self.crs.to_epsg() == 4326:
            return float(x), float(y)
        lon, lat = warp_transform(self.crs, "EPSG:4326", [x], [y])
        return float(lon[0]), float(lat[0])

    def bbox_to_lonlat(self, x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
        ax, ay = self.transform * (x0, y0)
        bx, by = self.transform * (x1, y1)
        left, right = min(ax, bx), max(ax, bx)
        bottom, top = min(ay, by), max(ay, by)
        if self.crs.to_epsg() == 4326:
            return left, bottom, right, top
        return tuple(float(v) for v in transform_bounds(self.crs, "EPSG:4326", left, bottom, right, top))


def bounds_lonlat(crs: CRS | None, bounds) -> list[float] | None:
    if crs is None or bounds is None:
        return None
    try:
        return [round(float(v), 6) for v in transform_bounds(crs, "EPSG:4326", *bounds)]
    except Exception:  # noqa: BLE001 -- malformed CRS: geolocation simply unavailable
        return None
