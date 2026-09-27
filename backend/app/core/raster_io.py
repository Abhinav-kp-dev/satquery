"""Loads the analysis-grid arrays every tool runs on.

- Reads each raster at its bounded analysis resolution (average resampling;
  GDAL serves this from internal overviews when the file has them, which is
  what keeps gigapixel COGs fast).
- Brings image B of a pair onto image A's analysis grid: reprojection when
  both are georeferenced, pixel-grid resampling otherwise.
- For bi-temporal pairs of the same modality, estimates and corrects any
  residual shift by phase correlation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject

from app.models.schemas import InputConfig, InputManifest, Modality
from app.tools.coregistration import coregister
from app.tools.georef import GeoContext
from app.tools.sar_backscatter import is_db


@dataclass
class LoadedScene:
    arrays: list[np.ndarray]  # float32 (bands, H, W), nodata -> NaN
    geo: GeoContext  # analysis grid of image A
    nodata: list[float | None]


def _read_decimated(path: str, out_h: int, out_w: int) -> tuple[np.ndarray, Affine | None, rasterio.crs.CRS | None, float | None]:
    with rasterio.open(path) as src:
        data = src.read(out_shape=(src.count, out_h, out_w), resampling=Resampling.average).astype("float32")
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        transform = src.transform
        has_geo = transform is not None and not transform.is_identity
        if has_geo:
            transform = transform * Affine.scale(src.width / out_w, src.height / out_h)
        return data, (transform if has_geo else None), src.crs, nodata


def _reproject_onto(path: str, dst_shape: tuple[int, int], dst_transform: Affine, dst_crs) -> np.ndarray:
    with rasterio.open(path) as src:
        out = np.full((src.count, *dst_shape), np.nan, dtype="float32")
        for i in range(src.count):
            reproject(
                source=rasterio.band(src, i + 1),
                destination=out[i],
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
        return out


def load_scene(paths: list[str], manifest: InputManifest) -> LoadedScene:
    a_img = manifest.images[0]
    a, a_transform, a_crs, a_nodata = _read_decimated(paths[0], a_img.analysis_height, a_img.analysis_width)
    arrays = [a]
    nodatas = [a_nodata]

    if len(paths) == 2:
        b_img = manifest.images[1]
        method = manifest.alignment.method if manifest.alignment else "pixel_grid_assumed"
        if method == "reprojected":
            b = _reproject_onto(paths[1], a.shape[1:], a_transform, a_crs)
            b_nodata = None
        else:
            b, _, _, b_nodata = _read_decimated(paths[1], a_img.analysis_height, a_img.analysis_width)

        if manifest.config == InputConfig.BI_TEMPORAL_PAIR and manifest.alignment is not None:
            # Linear SAR intensity is registered in log space, where speckle is additive.
            log_domain = b_img.modality == Modality.SAR and not is_db(b[0])
            fwd = (lambda x: np.log10(np.clip(np.nan_to_num(x, nan=1e-6), 1e-6, None))) if log_domain else np.nan_to_num
            b_corr, info = coregister(fwd(a[0]), fwd(b))
            manifest.alignment.residual_shift_px = info["residual_shift_px"]
            manifest.alignment.shift_corrected = info["corrected"]
            if info["corrected"]:
                b = (np.power(10.0, b_corr) if log_domain else b_corr).astype("float32")
                manifest.alignment.notes.append(
                    f"Residual shift of {info['magnitude_px']} px (dy, dx = {info['residual_shift_px']}) "
                    "estimated by phase correlation and corrected before change measurement."
                )
        arrays.append(b)
        nodatas.append(b_nodata)

    return LoadedScene(arrays=arrays, geo=GeoContext(transform=a_transform, crs=a_crs), nodata=nodatas)
