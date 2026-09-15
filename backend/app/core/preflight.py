"""Deterministic input validation.

Opens each uploaded raster with rasterio and reads band count, dtype, CRS,
and pixel size directly from the file's own metadata to decide modality and
GSD. Nothing here is inferred by a model, and nothing here is learned --
this stage exists specifically so a malformed or incompatible upload is
rejected before any model call happens, per the problem statement's
requirement that the agent "check the number, modality, format, metadata,
and compatibility of the input images."
"""
from __future__ import annotations

import numpy as np
import rasterio

from app.models.schemas import ImageManifest, InputConfig, InputManifest, Modality, Task

# SAR products are near-universally 1-2 bands (VV, VH or VV, HH).
# Optical multispectral scenes (Sentinel-2, Cartosat-2S, etc.) carry 3+ bands.
SAR_BAND_RANGE = (1, 2)

LEGAL_TASKS_BY_CONFIG: dict[InputConfig, list[Task]] = {
    InputConfig.SINGLE_IMAGE: [Task.VQA, Task.CAPTION, Task.GROUNDING],
    InputConfig.CROSS_MODAL_PAIR: [Task.FUSION, Task.VQA, Task.GROUNDING],
    InputConfig.BI_TEMPORAL_PAIR: [Task.CHANGE, Task.VQA],
}


def _detect_modality(band_count: int, dtype: str) -> Modality:
    if band_count in SAR_BAND_RANGE:
        return Modality.SAR
    if band_count >= 3:
        return Modality.OPTICAL
    return Modality.UNKNOWN


def _read_manifest(path: str, filename: str) -> ImageManifest:
    warnings: list[str] = []
    with rasterio.open(path) as src:
        band_count = src.count
        dtype = src.dtypes[0]
        width, height = src.width, src.height
        crs = src.crs.to_string() if src.crs else None
        transform = src.transform
        gsd_m = float(abs(transform.a)) if transform and transform.a else None

        if crs is None:
            warnings.append("No CRS found in file metadata; geolocation of results will be unavailable.")

        sample = src.read(1, out_shape=(min(height, 512), min(width, 512)))
        nodata = src.nodata
        if nodata is not None:
            nodata_fraction = float(np.mean(sample == nodata))
        else:
            nodata_fraction = float(np.mean(~np.isfinite(sample.astype("float64"))))
        if nodata_fraction > 0.15:
            warnings.append(f"{nodata_fraction:.0%} of sampled pixels are nodata/invalid.")

    modality = _detect_modality(band_count, dtype)
    if modality == Modality.UNKNOWN:
        warnings.append(f"Could not confidently classify modality from {band_count} band(s); treating as optical.")
        modality = Modality.OPTICAL

    return ImageManifest(
        filename=filename,
        modality=modality,
        band_count=band_count,
        width=width,
        height=height,
        gsd_m=gsd_m,
        crs=crs,
        dtype=dtype,
        nodata_fraction=nodata_fraction,
        warnings=warnings,
    )


def build_manifest(paths: list[tuple[str, str]], max_dim: int) -> InputManifest:
    """paths: list of (filesystem_path, original_filename), length 1 or 2."""
    if len(paths) not in (1, 2):
        return InputManifest(
            config=InputConfig.INVALID,
            images=[],
            legal_tasks=[],
            rejection_reason="Upload exactly one image, or two co-registered images (cross-modal or bi-temporal).",
        )

    try:
        images = [_read_manifest(p, name) for p, name in paths]
    except rasterio.errors.RasterioIOError as exc:
        return InputManifest(
            config=InputConfig.INVALID,
            images=[],
            legal_tasks=[],
            rejection_reason=f"Could not read file as GeoTIFF/TIFF: {exc}",
        )

    for img in images:
        if img.width > max_dim or img.height > max_dim:
            return InputManifest(
                config=InputConfig.INVALID,
                images=images,
                legal_tasks=[],
                rejection_reason=(
                    f"'{img.filename}' is {img.width}x{img.height}px, exceeding the "
                    f"{max_dim}px prototype cap. Tile the scene before uploading."
                ),
            )

    if len(images) == 1:
        config = InputConfig.SINGLE_IMAGE
    else:
        a, b = images
        if a.width != b.width or a.height != b.height:
            return InputManifest(
                config=InputConfig.INVALID,
                images=images,
                legal_tasks=[],
                rejection_reason=(
                    f"Image dimensions do not match ({a.width}x{a.height} vs "
                    f"{b.width}x{b.height}); a two-image query requires co-registered inputs."
                ),
            )
        if a.modality != b.modality:
            config = InputConfig.CROSS_MODAL_PAIR
            a.acquisition_tag, b.acquisition_tag = "optical" if a.modality == Modality.OPTICAL else "sar", (
                "optical" if b.modality == Modality.OPTICAL else "sar"
            )
        else:
            config = InputConfig.BI_TEMPORAL_PAIR
            a.acquisition_tag, b.acquisition_tag = "T1", "T2"

    return InputManifest(
        config=config,
        images=images,
        legal_tasks=LEGAL_TASKS_BY_CONFIG.get(config, []),
        rejection_reason=None,
    )
