"""Stage 0 -- deterministic input validation.

Opens each uploaded raster with rasterio and reads band count, band
descriptions, dtype, CRS, bounds, pixel size and acquisition date directly
from the file's own metadata to decide modality, GSD and pair
compatibility. Nothing here is inferred by a model and nothing is learned:
a malformed or incompatible upload is rejected before any model call,
per the problem statement's requirement that the agent "check the number,
modality, format, metadata, and compatibility of the input images."

Large scenes are not rejected. They are assigned a bounded analysis grid
(read decimated, through the file's overviews when present) and the
coarser effective GSD is recorded, so area measurements stay correct.
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import rasterio
from rasterio.warp import transform_bounds

from app.models.schemas import AlignmentRecord, ImageManifest, InputConfig, InputManifest, Modality, Task
from app.tools.bands import looks_like_sar, resolve_optical_bands
from app.tools.georef import bounds_lonlat

SAR_BAND_RANGE = (1, 2)
MIN_PAIR_OVERLAP = 0.5

LEGAL_TASKS_BY_CONFIG: dict[InputConfig, list[Task]] = {
    InputConfig.SINGLE_IMAGE: [Task.VQA, Task.CAPTION, Task.GROUNDING],
    InputConfig.CROSS_MODAL_PAIR: [Task.FUSION, Task.VQA, Task.CAPTION, Task.GROUNDING],
    InputConfig.BI_TEMPORAL_PAIR: [Task.CHANGE, Task.VQA, Task.GROUNDING],
}

_DATE_TAGS = ("ACQUISITION_DATE", "acquisition_date", "DATE_ACQUIRED", "TIFFTAG_DATETIME", "datetime", "SENSING_TIME")


def _detect_modality(band_count: int, names: list[str]) -> Modality:
    if names and looks_like_sar(names):
        return Modality.SAR
    if band_count in SAR_BAND_RANGE:
        return Modality.SAR
    if band_count >= 3:
        return Modality.OPTICAL
    return Modality.UNKNOWN


def _parse_date(tags: dict) -> str | None:
    for key in _DATE_TAGS:
        raw = tags.get(key)
        if not raw:
            continue
        raw = str(raw).strip()
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(raw[: len(datetime.now().strftime(fmt))], fmt).date().isoformat()
            except ValueError:
                continue
    return None


def _read_manifest(path: str, filename: str, analysis_max_dim: int) -> ImageManifest:
    warnings: list[str] = []
    with rasterio.open(path) as src:
        band_count = src.count
        names = [d or "" for d in src.descriptions]
        dtype = src.dtypes[0]
        width, height = src.width, src.height
        crs = src.crs.to_string() if src.crs else None
        transform = src.transform
        has_geotransform = transform is not None and not transform.is_identity
        gsd_m = float(abs(transform.a)) if has_geotransform else None
        if src.crs is not None and src.crs.is_geographic and gsd_m is not None:
            # Degrees -> metres at the scene's latitude.
            lat = (src.bounds.top + src.bounds.bottom) / 2
            gsd_m = gsd_m * 111_320 * math.cos(math.radians(lat))
        bounds = list(src.bounds) if has_geotransform else None
        tags = {**src.tags(), **src.tags(ns="IMAGE_STRUCTURE")}
        acquisition_date = _parse_date(tags)

        if crs is None:
            warnings.append("No CRS in file metadata; results cannot be geolocated and pairs must share a pixel grid.")
        if gsd_m is None:
            warnings.append("No pixel size in metadata; areas assume 10 m GSD.")

        sample = src.read(1, out_shape=(min(height, 512), min(width, 512))).astype("float64")
        nodata = src.nodata
        if nodata is not None:
            nodata_fraction = float(np.mean(sample == nodata))
        else:
            nodata_fraction = float(np.mean(~np.isfinite(sample)))
        if nodata_fraction > 0.15:
            warnings.append(f"{nodata_fraction:.0%} of sampled pixels are nodata/invalid.")

    modality = _detect_modality(band_count, names)
    if modality == Modality.UNKNOWN:
        warnings.append(f"Could not confidently classify modality from {band_count} band(s); treating as optical.")
        modality = Modality.OPTICAL

    layout = "sar_dual_pol" if modality == Modality.SAR and band_count >= 2 else "sar_single_pol"
    if modality == Modality.OPTICAL:
        slots = resolve_optical_bands(band_count, names)
        layout = slots.layout
        warnings.extend(slots.warnings)

    decimation = max(1.0, max(width, height) / analysis_max_dim)
    a_w, a_h = max(1, round(width / decimation)), max(1, round(height / decimation))
    if decimation > 1.0:
        warnings.append(
            f"Scene is {width}x{height}px; analysed on a {a_w}x{a_h} grid "
            f"({decimation:.1f}x decimation, effective GSD {((gsd_m or 10.0) * decimation):.1f} m)."
        )

    return ImageManifest(
        filename=filename,
        modality=modality,
        band_count=band_count,
        band_names=names,
        band_layout=layout,
        width=width,
        height=height,
        gsd_m=gsd_m,
        analysis_width=a_w,
        analysis_height=a_h,
        analysis_gsd_m=(gsd_m * decimation) if gsd_m else None,
        decimation=decimation,
        crs=crs,
        bounds=bounds,
        bounds_lonlat=bounds_lonlat(rasterio.crs.CRS.from_string(crs), bounds) if crs and bounds else None,
        dtype=dtype,
        nodata_fraction=nodata_fraction,
        acquisition_date=acquisition_date,
        warnings=warnings,
    )


def _overlap_fraction(a: ImageManifest, b: ImageManifest) -> float:
    b_bounds = transform_bounds(b.crs, a.crs, *b.bounds) if b.crs != a.crs else tuple(b.bounds)
    left, bottom = max(a.bounds[0], b_bounds[0]), max(a.bounds[1], b_bounds[1])
    right, top = min(a.bounds[2], b_bounds[2]), min(a.bounds[3], b_bounds[3])
    if right <= left or top <= bottom:
        return 0.0
    inter = (right - left) * (top - bottom)
    area_a = (a.bounds[2] - a.bounds[0]) * (a.bounds[3] - a.bounds[1])
    return float(inter / area_a) if area_a > 0 else 0.0


def _invalid(reason: str, images: list[ImageManifest] | None = None) -> InputManifest:
    return InputManifest(config=InputConfig.INVALID, images=images or [], legal_tasks=[], rejection_reason=reason)


def build_manifest(
    paths: list[tuple[str, str]], analysis_max_dim: int, max_native_dim: int = 60_000
) -> tuple[InputManifest, list[tuple[str, str]]]:
    """paths: list of (filesystem_path, original_filename), length 1 or 2.

    Returns the manifest and the paths reordered to match manifest.images
    (bi-temporal pairs are ordered earlier -> later when dates are known)."""
    if len(paths) not in (1, 2):
        return _invalid("Upload exactly one image, or two co-registered images (cross-modal or bi-temporal)."), paths

    try:
        images = [_read_manifest(p, name, analysis_max_dim) for p, name in paths]
    except rasterio.errors.RasterioIOError as exc:
        return _invalid(f"Could not read file as GeoTIFF/TIFF: {exc}"), paths

    for img in images:
        if img.width > max_native_dim or img.height > max_native_dim:
            return _invalid(f"'{img.filename}' is {img.width}x{img.height}px, beyond the {max_native_dim}px limit.", images), paths
        if img.nodata_fraction > 0.95:
            return _invalid(f"'{img.filename}' is almost entirely nodata ({img.nodata_fraction:.0%}).", images), paths

    if len(images) == 1:
        return InputManifest(config=InputConfig.SINGLE_IMAGE, images=images, legal_tasks=LEGAL_TASKS_BY_CONFIG[InputConfig.SINGLE_IMAGE]), paths

    a, b = images
    alignment: AlignmentRecord
    if a.crs and b.crs and a.bounds and b.bounds:
        try:
            overlap = _overlap_fraction(a, b)
        except Exception as exc:  # noqa: BLE001
            return _invalid(f"Could not compare the two images' footprints: {exc}", images), paths
        if overlap < MIN_PAIR_OVERLAP:
            return _invalid(
                f"The two images overlap by only {overlap:.0%} of the first image's footprint; "
                "a two-image query needs both to cover the same place.",
                images,
            ), paths
        same_grid = a.crs == b.crs and a.width == b.width and a.height == b.height and np.allclose(a.bounds, b.bounds)
        alignment = AlignmentRecord(
            method="identical_grid" if same_grid else "reprojected",
            overlap_fraction=round(overlap, 4),
            notes=[] if same_grid else [f"'{b.filename}' was reprojected onto '{a.filename}''s grid ({a.crs})."],
        )
    else:
        if a.width != b.width or a.height != b.height:
            return _invalid(
                f"Image dimensions do not match ({a.width}x{a.height} vs {b.width}x{b.height}) and the files carry "
                "no georeferencing to align them; a two-image query requires co-registered inputs.",
                images,
            ), paths
        alignment = AlignmentRecord(
            method="pixel_grid_assumed",
            notes=["No georeferencing on at least one image; pixel grids assumed to be co-registered."],
        )

    if a.modality != b.modality:
        config = InputConfig.CROSS_MODAL_PAIR
        a.acquisition_tag = a.modality.value
        b.acquisition_tag = b.modality.value
        if a.acquisition_date and b.acquisition_date and a.acquisition_date != b.acquisition_date:
            alignment.notes.append(
                f"Optical and SAR were acquired on different dates ({a.acquisition_date} vs {b.acquisition_date}); "
                "disagreement may reflect real change, not sensor error."
            )
    else:
        config = InputConfig.BI_TEMPORAL_PAIR
        if a.acquisition_date and b.acquisition_date and b.acquisition_date < a.acquisition_date:
            images = [b, a]
            paths = [paths[1], paths[0]]
            alignment.notes.append("Images reordered by acquisition date so T1 is the earlier scene.")
            a, b = images
        a.acquisition_tag, b.acquisition_tag = "T1", "T2"

    return InputManifest(config=config, images=images, legal_tasks=LEGAL_TASKS_BY_CONFIG[config], alignment=alignment), paths
