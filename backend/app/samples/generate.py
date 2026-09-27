"""Synthetic, georeferenced demo scenes with known ground truth.

Real Sentinel/Cartosat/RISAT scenes are large and licence-bound, so the
repo ships a generator instead: a physically-motivated label map (river,
lake, forest, cropland, town, bare soil) is rendered into
  - Sentinel-2 L2A-style 12-band surface reflectance (uint16, x10000) and
  - Sentinel-1-style dual-pol VV/VH linear intensity with gamma speckle,
georeferenced in UTM 46N over the Brahmaputra near Guwahati, with band
descriptions and acquisition dates in the GeoTIFF tags. Because the label
maps are known, `ground_truth.json` lets the eval harness score every tool.

These are for demonstrating and testing the pipeline end to end; accuracy
on them is not a claim about accuracy on real imagery.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter

SIZE = 512
GSD = 10.0
CRS = "EPSG:32646"
ORIGIN = (368_000.0, 2_900_000.0)

WATER, FOREST, CROP, BUILT, BARE = 1, 2, 3, 4, 5
CLASS_NAMES = {WATER: "water", FOREST: "vegetation", CROP: "vegetation", BUILT: "built_up", BARE: "bare_soil"}

S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
# Surface reflectance per class, in S2_BANDS order.
REFLECTANCE = {
    WATER: [0.07, 0.07, 0.06, 0.04, 0.03, 0.02, 0.02, 0.02, 0.015, 0.01, 0.01, 0.008],
    FOREST: [0.03, 0.03, 0.06, 0.03, 0.10, 0.25, 0.32, 0.36, 0.38, 0.35, 0.17, 0.08],
    CROP: [0.05, 0.05, 0.08, 0.06, 0.14, 0.26, 0.29, 0.31, 0.32, 0.30, 0.21, 0.12],
    BUILT: [0.12, 0.12, 0.13, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.19, 0.30, 0.27],
    BARE: [0.13, 0.13, 0.17, 0.23, 0.26, 0.28, 0.29, 0.30, 0.31, 0.29, 0.36, 0.33],
}
# (VV dB, VH dB) per class
# Urban double-bounce: bright VV, large VV-VH gap. Forest volume scattering: small gap.
BACKSCATTER = {WATER: (-21.0, -28.0), FOREST: (-8.0, -13.5), CROP: (-11.0, -17.0), BUILT: (-2.5, -11.0), BARE: (-13.0, -21.0)}


def _noise(rng: np.random.Generator, sigma: float) -> np.ndarray:
    n = gaussian_filter(rng.standard_normal((SIZE, SIZE)), sigma)
    return (n - n.mean()) / (n.std() + 1e-9)


def base_labels(seed: int = 7) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype("float64")
    labels = np.full((SIZE, SIZE), CROP, dtype="uint8")

    forest = _noise(rng, 18) > 0.55
    labels[forest] = FOREST

    # Town in the north-east with a road grid.
    town = ((xx - 400) ** 2 / 70**2 + (yy - 110) ** 2 / 55**2) < 1
    labels[town] = BUILT

    # Scattered bare fields.
    bare = (_noise(rng, 6) > 1.9) & ~town
    labels[bare] = BARE

    # Sinuous river, west to east, plus a lake in the south-west.
    centre = 300 + 45 * np.sin(xx / 70.0) + 12 * np.sin(xx / 23.0)
    dist = np.abs(yy - centre)
    river = dist < 11
    lake = ((xx - 120) ** 2 + (yy - 420) ** 2) < 38**2
    labels[river | lake] = WATER
    return labels, {"river_dist": dist, "xx": xx, "yy": yy}


def flooded(labels: np.ndarray, aux: dict, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = labels.copy()
    plain = (aux["river_dist"] < 38 + 22 * _noise(rng, 20)) & (labels != BUILT)
    out[plain] = WATER
    return out


def mined(labels: np.ndarray, aux: dict, seed: int = 13) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = labels.copy()
    xx, yy = aux["xx"], aux["yy"]
    pit = (((xx - 300) ** 2 / 48**2 + (yy - 440) ** 2 / 30**2) < 1 + 0.25 * _noise(rng, 5)) & (labels != WATER)
    haul_road = (np.abs(yy - (440 - (xx - 300) * 0.9)) < 2.5) & (xx > 300) & (xx < 420)
    out[pit | haul_road] = BARE
    return out


def render_optical(labels: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((len(S2_BANDS), SIZE, SIZE), dtype="float64")
    texture = 1 + 0.06 * _noise(rng, 2)
    for cls, refl in REFLECTANCE.items():
        m = labels == cls
        for b, r in enumerate(refl):
            out[b][m] = r
    out *= texture
    out *= 1 + 0.03 * rng.standard_normal(out.shape)
    return np.clip(out * 10_000, 1, 10_000).astype("uint16")


def render_sar(labels: np.ndarray, seed: int, looks: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((2, SIZE, SIZE), dtype="float64")
    for cls, (vv, vh) in BACKSCATTER.items():
        m = labels == cls
        out[0][m] = 10 ** (vv / 10)
        out[1][m] = 10 ** (vh / 10)
    speckle = rng.gamma(looks, 1.0 / looks, size=out.shape)
    return (out * speckle).astype("float32")


def write_tif(path: Path, data: np.ndarray, names: list[str], date: str) -> None:
    profile = dict(
        driver="GTiff", width=SIZE, height=SIZE, count=data.shape[0], dtype=str(data.dtype), crs=CRS,
        transform=from_origin(ORIGIN[0], ORIGIN[1], GSD, GSD), compress="deflate", tiled=True, blockxsize=256, blockysize=256,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        for i, n in enumerate(names, start=1):
            dst.set_band_description(i, n)
        dst.update_tags(ACQUISITION_DATE=date)


def _truth(labels: np.ndarray) -> dict:
    ha = GSD * GSD / 10_000
    out = {}
    for name in ("water", "vegetation", "built_up", "bare_soil"):
        m = np.isin(labels, [c for c, n in CLASS_NAMES.items() if n == name])
        out[name] = {"fraction": round(float(m.mean()), 5), "area_ha": round(float(m.sum() * ha), 2)}
    return out


def _change_truth(t1: np.ndarray, t2: np.ndarray, name: str) -> dict:
    ha = GSD * GSD / 10_000
    codes = [c for c, n in CLASS_NAMES.items() if n == name]
    a, b = np.isin(t1, codes), np.isin(t2, codes)
    return {"indicator": name, "gained_ha": round(float((b & ~a).sum() * ha), 2), "lost_ha": round(float((a & ~b).sum() * ha), 2)}


SAMPLES = [
    {
        "sample_id": "flood_single",
        "config": "single_image",
        "title": "Post-flood optical scene",
        "description": "One Sentinel-2-style image after monsoon flooding on the Brahmaputra. Shows evidence-sufficiency: 'is this flooded?' cannot be settled without a baseline.",
        "files": ["s2_2024-07-08_post.tif"],
        "suggested_questions": [
            "Is this area flooded?",
            "How many hectares of water are in this scene?",
            "Describe the land cover in this image.",
            "Where are the largest water bodies? Give coordinates.",
        ],
    },
    {
        "sample_id": "flood_pair",
        "config": "bi_temporal_pair",
        "title": "Flood: before / after",
        "description": "Two Sentinel-2-style acquisitions of the same place, pre- and post-flood. Change is measured in hectares after residual co-registration.",
        "files": ["s2_2024-03-15_pre.tif", "s2_2024-07-08_post.tif"],
        "suggested_questions": [
            "How much did the water extent change between the two dates?",
            "Where did new flooding occur?",
            "Is this area flooded?",
        ],
    },
    {
        "sample_id": "optical_sar",
        "config": "cross_modal_pair",
        "title": "Optical + SAR, same day",
        "description": "Co-registered optical and dual-pol SAR over the flooded scene. Agreement between the two sensors becomes the confidence score.",
        "files": ["s2_2024-07-08_post.tif", "s1_2024-07-08_vvvh.tif"],
        "suggested_questions": [
            "Use both sensors to map the water. Do they agree?",
            "How much built-up area is there?",
            "Describe this scene.",
        ],
    },
    {
        "sample_id": "mining_pair",
        "config": "bi_temporal_pair",
        "title": "New excavation between dates",
        "description": "Before/after pair where an open-cast pit and haul road appear in the south of the scene.",
        "files": ["s2_2023-11-02_baseline.tif", "s2_2024-11-05_pit.tif"],
        "suggested_questions": [
            "Has any new mining or excavation appeared between these dates?",
            "Where is the new bare soil? Give coordinates.",
        ],
    },
    {
        "sample_id": "sar_single",
        "config": "single_image",
        "title": "SAR only",
        "description": "A single dual-pol SAR scene. Shows the agent declining or scoping questions SAR alone cannot answer.",
        "files": ["s1_2024-07-08_vvvh.tif"],
        "suggested_questions": [
            "How much of this scene is water?",
            "How much vegetation is there?",
            "What changed since last year?",
        ],
    },
]


def generate(out_dir: Path, force: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "ground_truth.json"
    if marker.exists() and not force:
        return out_dir

    labels, aux = base_labels()
    post = flooded(labels, aux)
    pit = mined(labels, aux)

    write_tif(out_dir / "s2_2024-03-15_pre.tif", render_optical(labels, 21), S2_BANDS, "2024-03-15")
    write_tif(out_dir / "s2_2024-07-08_post.tif", render_optical(post, 22), S2_BANDS, "2024-07-08")
    write_tif(out_dir / "s1_2024-07-08_vvvh.tif", render_sar(post, 23), ["VV", "VH"], "2024-07-08")
    write_tif(out_dir / "s2_2023-11-02_baseline.tif", render_optical(labels, 24), S2_BANDS, "2023-11-02")
    # Shift the later mining scene by (2, -1) px to exercise co-registration.
    write_tif(out_dir / "s2_2024-11-05_pit.tif", np.roll(render_optical(pit, 25), (2, -1), axis=(1, 2)), S2_BANDS, "2024-11-05")

    truth = {
        "gsd_m": GSD,
        "scenes": {
            "s2_2024-03-15_pre.tif": _truth(labels),
            "s2_2024-07-08_post.tif": _truth(post),
            "s1_2024-07-08_vvvh.tif": _truth(post),
            "s2_2023-11-02_baseline.tif": _truth(labels),
            "s2_2024-11-05_pit.tif": _truth(pit),
        },
        "changes": {
            "flood_pair": _change_truth(labels, post, "water"),
            "mining_pair": _change_truth(labels, pit, "bare_soil"),
        },
        "injected_shift_px": {"mining_pair": [2, -1]},
    }
    (out_dir / "samples.json").write_text(json.dumps(SAMPLES, indent=2))
    marker.write_text(json.dumps(truth, indent=2))
    return out_dir


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "storage" / "samples"
    print(generate(target, force=True))
