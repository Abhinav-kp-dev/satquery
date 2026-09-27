import numpy as np

from app.core.planner import waves
from app.models.schemas import PlanNode
from app.tools.bands import resolve_optical_bands
from app.tools.coregistration import coregister
from app.tools.delta_change import compute_delta
from app.tools.georef import GeoContext
from app.tools.grounding import extract_regions, regions_to_geojson
from app.tools.sar_backscatter import compute_sar_backscatter, lee_filter
from app.tools.spectral_index import compute_spectral_indices
from app.tools.threshold import clean_mask, constrained_otsu


def test_sentinel2_band_layouts():
    assert resolve_optical_bands(13).swir1 == 11  # B11, not B10 cirrus
    assert resolve_optical_bands(12).swir1 == 10
    assert resolve_optical_bands(10).nir == 6  # BigEarthNet B02..B12
    named = resolve_optical_bands(4, ["red", "green", "blue", "nir"])
    assert (named.red, named.green, named.blue, named.nir) == (0, 1, 2, 3)


def _four_band(water_rows=20, size=64):
    raw = np.zeros((4, size, size), dtype="float64")
    raw[:] = np.array([0.04, 0.08, 0.05, 0.40])[:, None, None]  # vegetation
    raw[:, :water_rows] = np.array([0.07, 0.06, 0.04, 0.02])[:, None, None]  # water
    return raw * 10_000


def test_ndwi_measures_known_water_fraction():
    res = compute_spectral_indices(_four_band(20), 4, 10.0)
    assert abs(res["ndwi"]["water_fraction"] - 20 / 64) < 1e-6
    assert abs(res["ndwi"]["water_area_ha"] - 20 * 64 * 0.01) < 1e-6
    assert res["ndwi"]["separability"] > 0.9


def test_otsu_is_clamped_to_physical_range():
    idx = np.concatenate([np.full(900, -0.6), np.full(100, -0.2)])  # no water at all
    res = constrained_otsu(idx, None, (-0.05, 0.3), above=True)
    assert res["clamped"] and res["mask"].sum() == 0


def test_min_mapping_unit_removes_speckle():
    m = np.zeros((30, 30), dtype=bool)
    m[5, 5] = True
    m[10:20, 10:20] = True
    assert clean_mask(m, 9).sum() == 100


def test_lee_filter_reduces_speckle_variance():
    rng = np.random.default_rng(0)
    img = rng.gamma(4, 0.25, size=(128, 128))
    assert lee_filter(img).std() < img.std() * 0.7


def test_sar_water_on_speckled_scene():
    rng = np.random.default_rng(1)
    vv = np.full((128, 128), 10 ** (-9 / 10))
    vv[:40] = 10 ** (-21 / 10)
    vh = vv / 10 ** 0.6
    raw = np.stack([vv, vh]) * rng.gamma(4, 0.25, size=(2, 128, 128))
    res = compute_sar_backscatter(raw, 2, 10.0, ["VV", "VH"])
    assert abs(res["water_fraction"] - 40 / 128) < 0.02


def test_sar_accepts_db_input():
    raw = np.full((1, 64, 64), -9.0)
    raw[0, :16] = -22.0
    res = compute_sar_backscatter(raw, 1, 10.0)
    assert res["input_scale"] == "dB" and abs(res["water_fraction"] - 0.25) < 0.02


def test_delta_change_direction_and_area():
    t1 = np.zeros((50, 50), dtype=bool)
    t1[:10] = True
    t2 = t1.copy()
    t2[10:20] = True
    d = compute_delta(t1, t2, 10.0, "water")
    assert d["direction"] == "increase" and d["water_gained_ha"] == 500 * 0.01 and d["water_lost_ha"] == 0


def test_coregistration_recovers_injected_shift():
    rng = np.random.default_rng(3)
    from scipy.ndimage import gaussian_filter

    ref = gaussian_filter(rng.standard_normal((128, 128)), 3)
    moving = np.roll(ref, (3, -2), axis=(0, 1))[None]
    corrected, info = coregister(ref, moving)
    assert info["corrected"] and info["residual_shift_px"] == [-3.0, 2.0]
    assert np.abs(corrected[0, 10:-10, 10:-10] - ref[10:-10, 10:-10]).max() < 1e-6


def test_grounding_regions_with_coordinates():
    from affine import Affine
    from rasterio.crs import CRS

    m = np.zeros((100, 100), dtype=bool)
    m[10:30, 10:40] = True
    m[60:65, 70:75] = True
    geo = GeoContext(transform=Affine(10, 0, 368_000, 0, -10, 2_900_000), crs=CRS.from_epsg(32646))
    regions = extract_regions(m, "water", 10.0, geo)
    assert [r["pixel_count"] for r in regions] == [600, 25]
    assert regions[0]["bbox_px"] == [10, 10, 40, 30] and regions[0]["area_ha"] == 6.0
    lon, lat = regions[0]["centroid_lonlat"]
    assert 91.6 < lon < 91.8 and 26.1 < lat < 26.3
    fc = regions_to_geojson(regions)
    assert len(fc["features"]) == 2 and fc["features"][0]["geometry"]["type"] == "Polygon"


def test_plan_waves_parallelise_independent_nodes():
    nodes = [
        PlanNode(node_id="a#0", tool="spectral_index"),
        PlanNode(node_id="b#1", tool="sar_backscatter"),
        PlanNode(node_id="f", tool="fusion_agreement", depends_on=["a#0", "b#1"]),
        PlanNode(node_id="g", tool="grounding", depends_on=["f"]),
    ]
    assert [[n.node_id for n in w] for w in waves(nodes)] == [["a#0", "b#1"], ["f"], ["g"]]
