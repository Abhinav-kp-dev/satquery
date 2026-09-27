from app.core.parser import parse_question
from app.core.preflight import LEGAL_TASKS_BY_CONFIG
from app.core.sufficiency import check_sufficiency
from app.models.schemas import ImageManifest, InputConfig as C, InputManifest, Modality, SufficiencyVerdict as V


def manifest(config, *imgs):
    images = [
        ImageManifest(filename=f"{m}.tif", modality=Modality(m), band_count=b, band_names=[], width=10, height=10, dtype="uint16")
        for m, b in imgs
    ]
    return InputManifest(config=config, images=images, legal_tasks=LEGAL_TASKS_BY_CONFIG[config])


def verdict(q, m):
    spec = parse_question(q, m.config, m.legal_tasks)
    return check_sufficiency(spec, m)


def test_flooded_single_image_scopes_down_and_names_baseline():
    v, reason, missing = verdict("Is this area flooded?", manifest(C.SINGLE_IMAGE, ("optical", 12)))
    assert v == V.SCOPE_DOWN and "pre-event" in missing


def test_change_on_single_image_requests_input():
    v, _, missing = verdict("What changed since last year?", manifest(C.SINGLE_IMAGE, ("optical", 4)))
    assert v == V.REQUEST_INPUT and "earlier" in missing


def test_fusion_without_sar_requests_sar():
    v, _, missing = verdict("Combine optical and SAR to find water", manifest(C.SINGLE_IMAGE, ("optical", 4)))
    assert v == V.REQUEST_INPUT and "SAR" in missing


def test_vegetation_from_sar_is_scoped_by_capability():
    v, reason, missing = verdict("How much vegetation is there?", manifest(C.SINGLE_IMAGE, ("sar", 2)))
    assert v == V.SCOPE_DOWN and "near-infrared" in missing


def test_built_up_needs_swir_on_optical():
    v, _, _ = verdict("How much built-up area is there?", manifest(C.SINGLE_IMAGE, ("optical", 4)))
    assert v == V.SCOPE_DOWN
    v, _, _ = verdict("How much built-up area is there?", manifest(C.SINGLE_IMAGE, ("optical", 12)))
    assert v == V.PROCEED


def test_proceed_on_supported_question():
    v, _, _ = verdict("How much water is there?", manifest(C.SINGLE_IMAGE, ("optical", 4)))
    assert v == V.PROCEED
