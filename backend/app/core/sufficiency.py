"""Stage 2 -- ASSESS: evidence-sufficiency reasoning, the central novelty.

Before any tool runs, decide whether the evidence that was uploaded can
answer the question that was asked:

- REQUEST_INPUT: the question needs an observation that is structurally
  absent (a change question with one image; a fusion question with no
  SAR). Nothing is measured; the answer names exactly what to upload.
- SCOPE_DOWN: part of the question is answerable. "Is this area flooded?"
  on one image: water extent is measurable, "flooded" presupposes a
  baseline that was never provided. Answer the measurable part, declare
  the boundary, request the baseline.
- PROCEED: the evidence supports a direct answer.

Also checks sensor capability: vegetation cannot be measured from SAR
backscatter with this tool set, built-up needs SWIR or dual-pol, bare soil
needs SWIR.
"""
from __future__ import annotations

from app.models.schemas import ImageManifest, InputConfig, InputManifest, Modality, QuerySpec, SufficiencyVerdict, Target, Task
from app.tools.bands import resolve_optical_bands


def measurable_targets(img: ImageManifest) -> dict[Target, str]:
    """target -> how it is measured (quality note) for one image."""
    out: dict[Target, str] = {}
    if img.modality == Modality.SAR:
        out[Target.WATER] = "SAR low backscatter"
        if img.band_count >= 2:
            out[Target.BUILT_UP] = "SAR double-bounce (co-/cross-pol)"
        return out
    slots = resolve_optical_bands(img.band_count, img.band_names)
    if slots.nir is not None and slots.green is not None:
        out[Target.WATER] = "MNDWI" if slots.swir1 is not None else "NDWI"
    elif slots.green is not None:
        out[Target.WATER] = "RGB brightness heuristic (weak)"
    if slots.nir is not None and slots.red is not None:
        out[Target.VEGETATION] = "NDVI"
    if slots.swir1 is not None and slots.nir is not None:
        out[Target.BUILT_UP] = "NDBI"
    if slots.swir1 is not None and slots.nir is not None and slots.blue is not None:
        out[Target.BARE_SOIL] = "BSI"
    return out


_TARGET_WORDS = {
    Target.WATER: "water",
    Target.VEGETATION: "vegetation",
    Target.BUILT_UP: "built-up area",
    Target.BARE_SOIL: "bare soil / excavation",
}


def _needs_for(target: Target) -> str:
    return {
        Target.VEGETATION: "an optical image with red and near-infrared bands",
        Target.BUILT_UP: "an optical image with a SWIR band (e.g. Sentinel-2 B11) or dual-polarisation SAR",
        Target.BARE_SOIL: "an optical image with blue, red, NIR and SWIR bands",
        Target.WATER: "an optical or SAR image",
    }.get(target, "a compatible image")


def check_sufficiency(spec: QuerySpec, manifest: InputManifest) -> tuple[SufficiencyVerdict, str, str | None]:
    """Returns (verdict, reason, missing_input)."""
    config = manifest.config
    task = spec.task

    if task == Task.CHANGE and config != InputConfig.BI_TEMPORAL_PAIR:
        missing = "an earlier (pre-event) image of the same location and the same sensor type"
        return (
            SufficiencyVerdict.REQUEST_INPUT,
            "This question asks what changed over time, which needs two co-registered images of the same modality "
            f"from different dates. Only {'one image was' if config == InputConfig.SINGLE_IMAGE else 'an optical/SAR pair was'} "
            f"provided. To answer it, upload {missing}.",
            missing,
        )

    if task == Task.FUSION and config != InputConfig.CROSS_MODAL_PAIR:
        have = manifest.images[0].modality
        other = "SAR" if have == Modality.OPTICAL else "optical"
        missing = f"a co-registered {other} image of the same location and date"
        return (
            SufficiencyVerdict.REQUEST_INPUT,
            f"This question asks to combine optical and SAR evidence, but only {have.value} imagery was provided. "
            f"To answer it, upload {missing}.",
            missing,
        )

    if task not in manifest.legal_tasks:
        return (
            SufficiencyVerdict.REQUEST_INPUT,
            f"A {task.value} question is not supported for a {config.value.replace('_', ' ')} input.",
            None,
        )

    # Sensor capability for the requested target.
    if spec.target not in (Target.LAND_COVER,):
        capable = [measurable_targets(img) for img in manifest.images]
        if not any(spec.target in c for c in capable):
            need = _needs_for(spec.target)
            measurable = sorted({t for c in capable for t in c}, key=lambda t: t.value)
            measurable_words = ", ".join(_TARGET_WORDS[t] for t in measurable) or "nothing"
            return (
                SufficiencyVerdict.SCOPE_DOWN,
                f"{_TARGET_WORDS[spec.target].capitalize()} cannot be measured from this input "
                f"({', '.join(f'{i.modality.value} {i.band_count}-band' for i in manifest.images)}). "
                f"The answer will report what can be measured ({measurable_words}); {need} would be needed for the rest.",
                need,
            )

    if config == InputConfig.SINGLE_IMAGE and spec.implied_baseline:
        return (
            SufficiencyVerdict.SCOPE_DOWN,
            "This question implies a comparison against a baseline or normal state, but only one image was provided. "
            "The answer reports what this single acquisition measures and states that a pre-event (reference) image "
            "is needed to establish whether anything changed.",
            "a pre-event image of the same location",
        )

    return (
        SufficiencyVerdict.PROCEED,
        "The uploaded input configuration provides sufficient evidence to answer this question directly.",
        None,
    )
