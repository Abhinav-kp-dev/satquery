"""Stage 6 -- CALIBRATE: confidence from signals this system can defend,
never a model's self-reported certainty.

Components (each 0..1, each read from the Evidence Ledger):
  adjudication            did the narration's numbers survive checking?
  measurement_separability how cleanly the index/backscatter histogram split
                          at the chosen threshold (Otsu's eta)
  cross_sensor_agreement  IoU between independently derived optical and
                          SAR masks -- two sensors with different failure
                          modes concurring (cross-modal inputs only)
  data_quality            valid-pixel fraction, weak fallbacks (RGB-only
                          water), heavy decimation, assumed co-registration

The overall score is a weighted geometric mean, so one weak link pulls
the whole score down rather than being averaged away. It is a transparent
heuristic, not a calibrated probability -- the component breakdown is
shown so a reader can judge it.
"""
from __future__ import annotations

import math

from app.models.schemas import ConfidenceBreakdown, InputManifest, SufficiencyVerdict, Target

_ADJUDICATION = {"consistent": 1.0, "unverified": 0.75, "corrected": 0.55}
_WEIGHTS = {"adjudication": 0.3, "measurement_separability": 0.3, "cross_sensor_agreement": 0.25, "data_quality": 0.15}


def _band(x: float) -> str:
    return "high" if x >= 0.75 else ("medium" if x >= 0.5 else "low")


def compute_confidence(
    adjudication_status: str,
    target: Target,
    measurements: dict[str, float],
    sufficiency: SufficiencyVerdict,
    manifest: InputManifest,
    weak_fallback: bool = False,
) -> ConfidenceBreakdown:
    components: dict[str, float] = {}
    notes: dict[str, str] = {}

    components["adjudication"] = _ADJUDICATION.get(adjudication_status, 0.6)
    notes["adjudication"] = f"narrated figures {adjudication_status} against the ledger"

    seps = [v for k, v in measurements.items() if "_separability" in k and (target == Target.LAND_COVER or k.startswith(target.value))]
    if not seps:
        seps = [v for k, v in measurements.items() if "_separability" in k]
    if seps:
        # eta ~0.5+ is already a clean split for real imagery; rescale so 0.6 -> 1.0.
        components["measurement_separability"] = round(min(1.0, (sum(seps) / len(seps)) / 0.6), 3)
        notes["measurement_separability"] = f"mean Otsu separability {sum(seps) / len(seps):.2f} over {len(seps)} measurement(s)"

    if "iou" in measurements:
        components["cross_sensor_agreement"] = round(measurements["iou"], 3)
        notes["cross_sensor_agreement"] = f"optical vs SAR mask IoU {measurements['iou']:.2f}"

    dq = 1.0
    dq_notes = []
    for img in manifest.images:
        dq *= 1.0 - min(img.nodata_fraction, 0.9)
        if img.decimation > 4:
            dq *= 0.9
            dq_notes.append(f"{img.decimation:.0f}x decimated")
    if weak_fallback:
        dq *= 0.6
        dq_notes.append("RGB-only water heuristic")
    if manifest.alignment and manifest.alignment.method == "pixel_grid_assumed":
        dq *= 0.85
        dq_notes.append("co-registration assumed, not verified")
    if manifest.alignment and max(abs(v) for v in manifest.alignment.residual_shift_px) > 2:
        dq *= 0.9
        dq_notes.append("residual misregistration > 2 px corrected")
    components["data_quality"] = round(dq, 3)
    notes["data_quality"] = ", ".join(dq_notes) or "no data-quality issues detected"

    total_w = sum(_WEIGHTS[k] for k in components)
    overall = math.exp(sum(_WEIGHTS[k] * math.log(max(v, 1e-3)) for k, v in components.items()) / total_w)

    basis = [notes["adjudication"]]
    if "cross_sensor_agreement" in components:
        basis.append(notes["cross_sensor_agreement"])
    if "measurement_separability" in components:
        basis.append(notes["measurement_separability"])
    if sufficiency == SufficiencyVerdict.SCOPE_DOWN:
        overall = min(overall, 0.74)
        basis.append("answer deliberately scoped to what the evidence supports")

    return ConfidenceBreakdown(
        overall=round(overall, 3), band=_band(overall), basis="; ".join(basis), components=components, component_notes=notes
    )
