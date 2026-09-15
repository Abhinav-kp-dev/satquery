"""Confidence is derived from two things this system can actually defend:
whether the narration's claims survived adjudication against measured
evidence, and -- when two independent sensors were used -- how much they
agree. Neither is a raw softmax probability; both are explainable.
"""
from __future__ import annotations

from app.models.schemas import ConfidenceBreakdown, SufficiencyVerdict, Task

_ADJUDICATION_SCORE = {"consistent": 1.0, "overridden": 0.35, "unverified": 0.65}


def compute_confidence(
    adjudication_status: str, task: Task, evidence: dict, sufficiency: SufficiencyVerdict
) -> ConfidenceBreakdown:
    components = {"adjudication": _ADJUDICATION_SCORE.get(adjudication_status, 0.5)}
    basis_parts = [f"answer claims {adjudication_status} with measured evidence"]

    if task == Task.FUSION and "iou" in evidence:
        components["cross_sensor_agreement"] = float(evidence["iou"])
        basis_parts.append(f"optical/SAR mask agreement (IoU {evidence['iou']:.2f})")
        overall = 0.5 * components["adjudication"] + 0.5 * components["cross_sensor_agreement"]
    else:
        overall = components["adjudication"]

    if sufficiency == SufficiencyVerdict.SCOPE_DOWN:
        basis_parts.append("response intentionally scoped to available evidence")

    if overall >= 0.75:
        band = "high"
    elif overall >= 0.5:
        band = "medium"
    else:
        band = "low"

    return ConfidenceBreakdown(overall=round(overall, 3), band=band, basis="; ".join(basis_parts), components=components)
