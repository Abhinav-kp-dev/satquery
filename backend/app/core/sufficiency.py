"""Evidence-sufficiency reasoning -- the system's core differentiator.

A single snapshot can measure what is present (e.g. "41% of this scene is
water") but cannot, by itself, support a claim that requires a baseline
(e.g. "this area is flooded" implies "...compared to how it normally
looks"). Rather than let the narration model guess at a baseline it was
never given, this stage recognizes the mismatch deterministically and
scopes the answer down to what the evidence actually supports.
"""
from __future__ import annotations

from app.models.schemas import InputConfig, SufficiencyVerdict, Task

# Phrasing that presupposes a "before" state, without necessarily using an
# explicit change verb (those are already handled by task classification).
_IMPLIED_BASELINE_LANGUAGE = (
    "flooded", "flooding", "flood", "encroachment", "encroached",
    "illegal construction", "illegal mining", "deforestation", "deforested",
    "damage", "damaged", "eroded", "erosion", "degraded", "degradation",
    "depleted", "decline", "declined", "improved", "improvement",
    "loss of", "gain in", "abnormal", "unusual",
)


def check_sufficiency(
    question: str, config: InputConfig, task: Task
) -> tuple[SufficiencyVerdict, str]:
    q = question.lower()

    if task == Task.CHANGE and config != InputConfig.BI_TEMPORAL_PAIR:
        return (
            SufficiencyVerdict.REQUEST_INPUT,
            "This question asks about change over time, which requires two co-registered "
            "images of the same modality from different dates. Upload a before/after pair to proceed.",
        )

    if task == Task.FUSION and config != InputConfig.CROSS_MODAL_PAIR:
        return (
            SufficiencyVerdict.REQUEST_INPUT,
            "This question asks for combined optical+SAR evidence, which requires a co-registered "
            "optical/SAR pair of the same location and time.",
        )

    if config == InputConfig.SINGLE_IMAGE and any(term in q for term in _IMPLIED_BASELINE_LANGUAGE):
        return (
            SufficiencyVerdict.SCOPE_DOWN,
            "This question implies a comparison against a baseline or prior state, but only one "
            "image was provided. The answer will report what can be measured from this single "
            "acquisition and name what a second (reference) image would add.",
        )

    return (
        SufficiencyVerdict.PROCEED,
        "The uploaded input configuration provides sufficient evidence to answer this question directly.",
    )
