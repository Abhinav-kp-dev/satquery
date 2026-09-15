"""Layer 2 of the router: constrained keyword classification.

Layer 1 (deterministic, done in preflight) narrows the legal task set from
the input configuration alone -- a single image structurally cannot support
change or fusion, no matter what the question says. Layer 2 only picks
among whatever survived that narrowing. This keeps the router auditable:
the judges' stated scoring criterion is the observable execution trace, so
this stays a plain rule table rather than free-form LLM reasoning.
"""
from __future__ import annotations

from app.models.schemas import InputConfig, Task

_KEYWORDS: list[tuple[Task, tuple[str, ...]]] = [
    (
        Task.CHANGE,
        (
            "changed", "change", "increased", "decreased", "increase", "decrease",
            "grown", "shrunk", "since", "before and after", "difference between",
            "compare these", "over time", "new construction", "newly",
        ),
    ),
    (
        Task.FUSION,
        (
            "both images", "combine", "using both", "optical and sar", "sar and optical",
            "fuse", "together", "cross-modal", "cross modal", "each sensor",
        ),
    ),
    (
        Task.GROUNDING,
        (
            "highlight", "locate", "where is", "point to", "show me the", "mark the",
            "find the", "identify the region", "bounding box",
        ),
    ),
    (
        Task.CAPTION,
        (
            "describe", "what is in this", "land cover", "summarize the scene",
            "caption", "overview of this",
        ),
    ),
]


def classify(question: str, legal_tasks: list[Task]) -> tuple[Task, str]:
    """Returns (task, routing_layer_description)."""
    q = question.lower()
    for task, keywords in _KEYWORDS:
        if task in legal_tasks and any(kw in q for kw in keywords):
            return task, f"layer2_keyword_rule:{task.value}"

    if Task.VQA in legal_tasks:
        return Task.VQA, "layer2_default_fallback:vqa"
    return legal_tasks[0], f"layer2_default_fallback:{legal_tasks[0].value}"
