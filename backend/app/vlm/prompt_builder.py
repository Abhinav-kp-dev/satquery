"""Specialist prompt templates -- the no-training substitute for one LoRA
adapter per task family. Each task gets its own instruction template; the
model itself is never retrained, only conditioned differently per call.
"""
from __future__ import annotations

from app.config import settings
from app.models.schemas import SufficiencyVerdict, Task
from app.vlm.retrieval import get_example_pool

DOMAIN_CONTEXT = """You are a remote-sensing imagery analyst. You are shown a rendered composite of a \
satellite scene alongside evidence already measured by deterministic tools -- treat that evidence as \
ground truth, not as something to re-derive from the image yourself.

Key terms: NDWI (a water index derived from green/near-infrared reflectance), NDBI (a built-up index \
from short-wave-infrared/near-infrared reflectance), NDVI (a vegetation index from near-infrared/red \
reflectance), SAR backscatter (radar reflection strength in decibels -- low VV backscatter indicates \
water because it reflects radar away from the sensor; high VV with a small VV-VH gap indicates \
built-up surfaces, from double-bounce reflection off walls and ground), GSD (ground sample distance, \
the resolution of the image in metres per pixel)."""

TASK_TEMPLATES: dict[Task, str] = {
    Task.VQA: "Answer the question about this scene using only the tool evidence provided. Do not invent numbers not present in the evidence.",
    Task.CAPTION: "Describe the scene and its dominant land cover, grounding any quantitative statement in the tool evidence provided.",
    Task.GROUNDING: "Identify and describe the location of the region the question refers to, in relative terms (e.g. 'northeast quadrant'). An overlay will render the corresponding tool-derived mask.",
    Task.CHANGE: "Compare the two time points using the change evidence provided. State the direction of change and the measured area, in hectares, of the change.",
    Task.FUSION: "Combine the optical-derived and SAR-derived evidence below. Explicitly state where the two sensors agree and where they disagree, using the agreement/IoU figure provided.",
}

SCOPE_DOWN_INSTRUCTION = """The question implies a comparison against a baseline or prior state that this \
single image cannot provide. Answer only what the tool evidence directly supports, then plainly state \
what additional input (e.g. a prior-date image) would be needed to fully answer the question. Do not \
guess at whether a change or anomaly has occurred."""

OUTPUT_FORMAT_INSTRUCTION = """Respond in exactly this format, with no text outside the tags:
<ANSWER>your natural-language answer, written for a non-specialist reader</ANSWER>

Wrap every specific number you state (a percentage, an area, an index value) in a CLAIM tag inside the \
ANSWER, like this: <CLAIM value="41.2" unit="percent">41.2%</CLAIM>. Every claimed number must come \
directly from the tool evidence below -- never state a number that is not present in it."""


def format_evidence_block(evidence: dict) -> str:
    lines = []
    for key, value in evidence.items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.3f}")
        else:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines) if lines else "- (no numeric evidence available)"


def build_prompt(
    question: str,
    task: Task,
    evidence: dict,
    sufficiency: SufficiencyVerdict,
    manifest_line: str,
) -> tuple[str, list[str]]:
    pool = get_example_pool()
    examples = pool.retrieve(question, task, k=settings.few_shot_k)

    used_ids = [f"{task.value}:{ex['question'][:40]}" for ex in examples]

    if sufficiency == SufficiencyVerdict.SCOPE_DOWN:
        suff_example = pool.force_sufficiency_example()
        if suff_example and suff_example not in examples:
            examples = examples[:-1] + [suff_example] if examples else [suff_example]
            used_ids.append(f"sufficiency:{suff_example['question'][:40]}")

    few_shot_block = "\n\n".join(
        f"Example question: {ex['question']}\nExample answer: {ex['answer']}" for ex in examples
    )

    parts = [
        DOMAIN_CONTEXT,
        manifest_line,
        TASK_TEMPLATES[task],
    ]
    if sufficiency == SufficiencyVerdict.SCOPE_DOWN:
        parts.append(SCOPE_DOWN_INSTRUCTION)
    if few_shot_block:
        parts.append("Reference examples (for style and format only -- do not copy their numbers):\n\n" + few_shot_block)
    parts.append("Tool evidence for THIS scene:\n" + format_evidence_block(evidence))
    parts.append(OUTPUT_FORMAT_INSTRUCTION)
    parts.append(f"Question: {question}")

    return "\n\n".join(parts), used_ids
