"""Specialist prompt templates plus retrieval-augmented in-context examples.

With the `openai_compat` backend each task family is additionally served by
its own LoRA adapter (see registry.yaml); with the mock/Anthropic backends
the task template and retrieved exemplars are the specialisation.

The evidence block lists ledger entries by id, so the narrator is grounded
in -- and cites -- the same records the adjudicator later checks.
"""
from __future__ import annotations

from app.config import settings
from app.models.schemas import LedgerEntry, QuerySpec, SufficiencyVerdict, Task
from app.vlm.retrieval import get_example_pool

DOMAIN_CONTEXT = """You are a remote-sensing imagery analyst. You are shown rendered composite(s) of a \
satellite scene alongside evidence already measured by deterministic tools and recorded in an evidence \
ledger. Treat that evidence as ground truth -- never re-derive numbers from the image yourself. Use the \
image only for qualitative context (texture, shape, arrangement) that the numbers do not capture.

Key terms: NDWI/MNDWI (water indices from green vs near-infrared / short-wave-infrared reflectance), \
NDVI (vegetation index, near-infrared vs red), NDBI (built-up index, SWIR vs NIR), BSI (bare-soil index), \
SAR backscatter (radar return in dB: low co-pol = smooth water; bright co-pol with a small co-/cross-pol \
gap = double-bounce from buildings), IoU (overlap between two masks, 0..1), GSD (metres per pixel)."""

TASK_TEMPLATES: dict[Task, str] = {
    Task.VQA: "Answer the question about this scene using only the ledger evidence. Lead with the direct answer.",
    Task.CAPTION: "Describe the scene and its dominant land cover in 2-3 sentences, grounding every quantity in the ledger evidence.",
    Task.GROUNDING: "Say where the region the question refers to is (relative position, and coordinates if the ledger has them). The overlay draws the tool-derived regions.",
    Task.CHANGE: "Compare the two dates using the change evidence. State the direction of change and the measured area in hectares; mention where it is concentrated.",
    Task.FUSION: "Combine the optical-derived and SAR-derived evidence. State where the two sensors agree and disagree, citing the IoU, and what that means for how much to trust the result.",
}

SCOPE_DOWN_INSTRUCTION = """The question cannot be fully answered from this input: {reason} \
Answer only what the ledger evidence supports, then state plainly what additional input would be needed \
({missing}). Do not guess whether a change or anomaly occurred."""

OUTPUT_FORMAT_INSTRUCTION = """Respond in exactly this format, with no text outside the tags:
<ANSWER>your answer, written for a non-specialist reader, at most 4 sentences</ANSWER>

Wrap every number you state in a CLAIM tag inside the ANSWER, e.g. \
<CLAIM value="41.2" unit="percent">41.2%</CLAIM> or <CLAIM value="1078.5" unit="hectares">1,078.5 ha</CLAIM>. \
Units: percent, hectares, km2, iou, index, dB, count. Every claimed number must come from a ledger entry \
below (fractions are shown with their percentage). Numbers that do not match the ledger are replaced \
with the measured value before the answer is delivered."""

_HIDDEN = ("_separability", "valid_fraction")


def format_evidence_block(entries: dict[str, LedgerEntry], decisions: dict[str, object]) -> str:
    lines = []
    for key, e in entries.items():
        if any(h in key for h in _HIDDEN):
            continue
        v = e.value
        if e.unit == "fraction":
            lines.append(f"[{e.entry_id}] {key} = {v:.4f}  ({v * 100:.1f}%)")
        elif e.unit == "ha":
            lines.append(f"[{e.entry_id}] {key} = {v:,.2f} ha")
        elif isinstance(v, float):
            lines.append(f"[{e.entry_id}] {key} = {v:.3f} {e.unit or ''}".rstrip())
        else:
            lines.append(f"[{e.entry_id}] {key} = {v} {e.unit or ''}".rstrip())
    for key, v in decisions.items():
        lines.append(f"- {key}: {v}")
    return "\n".join(lines) if lines else "- (no numeric evidence available)"


def build_prompt(
    question: str,
    spec: QuerySpec,
    entries: dict[str, LedgerEntry],
    decisions: dict[str, object],
    sufficiency: SufficiencyVerdict,
    sufficiency_reason: str,
    missing_input: str | None,
    manifest_line: str,
    history: list[tuple[str, str]] | None = None,
) -> tuple[str, list[str]]:
    pool = get_example_pool()
    examples = pool.retrieve(question, spec.task, k=settings.few_shot_k)
    used_ids = [f"{spec.task.value}:{ex['question'][:48]}" for ex in examples]

    if sufficiency == SufficiencyVerdict.SCOPE_DOWN:
        suff = pool.force_sufficiency_example()
        if suff and suff not in examples:
            examples = (examples[:-1] if examples else []) + [suff]
            used_ids.append(f"sufficiency:{suff['question'][:48]}")

    parts = [DOMAIN_CONTEXT, manifest_line, TASK_TEMPLATES[spec.task]]
    parts.append(f"Parsed query: target={spec.target.value}, metric={spec.metric}, wants_location={spec.wants_location}")
    if sufficiency == SufficiencyVerdict.SCOPE_DOWN:
        parts.append(SCOPE_DOWN_INSTRUCTION.format(reason=sufficiency_reason, missing=missing_input or "a reference image"))
    if history:
        parts.append("Earlier turns in this session (for context only):\n" + "\n".join(f"Q: {q}\nA: {a}" for q, a in history[-3:]))
    if examples:
        parts.append(
            "Reference examples (style and format only -- do not copy their numbers):\n\n"
            + "\n\n".join(f"Example question: {ex['question']}\nExample answer: {ex['answer']}" for ex in examples)
        )
    parts.append("Evidence ledger for THIS scene:\n" + format_evidence_block(entries, decisions))
    parts.append(OUTPUT_FORMAT_INSTRUCTION)
    parts.append(f"Question: {question}")
    return "\n\n".join(parts), used_ids
