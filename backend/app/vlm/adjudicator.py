"""Parses the model's <ANSWER>/<CLAIM> output and checks every claimed
number against the evidence ledger. On a no-training (prompted, not
fine-tuned) model this is expected to catch more disagreements than a
fine-tuned model would produce -- that is the adjudicator's job, not a
bug: a numeric tool always wins over the language model on conflict.
"""
from __future__ import annotations

import re

from app.models.schemas import ClaimRecord

ANSWER_RE = re.compile(r"<ANSWER>(.*?)</ANSWER>", re.DOTALL)
CLAIM_RE = re.compile(r'<CLAIM value="([\-\d.]+)" unit="([^"]+)">(.*?)</CLAIM>', re.DOTALL)

TOLERANCE = 0.1  # 10% relative tolerance between a claimed number and the nearest ledger value


def _ledger_scalars(evidence: dict) -> dict[str, float]:
    scalars = {}
    for key, value in evidence.items():
        if isinstance(value, (int, float)):
            scalars[key] = float(value)
            if key.endswith("_fraction"):
                scalars[key.replace("_fraction", "_percent")] = float(value) * 100
    return scalars


def _nearest_match(value: float, unit: str, scalars: dict[str, float]) -> tuple[str | None, bool]:
    best_key, best_rel_err = None, None
    for key, ledger_val in scalars.items():
        denom = max(abs(ledger_val), 1e-6)
        rel_err = abs(value - ledger_val) / denom
        if best_rel_err is None or rel_err < best_rel_err:
            best_key, best_rel_err = key, rel_err
    if best_key is None:
        return None, False
    return best_key, best_rel_err <= TOLERANCE


def adjudicate(raw_output: str, evidence: dict) -> tuple[str, str, list[ClaimRecord], bool]:
    """Returns (status, answer_text, claim_records, llm_override)."""
    answer_match = ANSWER_RE.search(raw_output)
    raw_answer = answer_match.group(1).strip() if answer_match else raw_output.strip()
    # Keep the human-readable text inside each CLAIM tag, drop the markup.
    answer_text = CLAIM_RE.sub(lambda m: m.group(3), raw_answer).strip()

    scalars = _ledger_scalars(evidence)
    claims: list[ClaimRecord] = []
    any_mismatch = False

    for value_str, unit, text in CLAIM_RE.findall(raw_output):
        value = float(value_str)
        matched_key, within = _nearest_match(value, unit, scalars)
        claims.append(
            ClaimRecord(text=text.strip(), value=value, unit=unit, matched_ledger_key=matched_key, within_tolerance=within)
        )
        if not within:
            any_mismatch = True

    if not claims:
        return "unverified", answer_text, claims, False

    if any_mismatch:
        # Strip the tags but keep the model's prose; the confidence stage
        # will downweight this response for the mismatch, and the trace
        # records exactly which claim(s) failed adjudication.
        return "overridden", answer_text, claims, True

    return "consistent", answer_text, claims, False
