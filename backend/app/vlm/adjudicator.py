"""Stage 5 -- ADJUDICATE: every number the narration states is checked
against the Evidence Ledger, and on conflict the measurement wins --
enforced in code, not requested in a prompt.

1. Numbers inside <CLAIM value=".." unit=".."> tags and bare numbers with a
   unit found anywhere else in the prose ("about 40% water") are both
   checked -- a model cannot dodge adjudication by not tagging a figure.
2. Matching is unit-aware (a claimed 12 ha is never compared with a 12%
   fraction) and target-aware (a percentage stated next to the word
   "vegetation" is matched against vegetation entries first).
3. A figure outside tolerance is rewritten in the delivered answer to the
   ledger value it should have cited; a figure no measurement supports is
   withheld. Every correction is recorded, and the original model output is
   kept in the trace for audit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.models.schemas import ClaimRecord, LedgerEntry

ANSWER_RE = re.compile(r"<ANSWER>(.*?)</ANSWER>", re.DOTALL)
CLAIM_RE = re.compile(r'<CLAIM\s+value="([\-\d.,]+)"\s+unit="([^"]+)"\s*>(.*?)</CLAIM>', re.DOTALL)
UNTAGGED_RE = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(%|percent\b|per cent\b|hectares?\b|ha\b|km²|km2\b|sq\.?\s?km\b)",
    re.IGNORECASE,
)

REL_TOLERANCE = 0.10
ABS_TOLERANCE = {"percent": 0.5, "ha": 0.5, "km2": 0.005, "iou": 0.02, "dice": 0.02, "index": 0.02, "dB": 0.3, "count": 0.0, "eta": 0.02}

_CLASS_WORDS = {
    "water": ("water", "flood", "inundat", "river", "lake", "reservoir", "wet"),
    "vegetation": ("veget", "forest", "crop", "green", "tree", "canopy"),
    "built_up": ("built", "urban", "settlement", "building", "construction", "impervious"),
    "bare_soil": ("bare", "soil", "mining", "excavat", "quarr"),
}
_QUALIFIERS = {
    "_gained": ("gain", "new", "increase", "expan", "added", "appeared"),
    "_lost": ("lost", "loss", "decrease", "receded", "removed", "disappeared", "cleared"),
    "_net": ("net",),
    "_sar": ("sar", "radar", "backscatter"),
    "_optical": ("optical", "spectral"),
    "_t1": ("earlier", "before", "t1", "first date", "pre-"),
    "_t2": ("later", "after", "t2", "second date", "post-", "now"),
    "relative_change": ("relative", "compared to", "grew by", "shrank by"),
}


def canonical_unit(unit: str) -> str:
    u = unit.strip().lower()
    if u in ("%", "percent", "percentage", "per cent", "pct"):
        return "percent"
    if u in ("ha", "hectare", "hectares"):
        return "ha"
    if u in ("km2", "km²", "sq km", "sq. km", "square kilometres", "square kilometers"):
        return "km2"
    return u


def _candidates(unit: str, ledger: dict[str, LedgerEntry]) -> dict[str, tuple[float, LedgerEntry]]:
    """ledger key -> (value expressed in the claim's unit, entry)."""
    out: dict[str, tuple[float, LedgerEntry]] = {}
    for key, e in ledger.items():
        v = float(e.value)
        if unit == "percent":
            if e.unit == "fraction":
                out[key] = (v * 100, e)
            elif e.unit == "percent":
                out[key] = (v, e)
        elif unit == "ha" and e.unit == "ha":
            out[key] = (v, e)
        elif unit == "km2" and e.unit == "ha":
            out[key] = (v / 100, e)
        elif e.unit == unit:
            out[key] = (v, e)
    return out


_CLAUSE_BREAK = re.compile(r"[.;:(]|,|\band\b|\bwhile\b|\bwhereas\b")


def local_clause(context: str) -> str:
    """The text after the last clause boundary: 'water 17.8%, built-up (' ->
    'built-up'. Keeps a figure's class words from leaking into the next one."""
    parts = _CLAUSE_BREAK.split(context)
    tail = parts[-1] if parts else context
    # "built-up surfaces (" leaves an empty tail: step back one piece.
    if not tail.strip() and len(parts) > 1:
        tail = parts[-2]
    return tail


def _class_of(text: str) -> str | None:
    t = text.lower()
    for cls, words in _CLASS_WORDS.items():
        if any(w in t for w in words):
            return cls
    return None


def _narrow_class(cands: dict, clause: str, sentence: str) -> dict:
    cls = _class_of(clause) or _class_of(sentence)
    if cls:
        narrowed = {k: v for k, v in cands.items() if k.startswith(cls)}
        if narrowed:
            return narrowed
    return cands


def _narrow_qualifiers(cands: dict, clause: str) -> dict:
    ctx = clause.lower()
    for suffix, words in _QUALIFIERS.items():
        if any(w in ctx for w in words):
            narrowed = {k: v for k, v in cands.items() if suffix in k}
            if narrowed:
                cands = narrowed
    return cands


def _within(claimed: float, actual: float, unit: str) -> bool:
    if abs(claimed - actual) <= ABS_TOLERANCE.get(unit, 0.0):
        return True
    return abs(claimed - actual) / max(abs(actual), 1e-6) <= REL_TOLERANCE


def format_value(value: float, unit: str) -> str:
    if unit == "percent":
        return f"{value:.1f}%"
    if unit == "ha":
        return f"{value:,.1f} ha"
    if unit == "km2":
        return f"{value:.2f} km²"
    if unit in ("iou", "dice", "index", "eta"):
        return f"{value:.2f}"
    if unit == "dB":
        return f"{value:.1f} dB"
    if unit == "count":
        return f"{int(round(value))}"
    return f"{value:.3g}"


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


@dataclass
class Adjudication:
    status: str  # "consistent" | "corrected" | "unverified"
    answer: str
    claims: list[ClaimRecord] = field(default_factory=list)
    corrections: int = 0


def _judge(value: float, unit_raw: str, text: str, context: str, source: str, ledger: dict[str, LedgerEntry]) -> tuple[ClaimRecord, str]:
    """Returns the claim record and the text to deliver in its place."""
    unit = canonical_unit(unit_raw)
    sentence = re.split(r"[.!?]\s", context)[-1]
    clause = local_clause(context) + " " + text
    cands = _narrow_class(_candidates(unit, ledger), clause, sentence + " " + text)
    if not cands:
        rec = ClaimRecord(text=text.strip(), value=value, unit=unit, source=source, within_tolerance=False, corrected=True)
        return rec, "[figure withheld: no measurement supports it]"
    # Consistent if the figure matches any measurement of the right class;
    # only a genuine mismatch needs the qualifiers to pick the intended entry.
    matching = {k: v for k, v in cands.items() if _within(value, v[0], unit)}
    pool = matching or _narrow_qualifiers(cands, clause)
    key, (ledger_val, entry) = min(pool.items(), key=lambda kv: abs(kv[1][0] - value))
    ok = bool(matching)
    rec = ClaimRecord(
        text=text.strip(), value=value, unit=unit, source=source, matched_ledger_key=key,
        ledger_entry_id=entry.entry_id, ledger_value=round(ledger_val, 4), within_tolerance=ok, corrected=not ok,
    )
    if ok:
        return rec, text
    # Measurement wins: substitute the ledger value for the claimed number.
    replacement = format_value(ledger_val, unit)
    num = re.search(r"\d[\d,]*\.?\d*\s*(%|percent|hectares?|ha|km²|km2)?", text)
    delivered = text[: num.start()] + replacement + text[num.end():] if num else replacement
    rec.text = delivered.strip()
    return rec, delivered


def adjudicate(raw_output: str, ledger: dict[str, LedgerEntry]) -> Adjudication:
    """`ledger`: numeric measurement entries keyed by ledger key."""
    m = ANSWER_RE.search(raw_output)
    body = m.group(1).strip() if m else raw_output.strip()

    claims: list[ClaimRecord] = []
    pieces: list[str] = []
    cursor = 0

    def scan_untagged(segment: str, preceding: str) -> str:
        out, pos = [], 0
        for um in UNTAGGED_RE.finditer(segment):
            ctx = (preceding + segment[: um.start()])[-90:]
            rec, delivered = _judge(_to_float(um.group(1)), um.group(2), um.group(0), ctx, "untagged", ledger)
            claims.append(rec)
            out.append(segment[pos: um.start()])
            out.append(delivered)
            pos = um.end()
        out.append(segment[pos:])
        return "".join(out)

    for cm in CLAIM_RE.finditer(body):
        plain = body[cursor: cm.start()]
        pieces.append(scan_untagged(plain, "".join(pieces)))
        ctx = "".join(pieces)[-90:]
        try:
            value = _to_float(cm.group(1))
        except ValueError:
            value = float("nan")
        rec, delivered = _judge(value, cm.group(2), cm.group(3), ctx, "tagged", ledger)
        claims.append(rec)
        pieces.append(delivered)
        cursor = cm.end()
    pieces.append(scan_untagged(body[cursor:], "".join(pieces)))

    answer = re.sub(r"</?(ANSWER|CLAIM)[^>]*>", "", "".join(pieces))
    answer = re.sub(r"\s{2,}", " ", answer).strip()

    corrections = sum(1 for c in claims if c.corrected)
    if not claims:
        status = "unverified"
    elif corrections:
        status = "corrected"
    else:
        status = "consistent"
    return Adjudication(status=status, answer=answer, claims=claims, corrections=corrections)
