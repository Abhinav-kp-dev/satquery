"""Stage 1 -- PARSE: natural-language question -> typed QuerySpec.

A deterministic rule table, not free-form LLM reasoning, so routing is
reproducible and auditable (every rule that fired is recorded in
`matched_rules`). Two deliberate properties:

- The parser records what the question *asks for* (its intent) even when
  the uploaded input cannot support it. Deciding whether the evidence can
  answer is the ASSESS stage's job, and it needs to know the intent to name
  the missing observation ("upload a pre-event image").
- Follow-up turns inherit what they omit from the previous turn's spec
  ("and in hectares?" keeps the previous target), which is the ledger's
  agent-memory role at the routing level.
"""
from __future__ import annotations

import re

from app.models.schemas import InputConfig, QuerySpec, Target, Task

_TASK_RULES: list[tuple[Task, tuple[str, ...]]] = [
    (
        Task.CHANGE,
        (
            "changed", "change", "increased", "decreased", "increase", "decrease", "grown", "grew", "shrunk",
            "shrank", "since", "before and after", "difference between", "compare these", "compared to",
            "over time", "new construction", "newly", "expanded", "expansion", "receded", "loss of", "gain in",
            "between the two dates", "between these dates", "from t1", "earlier image", "later image", "pre-event",
            "post-event", "then and now", "appeared", "disappeared", "emerged", "new ", "any new",
        ),
    ),
    (
        Task.FUSION,
        (
            "both images", "combine", "combined", "using both", "optical and sar", "sar and optical", "fuse",
            "fusion", "cross-modal", "cross modal", "each sensor", "both sensors", "radar and optical",
            "optical and radar", "do the sensors agree", "agreement",
        ),
    ),
    (
        Task.GROUNDING,
        (
            "highlight", "locate", "where is", "where are", "where", "point to", "show me", "mark the", "find the",
            "identify the region", "bounding box", "coordinates", "latitude", "longitude", "outline", "delineate",
        ),
    ),
    (
        Task.CAPTION,
        (
            "describe", "what is in this", "what's in this", "land cover", "landcover", "summarize", "summarise",
            "caption", "overview", "what does this image show", "what do you see", "tell me about this",
        ),
    ),
]

_TARGET_RULES: list[tuple[Target, tuple[str, ...]]] = [
    (Target.BARE_SOIL, ("bare soil", "bare", "mining", "mine ", "mines ", "excavat", "quarr", "barren", "open-cast", "opencast")),
    (Target.WATER, ("water", "flood", "inundat", "river", "lake", "reservoir", "pond", "wetland", "sea ", "coast", "shoreline", "ndwi", "tank")),
    (Target.VEGETATION, ("veget", "forest", "tree", "crop", "green", "agricultur", "farm", "deforest", "canopy", "ndvi", "plantation", "grass")),
    (Target.BUILT_UP, ("built", "urban", "building", "construction", "settlement", "city", "town", "village", "infrastructure", "encroach", "ndbi", "impervious", "structures")),
]

# Wording that presupposes a "before" state without an explicit change verb.
IMPLIED_BASELINE = (
    "flooded", "flooding", "flood", "encroachment", "encroached", "illegal construction", "illegal mining",
    "deforestation", "deforested", "damage", "damaged", "eroded", "erosion", "degraded", "degradation",
    "depleted", "decline", "declined", "improved", "improvement", "abnormal", "unusual", "normal level",
    "higher than usual", "lower than usual", "dried up", "drought",
)

_AREA = ("hectare", "ha ", "km2", "km²", "sq km", "square", "area", "how large", "how big", "extent", "size", "acres")
_FRACTION = ("percent", "%", "fraction", "proportion", "share", "what part", "how much of", "coverage", "cover")
_PRESENCE = ("is there", "are there", "any ", "does this", "is any", "present")
_FOLLOW_UP_OPENERS = ("and ", "what about", "how about", "also", "same for", "in hectares", "in percent", "where is it", "where are they", "and where", "now ")
_PRONOUNS = re.compile(r"\b(it|that|this one|those|them|they)\b")


def _match(q: str, words: tuple[str, ...]) -> str | None:
    """Word-start matching, so stems work ('veget' -> 'vegetation') without
    substrings misfiring ('tree' in 'street'). A trailing space in a rule
    means whole-word only ('sea ' does not match 'season')."""
    for w in words:
        core = w.strip()
        if not core[0].isalnum():
            if core in q:
                return w
            continue
        pattern = r"\b" + re.escape(core) + (r"\b" if w.endswith(" ") else "")
        if re.search(pattern, q):
            return w
    return None


def parse_question(
    question: str,
    config: InputConfig,
    legal_tasks: list[Task],
    previous: QuerySpec | None = None,
    previous_query_id: str | None = None,
) -> QuerySpec:
    q = " " + re.sub(r"[?.,!;:\"()]", " ", question.lower()).strip() + " "
    rules: list[str] = []

    # ---- target
    target: Target | None = None
    for t, words in _TARGET_RULES:
        hit = _match(q, words)
        if hit:
            target = t
            rules.append(f"target:{t.value}<-'{hit.strip()}'")
            break

    # ---- intent (not yet filtered by legality: ASSESS handles that)
    intent: Task | None = None
    for task, words in _TASK_RULES:
        hit = _match(q, words)
        if hit:
            intent = task
            rules.append(f"task:{task.value}<-'{hit.strip()}'")
            break

    baseline_hit = _match(q, IMPLIED_BASELINE)
    implied_baseline = baseline_hit is not None
    if implied_baseline:
        rules.append(f"implied_baseline<-'{baseline_hit}'")
        # On a before/after pair, "is it flooded?" *is* a change question.
        if config == InputConfig.BI_TEMPORAL_PAIR and intent in (None, Task.CAPTION):
            intent = Task.CHANGE
            rules.append("task:change<-implied_baseline+bi_temporal_pair")

    # ---- follow-up inheritance
    follow_up = False
    inherited_from = None
    stripped = q.strip()
    opener = stripped.startswith(_FOLLOW_UP_OPENERS)
    elliptical = target is None and (bool(_PRONOUNS.search(q)) or len(stripped.split()) <= 4)
    if previous is not None and (opener or elliptical) and (target is None or intent is None):
        follow_up = True
        inherited_from = previous_query_id
        if target is None and intent != Task.CAPTION and previous.target != Target.LAND_COVER:
            target = previous.target
            rules.append(f"target:{target.value}<-inherited")
        if intent is None and previous.task != Task.CAPTION and (opener or target is None):
            asks_amount = _match(q, _AREA + _FRACTION) is not None
            if asks_amount and previous.task in (Task.FUSION, Task.GROUNDING):
                intent = Task.VQA  # "and in hectares?" after a fusion/where question wants the amount
                rules.append("task:vqa<-follow_up_asks_amount")
            else:
                intent = previous.task
                rules.append(f"task:{intent.value}<-inherited")

    wants_location = _match(q, ("where", "locate", "highlight", "show me", "mark", "find the", "coordinates", "latitude", "outline", "delineate")) is not None

    # ---- defaults
    if intent is None:
        if config == InputConfig.CROSS_MODAL_PAIR and target in (Target.WATER, Target.BUILT_UP, None):
            intent = Task.FUSION if _match(q, ("confiden", "sure", "reliab", "trust")) else Task.VQA
        else:
            intent = Task.VQA
        rules.append(f"task:{intent.value}<-default")
    if intent == Task.CAPTION or target is None:
        if target is None:
            target = Target.LAND_COVER
            rules.append("target:land_cover<-default")

    # ---- metric
    if intent == Task.CHANGE:
        metric = "change"
    elif intent == Task.FUSION:
        metric = "agreement"
    elif intent == Task.CAPTION or target == Target.LAND_COVER:
        metric = "describe"
    elif intent == Task.GROUNDING or (wants_location and not _match(q, _AREA + _FRACTION)):
        metric = "location"
    elif _match(q, _AREA):
        metric = "area"
    elif _match(q, _FRACTION):
        metric = "fraction"
    elif _match(q, _PRESENCE):
        metric = "presence"
    else:
        metric = "fraction"

    layer = "layer2_keyword_rule" if any(r.startswith("task:") and "default" not in r for r in rules) else "layer2_default_fallback"
    if follow_up:
        layer = "layer2_follow_up_inheritance"
    if intent not in legal_tasks:
        layer = f"{layer}+intent_not_supported_by_input"

    return QuerySpec(
        task=intent,
        target=target,
        metric=metric,
        implied_baseline=implied_baseline,
        wants_location=wants_location,
        follow_up=follow_up,
        inherited_from=inherited_from,
        routing_layer=f"{layer}:{intent.value}",
        matched_rules=rules,
    )
