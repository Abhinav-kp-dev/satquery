"""The agentic controller.

    preflight (session creation) ->
    1 PARSE      question -> typed QuerySpec (with follow-up inheritance)
    2 ASSESS     evidence sufficiency: proceed / scope_down / request_input
    3 PLAN       registry lookup -> tool DAG
    4 EXECUTE    parallel deterministic tools; measurements -> ledger
    5 ADJUDICATE narration checked against the ledger; measurement wins
    6 CALIBRATE  confidence from adjudication, separability, cross-sensor IoU, data quality
    7 REPORT     answer + overlay + regions + confidence, ledger committed (hash-chained)

Every stage appends to the same Evidence Ledger and records a StageRecord,
so the trace shows exactly what ran, what it measured and why.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone

import numpy as np

from app.config import RENDER_DIR, settings
from app.core.executor import CLASSES, MEMORY, ExecContext, execute_plan
from app.core.parser import parse_question
from app.core.planner import build_plan
from app.core.preflight import build_manifest
from app.core.raster_io import LoadedScene, load_scene
from app.core.registry import adapter_for
from app.core.sufficiency import check_sufficiency
from app.ledger.ledger import LedgerWriter
from app.ledger.store import commit_ledger, get_session, list_traces, save_session, save_trace
from app.models.schemas import (
    ConfidenceBreakdown,
    InputConfig,
    InputManifest,
    Layer,
    Modality,
    QueryTrace,
    Region,
    SessionInfo,
    StageRecord,
    SufficiencyVerdict,
)
from app.tools.rendering import MASK_COLORS, render_composite, render_layer, render_overlay
from app.utils.confidence import compute_confidence
from app.vlm.adjudicator import adjudicate
from app.vlm.client import get_mock_client, get_vlm_client
from app.vlm.prompt_builder import build_prompt


class InputRejected(Exception):
    def __init__(self, reason: str, manifest: InputManifest):
        self.reason = reason
        self.manifest = manifest
        super().__init__(reason)


class SessionNotFound(Exception):
    pass


# ------------------------------------------------------------ scene cache

_scenes: OrderedDict[str, LoadedScene] = OrderedDict()
_scene_lock = threading.Lock()
_MAX_SCENES = 6


def _scene_for(session_id: str, paths: list[str], manifest: InputManifest) -> LoadedScene:
    with _scene_lock:
        if session_id in _scenes:
            _scenes.move_to_end(session_id)
            return _scenes[session_id]
    scene = load_scene(paths, manifest)
    with _scene_lock:
        _scenes[session_id] = scene
        while len(_scenes) > _MAX_SCENES:
            evicted, _ = _scenes.popitem(last=False)
            MEMORY.forget(evicted)
    return scene


# ------------------------------------------------------------ sessions


def create_session(paths: list[tuple[str, str]], sample_id: str | None = None) -> SessionInfo:
    manifest, ordered = build_manifest(paths, settings.analysis_max_dim, settings.max_native_dim)
    if manifest.config == InputConfig.INVALID:
        raise InputRejected(manifest.rejection_reason or "Invalid input.", manifest)

    session_id = "s" + uuid.uuid4().hex[:11]
    fs_paths = [p for p, _ in ordered]
    scene = _scene_for(session_id, fs_paths, manifest)  # also fills alignment residuals

    render_urls = []
    for i, (arr, img) in enumerate(zip(scene.arrays, manifest.images)):
        fname = f"{session_id}_render_{i}.png"
        render_composite(arr, img.band_count, img.modality, img.band_names).save(RENDER_DIR / fname)
        render_urls.append(f"/renders/{fname}")

    info = SessionInfo(
        session_id=session_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        manifest=manifest,
        render_urls=render_urls,
        sample_id=sample_id,
    )
    save_session(info, fs_paths)
    return info


def load_session(session_id: str) -> tuple[SessionInfo, list[str]]:
    found = get_session(session_id)
    if found is None:
        raise SessionNotFound(session_id)
    return found


# ------------------------------------------------------------ helpers


def _manifest_line(m: InputManifest) -> str:
    def one(img):
        gsd = f"{img.analysis_gsd_m:.1f}" if img.analysis_gsd_m else "unknown"
        date = f" | date {img.acquisition_date}" if img.acquisition_date else ""
        return f"{img.modality.value} {img.band_count}-band ({img.band_layout}) | GSD {gsd} m{date}"

    if m.config == InputConfig.SINGLE_IMAGE:
        return f"[SINGLE IMAGE: {one(m.images[0])}]"
    if m.config == InputConfig.CROSS_MODAL_PAIR:
        return f"[CROSS-MODAL PAIR: {one(m.images[0])} + {one(m.images[1])}]"
    return f"[BI-TEMPORAL PAIR: T1 {one(m.images[0])} -> T2 {one(m.images[1])}]"


def _base_index(manifest: InputManifest) -> int:
    if manifest.config == InputConfig.BI_TEMPORAL_PAIR:
        return 1
    if manifest.config == InputConfig.CROSS_MODAL_PAIR:
        return 0 if manifest.images[0].modality == Modality.OPTICAL else 1
    return 0


def _layer_masks(results: dict, spec, manifest: InputManifest) -> dict[str, np.ndarray]:
    if "delta_change" in results and results["delta_change"].masks:
        return {k: results["delta_change"].masks[k] for k in ("gained", "lost")}
    if "fusion_agreement" in results and results["fusion_agreement"].masks and spec.task.value == "fusion":
        return {k: results["fusion_agreement"].masks[k] for k in ("agreement", "optical_only", "sar_only")}
    idx = _base_index(manifest)
    src = next((r for nid, r in results.items() if nid.endswith(f"#{idx}")), None)
    if src is None:
        return {}
    if spec.target.value in src.masks:
        return {spec.target.value: src.masks[spec.target.value]}
    return {c: src.masks[c] for c in CLASSES if c in src.masks}


def _session_history(session_id: str | None) -> list[QueryTrace]:
    return list_traces(limit=50, session_id=session_id) if session_id else []


# ------------------------------------------------------------ the pipeline


def run_query(session_id: str, question: str, perturb: bool = False) -> QueryTrace:
    start = time.perf_counter()
    info, paths = load_session(session_id)
    manifest = info.manifest
    query_id = uuid.uuid4().hex[:12]
    ledger = LedgerWriter(query_id)
    stages: list[StageRecord] = []

    # preflight facts, recorded so every trace is self-contained
    ledger.append("preflight", "metadata", "input_config", manifest.config.value, source="preflight")
    for i, img in enumerate(manifest.images):
        ledger.append(
            "preflight", "metadata", f"image_{i}",
            {"file": img.filename, "modality": img.modality.value, "bands": img.band_count, "layout": img.band_layout,
             "gsd_m": img.analysis_gsd_m, "crs": img.crs, "date": img.acquisition_date, "decimation": round(img.decimation, 2)},
            source="preflight",
        )
    if manifest.alignment:
        ledger.append("preflight", "metadata", "alignment", manifest.alignment.model_dump(), source="preflight")

    # 1 PARSE ----------------------------------------------------------
    t0 = time.perf_counter()
    history = _session_history(session_id)
    prev = history[-1] if history else None
    spec = parse_question(question, manifest.config, manifest.legal_tasks, prev.spec if prev else None, prev.query_id if prev else None)
    ledger.append("parse", "decision", "query_spec", spec.model_dump(mode="json"), source="parser")
    stages.append(StageRecord(
        name="parse", status="ok", runtime_ms=(time.perf_counter() - t0) * 1000,
        summary=f"task={spec.task.value}, target={spec.target.value}, metric={spec.metric}" + (" (follow-up)" if spec.follow_up else ""),
    ))

    # 2 ASSESS ---------------------------------------------------------
    t0 = time.perf_counter()
    verdict, reason, missing = check_sufficiency(spec, manifest)
    ledger.append("assess", "decision", "sufficiency", {"verdict": verdict.value, "reason": reason, "missing_input": missing}, source="sufficiency")
    stages.append(StageRecord(
        name="assess", status={"proceed": "ok", "scope_down": "warn", "request_input": "declined"}[verdict.value],
        summary=verdict.value.replace("_", " ") + (f": needs {missing}" if missing else ""), runtime_ms=(time.perf_counter() - t0) * 1000,
    ))

    tool_calls, plan_nodes, regions, layers = [], [], [], []
    overlay_url = None
    claims = []
    prompt_examples: list[str] = []
    adapter = None
    narrator = "none"
    answer_raw = None
    adjudication_status = "unverified"

    if verdict == SufficiencyVerdict.REQUEST_INPUT:
        for name in ("plan", "execute", "adjudicate"):
            stages.append(StageRecord(name=name, status="skipped", summary="declined before measurement: required input missing"))
        answer = reason
        confidence = ConfidenceBreakdown(overall=0.0, band="low", basis="query declined: the evidence needed was not provided")
        stages.append(StageRecord(name="calibrate", status="skipped", summary="no answer to calibrate"))
    else:
        # 3 PLAN -------------------------------------------------------
        t0 = time.perf_counter()
        plan_nodes = build_plan(spec, manifest)
        ledger.append("plan", "decision", "tool_plan", [{"node": n.node_id, "after": n.depends_on} for n in plan_nodes], source="planner")
        stages.append(StageRecord(name="plan", status="ok", runtime_ms=(time.perf_counter() - t0) * 1000,
                                  summary=" -> ".join(n.node_id for n in plan_nodes)))

        # 4 EXECUTE ----------------------------------------------------
        t0 = time.perf_counter()
        scene = _scene_for(session_id, paths, manifest)
        result = execute_plan(plan_nodes, ExecContext(manifest=manifest, scene=scene), ledger, session_id)
        tool_calls = result.calls
        reused = sum(1 for c in tool_calls if c.reused_from)
        warnings = [w for r in result.results.values() for w in r.warnings]
        stages.append(StageRecord(
            name="execute", status="warn" if warnings else "ok", runtime_ms=(time.perf_counter() - t0) * 1000,
            summary=f"{len(tool_calls)} tool call(s), {len(ledger.measurements())} measurements"
            + (f", {reused} reused from session memory" if reused else "") + (f"; {warnings[0]}" if warnings else ""),
        ))

        decisions = {k: v for r in result.results.values() for k, v in r.decisions.items()}
        regions = [Region(**r) for r in (result.results["grounding"].regions if "grounding" in result.results else [])]

        # renders for this query
        masks = _layer_masks(result.results, spec, manifest)
        base_idx = _base_index(manifest)
        base_img = manifest.images[base_idx]
        base = render_composite(scene.arrays[base_idx], base_img.band_count, base_img.modality, base_img.band_names)
        for name, mask in masks.items():
            fname = f"{query_id}_layer_{name}.png"
            render_layer(mask, name).save(RENDER_DIR / fname)
            layers.append(Layer(name=name, url=f"/renders/{fname}", color=list(MASK_COLORS.get(name, (255, 255, 255))), fraction=float(mask.mean())))
        if masks:
            render_overlay(base, masks).save(RENDER_DIR / f"{query_id}_overlay.png")
            overlay_url = f"/renders/{query_id}_overlay.png"

        # 5 NARRATE + ADJUDICATE --------------------------------------
        t0 = time.perf_counter()
        numeric = ledger.numeric_measurements()
        measurements = {k: float(e.value) for k, e in numeric.items()}
        prompt, prompt_examples = build_prompt(
            question, spec, numeric, decisions, verdict, reason, missing, _manifest_line(manifest),
            history=[(t.question, t.answer) for t in history],
        )
        adapter = adapter_for(spec.task)
        composites = [render_composite(a, i.band_count, i.modality, i.band_names) for a, i in zip(scene.arrays, manifest.images)]
        context = {
            "spec": spec, "measurements": measurements, "decisions": decisions, "sufficiency": verdict,
            "sufficiency_reason": reason, "missing_input": missing, "config": manifest.config,
            "regions": [r.model_dump() for r in regions], "perturb": perturb,
        }
        client = get_vlm_client()
        narrator = client.name
        narration_note = ""
        try:
            answer_raw = client.generate(prompt, composites, context, adapter=adapter)
        except Exception as exc:  # noqa: BLE001 -- any backend failure degrades to offline narration
            if not settings.vlm_fallback_to_mock:
                raise
            narration_note = f"{narrator} narration failed ({type(exc).__name__}); used offline narrator"
            client = get_mock_client()
            narrator = f"mock (fallback from {narrator})"
            answer_raw = client.generate(prompt, composites, context)
        if narrator.startswith("mock"):
            adapter = None

        adj = adjudicate(answer_raw, numeric)
        adjudication_status = adj.status
        claims = adj.claims
        answer = adj.answer
        for c in claims:
            ledger.append(
                "adjudicate", "correction" if c.corrected else "claim", c.matched_ledger_key or "unsupported",
                {"claimed": c.value, "unit": c.unit, "ledger_entry": c.ledger_entry_id, "ledger_value": c.ledger_value,
                 "within_tolerance": c.within_tolerance, "delivered": c.text, "source": c.source},
                c.unit, source="adjudicator",
            )
        stages.append(StageRecord(
            name="adjudicate", status="warn" if adj.corrections or narration_note else "ok", runtime_ms=(time.perf_counter() - t0) * 1000,
            summary=f"{len(claims)} figure(s) checked, {adj.corrections} corrected to the measured value" + (f"; {narration_note}" if narration_note else ""),
        ))

        # 6 CALIBRATE -------------------------------------------------
        t0 = time.perf_counter()
        weak = any(str(v) == "rgb_dark_water_heuristic" for k, v in decisions.items() if k.startswith("water_index"))
        confidence = compute_confidence(adjudication_status, spec.target, measurements, verdict, manifest, weak_fallback=weak)
        ledger.append("calibrate", "confidence", "confidence", confidence.model_dump(), None, source="calibrator")
        stages.append(StageRecord(name="calibrate", status="ok", runtime_ms=(time.perf_counter() - t0) * 1000,
                                  summary=f"{confidence.band} ({confidence.overall:.2f}): " + ", ".join(f"{k} {v:.2f}" for k, v in confidence.components.items())))

    # 7 REPORT ---------------------------------------------------------
    t0 = time.perf_counter()
    ledger.append("report", "report", "answer", answer, source="controller")
    committed = commit_ledger(ledger.entries)
    stages.append(StageRecord(name="report", status="ok", runtime_ms=(time.perf_counter() - t0) * 1000,
                              summary=f"{len(committed)} ledger entries committed; head {committed[-1].hash[:12]}"))

    trace = QueryTrace(
        query_id=query_id,
        session_id=session_id,
        question=question,
        config=manifest.config,
        spec=spec,
        task_classified=spec.task,
        routing_layer=spec.routing_layer,
        legal_tasks=manifest.legal_tasks,
        sufficiency=verdict,
        sufficiency_reason=reason,
        missing_input=missing,
        stages=stages,
        plan=[n.node_id for n in plan_nodes],
        plan_nodes=plan_nodes,
        tool_calls=tool_calls,
        prompt_examples_used=prompt_examples,
        narrator=narrator,
        adapter=adapter,
        llm_override=adjudication_status == "corrected",
        adjudication_status=adjudication_status,
        claims=claims,
        regions=regions,
        confidence=confidence,
        answer=answer,
        answer_raw=answer_raw,
        ledger=committed,
        ledger_head=committed[-1].hash,
        render_urls=info.render_urls,
        overlay_url=overlay_url,
        layers=layers,
        total_runtime_ms=(time.perf_counter() - start) * 1000,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    save_trace(trace)
    return trace
