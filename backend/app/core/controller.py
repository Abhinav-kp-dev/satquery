"""The agentic controller: preflight -> sufficiency -> routing -> tool
execution -> narration -> adjudication -> confidence -> ledger. Every
stage's output is a typed object logged into the QueryTrace, which is
what gets persisted to the Evidence Ledger and shown in the UI's
"how this was computed" panel.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

import numpy as np
import rasterio

from app.config import RENDER_DIR, settings
from app.core.classifier import classify
from app.core.preflight import build_manifest
from app.core.sufficiency import check_sufficiency
from app.ledger.store import save_trace
from app.models.schemas import (
    ClaimRecord,
    ConfidenceBreakdown,
    ImageManifest,
    InputConfig,
    InputManifest,
    Modality,
    QueryTrace,
    SufficiencyVerdict,
    ToolCallRecord,
)
from app.tools.delta_change import compute_delta
from app.tools.fusion_agreement import compute_agreement
from app.tools.region import region_hint
from app.tools.rendering import render_composite, render_overlay
from app.tools.sar_backscatter import compute_sar_backscatter
from app.tools.spectral_index import compute_spectral_indices
from app.vlm.adjudicator import adjudicate
from app.vlm.client import get_vlm_client
from app.vlm.prompt_builder import build_prompt
from app.utils.confidence import compute_confidence


class InputRejected(Exception):
    def __init__(self, reason: str, manifest: InputManifest):
        self.reason = reason
        self.manifest = manifest
        super().__init__(reason)


def _load_raw(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read()


def _manifest_line(config: InputConfig, images: list[ImageManifest]) -> str:
    if config == InputConfig.SINGLE_IMAGE:
        img = images[0]
        return f"[SENSOR: {img.modality.value} | GSD: {img.gsd_m or 'unknown'}m | BANDS: {img.band_count}]"
    if config == InputConfig.CROSS_MODAL_PAIR:
        a, b = images
        return f"[CROSS-MODAL PAIR: optical + SAR | GSD: {a.gsd_m or 'unknown'}m]"
    a, b = images
    return f"[BI-TEMPORAL PAIR: {a.modality.value} | GSD: {a.gsd_m or 'unknown'}m | T1 vs T2]"


def _primary_water_mask(spectral: dict) -> np.ndarray | None:
    if "ndwi" in spectral:
        return spectral["ndwi"]["mask"]
    if "rgb_dark_water_heuristic" in spectral:
        return spectral["rgb_dark_water_heuristic"]["mask"]
    return None


def _tools_single_image(img: ImageManifest, raw: np.ndarray) -> tuple[dict, dict, list[ToolCallRecord]]:
    evidence: dict = {}
    masks: dict = {}
    calls: list[ToolCallRecord] = []

    t0 = time.perf_counter()
    if img.modality == Modality.OPTICAL:
        res = compute_spectral_indices(raw, img.band_count, img.gsd_m)
        if "ndwi" in res:
            evidence["water_fraction"] = res["ndwi"]["water_fraction"]
            evidence["water_area_ha"] = res["ndwi"]["water_area_ha"]
            masks["water"] = res["ndwi"]["mask"]
        elif "rgb_dark_water_heuristic" in res:
            evidence["water_fraction"] = res["rgb_dark_water_heuristic"]["water_fraction"]
            evidence["water_area_ha"] = res["rgb_dark_water_heuristic"]["water_area_ha"]
            masks["water"] = res["rgb_dark_water_heuristic"]["mask"]
        if "ndvi" in res:
            evidence["vegetation_fraction"] = res["ndvi"]["vegetation_fraction"]
            evidence["vegetation_area_ha"] = res["ndvi"]["vegetation_area_ha"]
            masks["vegetation"] = res["ndvi"]["mask"]
        if "ndbi" in res:
            evidence["built_up_fraction"] = res["ndbi"]["built_up_fraction"]
            evidence["built_up_area_ha"] = res["ndbi"]["built_up_area_ha"]
            masks["built_up"] = res["ndbi"]["mask"]
        calls.append(
            ToolCallRecord(
                tool="spectral_index",
                impl="NDWI/NDVI/NDBI (Otsu-thresholded)",
                params={"band_count": img.band_count, "gsd_m": img.gsd_m},
                output={k: v for k, v in evidence.items()},
                runtime_ms=(time.perf_counter() - t0) * 1000,
            )
        )
    else:
        res = compute_sar_backscatter(raw, img.band_count, img.gsd_m)
        evidence["water_fraction"] = res["water_fraction"]
        evidence["water_area_ha"] = res["water_area_ha"]
        masks["water"] = res["mask"]
        if "built_up_fraction" in res:
            evidence["built_up_fraction"] = res["built_up_fraction"]
            evidence["built_up_area_ha"] = res["built_up_area_ha"]
            masks["built_up"] = res["built_up_mask"]
        calls.append(
            ToolCallRecord(
                tool="sar_backscatter",
                impl="VV/VH dB thresholding (Otsu)",
                params={"band_count": img.band_count, "gsd_m": img.gsd_m},
                output={k: v for k, v in evidence.items()},
                runtime_ms=(time.perf_counter() - t0) * 1000,
            )
        )

    primary_mask = masks.get("water")
    if primary_mask is None:
        primary_mask = masks.get("built_up")
    if primary_mask is None:
        primary_mask = masks.get("vegetation")
    evidence["region_hint"] = region_hint(primary_mask) if primary_mask is not None else "the scene"
    return evidence, masks, calls


def _tools_cross_modal(images: list[ImageManifest], raws: list[np.ndarray]) -> tuple[dict, dict, list[ToolCallRecord]]:
    opt_idx = 0 if images[0].modality == Modality.OPTICAL else 1
    sar_idx = 1 - opt_idx
    opt_img, sar_img = images[opt_idx], images[sar_idx]
    opt_raw, sar_raw = raws[opt_idx], raws[sar_idx]

    evidence: dict = {}
    masks: dict = {}
    calls: list[ToolCallRecord] = []

    t0 = time.perf_counter()
    opt_res = compute_spectral_indices(opt_raw, opt_img.band_count, opt_img.gsd_m)
    opt_water = _primary_water_mask(opt_res)
    if opt_water is not None:
        evidence["water_fraction_optical"] = float(opt_water.mean())
    calls.append(
        ToolCallRecord(
            tool="spectral_index",
            impl="NDWI/NDVI/NDBI (Otsu-thresholded)",
            params={"band_count": opt_img.band_count},
            output={"water_fraction_optical": evidence.get("water_fraction_optical")},
            runtime_ms=(time.perf_counter() - t0) * 1000,
        )
    )

    t0 = time.perf_counter()
    sar_res = compute_sar_backscatter(sar_raw, sar_img.band_count, sar_img.gsd_m)
    sar_water = sar_res["mask"]
    evidence["water_fraction_sar"] = float(sar_water.mean())
    if "built_up_fraction" in sar_res:
        evidence["built_up_fraction_sar"] = sar_res["built_up_fraction"]
    calls.append(
        ToolCallRecord(
            tool="sar_backscatter",
            impl="VV/VH dB thresholding (Otsu)",
            params={"band_count": sar_img.band_count},
            output={"water_fraction_sar": evidence["water_fraction_sar"]},
            runtime_ms=(time.perf_counter() - t0) * 1000,
        )
    )

    if "built_up_fraction" in opt_res.get("ndbi", {}):
        evidence["built_up_fraction_optical"] = opt_res["ndbi"]["built_up_fraction"]

    if opt_water is not None:
        t0 = time.perf_counter()
        agreement = compute_agreement(opt_water, sar_water)
        evidence["iou"] = agreement["iou"]
        evidence["agreement_fraction"] = agreement["agreement_fraction"]
        evidence["disagreement_fraction"] = agreement["disagreement_fraction"]
        masks["agreement"] = agreement["agreement_mask"]
        masks["disagreement"] = agreement["disagreement_mask"]
        calls.append(
            ToolCallRecord(
                tool="fusion_agreement",
                impl="IoU between independently-derived water masks",
                params={},
                output={"iou": agreement["iou"], "agreement_fraction": agreement["agreement_fraction"]},
                runtime_ms=(time.perf_counter() - t0) * 1000,
            )
        )
        evidence["region_hint"] = region_hint(agreement["agreement_mask"] | agreement["disagreement_mask"])
    else:
        masks["water"] = sar_water
        evidence["region_hint"] = region_hint(sar_water)

    # render base uses the optical composite; report which index is index 0/1 for that
    evidence["_base_render_idx"] = opt_idx
    return evidence, masks, calls


def _tools_bi_temporal(question: str, images: list[ImageManifest], raws: list[np.ndarray]) -> tuple[dict, dict, list[ToolCallRecord]]:
    t1_img, t2_img = images
    t1_raw, t2_raw = raws
    evidence: dict = {}
    masks: dict = {}
    calls: list[ToolCallRecord] = []
    q = question.lower()

    if t1_img.modality == Modality.OPTICAL:
        t0 = time.perf_counter()
        t1_res = compute_spectral_indices(t1_raw, t1_img.band_count, t1_img.gsd_m)
        t2_res = compute_spectral_indices(t2_raw, t2_img.band_count, t2_img.gsd_m)
        calls.append(
            ToolCallRecord(
                tool="spectral_index",
                impl="NDWI/NDVI/NDBI (Otsu-thresholded), both dates",
                params={"band_count": t1_img.band_count},
                output={"t1_indices": list(t1_res.keys()), "t2_indices": list(t2_res.keys())},
                runtime_ms=(time.perf_counter() - t0) * 1000,
            )
        )

        candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if "ndwi" in t1_res and "ndwi" in t2_res:
            candidates["water"] = (t1_res["ndwi"]["mask"], t2_res["ndwi"]["mask"])
        if "ndbi" in t1_res and "ndbi" in t2_res:
            candidates["built_up"] = (t1_res["ndbi"]["mask"], t2_res["ndbi"]["mask"])
        if "ndvi" in t1_res and "ndvi" in t2_res:
            candidates["vegetation"] = (t1_res["ndvi"]["mask"], t2_res["ndvi"]["mask"])

        if any(k in q for k in ("water", "flood", "flooded", "inundat")) and "water" in candidates:
            label = "water"
        elif any(k in q for k in ("built", "urban", "construction", "settlement")) and "built_up" in candidates:
            label = "built_up"
        elif any(k in q for k in ("veget", "crop", "forest", "green", "deforest")) and "vegetation" in candidates:
            label = "vegetation"
        elif candidates:
            # No keyword match: pick whichever indicator moved the most between dates.
            label = max(
                candidates,
                key=lambda k: abs(int(candidates[k][1].sum()) - int(candidates[k][0].sum())),
            )
        else:
            label = None
    else:
        t0 = time.perf_counter()
        t1_res = compute_sar_backscatter(t1_raw, t1_img.band_count, t1_img.gsd_m)
        t2_res = compute_sar_backscatter(t2_raw, t2_img.band_count, t2_img.gsd_m)
        calls.append(
            ToolCallRecord(
                tool="sar_backscatter",
                impl="VV/VH dB thresholding (Otsu), both dates",
                params={"band_count": t1_img.band_count},
                output={},
                runtime_ms=(time.perf_counter() - t0) * 1000,
            )
        )
        candidates = {"water": (t1_res["mask"], t2_res["mask"])}
        label = "water"

    if label is not None:
        t0 = time.perf_counter()
        mask_t1, mask_t2 = candidates[label]
        delta = compute_delta(mask_t1, mask_t2, t2_img.gsd_m, label)
        evidence["change_indicator"] = label
        evidence["direction"] = delta["direction"]
        evidence["t1_fraction"] = delta["t1_fraction"]
        evidence["t2_fraction"] = delta["t2_fraction"]
        evidence[f"{label}_gained_ha"] = delta[f"{label}_gained_ha"]
        evidence[f"{label}_lost_ha"] = delta[f"{label}_lost_ha"]
        evidence[f"{label}_net_change_ha"] = delta[f"{label}_net_change_ha"]
        masks["gained"] = delta["gained_mask"]
        masks["lost"] = delta["lost_mask"]
        calls.append(
            ToolCallRecord(
                tool="delta_change",
                impl="T2 minus T1 mask arithmetic",
                params={"indicator": label},
                output={
                    "direction": delta["direction"],
                    f"{label}_net_change_ha": delta[f"{label}_net_change_ha"],
                },
                runtime_ms=(time.perf_counter() - t0) * 1000,
            )
        )
        evidence["region_hint"] = region_hint(delta["gained_mask"] | delta["lost_mask"])
    else:
        evidence["region_hint"] = "the scene"

    return evidence, masks, calls


def run_query(paths: list[tuple[str, str]], question: str, max_dim: int = 4096) -> tuple[QueryTrace, list[str], str | None]:
    start = time.perf_counter()
    query_id = uuid.uuid4().hex[:12]

    manifest = build_manifest(paths, max_dim)
    if manifest.config == InputConfig.INVALID:
        raise InputRejected(manifest.rejection_reason or "Invalid input.", manifest)

    task, routing_layer = classify(question, manifest.legal_tasks)
    sufficiency, sufficiency_reason = check_sufficiency(question, manifest.config, task)

    raws = [_load_raw(p) for p, _ in paths]

    render_urls: list[str] = []
    overlay_url: str | None = None
    claims: list[ClaimRecord] = []
    tool_calls: list[ToolCallRecord] = []
    answer_text: str
    adjudication_status = "unverified"
    llm_override = False
    prompt_examples: list[str] = []

    if sufficiency == SufficiencyVerdict.REQUEST_INPUT:
        evidence: dict = {}
        answer_text = sufficiency_reason
        confidence = ConfidenceBreakdown(
            overall=0.0, band="low", basis="query declined: required input configuration not provided"
        )
    else:
        if manifest.config == InputConfig.SINGLE_IMAGE:
            evidence, masks, tool_calls = _tools_single_image(manifest.images[0], raws[0])
            base_composite = render_composite(raws[0], manifest.images[0].band_count, manifest.images[0].modality)
            composites_for_vlm = [base_composite]
        elif manifest.config == InputConfig.CROSS_MODAL_PAIR:
            evidence, masks, tool_calls = _tools_cross_modal(manifest.images, raws)
            base_idx = evidence.pop("_base_render_idx", 0)
            base_composite = render_composite(raws[base_idx], manifest.images[base_idx].band_count, manifest.images[base_idx].modality)
            composites_for_vlm = [
                render_composite(r, img.band_count, img.modality) for r, img in zip(raws, manifest.images)
            ]
        else:  # bi-temporal
            evidence, masks, tool_calls = _tools_bi_temporal(question, manifest.images, raws)
            base_composite = render_composite(raws[1], manifest.images[1].band_count, manifest.images[1].modality)
            composites_for_vlm = [
                render_composite(r, img.band_count, img.modality) for r, img in zip(raws, manifest.images)
            ]

        for i, comp in enumerate(composites_for_vlm):
            fname = f"{query_id}_render_{i}.png"
            comp.save(RENDER_DIR / fname)
            render_urls.append(f"/renders/{fname}")

        if masks:
            overlay = render_overlay(base_composite, masks)
            overlay_fname = f"{query_id}_overlay.png"
            overlay.save(RENDER_DIR / overlay_fname)
            overlay_url = f"/renders/{overlay_fname}"

        manifest_line = _manifest_line(manifest.config, manifest.images)
        prompt, prompt_examples = build_prompt(question, task, evidence, sufficiency, manifest_line)

        client = get_vlm_client()
        raw_output = client.generate(
            prompt, composites_for_vlm, context={"task": task, "evidence": evidence, "sufficiency": sufficiency}
        )

        adjudication_status, answer_text, claims, llm_override = adjudicate(raw_output, evidence)
        confidence = compute_confidence(adjudication_status, task, evidence, sufficiency)

    trace = QueryTrace(
        query_id=query_id,
        question=question,
        config=manifest.config,
        task_classified=task,
        routing_layer=routing_layer,
        legal_tasks=manifest.legal_tasks,
        sufficiency=sufficiency,
        sufficiency_reason=sufficiency_reason,
        plan=[c.tool for c in tool_calls],
        tool_calls=tool_calls,
        prompt_examples_used=prompt_examples,
        llm_override=llm_override,
        claims=claims,
        confidence=confidence,
        answer=answer_text,
        total_runtime_ms=(time.perf_counter() - start) * 1000,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    save_trace(trace)
    return trace, render_urls, overlay_url
