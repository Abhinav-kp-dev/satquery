"""Stage 4 -- EXECUTE: run the plan DAG wave by wave.

Nodes within a wave are independent (e.g. T1 and T2 index computation, or
the optical and SAR branches of a fusion query) and run concurrently on a
thread pool -- NumPy releases the GIL for the heavy array work.

Every scalar a node produces is appended to the Evidence Ledger as a
`measurement` entry. Session memory: when a follow-up turn in the same
session schedules a node with an identical signature, the cached result is
reused and its ledger entries are re-appended with `reused_from` pointing
at the original entry -- the ledger's agent-memory role.
"""
from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.core.planner import waves
from app.core.raster_io import LoadedScene
from app.core.registry import tool_entry
from app.ledger.ledger import LedgerWriter
from app.models.schemas import InputManifest, PlanNode, ToolCallRecord
from app.tools.delta_change import compute_delta
from app.tools.fusion_agreement import compute_agreement
from app.tools.grounding import extract_regions
from app.tools.region import region_hint
from app.tools.sar_backscatter import compute_sar_backscatter
from app.tools.spectral_index import compute_spectral_indices, primary_water

_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool")
CLASSES = ("water", "vegetation", "built_up", "bare_soil")


@dataclass
class NodeResult:
    scalars: dict[str, tuple[float, str, str]] = field(default_factory=dict)  # key -> (value, unit, kind)
    decisions: dict[str, Any] = field(default_factory=dict)
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    regions: list[dict] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ExecContext:
    manifest: InputManifest
    scene: LoadedScene


@dataclass
class ExecResult:
    results: dict[str, NodeResult]
    calls: list[ToolCallRecord]


# ------------------------------------------------------------ node functions


def _gsd(ctx: ExecContext, i: int) -> float | None:
    img = ctx.manifest.images[i]
    return img.analysis_gsd_m or img.gsd_m


def _run_spectral(ctx: ExecContext, node: PlanNode, deps: dict[str, NodeResult]) -> NodeResult:
    i = node.image_index[0]
    img = ctx.manifest.images[i]
    s = node.params.get("suffix", "")
    res = compute_spectral_indices(ctx.scene.arrays[i], img.band_count, _gsd(ctx, i), img.band_names)
    out = NodeResult(warnings=list(res["warnings"]))
    out.params = {"layout": res["layout"], "gsd_m": _gsd(ctx, i)}

    water = primary_water(res)
    per_class = {"water": water[1] if water else None, "vegetation": res.get("ndvi"), "built_up": res.get("ndbi"), "bare_soil": res.get("bsi")}
    if water:
        out.decisions[f"water_index{s}"] = water[0]
    for cls, r in per_class.items():
        if r is None:
            continue
        out.scalars[f"{cls}_fraction{s}"] = (r[f"{cls}_fraction"], "fraction", "measurement")
        out.scalars[f"{cls}_area_ha{s}"] = (r[f"{cls}_area_ha"], "ha", "measurement")
        out.scalars[f"{cls}_separability{s}"] = (r["separability"], "eta", "quality")
        out.params[f"{cls}_threshold"] = round(r["threshold"], 4)
        if r.get("clamped"):
            out.params[f"{cls}_otsu_clamped_from"] = round(r["otsu_raw"], 4)
        out.masks[cls] = r["mask"]
    for idx in ("ndwi", "mndwi", "ndvi", "ndbi", "bsi"):
        if idx in res:
            out.scalars[f"{idx}_mean{s}"] = (res[idx]["mean_index"], "index", "measurement")
    out.scalars[f"valid_fraction{s}"] = (res["valid_fraction"], "fraction", "quality")
    return out


def _run_sar(ctx: ExecContext, node: PlanNode, deps: dict[str, NodeResult]) -> NodeResult:
    i = node.image_index[0]
    img = ctx.manifest.images[i]
    s = node.params.get("suffix", "")
    res = compute_sar_backscatter(ctx.scene.arrays[i], img.band_count, _gsd(ctx, i), img.band_names)
    out = NodeResult(warnings=list(res["warnings"]))
    out.params = {"input_scale": res["input_scale"], "speckle_filter": res["speckle_filter"], "water_threshold_db": round(res["threshold_db"], 2)}
    out.scalars[f"water_fraction{s}"] = (res["water_fraction"], "fraction", "measurement")
    out.scalars[f"water_area_ha{s}"] = (res["water_area_ha"], "ha", "measurement")
    out.scalars[f"water_separability{s}"] = (res["separability"], "eta", "quality")
    out.scalars[f"mean_co_pol_db{s}"] = (res["mean_co_pol_db"], "dB", "measurement")
    out.masks["water"] = res["mask"]
    if "built_up_mask" in res:
        out.scalars[f"built_up_fraction{s}"] = (res["built_up_fraction"], "fraction", "measurement")
        out.scalars[f"built_up_area_ha{s}"] = (res["built_up_area_ha"], "ha", "measurement")
        out.params["built_up_threshold_db"] = round(res["built_up_threshold_db"], 2)
        out.masks["built_up"] = res["built_up_mask"]
    return out


def _pick_common_target(a: NodeResult, b: NodeResult, wanted: str) -> str | None:
    common = [c for c in CLASSES if c in a.masks and c in b.masks]
    if wanted in common:
        return wanted
    if not common:
        return None
    return max(common, key=lambda c: int((a.masks[c] ^ b.masks[c]).sum()))


def _run_delta(ctx: ExecContext, node: PlanNode, deps: dict[str, NodeResult]) -> NodeResult:
    a, b = (deps[d] for d in node.depends_on)
    label = _pick_common_target(a, b, node.params.get("target", "land_cover"))
    out = NodeResult()
    if label is None:
        out.warnings.append("No indicator could be measured on both dates; change cannot be quantified.")
        return out
    valid = np.all(np.isfinite(ctx.scene.arrays[0]), axis=0) & np.all(np.isfinite(ctx.scene.arrays[1]), axis=0)
    d = compute_delta(a.masks[label], b.masks[label], _gsd(ctx, 0), label, valid)
    out.decisions["change_indicator"] = label
    out.decisions["direction"] = d["direction"]
    if node.params.get("target") not in (label,):
        out.decisions["indicator_selection"] = "largest measured change among indicators available on both dates"
    for k in (f"{label}_gained_ha", f"{label}_lost_ha", f"{label}_net_change_ha", f"{label}_t1_area_ha", f"{label}_t2_area_ha"):
        out.scalars[k] = (d[k], "ha", "measurement")
    out.scalars[f"{label}_t1_fraction"] = (d["t1_fraction"], "fraction", "measurement")
    out.scalars[f"{label}_t2_fraction"] = (d["t2_fraction"], "fraction", "measurement")
    out.scalars["changed_fraction"] = (d["changed_fraction"], "fraction", "measurement")
    out.scalars[f"{label}_relative_change_percent"] = (d["relative_change_percent"], "percent", "measurement")
    out.masks["gained"] = d["gained_mask"]
    out.masks["lost"] = d["lost_mask"]
    out.params = {"indicator": label, "min_mapping_unit_px": 9}
    return out


def _run_fusion(ctx: ExecContext, node: PlanNode, deps: dict[str, NodeResult]) -> NodeResult:
    r0, r1 = (deps[d] for d in node.depends_on)
    opt_first = ctx.manifest.images[node.image_index[0]].modality.value == "optical"
    opt, sar = (r0, r1) if opt_first else (r1, r0)
    wanted = node.params.get("target", "water")
    label = wanted if wanted in opt.masks and wanted in sar.masks else ("water" if "water" in opt.masks else None)
    out = NodeResult()
    if label is None:
        out.warnings.append("No phenomenon measurable by both sensors; agreement unavailable.")
        return out
    ag = compute_agreement(opt.masks[label], sar.masks[label])
    out.decisions["agreement_target"] = label
    out.scalars["iou"] = (ag["iou"], "iou", "agreement")
    out.scalars["dice"] = (ag["dice"], "dice", "agreement")
    out.scalars["agreement_fraction"] = (ag["agreement_fraction"], "fraction", "measurement")
    out.scalars["optical_only_fraction"] = (ag["optical_only_fraction"], "fraction", "measurement")
    out.scalars["sar_only_fraction"] = (ag["sar_only_fraction"], "fraction", "measurement")
    out.masks["agreement"] = ag["agreement_mask"]
    out.masks["optical_only"] = ag["optical_only_mask"]
    out.masks["sar_only"] = ag["sar_only_mask"]
    out.params = {"target": label}
    return out


def _run_grounding(ctx: ExecContext, node: PlanNode, deps: dict[str, NodeResult]) -> NodeResult:
    src_id = node.depends_on[0]
    src = deps[src_id]
    target = node.params.get("target", "land_cover")
    gsd = _gsd(ctx, 0)
    out = NodeResult()

    if src_id == "delta_change":
        indicator = src.decisions.get("change_indicator", "change")
        selected = [(f"{indicator}_gained", src.masks.get("gained")), (f"{indicator}_lost", src.masks.get("lost"))]
        img_idx = 1
    elif src_id == "fusion_agreement":
        t = src.decisions.get("agreement_target", "water")
        selected = [(f"{t}_agreement", src.masks.get("agreement"))]
        img_idx = 0
    else:
        img_idx = int(src_id.split("#")[-1]) if "#" in src_id else 0
        if target in src.masks:
            selected = [(target, src.masks[target])]
        else:
            selected = [(c, src.masks[c]) for c in CLASSES if c in src.masks]

    hint_mask = None
    per_label = 8 if len(selected) == 1 else 4
    for label, mask in selected:
        if mask is None:
            continue
        regions = extract_regions(mask, label, gsd, ctx.scene.geo, image_index=img_idx, max_regions=per_label)
        out.regions.extend(regions)
        out.scalars[f"region_count_{label}"] = (float(len(regions)), "count", "measurement")
        if regions:
            out.scalars[f"largest_region_ha_{label}"] = (regions[0]["area_ha"], "ha", "measurement")
        if hint_mask is None and mask.any():
            hint_mask = mask
    out.decisions["region_hint"] = region_hint(hint_mask) if hint_mask is not None else "the scene"
    out.params = {"source": src_id, "labels": [l for l, _ in selected]}
    return out


NODE_FUNCS = {
    "spectral_index": _run_spectral,
    "sar_backscatter": _run_sar,
    "delta_change": _run_delta,
    "fusion_agreement": _run_fusion,
    "grounding": _run_grounding,
}

# ------------------------------------------------------------ session memory


class SessionMemory:
    """Per-session cache of node results, bounded LRU across sessions."""

    def __init__(self, max_sessions: int = 8):
        self._data: OrderedDict[str, dict[str, tuple[NodeResult, str, dict[str, str]]]] = OrderedDict()
        self._lock = threading.Lock()
        self.max_sessions = max_sessions

    @staticmethod
    def signature(node: PlanNode) -> str:
        return json.dumps({"tool": node.tool, "img": node.image_index, "deps": node.depends_on, "params": node.params}, sort_keys=True)

    def get(self, session_id: str | None, node: PlanNode):
        if not session_id:
            return None
        with self._lock:
            sess = self._data.get(session_id)
            if sess is None:
                return None
            self._data.move_to_end(session_id)
            return sess.get(self.signature(node))

    def put(self, session_id: str | None, node: PlanNode, result: NodeResult, query_id: str, entry_ids: dict[str, str]):
        if not session_id:
            return
        with self._lock:
            sess = self._data.setdefault(session_id, {})
            self._data.move_to_end(session_id)
            sess[self.signature(node)] = (result, query_id, entry_ids)
            while len(self._data) > self.max_sessions:
                self._data.popitem(last=False)

    def forget(self, session_id: str):
        with self._lock:
            self._data.pop(session_id, None)


MEMORY = SessionMemory()

# ------------------------------------------------------------ executor


def _record(ledger: LedgerWriter, node: PlanNode, result: NodeResult, reused: tuple[str, dict[str, str]] | None) -> dict[str, str]:
    ids: dict[str, str] = {}
    for key, (value, unit, kind) in result.scalars.items():
        reused_from = f"{reused[0]}:{reused[1][key]}" if reused and key in reused[1] else None
        ids[key] = ledger.append("execute", "measurement", key, value, unit, source=node.node_id, reused_from=reused_from)
    for key, value in result.decisions.items():
        ledger.append("execute", "decision", key, value, None, source=node.node_id)
    for r in result.regions:
        r["ledger_entry_id"] = ledger.append(
            "execute", "region", r["region_id"],
            {"bbox_px": r["bbox_px"], "area_ha": r["area_ha"], "centroid_lonlat": r["centroid_lonlat"]},
            "ha", source=node.node_id,
        )
    return ids


def execute_plan(
    nodes: list[PlanNode], ctx: ExecContext, ledger: LedgerWriter, session_id: str | None = None
) -> ExecResult:
    results: dict[str, NodeResult] = {}
    calls: list[ToolCallRecord] = []

    for wave in waves(nodes):
        def run(node: PlanNode):
            cached = MEMORY.get(session_id, node) if node.tool != "grounding" else None
            t0 = time.perf_counter()
            if cached is not None:
                return node, cached[0], (cached[1], cached[2]), (time.perf_counter() - t0) * 1000, "session-memory"
            deps = {d: results[d] for d in node.depends_on}
            res = NODE_FUNCS[node.tool](ctx, node, deps)
            return node, res, None, (time.perf_counter() - t0) * 1000, threading.current_thread().name

        outcomes = list(_POOL.map(run, wave)) if len(wave) > 1 else [run(wave[0])]
        for node, res, reused, ms, worker in outcomes:  # ledger order is plan order, not completion order
            results[node.node_id] = res
            ids = _record(ledger, node, res, reused)
            if reused is None and node.tool != "grounding":
                MEMORY.put(session_id, node, res, ledger.query_id, ids)
            calls.append(
                ToolCallRecord(
                    node_id=node.node_id,
                    tool=node.tool,
                    impl=tool_entry(node.tool).get("impl", node.tool),
                    params={**node.params, **res.params},
                    output={k: v[0] for k, v in res.scalars.items()} | {k: v for k, v in res.decisions.items()},
                    runtime_ms=ms,
                    ledger_entries=list(ids.values()),
                    reused_from=f"{reused[0]}:{node.node_id}" if reused else None,
                    worker=worker,
                )
            )
    return ExecResult(results=results, calls=calls)
