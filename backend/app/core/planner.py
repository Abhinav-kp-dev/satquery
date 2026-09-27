"""Stage 3 -- PLAN: registry lookup -> tool DAG.

Per-image analysis nodes have no dependencies on each other, so they form
the first execution wave and run in parallel; pair tools (change, fusion)
depend on both; grounding depends on whatever produced the mask the answer
is about. Only tools the registry lists for the task are scheduled.
"""
from __future__ import annotations

from app.core.registry import allowed_tools
from app.models.schemas import InputConfig, InputManifest, Modality, PlanNode, QuerySpec, Task


def _suffix(config: InputConfig, manifest: InputManifest, i: int) -> str:
    if config == InputConfig.CROSS_MODAL_PAIR:
        return f"_{manifest.images[i].modality.value}"
    if config == InputConfig.BI_TEMPORAL_PAIR:
        return f"_t{i + 1}"
    return ""


def build_plan(spec: QuerySpec, manifest: InputManifest) -> list[PlanNode]:
    allowed = set(allowed_tools(spec.task))
    nodes: list[PlanNode] = []
    analysis_ids: list[str] = []

    for i, img in enumerate(manifest.images):
        tool = "spectral_index" if img.modality == Modality.OPTICAL else "sar_backscatter"
        if tool not in allowed:
            continue
        node_id = f"{tool}#{i}"
        nodes.append(PlanNode(node_id=node_id, tool=tool, image_index=[i], params={"suffix": _suffix(manifest.config, manifest, i)}))
        analysis_ids.append(node_id)

    mask_source = analysis_ids[:1]
    if manifest.config == InputConfig.BI_TEMPORAL_PAIR and "delta_change" in allowed and len(analysis_ids) == 2:
        nodes.append(
            PlanNode(node_id="delta_change", tool="delta_change", image_index=[0, 1], depends_on=analysis_ids, params={"target": spec.target.value})
        )
        if spec.task in (Task.CHANGE, Task.GROUNDING):
            mask_source = ["delta_change"]
    if manifest.config == InputConfig.CROSS_MODAL_PAIR and "fusion_agreement" in allowed and len(analysis_ids) == 2:
        nodes.append(
            PlanNode(
                node_id="fusion_agreement", tool="fusion_agreement", image_index=[0, 1], depends_on=analysis_ids,
                params={"target": spec.target.value},
            )
        )
        if spec.task == Task.FUSION:
            mask_source = ["fusion_agreement"]

    if "grounding" in allowed and mask_source:
        nodes.append(
            PlanNode(node_id="grounding", tool="grounding", depends_on=mask_source, params={"target": spec.target.value, "task": spec.task.value})
        )
    return nodes


def waves(nodes: list[PlanNode]) -> list[list[PlanNode]]:
    """Topological layering: every node in a wave depends only on earlier waves."""
    done: set[str] = set()
    remaining = list(nodes)
    out: list[list[PlanNode]] = []
    while remaining:
        ready = [n for n in remaining if all(d in done for d in n.depends_on)]
        if not ready:
            raise ValueError(f"Plan has a cycle or a missing dependency: {[n.node_id for n in remaining]}")
        out.append(ready)
        done.update(n.node_id for n in ready)
        remaining = [n for n in remaining if n.node_id not in done]
    return out
