from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Modality(str, Enum):
    OPTICAL = "optical"
    SAR = "sar"
    UNKNOWN = "unknown"


class InputConfig(str, Enum):
    SINGLE_IMAGE = "single_image"
    CROSS_MODAL_PAIR = "cross_modal_pair"
    BI_TEMPORAL_PAIR = "bi_temporal_pair"
    INVALID = "invalid"


class Task(str, Enum):
    VQA = "vqa"
    CAPTION = "caption"
    GROUNDING = "grounding"
    CHANGE = "change"
    FUSION = "fusion"


class SufficiencyVerdict(str, Enum):
    PROCEED = "proceed"
    SCOPE_DOWN = "scope_down"
    REQUEST_INPUT = "request_input"


class ImageManifest(BaseModel):
    """What preflight learns about a single uploaded raster, read from its
    pixels and georeferencing metadata alone -- before any model touches it."""

    filename: str
    modality: Modality
    band_count: int
    width: int
    height: int
    gsd_m: float | None = None
    crs: str | None = None
    dtype: str
    nodata_fraction: float = 0.0
    acquisition_tag: str | None = None  # "T1" / "T2", set once paired
    warnings: list[str] = Field(default_factory=list)


class InputManifest(BaseModel):
    """The single object every downstream stage reads. Built once, at upload
    time, and never re-derived — this is the contract between preflight and
    everything after it."""

    config: InputConfig
    images: list[ImageManifest]
    legal_tasks: list[Task]
    rejection_reason: str | None = None


class ToolCallRecord(BaseModel):
    tool: str
    impl: str
    params: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    runtime_ms: float


class ClaimRecord(BaseModel):
    text: str
    value: float
    unit: str
    matched_ledger_key: str | None = None
    within_tolerance: bool | None = None


class ConfidenceBreakdown(BaseModel):
    overall: float
    band: str  # "high" / "medium" / "low"
    basis: str
    components: dict[str, float] = Field(default_factory=dict)


class QueryTrace(BaseModel):
    query_id: str
    question: str
    config: InputConfig
    task_classified: Task | None
    routing_layer: str
    legal_tasks: list[Task]
    sufficiency: SufficiencyVerdict
    sufficiency_reason: str
    plan: list[str]
    tool_calls: list[ToolCallRecord]
    prompt_examples_used: list[str]
    llm_override: bool
    claims: list[ClaimRecord]
    confidence: ConfidenceBreakdown
    answer: str
    total_runtime_ms: float
    created_at: str


class QueryResponse(BaseModel):
    trace: QueryTrace
    render_urls: list[str]
    overlay_url: str | None = None
