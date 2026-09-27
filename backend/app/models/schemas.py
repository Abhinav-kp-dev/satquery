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


class Target(str, Enum):
    """The physical phenomenon a question is about. Every deterministic tool
    measures one or more of these; the parser maps question wording onto
    them so the planner knows which measurement the answer hinges on."""

    WATER = "water"
    VEGETATION = "vegetation"
    BUILT_UP = "built_up"
    BARE_SOIL = "bare_soil"
    LAND_COVER = "land_cover"  # "all of it" -- captioning / overview questions


class SufficiencyVerdict(str, Enum):
    PROCEED = "proceed"
    SCOPE_DOWN = "scope_down"
    REQUEST_INPUT = "request_input"


# --------------------------------------------------------------------------
# Stage 0: preflight
# --------------------------------------------------------------------------


class ImageManifest(BaseModel):
    """What preflight learns about a single uploaded raster, read from its
    pixels and georeferencing metadata alone -- before any model touches it."""

    filename: str
    modality: Modality
    band_count: int
    band_names: list[str] = Field(default_factory=list)
    band_layout: str = "unknown"
    width: int
    height: int
    gsd_m: float | None = None
    # Analysis grid: gigapixel scenes are read decimated (via overviews when
    # the file has them), so tools run on a bounded grid at a coarser GSD.
    analysis_width: int = 0
    analysis_height: int = 0
    analysis_gsd_m: float | None = None
    decimation: float = 1.0
    crs: str | None = None
    bounds: list[float] | None = None  # [left, bottom, right, top] in `crs`
    bounds_lonlat: list[float] | None = None  # [west, south, east, north]
    dtype: str
    nodata_fraction: float = 0.0
    acquisition_tag: str | None = None  # "T1"/"T2" or "optical"/"sar", set once paired
    acquisition_date: str | None = None
    warnings: list[str] = Field(default_factory=list)


class AlignmentRecord(BaseModel):
    """How the second image of a pair was brought onto the first's grid."""

    method: str  # "identical_grid" | "reprojected" | "pixel_grid_assumed"
    overlap_fraction: float = 1.0
    residual_shift_px: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    shift_corrected: bool = False
    notes: list[str] = Field(default_factory=list)


class InputManifest(BaseModel):
    """The single object every downstream stage reads. Built once, at upload
    time, and never re-derived -- this is the contract between preflight and
    everything after it."""

    config: InputConfig
    images: list[ImageManifest]
    legal_tasks: list[Task]
    alignment: AlignmentRecord | None = None
    rejection_reason: str | None = None


# --------------------------------------------------------------------------
# Stage 1: parse
# --------------------------------------------------------------------------


class QuerySpec(BaseModel):
    """The typed task specification a natural-language question is parsed
    into. Everything downstream reads this, never the raw question text."""

    task: Task
    target: Target
    metric: str  # "fraction" | "area" | "location" | "describe" | "change" | "agreement" | "presence"
    implied_baseline: bool = False
    wants_location: bool = False
    follow_up: bool = False
    inherited_from: str | None = None  # query_id whose spec filled in omitted parts
    routing_layer: str
    matched_rules: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Stage 3/4: plan + execute
# --------------------------------------------------------------------------


class PlanNode(BaseModel):
    node_id: str
    tool: str
    image_index: list[int] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class ToolCallRecord(BaseModel):
    node_id: str = ""
    tool: str
    impl: str
    params: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    runtime_ms: float
    ledger_entries: list[str] = Field(default_factory=list)
    reused_from: str | None = None  # "<query_id>:<node_id>" when served from session memory
    worker: str | None = None


# --------------------------------------------------------------------------
# Evidence ledger
# --------------------------------------------------------------------------


class LedgerEntry(BaseModel):
    """One append-only fact. The same structure plays four roles:
    agent memory (follow-up turns reuse entries), generation grounding (the
    narrator is shown entries by id), confidence substrate (calibration reads
    entries), and audit trail (entries are hash-chained)."""

    entry_id: str  # "E7" -- local to the query, stable for citation
    query_id: str
    seq: int = 0  # global position in the chain, assigned on commit
    stage: str
    kind: str  # "metadata" | "decision" | "measurement" | "claim" | "correction" | "confidence" | "report"
    key: str
    value: Any = None
    unit: str | None = None
    source: str
    reused_from: str | None = None
    prev_hash: str = ""
    hash: str = ""


class StageRecord(BaseModel):
    name: str  # parse | assess | plan | execute | adjudicate | calibrate | report
    status: str  # ok | warn | declined | skipped
    summary: str
    runtime_ms: float = 0.0


class ClaimRecord(BaseModel):
    text: str
    value: float
    unit: str
    source: str = "tagged"  # "tagged" (<CLAIM>) | "untagged" (bare number found in prose)
    matched_ledger_key: str | None = None
    ledger_entry_id: str | None = None
    ledger_value: float | None = None
    within_tolerance: bool | None = None
    corrected: bool = False


class Region(BaseModel):
    """A grounded region: a connected component of a tool-derived mask."""

    region_id: str
    label: str
    image_index: int = 0
    bbox_px: list[int]  # [x0, y0, x1, y1] on the analysis grid
    bbox_norm: list[float]  # same, normalised to 0..1
    area_ha: float
    pixel_count: int
    centroid_px: list[float]
    centroid_lonlat: list[float] | None = None
    bbox_lonlat: list[float] | None = None  # [west, south, east, north]
    ledger_entry_id: str | None = None


class ConfidenceBreakdown(BaseModel):
    overall: float
    band: str  # "high" / "medium" / "low"
    basis: str
    components: dict[str, float] = Field(default_factory=dict)
    component_notes: dict[str, str] = Field(default_factory=dict)


class Layer(BaseModel):
    name: str
    url: str
    color: list[int]
    fraction: float | None = None


class QueryTrace(BaseModel):
    query_id: str
    session_id: str | None = None
    question: str
    config: InputConfig
    spec: QuerySpec | None = None
    task_classified: Task | None
    routing_layer: str
    legal_tasks: list[Task]
    sufficiency: SufficiencyVerdict
    sufficiency_reason: str
    missing_input: str | None = None
    stages: list[StageRecord] = Field(default_factory=list)
    plan: list[str]
    plan_nodes: list[PlanNode] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord]
    prompt_examples_used: list[str]
    narrator: str = "mock"
    adapter: str | None = None
    llm_override: bool
    adjudication_status: str = "unverified"
    claims: list[ClaimRecord]
    regions: list[Region] = Field(default_factory=list)
    confidence: ConfidenceBreakdown
    answer: str
    answer_raw: str | None = None
    ledger: list[LedgerEntry] = Field(default_factory=list)
    ledger_head: str | None = None
    render_urls: list[str] = Field(default_factory=list)
    overlay_url: str | None = None
    layers: list[Layer] = Field(default_factory=list)
    total_runtime_ms: float
    created_at: str


class QueryResponse(BaseModel):
    trace: QueryTrace
    render_urls: list[str]
    overlay_url: str | None = None


class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    manifest: InputManifest
    render_urls: list[str]
    sample_id: str | None = None
    query_ids: list[str] = Field(default_factory=list)


class SampleInfo(BaseModel):
    sample_id: str
    title: str
    description: str
    config: InputConfig
    suggested_questions: list[str]
