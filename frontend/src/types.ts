export type Modality = "optical" | "sar" | "unknown"
export type InputConfig = "single_image" | "cross_modal_pair" | "bi_temporal_pair" | "invalid"
export type Task = "vqa" | "caption" | "grounding" | "change" | "fusion"
export type Target = "water" | "vegetation" | "built_up" | "bare_soil" | "land_cover"
export type SufficiencyVerdict = "proceed" | "scope_down" | "request_input"

export interface ImageManifest {
  filename: string
  modality: Modality
  band_count: number
  band_names: string[]
  band_layout: string
  width: number
  height: number
  gsd_m: number | null
  analysis_width: number
  analysis_height: number
  analysis_gsd_m: number | null
  decimation: number
  crs: string | null
  bounds_lonlat: number[] | null
  dtype: string
  nodata_fraction: number
  acquisition_tag: string | null
  acquisition_date: string | null
  warnings: string[]
}

export interface AlignmentRecord {
  method: string
  overlap_fraction: number
  residual_shift_px: number[]
  shift_corrected: boolean
  notes: string[]
}

export interface InputManifest {
  config: InputConfig
  images: ImageManifest[]
  legal_tasks: Task[]
  alignment: AlignmentRecord | null
}

export interface SessionInfo {
  session_id: string
  created_at: string
  manifest: InputManifest
  render_urls: string[]
  sample_id: string | null
}

export interface SampleInfo {
  sample_id: string
  title: string
  description: string
  config: InputConfig
  suggested_questions: string[]
}

export interface QuerySpec {
  task: Task
  target: Target
  metric: string
  implied_baseline: boolean
  wants_location: boolean
  follow_up: boolean
  inherited_from: string | null
  routing_layer: string
  matched_rules: string[]
}

export interface PlanNode {
  node_id: string
  tool: string
  image_index: number[]
  depends_on: string[]
  params: Record<string, unknown>
}

export interface ToolCallRecord {
  node_id: string
  tool: string
  impl: string
  params: Record<string, unknown>
  output: Record<string, unknown>
  runtime_ms: number
  ledger_entries: string[]
  reused_from: string | null
  worker: string | null
}

export interface LedgerEntry {
  entry_id: string
  query_id: string
  seq: number
  stage: string
  kind: string
  key: string
  value: unknown
  unit: string | null
  source: string
  reused_from: string | null
  prev_hash: string
  hash: string
}

export interface StageRecord {
  name: string
  status: "ok" | "warn" | "declined" | "skipped"
  summary: string
  runtime_ms: number
}

export interface ClaimRecord {
  text: string
  value: number
  unit: string
  source: string
  matched_ledger_key: string | null
  ledger_entry_id: string | null
  ledger_value: number | null
  within_tolerance: boolean | null
  corrected: boolean
}

export interface Region {
  region_id: string
  label: string
  image_index: number
  bbox_px: number[]
  bbox_norm: number[]
  area_ha: number
  pixel_count: number
  centroid_px: number[]
  centroid_lonlat: number[] | null
  bbox_lonlat: number[] | null
  ledger_entry_id: string | null
}

export interface ConfidenceBreakdown {
  overall: number
  band: "high" | "medium" | "low"
  basis: string
  components: Record<string, number>
  component_notes: Record<string, string>
}

export interface Layer {
  name: string
  url: string
  color: number[]
  fraction: number | null
}

export interface QueryTrace {
  query_id: string
  session_id: string | null
  question: string
  config: InputConfig
  spec: QuerySpec | null
  task_classified: Task | null
  routing_layer: string
  sufficiency: SufficiencyVerdict
  sufficiency_reason: string
  missing_input: string | null
  stages: StageRecord[]
  plan: string[]
  plan_nodes: PlanNode[]
  tool_calls: ToolCallRecord[]
  prompt_examples_used: string[]
  narrator: string
  adapter: string | null
  llm_override: boolean
  adjudication_status: "consistent" | "corrected" | "unverified"
  claims: ClaimRecord[]
  regions: Region[]
  confidence: ConfidenceBreakdown
  answer: string
  answer_raw: string | null
  ledger: LedgerEntry[]
  ledger_head: string | null
  render_urls: string[]
  overlay_url: string | null
  layers: Layer[]
  total_runtime_ms: number
  created_at: string
}

export interface Health {
  status: string
  vlm_provider: string
  vlm_model: string
}

export interface LedgerVerification {
  valid: boolean
  entries: number
  head?: string
  first_bad_seq?: number
}
