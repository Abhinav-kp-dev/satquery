export type Modality = "optical" | "sar" | "unknown"

export type InputConfig =
  | "single_image"
  | "cross_modal_pair"
  | "bi_temporal_pair"
  | "invalid"

export type Task = "vqa" | "caption" | "grounding" | "change" | "fusion"

export type SufficiencyVerdict = "proceed" | "scope_down" | "request_input"

export interface ToolCallRecord {
  tool: string
  impl: string
  params: Record<string, unknown>
  output: Record<string, unknown>
  runtime_ms: number
}

export interface ClaimRecord {
  text: string
  value: number
  unit: string
  matched_ledger_key: string | null
  within_tolerance: boolean | null
}

export interface ConfidenceBreakdown {
  overall: number
  band: "high" | "medium" | "low"
  basis: string
  components: Record<string, number>
}

export interface QueryTrace {
  query_id: string
  question: string
  config: InputConfig
  task_classified: Task | null
  routing_layer: string
  legal_tasks: Task[]
  sufficiency: SufficiencyVerdict
  sufficiency_reason: string
  plan: string[]
  tool_calls: ToolCallRecord[]
  prompt_examples_used: string[]
  llm_override: boolean
  claims: ClaimRecord[]
  confidence: ConfidenceBreakdown
  answer: string
  total_runtime_ms: number
  created_at: string
}

export interface QueryResponse {
  trace: QueryTrace
  render_urls: string[]
  overlay_url: string | null
}

export interface ApiError {
  detail: string
}
