import {
  CircleAlert,
  CornerDownRight,
  FileDown,
  FileJson,
  GitBranch,
  Loader2,
  MapPinned,
  Send,
  ShieldAlert,
  Sparkles,
  Upload,
} from "lucide-react"
import { useEffect, useRef, useState } from "react"
import { reportUrl } from "../api/client"
import { TARGET_LABEL, TASK_LABEL } from "../lib/format"
import type { ClaimRecord, QueryTrace, SufficiencyVerdict } from "../types"
import { ConfidenceMeter } from "./ConfidenceMeter"

const VERDICT: Record<SufficiencyVerdict, { label: string; cls: string }> = {
  proceed: { label: "Evidence sufficient", cls: "text-confidence-high bg-confidence-high/10" },
  scope_down: { label: "Scoped to evidence", cls: "text-accent-amber bg-accent-amber/10" },
  request_input: { label: "Input required", cls: "text-confidence-low bg-confidence-low/10" },
}

const STAGES = ["Parse", "Assess", "Plan", "Execute", "Adjudicate", "Calibrate", "Report"]

function AnnotatedAnswer({ trace, onClaim }: { trace: QueryTrace; onClaim: (entryId: string) => void }) {
  const parts: React.ReactNode[] = []
  let cursor = 0
  const text = trace.answer
  trace.claims.forEach((c: ClaimRecord, i) => {
    const at = text.indexOf(c.text, cursor)
    if (at < 0 || !c.text) return
    parts.push(text.slice(cursor, at))
    parts.push(
      <button
        key={i}
        onClick={() => c.ledger_entry_id && onClaim(c.ledger_entry_id)}
        title={
          c.ledger_entry_id
            ? `${c.matched_ledger_key} = ${c.ledger_value} (ledger ${c.ledger_entry_id})${c.corrected ? ` - narrator said ${c.value}` : ""}`
            : "No supporting measurement"
        }
        className={`rounded px-0.5 font-medium underline decoration-dotted underline-offset-4 transition-colors ${
          c.corrected
            ? "bg-accent-amber/10 text-accent-amber decoration-accent-amber/60 hover:bg-accent-amber/20"
            : "text-accent-teal decoration-accent-teal/50 hover:bg-accent-teal/10"
        }`}
      >
        {c.text}
      </button>,
    )
    cursor = at + c.text.length
  })
  parts.push(text.slice(cursor))
  return <p className="text-[15px] leading-relaxed text-ink">{parts}</p>
}

function TurnCard({
  trace,
  active,
  onSelect,
  onClaim,
  onUploadMore,
}: {
  trace: QueryTrace
  active: boolean
  onSelect: () => void
  onClaim: (entryId: string) => void
  onUploadMore: () => void
}) {
  const v = VERDICT[trace.sufficiency]
  const corrected = trace.claims.filter((c) => c.corrected)
  return (
    <div className="animate-fade-in space-y-2">
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-brand-500/15 px-4 py-2.5 text-[14px] text-ink">{trace.question}</div>
      </div>
      <div
        onClick={onSelect}
        className={`cursor-pointer rounded-xl border bg-surface p-4 transition-colors ${
          active ? "border-brand-500/50 ring-1 ring-brand-500/20" : "border-border-soft hover:border-border"
        }`}
      >
        <div className="mb-3 flex flex-wrap items-center gap-1.5">
          {trace.spec && (
            <>
              <span className="rounded-md border border-border-soft bg-surface-raised px-2 py-0.5 text-[11.5px] text-ink-muted">
                {TASK_LABEL[trace.spec.task]}
              </span>
              <span className="rounded-md border border-border-soft bg-surface-raised px-2 py-0.5 text-[11.5px] text-ink-muted">
                {TARGET_LABEL[trace.spec.target]}
              </span>
            </>
          )}
          <span className={`inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11.5px] font-medium ${v.cls}`}>
            <GitBranch size={11} />
            {v.label}
          </span>
          {trace.spec?.follow_up && (
            <span className="inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11.5px] text-accent-violet">
              <CornerDownRight size={11} /> follow-up
            </span>
          )}
          {trace.regions.length > 0 && (
            <span className="inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[11.5px] text-ink-faint">
              <MapPinned size={11} /> {trace.regions.length} region{trace.regions.length > 1 ? "s" : ""}
            </span>
          )}
        </div>

        <AnnotatedAnswer trace={trace} onClaim={onClaim} />

        {trace.sufficiency === "request_input" && trace.missing_input && (
          <div className="mt-3 flex items-start gap-2 rounded-lg border border-confidence-low/25 bg-confidence-low/[0.05] px-3 py-2.5 text-[12.5px] text-ink-muted">
            <Upload size={14} className="mt-0.5 shrink-0 text-confidence-low" />
            <div>
              Needed: <span className="text-ink">{trace.missing_input}</span>. Nothing was measured, because a guess
              here would be unsupported.
              <button onClick={onUploadMore} className="ml-1 text-brand-400 hover:underline">
                Start a session with it
              </button>
            </div>
          </div>
        )}

        {corrected.length > 0 && (
          <div className="mt-3 flex items-start gap-2 rounded-lg border border-accent-amber/25 bg-accent-amber/[0.05] px-3 py-2.5 text-[12.5px] text-ink-muted">
            <ShieldAlert size={14} className="mt-0.5 shrink-0 text-accent-amber" />
            <div>
              The narrator stated {corrected.length === 1 ? "a figure" : `${corrected.length} figures`} the measurements
              don't support ({corrected.map((c) => `${c.value} ${c.unit}`).join(", ")}).{" "}
              {corrected.length === 1 ? "It was" : "They were"} replaced with the measured value
              {corrected.length === 1 ? "" : "s"} before delivery: the measurement wins.
            </div>
          </div>
        )}

        <div className="mt-4 border-t border-border-soft pt-3">
          <ConfidenceMeter confidence={trace.confidence} compact />
        </div>

        <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-[11px] text-ink-faint">
          <span>
            {trace.total_runtime_ms.toFixed(0)} ms · narrator {trace.narrator}
            {trace.adapter ? ` (${trace.adapter})` : ""} · {trace.ledger.length} ledger entries
          </span>
          <span className="flex gap-1" onClick={(e) => e.stopPropagation()}>
            <a href={reportUrl(trace.query_id, "pdf")} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 rounded border border-border-soft px-1.5 py-0.5 hover:text-ink">
              <FileDown size={11} /> PDF
            </a>
            <a href={reportUrl(trace.query_id, "json")} className="inline-flex items-center gap-1 rounded border border-border-soft px-1.5 py-0.5 hover:text-ink">
              <FileJson size={11} /> JSON
            </a>
            {trace.regions.some((r) => r.centroid_lonlat) && (
              <a href={reportUrl(trace.query_id, "geojson")} className="inline-flex items-center gap-1 rounded border border-border-soft px-1.5 py-0.5 hover:text-ink">
                <MapPinned size={11} /> GeoJSON
              </a>
            )}
          </span>
        </div>
      </div>
    </div>
  )
}

function PendingTurn({ question }: { question: string }) {
  const [i, setI] = useState(0)
  useEffect(() => {
    const id = setInterval(() => setI((x) => Math.min(x + 1, STAGES.length - 1)), 120)
    return () => clearInterval(id)
  }, [])
  return (
    <div className="space-y-2">
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-brand-500/15 px-4 py-2.5 text-[14px] text-ink">{question}</div>
      </div>
      <div className="flex items-center gap-2 rounded-xl border border-border-soft bg-surface px-4 py-3 text-[12.5px] text-ink-muted">
        <Loader2 size={14} className="animate-spin text-brand-400" />
        {STAGES.map((s, k) => (
          <span key={s} className={k <= i ? "text-ink" : "text-ink-faint"}>
            {s}
            {k < STAGES.length - 1 && <span className="mx-1 text-ink-faint">›</span>}
          </span>
        ))}
      </div>
    </div>
  )
}

export function Conversation({
  turns,
  pending,
  activeId,
  onSelect,
  onClaim,
  onUploadMore,
}: {
  turns: QueryTrace[]
  pending: string | null
  activeId: string | null
  onSelect: (id: string) => void
  onClaim: (queryId: string, entryId: string) => void
  onUploadMore: () => void
}) {
  const end = useRef<HTMLDivElement>(null)
  useEffect(() => end.current?.scrollIntoView({ behavior: "smooth", block: "end" }), [turns.length, pending])
  return (
    <div className="flex flex-col gap-5">
      {turns.map((t) => (
        <TurnCard
          key={t.query_id}
          trace={t}
          active={t.query_id === activeId}
          onSelect={() => onSelect(t.query_id)}
          onClaim={(e) => onClaim(t.query_id, e)}
          onUploadMore={onUploadMore}
        />
      ))}
      {pending && <PendingTurn question={pending} />}
      <div ref={end} />
    </div>
  )
}

export function Composer({
  suggestions,
  loading,
  error,
  onAsk,
}: {
  suggestions: string[]
  loading: boolean
  error: string | null
  onAsk: (q: string, stress: boolean) => void
}) {
  const [q, setQ] = useState("")
  const [stress, setStress] = useState(false)
  const submit = (text: string) => {
    if (!text.trim() || loading) return
    onAsk(text.trim(), stress)
    setQ("")
  }
  return (
    <div className="rounded-xl border border-border-soft bg-surface p-3">
      {suggestions.length > 0 && (
        <div className="mb-2.5 flex flex-wrap gap-1.5">
          {suggestions.map((s) => (
            <button
              key={s}
              disabled={loading}
              onClick={() => submit(s)}
              className="rounded-full border border-border-soft px-2.5 py-1 text-[11.5px] text-ink-faint transition-colors hover:border-border hover:text-ink-muted"
            >
              {s}
            </button>
          ))}
        </div>
      )}
      <div className="flex items-end gap-2">
        <textarea
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault()
              submit(q)
            }
          }}
          rows={2}
          placeholder="Ask about this imagery in plain language. Follow-ups like “and in hectares?” work."
          className="min-h-[44px] flex-1 resize-none rounded-lg border border-border-soft bg-surface-raised px-3 py-2.5 text-[13.5px] text-ink placeholder:text-ink-faint focus:border-brand-500/50 focus:outline-none"
        />
        <button
          onClick={() => submit(q)}
          disabled={!q.trim() || loading}
          aria-label="Ask"
          className="flex h-11 w-11 items-center justify-center rounded-lg bg-brand-500 text-white transition-colors hover:bg-brand-600 disabled:bg-surface-raised disabled:text-ink-faint"
        >
          {loading ? <Loader2 size={16} className="animate-spin" /> : <Send size={15} />}
        </button>
      </div>
      <label className="mt-2 flex cursor-pointer items-center gap-2 text-[11.5px] text-ink-faint">
        <input type="checkbox" checked={stress} onChange={(e) => setStress(e.target.checked)} className="accent-accent-amber" />
        <Sparkles size={11} className={stress ? "text-accent-amber" : ""} />
        Stress test: make the narrator misstate a figure, to watch adjudication catch it
      </label>
      {error && (
        <div className="mt-2 flex items-start gap-2 text-[12px] text-confidence-low">
          <CircleAlert size={13} className="mt-0.5 shrink-0" /> {error}
        </div>
      )}
    </div>
  )
}
