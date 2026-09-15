import { ChevronDown, Clock, ListTree, Route, ShieldCheck, Sigma } from "lucide-react"
import { useState } from "react"
import type { QueryTrace } from "../types"

function Section({
  title,
  icon,
  defaultOpen,
  children,
}: {
  title: string
  icon: React.ReactNode
  defaultOpen?: boolean
  children: React.ReactNode
}) {
  const [open, setOpen] = useState(Boolean(defaultOpen))
  return (
    <div className="border-b border-border-soft last:border-b-0">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-2 px-1 py-3 text-left"
      >
        <span className="flex items-center gap-2 text-[13px] font-medium text-ink-muted">
          {icon}
          {title}
        </span>
        <ChevronDown size={14} className={`text-ink-faint transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && <div className="animate-fade-in px-1 pb-4">{children}</div>}
    </div>
  )
}

function KeyValue({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1 text-[12.5px]">
      <span className="text-ink-faint">{label}</span>
      <span className="text-right text-ink-muted">{value}</span>
    </div>
  )
}

function formatValue(v: unknown): string {
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(3)
  if (v === null || v === undefined) return "--"
  return String(v)
}

export function TracePanel({ trace }: { trace: QueryTrace }) {
  return (
    <div className="rounded-xl border border-border-soft bg-surface px-4">
      <div className="border-b border-border-soft px-1 py-3">
        <span className="text-[13px] font-semibold text-ink">Evidence ledger</span>
        <span className="ml-2 text-[11px] text-ink-faint">query {trace.query_id}</span>
      </div>

      <Section title="Routing" icon={<Route size={14} />} defaultOpen>
        <KeyValue label="Task classified" value={trace.task_classified ?? "declined"} />
        <KeyValue label="Routing layer" value={trace.routing_layer} />
        <KeyValue label="Legal tasks for this input" value={trace.legal_tasks.join(", ") || "none"} />
      </Section>

      <Section title="Sufficiency check" icon={<ShieldCheck size={14} />}>
        <p className="text-[12.5px] leading-relaxed text-ink-muted">{trace.sufficiency_reason}</p>
      </Section>

      <Section title={`Tool calls (${trace.tool_calls.length})`} icon={<ListTree size={14} />} defaultOpen>
        {trace.tool_calls.length === 0 && <p className="text-[12.5px] text-ink-faint">No tools were executed.</p>}
        <div className="space-y-3">
          {trace.tool_calls.map((call, i) => (
            <div key={i} className="rounded-lg border border-border-soft bg-surface-raised p-3">
              <div className="mb-1 flex items-center justify-between">
                <span className="font-mono text-[12px] text-brand-400">{call.tool}</span>
                <span className="text-[11px] tabular-nums text-ink-faint">{call.runtime_ms.toFixed(1)} ms</span>
              </div>
              <div className="mb-2 text-[11.5px] text-ink-faint">{call.impl}</div>
              {Object.entries(call.output).length > 0 && (
                <div className="space-y-0.5 border-t border-border-soft pt-2">
                  {Object.entries(call.output).map(([k, v]) => (
                    <KeyValue key={k} label={k} value={formatValue(v)} />
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </Section>

      <Section title="Narration examples retrieved" icon={<Sigma size={14} />}>
        {trace.prompt_examples_used.length === 0 ? (
          <p className="text-[12.5px] text-ink-faint">No in-context examples were retrieved for this response.</p>
        ) : (
          <ul className="space-y-1">
            {trace.prompt_examples_used.map((ex, i) => (
              <li key={i} className="truncate text-[12px] text-ink-faint">
                {ex}
              </li>
            ))}
          </ul>
        )}
      </Section>

      <Section title="Timing" icon={<Clock size={14} />}>
        <KeyValue label="Total runtime" value={`${trace.total_runtime_ms.toFixed(1)} ms`} />
        <KeyValue label="Created" value={new Date(trace.created_at).toLocaleString()} />
      </Section>
    </div>
  )
}
