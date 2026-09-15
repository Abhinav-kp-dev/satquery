import { History } from "lucide-react"
import type { QueryTrace } from "../types"

const DOT: Record<string, string> = {
  high: "bg-confidence-high",
  medium: "bg-confidence-medium",
  low: "bg-confidence-low",
}

function timeAgo(iso: string): string {
  const diffMs = Date.now() - new Date(iso).getTime()
  const mins = Math.floor(diffMs / 60000)
  if (mins < 1) return "just now"
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  return `${Math.floor(hrs / 24)}d ago`
}

export function HistoryRail({
  items,
  activeId,
  onSelect,
}: {
  items: QueryTrace[]
  activeId: string | null
  onSelect: (trace: QueryTrace) => void
}) {
  if (items.length === 0) return null

  return (
    <div className="rounded-xl border border-border-soft bg-surface p-4">
      <div className="mb-2.5 flex items-center gap-2 text-[13px] font-medium text-ink-muted">
        <History size={14} />
        Recent queries
      </div>
      <div className="max-h-64 space-y-1 overflow-y-auto">
        {items.map((t) => (
          <button
            key={t.query_id}
            onClick={() => onSelect(t)}
            className={`flex w-full items-start gap-2.5 rounded-lg px-2.5 py-2 text-left transition-colors ${
              activeId === t.query_id ? "bg-surface-hover" : "hover:bg-surface-hover/60"
            }`}
          >
            <span className={`mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full ${DOT[t.confidence.band] ?? "bg-ink-faint"}`} />
            <span className="min-w-0 flex-1">
              <span className="block truncate text-[12.5px] text-ink-muted">{t.question}</span>
              <span className="block text-[11px] text-ink-faint">{timeAgo(t.created_at)}</span>
            </span>
          </button>
        ))}
      </div>
    </div>
  )
}
