import type { ConfidenceBreakdown } from "../types"

const BAND_STYLES: Record<string, { dot: string; text: string; ring: string }> = {
  high: { dot: "bg-confidence-high", text: "text-confidence-high", ring: "ring-confidence-high/25" },
  medium: { dot: "bg-confidence-medium", text: "text-confidence-medium", ring: "ring-confidence-medium/25" },
  low: { dot: "bg-confidence-low", text: "text-confidence-low", ring: "ring-confidence-low/25" },
}

export function ConfidenceBadge({ confidence }: { confidence: ConfidenceBreakdown }) {
  const style = BAND_STYLES[confidence.band] ?? BAND_STYLES.medium
  return (
    <div
      className={`inline-flex items-center gap-2 rounded-full bg-surface-raised px-3 py-1.5 ring-1 ${style.ring}`}
      title={confidence.basis}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${style.dot}`} />
      <span className={`text-[12px] font-medium capitalize ${style.text}`}>{confidence.band} confidence</span>
      <span className="text-[12px] tabular-nums text-ink-faint">{confidence.overall.toFixed(2)}</span>
    </div>
  )
}
