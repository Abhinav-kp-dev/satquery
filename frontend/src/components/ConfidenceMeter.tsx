import type { ConfidenceBreakdown } from "../types"

const BAND: Record<string, { text: string; bar: string }> = {
  high: { text: "text-confidence-high", bar: "bg-confidence-high" },
  medium: { text: "text-confidence-medium", bar: "bg-confidence-medium" },
  low: { text: "text-confidence-low", bar: "bg-confidence-low" },
}

const NAMES: Record<string, string> = {
  adjudication: "Adjudication",
  measurement_separability: "Measurement separability",
  cross_sensor_agreement: "Cross-sensor agreement",
  data_quality: "Data quality",
}

export function ConfidenceMeter({ confidence, compact }: { confidence: ConfidenceBreakdown; compact?: boolean }) {
  const s = BAND[confidence.band] ?? BAND.medium
  const comps = Object.entries(confidence.components)
  return (
    <div>
      <div className="flex items-center justify-between gap-3">
        <span className="text-[12px] text-ink-faint">Confidence</span>
        <span className={`text-[12.5px] font-medium capitalize ${s.text}`}>
          {confidence.band} <span className="tabular-nums text-ink-faint">{confidence.overall.toFixed(2)}</span>
        </span>
      </div>
      <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-surface-raised">
        <div className={`h-full rounded-full ${s.bar}`} style={{ width: `${Math.max(confidence.overall * 100, 2)}%` }} />
      </div>
      {comps.length > 0 && (
        <div className={`mt-2.5 grid gap-x-4 gap-y-1.5 ${compact ? "grid-cols-2" : "grid-cols-1"}`}>
          {comps.map(([k, v]) => (
            <div key={k} title={confidence.component_notes[k]}>
              <div className="flex justify-between text-[11px]">
                <span className="text-ink-faint">{NAMES[k] ?? k}</span>
                <span className="tabular-nums text-ink-muted">{v.toFixed(2)}</span>
              </div>
              <div className="mt-0.5 h-1 overflow-hidden rounded-full bg-surface-raised">
                <div className="h-full rounded-full bg-ink-faint" style={{ width: `${v * 100}%` }} />
              </div>
              {!compact && confidence.component_notes[k] && (
                <div className="mt-0.5 text-[10.5px] text-ink-faint">{confidence.component_notes[k]}</div>
              )}
            </div>
          ))}
        </div>
      )}
      {!compact && <p className="mt-2 text-[11px] leading-snug text-ink-faint">{confidence.basis}</p>}
    </div>
  )
}
