import { CheckCircle2, CircleAlert } from "lucide-react"
import type { ClaimRecord } from "../types"
import { ConfidenceBadge } from "./ConfidenceBadge"
import type { QueryTrace } from "../types"

export function AnswerCard({ trace }: { trace: QueryTrace }) {
  return (
    <div className="rounded-xl border border-border-soft bg-surface p-5">
      <div className="mb-3 flex items-start justify-between gap-3">
        <span className="text-[13px] font-medium text-ink-muted">Answer</span>
        <ConfidenceBadge confidence={trace.confidence} />
      </div>

      <p className="text-[15px] leading-relaxed text-ink">{trace.answer}</p>

      {trace.claims.length > 0 && (
        <div className="mt-4 flex flex-wrap gap-2 border-t border-border-soft pt-4">
          {trace.claims.map((claim, i) => (
            <ClaimPill key={i} claim={claim} />
          ))}
        </div>
      )}

      {trace.llm_override && (
        <div className="mt-3 flex items-start gap-2 rounded-md border border-accent-rose/25 bg-accent-rose/[0.06] px-3 py-2 text-[12px] text-accent-rose">
          <CircleAlert size={14} className="mt-0.5 shrink-0" />
          <span>
            One or more narrated figures did not match the measured evidence within tolerance. The measurement is
            treated as authoritative -- see the claim{trace.claims.filter((c) => !c.within_tolerance).length > 1 ? "s" : ""}{" "}
            flagged above.
          </span>
        </div>
      )}
    </div>
  )
}

function ClaimPill({ claim }: { claim: ClaimRecord }) {
  const ok = claim.within_tolerance
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-[12px] tabular-nums ${
        ok === false
          ? "border-confidence-low/30 bg-confidence-low/[0.06] text-confidence-low"
          : "border-border-soft bg-surface-raised text-ink-muted"
      }`}
      title={claim.matched_ledger_key ? `Matched ledger key: ${claim.matched_ledger_key}` : "No matching ledger value"}
    >
      {ok === false ? <CircleAlert size={12} /> : <CheckCircle2 size={12} className="text-confidence-high" />}
      {claim.value} {claim.unit}
    </span>
  )
}
