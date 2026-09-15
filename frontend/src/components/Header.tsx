import { Satellite } from "lucide-react"

export function Header({ vlmProvider }: { vlmProvider: string | null }) {
  return (
    <header className="flex items-center justify-between border-b border-border-soft px-6 py-4 lg:px-8">
      <div className="flex items-center gap-3">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-brand-400 to-brand-600 text-white">
          <Satellite size={16} strokeWidth={2.25} />
        </div>
        <div>
          <div className="text-[14.5px] font-semibold leading-none tracking-tight text-ink">SatQuery AI</div>
          <div className="mt-1 text-[11px] leading-none text-ink-faint">Evidence-grounded remote sensing analysis</div>
        </div>
      </div>

      {vlmProvider && (
        <div className="hidden items-center gap-2 rounded-full border border-border-soft bg-surface-raised px-3 py-1.5 sm:flex">
          <span className="h-1.5 w-1.5 rounded-full bg-confidence-high" />
          <span className="text-[11.5px] text-ink-faint">
            narration: <span className="text-ink-muted">{vlmProvider}</span>
          </span>
        </div>
      )}
    </header>
  )
}
