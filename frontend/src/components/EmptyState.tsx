import { GitMerge, ImagePlus, ScanEye } from "lucide-react"

const CONFIGS = [
  {
    icon: ImagePlus,
    title: "Single image",
    body: "Describe the scene, answer a question, or locate a specific region.",
  },
  {
    icon: GitMerge,
    title: "Cross-modal pair",
    body: "Combine co-registered optical and SAR imagery for higher-confidence detection.",
  },
  {
    icon: ScanEye,
    title: "Bi-temporal pair",
    body: "Compare two dates of the same location and quantify what changed.",
  },
]

export function EmptyState() {
  return (
    <div className="flex h-full flex-col items-center justify-center rounded-xl border border-dashed border-border px-8 py-16 text-center">
      <div className="mb-6 max-w-sm">
        <h2 className="text-[15px] font-medium text-ink">Upload imagery to begin</h2>
        <p className="mt-2 text-[13px] leading-relaxed text-ink-faint">
          Every answer is grounded in tool-measured evidence, not visual guesswork -- and the system will tell you
          when your question needs more input than it was given.
        </p>
      </div>
      <div className="grid w-full max-w-lg grid-cols-1 gap-3 sm:grid-cols-3">
        {CONFIGS.map(({ icon: Icon, title, body }) => (
          <div key={title} className="rounded-lg border border-border-soft bg-surface p-3.5 text-left">
            <Icon size={16} className="mb-2 text-brand-400" />
            <div className="text-[12.5px] font-medium text-ink-muted">{title}</div>
            <div className="mt-1 text-[11.5px] leading-snug text-ink-faint">{body}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
