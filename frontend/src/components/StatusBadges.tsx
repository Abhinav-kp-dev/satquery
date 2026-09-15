import { GitBranch, Layers, ScanSearch } from "lucide-react"
import type { InputConfig, SufficiencyVerdict, Task } from "../types"

const CONFIG_LABEL: Record<InputConfig, string> = {
  single_image: "Single image",
  cross_modal_pair: "Cross-modal pair",
  bi_temporal_pair: "Bi-temporal pair",
  invalid: "Invalid input",
}

const TASK_LABEL: Record<Task, string> = {
  vqa: "Visual QA",
  caption: "Captioning",
  grounding: "Grounding",
  change: "Change analysis",
  fusion: "Sensor fusion",
}

const SUFFICIENCY_STYLE: Record<SufficiencyVerdict, { label: string; className: string }> = {
  proceed: { label: "Evidence sufficient", className: "text-confidence-high bg-confidence-high/10" },
  scope_down: { label: "Scoped to evidence", className: "text-accent-amber bg-accent-amber/10" },
  request_input: { label: "Input required", className: "text-confidence-low bg-confidence-low/10" },
}

function Chip({ icon, children }: { icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-md border border-border-soft bg-surface-raised px-2.5 py-1 text-[12px] text-ink-muted">
      {icon}
      {children}
    </span>
  )
}

export function StatusBadges({
  config,
  task,
  sufficiency,
}: {
  config: InputConfig
  task: Task | null
  sufficiency: SufficiencyVerdict
}) {
  const suff = SUFFICIENCY_STYLE[sufficiency]
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Chip icon={<Layers size={12} />}>{CONFIG_LABEL[config]}</Chip>
      {task && <Chip icon={<ScanSearch size={12} />}>{TASK_LABEL[task]}</Chip>}
      <span
        className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[12px] font-medium ${suff.className}`}
      >
        <GitBranch size={12} />
        {suff.label}
      </span>
    </div>
  )
}
