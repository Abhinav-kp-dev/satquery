import { Loader2, Send } from "lucide-react"
import { useEffect, useState } from "react"
import { UploadZone } from "./UploadZone"

const EXAMPLE_PROMPTS = [
  "Describe the land cover in this image.",
  "Is this area flooded?",
  "Highlight the water body in this image.",
  "What changed between these two dates, and where?",
  "Use both the optical and SAR images to find water and built-up regions.",
]

const STATUS_STAGES = [
  "Validating input manifest",
  "Checking evidence sufficiency",
  "Running deterministic tools",
  "Retrieving reference examples",
  "Generating narration",
  "Adjudicating claims",
]

interface Props {
  onSubmit: (question: string, imageA: File, imageB: File | null) => void
  loading: boolean
  error: string | null
}

export function QueryPanel({ onSubmit, loading, error }: Props) {
  const [imageA, setImageA] = useState<File | null>(null)
  const [imageB, setImageB] = useState<File | null>(null)
  const [question, setQuestion] = useState("")
  const [stageIdx, setStageIdx] = useState(0)

  useEffect(() => {
    if (!loading) {
      setStageIdx(0)
      return
    }
    const id = setInterval(() => setStageIdx((i) => Math.min(i + 1, STATUS_STAGES.length - 1)), 550)
    return () => clearInterval(id)
  }, [loading])

  const canSubmit = Boolean(imageA && question.trim()) && !loading

  return (
    <div className="flex flex-col gap-5">
      <div className="rounded-xl border border-border-soft bg-surface p-4">
        <div className="mb-3 text-[13px] font-medium text-ink-muted">Imagery</div>
        <div className="space-y-3">
          <UploadZone label="Primary image" hint="GeoTIFF / TIFF" file={imageA} onChange={setImageA} />
          <UploadZone
            label="Second image"
            hint="Cross-modal pair or bi-temporal pair"
            file={imageB}
            onChange={setImageB}
            optional
          />
        </div>
      </div>

      <div className="rounded-xl border border-border-soft bg-surface p-4">
        <div className="mb-3 text-[13px] font-medium text-ink-muted">Question</div>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask about the uploaded imagery in plain language..."
          rows={3}
          className="w-full resize-none rounded-lg border border-border-soft bg-surface-raised px-3 py-2.5 text-[13.5px] text-ink placeholder:text-ink-faint focus:border-brand-500/50 focus:outline-none"
        />
        <div className="mt-2.5 flex flex-wrap gap-1.5">
          {EXAMPLE_PROMPTS.map((p) => (
            <button
              key={p}
              type="button"
              onClick={() => setQuestion(p)}
              className="rounded-full border border-border-soft px-2.5 py-1 text-[11px] text-ink-faint transition-colors hover:border-border hover:text-ink-muted"
            >
              {p}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-confidence-low/30 bg-confidence-low/[0.06] px-3.5 py-3 text-[12.5px] text-confidence-low">
          {error}
        </div>
      )}

      <button
        disabled={!canSubmit}
        onClick={() => imageA && onSubmit(question.trim(), imageA, imageB)}
        className="flex items-center justify-center gap-2 rounded-lg bg-brand-500 px-4 py-3 text-[13.5px] font-medium text-white transition-colors hover:bg-brand-600 disabled:cursor-not-allowed disabled:bg-surface-raised disabled:text-ink-faint"
      >
        {loading ? (
          <>
            <Loader2 size={15} className="animate-spin" />
            {STATUS_STAGES[stageIdx]}...
          </>
        ) : (
          <>
            <Send size={14} />
            Run analysis
          </>
        )}
      </button>
    </div>
  )
}
