import { Layers } from "lucide-react"
import { useState } from "react"
import type { InputConfig } from "../types"

const LABELS: Record<InputConfig, string[]> = {
  single_image: ["Scene"],
  cross_modal_pair: ["Image A", "Image B"],
  bi_temporal_pair: ["T1 (earlier)", "T2 (later)"],
  invalid: [],
}

export function ImageViewer({
  renderUrls,
  overlayUrl,
  config,
}: {
  renderUrls: string[]
  overlayUrl: string | null
  config: InputConfig
}) {
  const [showOverlay, setShowOverlay] = useState(Boolean(overlayUrl))
  const [overlayFailed, setOverlayFailed] = useState(false)
  const labels = LABELS[config] ?? []

  if (renderUrls.length === 0) return null
  const overlayAvailable = Boolean(overlayUrl) && !overlayFailed

  return (
    <div className="rounded-xl border border-border-soft bg-surface p-4">
      <div className="mb-3 flex items-center justify-between">
        <span className="text-[13px] font-medium text-ink-muted">Imagery</span>
        {overlayAvailable && (
          <button
            onClick={() => setShowOverlay((v) => !v)}
            className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-[12px] transition-colors ${
              showOverlay
                ? "border-brand-500/40 bg-brand-500/10 text-brand-400"
                : "border-border-soft text-ink-faint hover:text-ink-muted"
            }`}
          >
            <Layers size={12} />
            Evidence overlay
          </button>
        )}
      </div>

      {showOverlay && overlayAvailable ? (
        <figure>
          <img
            src={overlayUrl!}
            alt="Evidence overlay"
            className="w-full rounded-lg border border-border-soft"
            onError={() => setOverlayFailed(true)}
          />
          <figcaption className="mt-2 text-[11px] text-ink-faint">
            Tool-derived masks rendered over the base composite -- not a model prediction.
          </figcaption>
        </figure>
      ) : (
        <div className={`grid gap-3 ${renderUrls.length > 1 ? "grid-cols-2" : "grid-cols-1"}`}>
          {renderUrls.map((url, i) => (
            <figure key={url}>
              <img src={url} alt={labels[i] ?? `Render ${i}`} className="w-full rounded-lg border border-border-soft" />
              <figcaption className="mt-1.5 text-[11px] text-ink-faint">{labels[i] ?? `Render ${i}`}</figcaption>
            </figure>
          ))}
        </div>
      )}
    </div>
  )
}
