import { AlertTriangle, ArrowRight, Crosshair, Layers, Loader2, Plus, Radar, Satellite } from "lucide-react"
import { useState } from "react"
import { CONFIG_LABEL, TASK_LABEL, imageLabel } from "../lib/format"
import type { SampleInfo, SessionInfo } from "../types"
import { UploadZone } from "./UploadZone"

function Card({ title, action, children }: { title: string; action?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-border-soft bg-surface p-4">
      <div className="mb-3 flex items-center justify-between">
        <span className="text-[13px] font-medium text-ink-muted">{title}</span>
        {action}
      </div>
      {children}
    </div>
  )
}

export function SessionPicker({
  samples,
  busy,
  error,
  onSample,
  onUpload,
}: {
  samples: SampleInfo[]
  busy: string | null
  error: string | null
  onSample: (id: string) => void
  onUpload: (a: File, b: File | null) => void
}) {
  const [a, setA] = useState<File | null>(null)
  const [b, setB] = useState<File | null>(null)

  return (
    <div className="flex flex-col gap-4">
      <Card title="Try a demo scene">
        <div className="space-y-2">
          {samples.map((s) => (
            <button
              key={s.sample_id}
              disabled={busy !== null}
              onClick={() => onSample(s.sample_id)}
              className="group w-full rounded-lg border border-border-soft bg-surface-raised px-3 py-2.5 text-left transition-colors hover:border-border hover:bg-surface-hover disabled:opacity-60"
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-[13px] font-medium text-ink">{s.title}</span>
                {busy === s.sample_id ? (
                  <Loader2 size={13} className="animate-spin text-brand-400" />
                ) : (
                  <ArrowRight size={13} className="text-ink-faint transition-transform group-hover:translate-x-0.5" />
                )}
              </div>
              <div className="mt-0.5 text-[11px] text-ink-faint">{CONFIG_LABEL[s.config]}</div>
              <div className="mt-1 text-[11.5px] leading-snug text-ink-muted">{s.description}</div>
            </button>
          ))}
        </div>
        <p className="mt-3 text-[11px] leading-snug text-ink-faint">
          Synthetic, georeferenced Sentinel-2/Sentinel-1-style scenes with known ground truth.
        </p>
      </Card>

      <Card title="Or upload your own">
        <div className="space-y-3">
          <UploadZone label="Image" hint="GeoTIFF / TIFF, any size" file={a} onChange={setA} />
          <UploadZone label="Second image" hint="Same place: another date, or SAR/optical" file={b} onChange={setB} optional />
        </div>
        <button
          disabled={!a || busy !== null}
          onClick={() => a && onUpload(a, b)}
          className="mt-3 flex w-full items-center justify-center gap-2 rounded-lg bg-brand-500 px-4 py-2.5 text-[13px] font-medium text-white transition-colors hover:bg-brand-600 disabled:cursor-not-allowed disabled:bg-surface-raised disabled:text-ink-faint"
        >
          {busy === "upload" ? <Loader2 size={14} className="animate-spin" /> : <Layers size={14} />}
          Run preflight and start session
        </button>
      </Card>

      {error && (
        <div className="rounded-lg border border-confidence-low/30 bg-confidence-low/[0.06] px-3.5 py-3 text-[12.5px] text-confidence-low">
          {error}
        </div>
      )}
    </div>
  )
}

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-0.5 text-[11.5px]">
      <span className="text-ink-faint">{k}</span>
      <span className="truncate text-right text-ink-muted">{v}</span>
    </div>
  )
}

export function ManifestCard({ session, onNew }: { session: SessionInfo; onNew: () => void }) {
  const m = session.manifest
  return (
    <Card
      title="Session"
      action={
        <button
          onClick={onNew}
          className="inline-flex items-center gap-1 rounded-md border border-border-soft px-2 py-1 text-[11.5px] text-ink-muted hover:border-border hover:text-ink"
        >
          <Plus size={12} /> New
        </button>
      }
    >
      <div className="mb-3 flex items-center gap-2">
        <span className="rounded-md bg-brand-500/10 px-2 py-0.5 text-[12px] font-medium text-brand-400">{CONFIG_LABEL[m.config]}</span>
        <span className="font-mono text-[10.5px] text-ink-faint">{session.session_id}</span>
      </div>

      <div className={`grid gap-2 ${session.render_urls.length > 1 ? "grid-cols-2" : "grid-cols-1"}`}>
        {session.render_urls.map((u, i) => (
          <figure key={u}>
            <img src={u} alt={imageLabel(session, i)} className="aspect-square w-full rounded-md border border-border-soft object-cover" />
            <figcaption className="mt-1 truncate text-[10.5px] text-ink-faint">{imageLabel(session, i)}</figcaption>
          </figure>
        ))}
      </div>

      <div className="mt-3 space-y-3">
        {m.images.map((img, i) => (
          <div key={i} className="rounded-lg border border-border-soft bg-surface-raised px-3 py-2">
            <div className="mb-1 flex items-center gap-1.5 text-[12px] font-medium text-ink">
              {img.modality === "sar" ? <Radar size={12} className="text-accent-violet" /> : <Satellite size={12} className="text-accent-teal" />}
              <span className="truncate">{img.filename}</span>
            </div>
            <Row k="Sensor" v={`${img.modality} · ${img.band_count} bands`} />
            <Row k="Layout" v={img.band_layout.replaceAll("_", " ")} />
            <Row k="Size" v={`${img.width.toLocaleString()} x ${img.height.toLocaleString()} px`} />
            <Row
              k="GSD"
              v={
                img.gsd_m
                  ? img.decimation > 1
                    ? `${img.gsd_m.toFixed(1)} m (analysed at ${img.analysis_gsd_m?.toFixed(1)} m)`
                    : `${img.gsd_m.toFixed(1)} m`
                  : "unknown"
              }
            />
            <Row k="CRS" v={img.crs ?? "none"} />
            {img.acquisition_date && <Row k="Acquired" v={img.acquisition_date} />}
            {img.bounds_lonlat && (
              <Row
                k="Centre"
                v={
                  <span className="inline-flex items-center gap-1">
                    <Crosshair size={10} />
                    {((img.bounds_lonlat[1] + img.bounds_lonlat[3]) / 2).toFixed(4)}°N,{" "}
                    {((img.bounds_lonlat[0] + img.bounds_lonlat[2]) / 2).toFixed(4)}°E
                  </span>
                }
              />
            )}
            {img.warnings.map((w) => (
              <div key={w} className="mt-1 flex gap-1.5 text-[11px] leading-snug text-accent-amber">
                <AlertTriangle size={11} className="mt-0.5 shrink-0" />
                {w}
              </div>
            ))}
          </div>
        ))}
      </div>

      {m.alignment && (
        <div className="mt-3 rounded-lg border border-border-soft px-3 py-2">
          <div className="mb-1 text-[11.5px] font-medium text-ink-muted">Pair alignment</div>
          <Row k="Method" v={m.alignment.method.replaceAll("_", " ")} />
          <Row k="Overlap" v={`${(m.alignment.overlap_fraction * 100).toFixed(0)}%`} />
          <Row
            k="Residual shift"
            v={`${m.alignment.residual_shift_px.map((v) => v.toFixed(1)).join(", ")} px${m.alignment.shift_corrected ? " · corrected" : ""}`}
          />
          {m.alignment.notes.map((n) => (
            <p key={n} className="mt-1 text-[11px] leading-snug text-ink-faint">
              {n}
            </p>
          ))}
        </div>
      )}

      <div className="mt-3">
        <div className="mb-1 text-[11px] text-ink-faint">Questions this input can answer</div>
        <div className="flex flex-wrap gap-1">
          {m.legal_tasks.map((t) => (
            <span key={t} className="rounded border border-border-soft px-1.5 py-0.5 text-[11px] text-ink-muted">
              {TASK_LABEL[t]}
            </span>
          ))}
        </div>
      </div>
    </Card>
  )
}
