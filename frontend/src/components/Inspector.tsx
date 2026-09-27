import {
  CheckCircle2,
  ChevronDown,
  CircleDashed,
  CircleSlash,
  Database,
  Eye,
  Layers,
  Link2,
  Loader2,
  MapPinned,
  Recycle,
  ShieldCheck,
  ShieldX,
  TriangleAlert,
  Workflow,
} from "lucide-react"
import { useEffect, useMemo, useRef, useState } from "react"
import { api } from "../api/client"
import { LAYER_LABEL, baseIndex, fmtValue, imageLabel, rgb } from "../lib/format"
import type { LedgerVerification, QueryTrace, SessionInfo, StageRecord } from "../types"
import { ConfidenceMeter } from "./ConfidenceMeter"

export type InspectorTab = "evidence" | "pipeline" | "ledger"

// ------------------------------------------------------------------ evidence

function EvidenceView({ trace, session }: { trace: QueryTrace; session: SessionInfo }) {
  const pair = session.render_urls.length > 1
  const bitemporal = session.manifest.config === "bi_temporal_pair"
  const base = baseIndex(session)
  const [view, setView] = useState<string>("overlay")
  const [hidden, setHidden] = useState<Record<string, boolean>>({})
  const [showBoxes, setShowBoxes] = useState(true)
  const [hover, setHover] = useState<string | null>(null)
  const [split, setSplit] = useState(50)

  if (trace.sufficiency === "request_input") {
    return (
      <div className="flex flex-col items-center gap-2 py-12 text-center text-[12.5px] text-ink-faint">
        <CircleSlash size={20} />
        No measurements were taken for this question: the required input was missing.
      </div>
    )
  }

  const views = [
    { id: "overlay", label: "Evidence" },
    ...session.render_urls.map((_, i) => ({ id: `img${i}`, label: imageLabel(session, i).split(" · ")[0] })),
    ...(bitemporal ? [{ id: "compare", label: "Compare" }] : []),
  ]

  return (
    <div>
      <div className="mb-3 flex flex-wrap gap-1">
        {views.map((v) => (
          <button
            key={v.id}
            onClick={() => setView(v.id)}
            className={`rounded-md px-2.5 py-1 text-[12px] transition-colors ${
              view === v.id ? "bg-surface-hover text-ink" : "text-ink-faint hover:text-ink-muted"
            }`}
          >
            {v.label}
          </button>
        ))}
      </div>

      <div className="relative overflow-hidden rounded-lg border border-border-soft bg-black">
        {view === "overlay" && (
          <>
            <img src={session.render_urls[base]} alt="Base composite" className="block w-full" />
            {trace.layers.map((l) =>
              hidden[l.name] ? null : <img key={l.name} src={l.url} alt={l.name} className="absolute inset-0 h-full w-full" />,
            )}
            {showBoxes && trace.regions.length > 0 && (
              <svg viewBox="0 0 1 1" preserveAspectRatio="none" className="absolute inset-0 h-full w-full">
                {trace.regions.map((r) => {
                  const [x0, y0, x1, y1] = r.bbox_norm
                  const on = hover === r.region_id
                  return (
                    <rect
                      key={r.region_id}
                      x={x0}
                      y={y0}
                      width={x1 - x0}
                      height={y1 - y0}
                      fill={on ? "rgba(255,230,0,0.12)" : "none"}
                      stroke={on ? "#ffe600" : "rgba(255,230,0,0.8)"}
                      strokeWidth={on ? 2.5 : 1.5}
                      vectorEffect="non-scaling-stroke"
                      onMouseEnter={() => setHover(r.region_id)}
                      onMouseLeave={() => setHover(null)}
                    />
                  )
                })}
              </svg>
            )}
          </>
        )}
        {view.startsWith("img") && <img src={session.render_urls[Number(view.slice(3))]} alt={view} className="block w-full" />}
        {view === "compare" && (
          <div className="relative select-none">
            <img src={session.render_urls[1]} alt="After" className="block w-full" />
            <img
              src={session.render_urls[0]}
              alt="Before"
              className="absolute inset-0 h-full w-full"
              style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }}
            />
            <div className="absolute inset-y-0 w-0.5 bg-white/80" style={{ left: `${split}%` }} />
            <span className="absolute left-2 top-2 rounded bg-black/60 px-1.5 py-0.5 text-[10.5px] text-white">before</span>
            <span className="absolute right-2 top-2 rounded bg-black/60 px-1.5 py-0.5 text-[10.5px] text-white">after</span>
            <input
              type="range"
              min={0}
              max={100}
              value={split}
              onChange={(e) => setSplit(Number(e.target.value))}
              aria-label="Before/after split"
              className="absolute inset-x-3 bottom-3 accent-white"
            />
          </div>
        )}
      </div>

      {view === "overlay" && (
        <div className="mt-3 flex flex-wrap items-center gap-1.5">
          {trace.layers.map((l) => (
            <button
              key={l.name}
              onClick={() => setHidden((h) => ({ ...h, [l.name]: !h[l.name] }))}
              className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11.5px] transition-opacity ${
                hidden[l.name] ? "border-border-soft opacity-45" : "border-border"
              }`}
            >
              <span className="h-2.5 w-2.5 rounded-sm" style={{ background: rgb(l.color) }} />
              <span className="text-ink-muted">{LAYER_LABEL[l.name] ?? l.name}</span>
              {l.fraction !== null && <span className="tabular-nums text-ink-faint">{(l.fraction * 100).toFixed(1)}%</span>}
            </button>
          ))}
          {trace.regions.length > 0 && (
            <button
              onClick={() => setShowBoxes((v) => !v)}
              className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11.5px] ${showBoxes ? "border-border" : "border-border-soft opacity-45"}`}
            >
              <span className="h-2.5 w-2.5 rounded-sm border border-[#ffe600]" />
              <span className="text-ink-muted">Regions</span>
            </button>
          )}
        </div>
      )}
      <p className="mt-2 text-[11px] text-ink-faint">
        Layers are the masks the deterministic tools measured. Nothing here is drawn by a language model.
        {pair && !bitemporal && " Base image: the optical composite."}
      </p>

      {trace.regions.length > 0 && (
        <div className="mt-4">
          <div className="mb-1.5 flex items-center gap-1.5 text-[12px] font-medium text-ink-muted">
            <MapPinned size={12} /> Grounded regions
          </div>
          <div className="overflow-hidden rounded-lg border border-border-soft">
            {trace.regions.map((r) => (
              <div
                key={r.region_id}
                onMouseEnter={() => setHover(r.region_id)}
                onMouseLeave={() => setHover(null)}
                className={`grid grid-cols-[1fr_auto] gap-x-3 border-b border-border-soft px-3 py-1.5 text-[11.5px] last:border-b-0 ${
                  hover === r.region_id ? "bg-surface-hover" : ""
                }`}
              >
                <span className="text-ink-muted">{r.region_id.replaceAll("_", " ")}</span>
                <span className="tabular-nums text-ink">{r.area_ha.toLocaleString(undefined, { maximumFractionDigits: 1 })} ha</span>
                <span className="font-mono text-[10.5px] text-ink-faint">
                  {r.centroid_lonlat ? `${r.centroid_lonlat[1].toFixed(5)}°N ${r.centroid_lonlat[0].toFixed(5)}°E` : `px ${r.centroid_px.join(", ")}`}
                </span>
                <span className="text-[10.5px] text-ink-faint">{r.ledger_entry_id}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="mt-4 rounded-lg border border-border-soft p-3">
        <ConfidenceMeter confidence={trace.confidence} />
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ pipeline

const STAGE_ICON: Record<StageRecord["status"], React.ReactNode> = {
  ok: <CheckCircle2 size={14} className="text-confidence-high" />,
  warn: <TriangleAlert size={14} className="text-accent-amber" />,
  declined: <CircleSlash size={14} className="text-confidence-low" />,
  skipped: <CircleDashed size={14} className="text-ink-faint" />,
}

function Collapsible({ title, children, open: initial }: { title: React.ReactNode; children: React.ReactNode; open?: boolean }) {
  const [open, setOpen] = useState(Boolean(initial))
  return (
    <div className="rounded-lg border border-border-soft">
      <button onClick={() => setOpen((v) => !v)} className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left">
        <span className="min-w-0 text-[12px] text-ink-muted">{title}</span>
        <ChevronDown size={13} className={`shrink-0 text-ink-faint transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && <div className="border-t border-border-soft px-3 py-2">{children}</div>}
    </div>
  )
}

function PipelineView({ trace, onEntry }: { trace: QueryTrace; onEntry: (id: string) => void }) {
  const depth = useMemo(() => {
    const d: Record<string, number> = {}
    for (const n of trace.plan_nodes) d[n.node_id] = n.depends_on.length ? Math.max(...n.depends_on.map((x) => d[x] ?? 0)) + 1 : 0
    return d
  }, [trace])
  const waves = useMemo(() => {
    const w: string[][] = []
    for (const n of trace.plan_nodes) (w[depth[n.node_id]] ??= []).push(n.node_id)
    return w
  }, [trace, depth])

  return (
    <div className="space-y-4">
      <ol className="relative space-y-0">
        {trace.stages.map((s, i) => (
          <li key={s.name} className="relative flex gap-3 pb-3">
            {i < trace.stages.length - 1 && <span className="absolute left-[6.5px] top-5 h-full w-px bg-border-soft" />}
            <span className="relative z-10 mt-0.5 bg-surface">{STAGE_ICON[s.status]}</span>
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-[12.5px] font-medium capitalize text-ink">
                  {i + 1}. {s.name}
                </span>
                {s.status !== "skipped" && <span className="text-[10.5px] tabular-nums text-ink-faint">{s.runtime_ms.toFixed(1)} ms</span>}
              </div>
              <div className="text-[11.5px] leading-snug text-ink-faint">{s.summary}</div>
            </div>
          </li>
        ))}
      </ol>

      {trace.spec && (
        <Collapsible title={<>Parsed query · <span className="font-mono">{trace.spec.routing_layer}</span></>}>
          <div className="flex flex-wrap gap-1">
            {trace.spec.matched_rules.map((r) => (
              <span key={r} className="rounded bg-surface-raised px-1.5 py-0.5 font-mono text-[10.5px] text-ink-muted">
                {r}
              </span>
            ))}
          </div>
          <p className="mt-2 text-[11.5px] text-ink-faint">{trace.sufficiency_reason}</p>
        </Collapsible>
      )}

      {waves.length > 0 && (
        <div>
          <div className="mb-1.5 text-[12px] font-medium text-ink-muted">Tool plan (each row runs in parallel)</div>
          <div className="space-y-1.5">
            {waves.map((w, i) => (
              <div key={i} className="flex flex-wrap items-center gap-1.5">
                <span className="w-12 text-[10.5px] text-ink-faint">wave {i + 1}</span>
                {w.map((id) => (
                  <span key={id} className="rounded-md border border-border bg-surface-raised px-2 py-0.5 font-mono text-[11px] text-brand-400">
                    {id}
                  </span>
                ))}
              </div>
            ))}
          </div>
        </div>
      )}

      {trace.tool_calls.length > 0 && (
        <div className="space-y-2">
          <div className="text-[12px] font-medium text-ink-muted">Tool calls</div>
          {trace.tool_calls.map((c) => (
            <Collapsible
              key={c.node_id}
              title={
                <span className="flex items-center gap-2">
                  <span className="font-mono text-brand-400">{c.node_id}</span>
                  {c.reused_from ? (
                    <span className="inline-flex items-center gap-1 text-[10.5px] text-accent-violet">
                      <Recycle size={10} /> session memory
                    </span>
                  ) : (
                    <span className="text-[10.5px] tabular-nums text-ink-faint">{c.runtime_ms.toFixed(1)} ms</span>
                  )}
                </span>
              }
            >
              <div className="mb-1.5 text-[11px] text-ink-faint">{c.impl}</div>
              {c.reused_from && <div className="mb-1.5 text-[11px] text-accent-violet">reused from {c.reused_from}</div>}
              <div className="space-y-0.5">
                {Object.entries(c.output).map(([k, v]) => (
                  <div key={k} className="flex justify-between gap-3 text-[11.5px]">
                    <span className="text-ink-faint">{k}</span>
                    <span className="truncate text-right tabular-nums text-ink-muted">{fmtValue(v)}</span>
                  </div>
                ))}
              </div>
              {Object.keys(c.params).length > 0 && (
                <div className="mt-2 border-t border-border-soft pt-1.5 font-mono text-[10.5px] text-ink-faint">
                  {Object.entries(c.params).map(([k, v]) => `${k}=${fmtValue(v)}`).join("  ")}
                </div>
              )}
              <div className="mt-2 flex flex-wrap gap-1">
                {c.ledger_entries.map((e) => (
                  <button key={e} onClick={() => onEntry(e)} className="rounded bg-surface-raised px-1.5 py-0.5 font-mono text-[10px] text-ink-faint hover:text-ink">
                    {e}
                  </button>
                ))}
              </div>
            </Collapsible>
          ))}
        </div>
      )}

      {trace.answer_raw && (
        <Collapsible title="Narrator output before adjudication">
          <pre className="whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed text-ink-muted">{trace.answer_raw}</pre>
          {trace.prompt_examples_used.length > 0 && (
            <div className="mt-2 text-[10.5px] text-ink-faint">In-context examples: {trace.prompt_examples_used.join(" · ")}</div>
          )}
        </Collapsible>
      )}
    </div>
  )
}

// ------------------------------------------------------------------ ledger

function LedgerView({ trace, highlight }: { trace: QueryTrace; highlight: string | null }) {
  const [stage, setStage] = useState<string>("all")
  const [check, setCheck] = useState<LedgerVerification | null>(null)
  const [checking, setChecking] = useState(false)
  const rowRefs = useRef<Record<string, HTMLTableRowElement | null>>({})
  const stages = ["all", ...Array.from(new Set(trace.ledger.map((e) => e.stage)))]

  useEffect(() => {
    if (highlight) {
      requestAnimationFrame(() => rowRefs.current[highlight]?.scrollIntoView({ block: "center", behavior: "smooth" }))
    }
  }, [highlight])

  const rows = trace.ledger.filter((e) => stage === "all" || e.stage === stage || e.entry_id === highlight)
  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap gap-1">
          {stages.map((s) => (
            <button
              key={s}
              onClick={() => setStage(s)}
              className={`rounded-md px-2 py-0.5 text-[11.5px] capitalize ${stage === s ? "bg-surface-hover text-ink" : "text-ink-faint hover:text-ink-muted"}`}
            >
              {s}
            </button>
          ))}
        </div>
        <button
          onClick={async () => {
            setChecking(true)
            try {
              setCheck(await api.verifyLedger())
            } finally {
              setChecking(false)
            }
          }}
          className="inline-flex items-center gap-1.5 rounded-md border border-border-soft px-2 py-1 text-[11.5px] text-ink-muted hover:border-border hover:text-ink"
        >
          {checking ? <Loader2 size={12} className="animate-spin" /> : <ShieldCheck size={12} />}
          Verify hash chain
        </button>
      </div>
      {check && (
        <div
          className={`mb-3 flex items-center gap-2 rounded-lg px-3 py-2 text-[12px] ${
            check.valid ? "bg-confidence-high/10 text-confidence-high" : "bg-confidence-low/10 text-confidence-low"
          }`}
        >
          {check.valid ? <ShieldCheck size={14} /> : <ShieldX size={14} />}
          {check.valid
            ? `All ${check.entries} entries verified; every hash links to the one before it.`
            : `Chain broken at entry #${check.first_bad_seq}: stored data was modified after it was written.`}
        </div>
      )}
      <div className="max-h-[560px] overflow-auto rounded-lg border border-border-soft">
        <table className="w-full text-left text-[11px]">
          <thead className="sticky top-0 bg-surface-raised text-ink-faint">
            <tr>
              <th className="px-2 py-1.5 font-medium">id</th>
              <th className="px-2 py-1.5 font-medium">stage · kind</th>
              <th className="px-2 py-1.5 font-medium">key</th>
              <th className="px-2 py-1.5 font-medium">value</th>
              <th className="px-2 py-1.5 font-medium">hash</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((e) => (
              <tr
                key={e.entry_id}
                ref={(el) => {
                  rowRefs.current[e.entry_id] = el
                }}
                className={`border-t border-border-soft align-top ${e.entry_id === highlight ? "bg-accent-teal/10" : ""}`}
              >
                <td className="px-2 py-1 font-mono text-ink-muted">{e.entry_id}</td>
                <td className="px-2 py-1 text-ink-faint">
                  {e.stage} · {e.kind}
                  {e.reused_from && (
                    <div className="inline-flex items-center gap-0.5 text-accent-violet" title={`reused from ${e.reused_from}`}>
                      <Link2 size={9} /> memory
                    </div>
                  )}
                </td>
                <td className="max-w-[140px] break-words px-2 py-1 text-ink-muted">{e.key}</td>
                <td className="max-w-[160px] truncate px-2 py-1 tabular-nums text-ink" title={fmtValue(e.value, e.unit)}>
                  {fmtValue(e.value, e.unit)}
                </td>
                <td className="px-2 py-1 font-mono text-[10px] text-ink-faint" title={`prev ${e.prev_hash}\nthis ${e.hash}`}>
                  {e.hash.slice(0, 10)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-[11px] leading-snug text-ink-faint">
        One append-only ledger acts as agent memory (follow-ups reuse entries), grounding (the narrator cites entry
        ids), confidence substrate, and audit trail (SHA-256 chained across every query).
      </p>
    </div>
  )
}

// ------------------------------------------------------------------ shell

export function Inspector({
  trace,
  session,
  tab,
  onTab,
  highlight,
  onEntry,
}: {
  trace: QueryTrace
  session: SessionInfo
  tab: InspectorTab
  onTab: (t: InspectorTab) => void
  highlight: string | null
  onEntry: (id: string) => void
}) {
  const tabs: { id: InspectorTab; label: string; icon: React.ReactNode }[] = [
    { id: "evidence", label: "Evidence", icon: <Eye size={13} /> },
    { id: "pipeline", label: "Pipeline", icon: <Workflow size={13} /> },
    { id: "ledger", label: `Ledger (${trace.ledger.length})`, icon: <Database size={13} /> },
  ]
  return (
    <div className="rounded-xl border border-border-soft bg-surface">
      <div className="flex items-center gap-1 border-b border-border-soft px-3 pt-2">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => onTab(t.id)}
            className={`-mb-px inline-flex items-center gap-1.5 border-b-2 px-2.5 py-2 text-[12.5px] transition-colors ${
              tab === t.id ? "border-brand-500 text-ink" : "border-transparent text-ink-faint hover:text-ink-muted"
            }`}
          >
            {t.icon}
            {t.label}
          </button>
        ))}
        <span className="ml-auto hidden items-center gap-1 text-[10.5px] text-ink-faint sm:flex">
          <Layers size={11} /> {trace.query_id}
        </span>
      </div>
      <div className="p-4">
        {tab === "evidence" && <EvidenceView key={trace.query_id} trace={trace} session={session} />}
        {tab === "pipeline" && <PipelineView trace={trace} onEntry={onEntry} />}
        {tab === "ledger" && <LedgerView trace={trace} highlight={highlight} />}
      </div>
    </div>
  )
}
