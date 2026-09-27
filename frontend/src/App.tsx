import { MessageSquareText } from "lucide-react"
import { useCallback, useEffect, useMemo, useState } from "react"
import { ApiRequestError, api } from "./api/client"
import { Composer, Conversation } from "./components/Conversation"
import { Header } from "./components/Header"
import { Inspector, type InspectorTab } from "./components/Inspector"
import { ManifestCard, SessionPicker } from "./components/SessionPanel"
import type { Health, QueryTrace, SampleInfo, SessionInfo } from "./types"

const errText = (e: unknown) => (e instanceof ApiRequestError ? e.message : "Could not reach the backend.")

function readSessionParam(): string | null {
  return new URLSearchParams(window.location.search).get("session")
}

function writeSessionParam(id: string | null) {
  const url = new URL(window.location.href)
  if (id) url.searchParams.set("session", id)
  else url.searchParams.delete("session")
  window.history.replaceState(null, "", url)
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [samples, setSamples] = useState<SampleInfo[]>([])
  const [session, setSession] = useState<SessionInfo | null>(null)
  const [turns, setTurns] = useState<QueryTrace[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [tab, setTab] = useState<InspectorTab>("evidence")
  const [highlight, setHighlight] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [pending, setPending] = useState<string | null>(null)
  const [sessionError, setSessionError] = useState<string | null>(null)
  const [askError, setAskError] = useState<string | null>(null)

  useEffect(() => {
    api.health().then(setHealth).catch(() => {})
    api.samples().then(setSamples).catch((e) => setSessionError(errText(e)))
    const id = readSessionParam()
    if (id)
      api
        .session(id)
        .then(({ session, traces }) => {
          setSession(session)
          setTurns(traces)
          setActiveId(traces.at(-1)?.query_id ?? null)
        })
        .catch(() => writeSessionParam(null))
  }, [])

  const openSession = useCallback((s: SessionInfo) => {
    setSession(s)
    setTurns([])
    setActiveId(null)
    setAskError(null)
    setTab("evidence")
    writeSessionParam(s.session_id)
  }, [])

  async function startSample(id: string) {
    setBusy(id)
    setSessionError(null)
    try {
      openSession(await api.sampleSession(id))
    } catch (e) {
      setSessionError(errText(e))
    } finally {
      setBusy(null)
    }
  }

  async function startUpload(a: File, b: File | null) {
    setBusy("upload")
    setSessionError(null)
    try {
      openSession(await api.uploadSession(a, b))
    } catch (e) {
      setSessionError(errText(e))
    } finally {
      setBusy(null)
    }
  }

  async function ask(q: string, stress: boolean) {
    if (!session) return
    setPending(q)
    setAskError(null)
    try {
      const t = await api.ask(session.session_id, q, stress)
      setTurns((ts) => [...ts, t])
      setActiveId(t.query_id)
      setHighlight(null)
      setTab("evidence")
    } catch (e) {
      setAskError(errText(e))
    } finally {
      setPending(null)
    }
  }

  function newSession() {
    setSession(null)
    setTurns([])
    setActiveId(null)
    writeSessionParam(null)
  }

  const active = useMemo(() => turns.find((t) => t.query_id === activeId) ?? null, [turns, activeId])
  const suggestions = useMemo(() => {
    const s = samples.find((x) => x.sample_id === session?.sample_id)
    if (s) return s.suggested_questions.filter((q) => !turns.some((t) => t.question === q))
    if (!session) return []
    const cfg = session.manifest.config
    if (cfg === "bi_temporal_pair") return ["What changed between the two dates?", "Where did the change happen?"]
    if (cfg === "cross_modal_pair") return ["Use both sensors to map the water. Do they agree?", "Describe this scene."]
    return ["Describe the land cover in this image.", "How much water is there?", "Is this area flooded?"]
  }, [samples, session, turns])

  return (
    <div className="min-h-screen bg-canvas">
      <Header health={health} />
      <main className="mx-auto grid max-w-[1600px] grid-cols-1 gap-5 px-4 py-5 sm:px-6 lg:grid-cols-[320px_minmax(0,1fr)] lg:px-8 2xl:grid-cols-[320px_minmax(0,1fr)_560px]">
        <aside className="flex flex-col gap-4">
          {session ? (
            <ManifestCard session={session} onNew={newSession} />
          ) : (
            <SessionPicker samples={samples} busy={busy} error={sessionError} onSample={startSample} onUpload={startUpload} />
          )}
        </aside>

        <section className="flex min-w-0 flex-col gap-4">
          {session ? (
            <>
              {turns.length === 0 && !pending && (
                <div className="rounded-xl border border-dashed border-border px-6 py-10 text-center">
                  <MessageSquareText size={22} className="mx-auto mb-2 text-ink-faint" />
                  <div className="text-[14px] text-ink">Ask a question about this imagery</div>
                  <p className="mx-auto mt-1 max-w-md text-[12.5px] leading-relaxed text-ink-faint">
                    Each question runs through seven stages: parse, assess, plan, execute, adjudicate, calibrate and
                    report. When the imagery can't answer a question, you'll be told what's missing.
                  </p>
                </div>
              )}
              <Conversation
                turns={turns}
                pending={pending}
                activeId={activeId}
                onSelect={(id) => {
                  setActiveId(id)
                  setHighlight(null)
                }}
                onClaim={(qid, entry) => {
                  setActiveId(qid)
                  setHighlight(entry)
                  setTab("ledger")
                }}
                onUploadMore={newSession}
              />
              <div className="sticky bottom-4 z-10">
                <Composer suggestions={suggestions} loading={pending !== null} error={askError} onAsk={ask} />
              </div>
            </>
          ) : (
            <div className="rounded-xl border border-border-soft bg-surface px-6 py-12">
              <div className="mx-auto max-w-xl text-center">
                <div className="text-[18px] font-semibold tracking-tight text-ink">Ask satellite imagery questions in plain English</div>
                <p className="mt-2 text-[13.5px] leading-relaxed text-ink-muted">
                  Upload a single image, an optical + SAR pair, or a before/after pair. Deterministic tools measure
                  water, vegetation, built-up area, bare soil and change directly from the bands. A language model
                  only writes the sentence, and every number it states is checked against the measurements before
                  you see it.
                </p>
                <div className="mt-6 grid grid-cols-1 gap-3 text-left sm:grid-cols-3">
                  {[
                    ["Knows what it can't know", "“Is this flooded?” on one image gets the measured water extent plus a request for a pre-event image, not a guess."],
                    ["Measurement wins", "Numbers the narrator states are matched to ledger entries. A wrong figure is replaced with the measured one."],
                    ["Two sensors, one confidence", "Optical and SAR are measured independently. How well they agree becomes the confidence score."],
                  ].map(([h, p]) => (
                    <div key={h} className="rounded-lg border border-border-soft bg-surface-raised p-3">
                      <div className="text-[12.5px] font-medium text-ink">{h}</div>
                      <div className="mt-1 text-[11.5px] leading-snug text-ink-faint">{p}</div>
                    </div>
                  ))}
                </div>
                <p className="mt-6 text-[12px] text-ink-faint">Pick a demo scene on the left to start.</p>
              </div>
            </div>
          )}
        </section>

        {session && active && (
          <aside className="min-w-0 lg:col-start-2 2xl:col-start-3 2xl:row-start-1">
            <div className="2xl:sticky 2xl:top-5">
              <Inspector
                trace={active}
                session={session}
                tab={tab}
                onTab={setTab}
                highlight={highlight}
                onEntry={(e) => {
                  setHighlight(e)
                  setTab("ledger")
                }}
              />
            </div>
          </aside>
        )}
      </main>
    </div>
  )
}
