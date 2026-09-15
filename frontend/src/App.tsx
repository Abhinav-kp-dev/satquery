import { useEffect, useState } from "react"
import { ApiRequestError, fetchHistory, submitQuery } from "./api/client"
import { AnswerCard } from "./components/AnswerCard"
import { EmptyState } from "./components/EmptyState"
import { Header } from "./components/Header"
import { HistoryRail } from "./components/HistoryRail"
import { ImageViewer } from "./components/ImageViewer"
import { QueryPanel } from "./components/QueryPanel"
import { StatusBadges } from "./components/StatusBadges"
import { TracePanel } from "./components/TracePanel"
import type { QueryTrace } from "./types"

interface ResultState {
  trace: QueryTrace
  renderUrls: string[]
  overlayUrl: string | null
}

function deriveRenderUrls(trace: QueryTrace): { renderUrls: string[]; overlayUrl: string | null } {
  if (trace.sufficiency === "request_input") return { renderUrls: [], overlayUrl: null }
  const count = trace.config === "single_image" ? 1 : 2
  const renderUrls = Array.from({ length: count }, (_, i) => `/renders/${trace.query_id}_render_${i}.png`)
  const overlayUrl = trace.tool_calls.length > 0 ? `/renders/${trace.query_id}_overlay.png` : null
  return { renderUrls, overlayUrl }
}

export default function App() {
  const [result, setResult] = useState<ResultState | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [history, setHistory] = useState<QueryTrace[]>([])
  const [vlmProvider, setVlmProvider] = useState<string | null>(null)

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((d) => setVlmProvider(d.vlm_provider))
      .catch(() => {})
    fetchHistory().then(setHistory).catch(() => {})
  }, [])

  async function handleSubmit(question: string, imageA: File, imageB: File | null) {
    setLoading(true)
    setError(null)
    try {
      const res = await submitQuery(question, imageA, imageB)
      setResult({ trace: res.trace, renderUrls: res.render_urls, overlayUrl: res.overlay_url })
      setHistory((h) => [res.trace, ...h.filter((t) => t.query_id !== res.trace.query_id)].slice(0, 20))
    } catch (e) {
      setError(e instanceof ApiRequestError ? e.message : "Something went wrong reaching the backend.")
    } finally {
      setLoading(false)
    }
  }

  function handleHistorySelect(trace: QueryTrace) {
    const { renderUrls, overlayUrl } = deriveRenderUrls(trace)
    setResult({ trace, renderUrls, overlayUrl })
    setError(null)
  }

  return (
    <div className="min-h-screen bg-canvas">
      <Header vlmProvider={vlmProvider} />

      <main className="mx-auto grid max-w-7xl grid-cols-1 gap-6 px-6 py-6 lg:grid-cols-[380px_1fr] lg:px-8 lg:py-8">
        <aside className="flex flex-col gap-5">
          <QueryPanel onSubmit={handleSubmit} loading={loading} error={error} />
          <HistoryRail items={history} activeId={result?.trace.query_id ?? null} onSelect={handleHistorySelect} />
        </aside>

        <section className="min-w-0">
          {result ? (
            <div className="animate-fade-in flex flex-col gap-5">
              <StatusBadges
                config={result.trace.config}
                task={result.trace.task_classified}
                sufficiency={result.trace.sufficiency}
              />
              <AnswerCard trace={result.trace} />
              <ImageViewer renderUrls={result.renderUrls} overlayUrl={result.overlayUrl} config={result.trace.config} />
              <TracePanel trace={result.trace} />
            </div>
          ) : (
            <EmptyState />
          )}
        </section>
      </main>
    </div>
  )
}
