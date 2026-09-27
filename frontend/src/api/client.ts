import type { Health, LedgerVerification, QueryTrace, SampleInfo, SessionInfo } from "../types"

export class ApiRequestError extends Error {}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    const detail = body?.detail
    throw new ApiRequestError(typeof detail === "string" ? detail : `Request failed (${res.status})`)
  }
  return res.json()
}

export const api = {
  health: () => fetch("/api/health").then((r) => json<Health>(r)),
  samples: () => fetch("/api/samples").then((r) => json<SampleInfo[]>(r)),
  sampleSession: (id: string) => fetch(`/api/sessions/sample/${id}`, { method: "POST" }).then((r) => json<SessionInfo>(r)),
  uploadSession: (a: File, b: File | null) => {
    const form = new FormData()
    form.append("image_a", a)
    if (b) form.append("image_b", b)
    return fetch("/api/sessions", { method: "POST", body: form }).then((r) => json<SessionInfo>(r))
  },
  session: (id: string) =>
    fetch(`/api/sessions/${id}`).then((r) => json<{ session: SessionInfo; traces: QueryTrace[] }>(r)),
  recentSessions: () => fetch("/api/sessions?limit=8").then((r) => json<SessionInfo[]>(r)),
  ask: (sessionId: string, question: string, stressTest: boolean) =>
    fetch(`/api/sessions/${sessionId}/query`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question, stress_test: stressTest }),
    }).then((r) => json<QueryTrace>(r)),
  verifyLedger: () => fetch("/api/ledger/verify").then((r) => json<LedgerVerification>(r)),
}

export const reportUrl = (queryId: string, ext: "pdf" | "json" | "geojson") => `/api/report/${queryId}.${ext}`
