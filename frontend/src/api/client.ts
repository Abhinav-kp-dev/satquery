import type { QueryResponse, QueryTrace } from "../types"

export class ApiRequestError extends Error {}

export async function submitQuery(
  question: string,
  imageA: File,
  imageB: File | null,
): Promise<QueryResponse> {
  const form = new FormData()
  form.append("question", question)
  form.append("image_a", imageA)
  if (imageB) form.append("image_b", imageB)

  const res = await fetch("/api/query", { method: "POST", body: form })
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    throw new ApiRequestError(body?.detail ?? `Request failed (${res.status})`)
  }
  return res.json()
}

export async function fetchHistory(limit = 20): Promise<QueryTrace[]> {
  const res = await fetch(`/api/history?limit=${limit}`)
  if (!res.ok) throw new ApiRequestError(`Could not load history (${res.status})`)
  return res.json()
}
