import type { InputConfig, SessionInfo, Target, Task } from "../types"

export const CONFIG_LABEL: Record<InputConfig, string> = {
  single_image: "Single image",
  cross_modal_pair: "Optical + SAR pair",
  bi_temporal_pair: "Before / after pair",
  invalid: "Invalid input",
}

export const TASK_LABEL: Record<Task, string> = {
  vqa: "Visual QA",
  caption: "Scene description",
  grounding: "Grounding",
  change: "Change analysis",
  fusion: "Sensor fusion",
}

export const TARGET_LABEL: Record<Target, string> = {
  water: "Water",
  vegetation: "Vegetation",
  built_up: "Built-up",
  bare_soil: "Bare soil",
  land_cover: "Land cover",
}

export function fmtValue(v: unknown, unit?: string | null): string {
  if (typeof v === "number") {
    if (unit === "fraction") return `${(v * 100).toFixed(2)}%`
    if (unit === "ha") return `${v.toLocaleString(undefined, { maximumFractionDigits: 1 })} ha`
    if (Number.isInteger(v)) return String(v)
    return `${v.toFixed(3)}${unit ? ` ${unit}` : ""}`
  }
  if (v === null || v === undefined) return "--"
  if (typeof v === "object") return JSON.stringify(v)
  return String(v)
}

export function imageLabel(session: SessionInfo, i: number): string {
  const img = session.manifest.images[i]
  const cfg = session.manifest.config
  const date = img.acquisition_date ? ` · ${img.acquisition_date}` : ""
  if (cfg === "bi_temporal_pair") return `${i === 0 ? "T1 (before)" : "T2 (after)"}${date}`
  if (cfg === "cross_modal_pair") return `${img.modality === "sar" ? "SAR" : "Optical"}${date}`
  return `${img.modality === "sar" ? "SAR" : "Optical"}${date}`
}

export function baseIndex(session: SessionInfo): number {
  const m = session.manifest
  if (m.config === "bi_temporal_pair") return 1
  if (m.config === "cross_modal_pair") return m.images[0].modality === "optical" ? 0 : 1
  return 0
}

export function rgb(c: number[], a = 1) {
  return `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${a})`
}

export const LAYER_LABEL: Record<string, string> = {
  water: "Water",
  vegetation: "Vegetation",
  built_up: "Built-up",
  bare_soil: "Bare soil",
  gained: "Gained",
  lost: "Lost",
  agreement: "Both sensors agree",
  optical_only: "Optical only",
  sar_only: "SAR only",
}
