# SatQuery AI

**SIH 2026 · Problem statement SIH26167 (ISRO / SAC): an interactive vision-language assistant for
multimodal remote-sensing image analysis through text queries.**

Upload a GeoTIFF, an optical + SAR pair, or a before/after pair, and ask a question in plain English.
Deterministic tools measure the answer from the raw bands. A vision-language model only writes the
sentence, and every number it states is checked against the measurements before you see it. When
the imagery can't answer the question, the system says so and names the image that would.

```
User question ──► 1 PARSE ──► 2 ASSESS ──┬─ request_input ─► "upload a pre-event image"
                  typed spec   sufficiency ├─ scope_down ───┐
                                           └─ proceed ──────┤
                                                            ▼
            3 PLAN ──► 4 EXECUTE ─────────► 5 ADJUDICATE ──► 6 CALIBRATE ──► 7 REPORT
            registry   parallel tool DAG    narration vs     adjudication,   answer, overlay,
            -> DAG     (NDWI/MNDWI, NDVI,   ledger;          separability,   regions + lat/lon,
                       NDBI, BSI, SAR,      measurement      cross-sensor    PDF / JSON / GeoJSON
                       change, fusion,      wins on          IoU, data
                       grounding)           conflict         quality
                 │            │                  │                │               │
                 └────────────┴──── Evidence Ledger (append-only, SHA-256 chained) ┘
```

## What the proposal promised, and where it lives

| Proposal claim | Implementation |
|---|---|
| **Evidence-sufficiency reasoning.** Knows what evidence a question needs before answering and requests a pre-event image when required | `backend/app/core/sufficiency.py`. The parser records the question's *intent* even when the input can't support it, so ASSESS can name the missing observation. It also checks sensor capability: vegetation can't be measured from SAR, and built-up needs SWIR or dual-pol. |
| **Measurement-grounded generation with enforced adjudication.** On conflict, the measurement wins | `backend/app/vlm/adjudicator.py`. Matching is unit-aware and class-aware, and numbers the model didn't tag are checked too. A mismatched figure is **rewritten to the ledger value** in the delivered answer, and the raw model output is kept in the trace. |
| **Cross-sensor agreement as calibrated confidence** | `tools/fusion_agreement.py` computes IoU/Dice between masks derived independently from optical and SAR. `utils/confidence.py` combines it with adjudication, Otsu separability and data quality in a weighted geometric mean. |
| **One backbone, three input configurations; swappable LoRA adapters** | `core/preflight.py` detects single, cross-modal and bi-temporal inputs. `data/registry.yaml` maps each task to a LoRA adapter, and `vlm/client.py`'s `openai_compat` backend routes to it (Qwen2.5-VL on vLLM). `training/` has the data-prep and LoRA scripts. |
| **The Evidence Ledger: one structure, four roles** | `ledger/ledger.py` and `ledger/store.py`. The same entries serve as **agent memory** (follow-up turns reuse them through `reused_from`), **grounding** (the narrator sees `[E7] water_fraction = ...`), **confidence substrate**, and **audit trail** (hash-chained across all queries, with SQLite triggers blocking UPDATE and DELETE, and `GET /api/ledger/verify` recomputing every hash). |
| **Plain English, no GIS expertise** | `core/parser.py` turns the question into a typed `QuerySpec` (task, target, metric, location) and inherits omitted parts on follow-ups ("and in hectares?"). |
| **Accepts GeoTIFF as agencies hold it** | Band layouts are resolved from the file's band descriptions first, then from known layouts (Sentinel-2 12/13-band, BigEarthNet 10-band, 4-band BGRN, RGB). SAR can be linear or dB, single- or dual-pol. Pairs on different grids are **reprojected** onto a common grid, and non-overlapping pairs are rejected. |
| **Operational scene size.** A 512×512 assumption fails on real scenes | Large scenes are **not rejected**. They're read decimated through GeoTIFF overviews onto a bounded analysis grid, and hectares use the effective GSD. A 67-megapixel scene answers in about 3.3 s (see the eval). |
| **Co-registration** | `tools/coregistration.py` uses phase correlation on bi-temporal pairs. It finds and corrects the (2, −1) px shift injected into the mining demo, so misregistration doesn't produce fake change. |
| **Visual grounding** | `tools/grounding.py` turns connected components into bounding boxes, areas and WGS84 centroids, drawn in the UI and exported as GeoJSON. |
| **Answer + confidence + map overlay + PDF** | `report/pdf.py`, plus JSON and GeoJSON exports, and toggleable per-layer mask PNGs in the UI. |

## Try it in 30 seconds

```bash
# backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# frontend (second terminal)
cd frontend && npm install && npm run dev      # http://localhost:5173
```

Or with Docker: `docker compose up --build`, then open http://localhost:8080.

On first start the backend generates five **synthetic, georeferenced demo scenes** with known ground
truth (`app/samples/generate.py`): Sentinel-2-style 12-band optical and Sentinel-1-style VV/VH SAR
over the Brahmaputra near Guwahati. Things to try:

| Demo | Ask | What happens |
|---|---|---|
| Post-flood optical scene | *Is this area flooded?* | **scope_down**: reports 17.8% water (466 ha) and says a pre-event image is needed to call it a flood |
| Flood: before / after | *How much did the water extent change?* | 308 ha newly water, located with boxes and lat/lon; compare slider |
| Optical + SAR | *Use both sensors to map the water. Do they agree?* | IoU 0.98 becomes the cross-sensor confidence component |
| SAR only | *What changed since last year?* | **request_input**: nothing is measured, and the answer names the missing image |
| Any | tick **Stress test**, then ask | the narrator misstates a figure, and the delivered answer shows the measured value |
| Any | click a highlighted number | jumps to the ledger entry it cites; **Verify hash chain** re-checks the whole ledger |

## Narration backends

Set in `backend/.env` (see `.env.example`). **No backend ever produces a number.** They only phrase
the measured evidence, and all output goes through the adjudicator.

| `SATQUERY_VLM_PROVIDER` | What it is |
|---|---|
| `mock` (default) | Offline, template-driven from the ledger. No credentials and no GPU. |
| `openai_compat` | Self-hosted open-weight VLM (e.g. Qwen2.5-VL on vLLM with `--lora-modules`). Each task is routed to its LoRA adapter by name. Falls back to `mock` if the server is unreachable. |
| `anthropic` | Claude via the official SDK. |

## Evaluation

`cd backend && python -m eval.run_eval`. The full output is in [`backend/eval/RESULTS.md`](backend/eval/RESULTS.md).

| | result |
|---|---|
| Routing + sufficiency, 60 labelled questions, 7 input types | 96.7% fully correct (task 98.3%, target 100%, verdict 98.3%) |
| Tool masks vs ground truth (synthetic scenes) | mean IoU 0.975; SAR water 0.98, SAR built-up 0.78 |
| Change area vs ground truth | 0.0% (flood), 0.1% (mining, after correcting a 2.2 px shift) |
| Adjudication | 0 false corrections in 43 honest figures; 14 of 14 injected errors caught and removed |
| Latency (CPU) | 161 ms median per query; 67 MP scene in 1.4 s session + 1.8 s query |

**Read these numbers with their caveats:**
- The routing set was written alongside the rules, so it measures coverage, not generalisation.
- The synthetic optical scenes are clean, so tool accuracy on them is an upper bound. Nothing here
  was measured on real Cartosat or RISAT imagery.
- The 67 MP test scene is a regular block pattern.

## Tests

```bash
cd backend && pip install -r requirements-dev.txt && python -m pytest -q     # 51 tests
cd frontend && npm run build && npx oxlint src
```

## API

| Method | Path | |
|---|---|---|
| GET | `/api/samples` | demo scenes and suggested questions |
| POST | `/api/sessions` | upload 1–2 GeoTIFFs; preflight runs and returns the manifest |
| POST | `/api/sessions/sample/{id}` | start a session on a demo scene |
| POST | `/api/sessions/{id}/query` | `{"question": "...", "stress_test": false}` returns the full trace |
| GET | `/api/sessions/{id}` | session plus all its traces |
| GET | `/api/report/{query_id}.pdf` / `.json` / `.geojson` | exports |
| GET | `/api/ledger/verify` | recompute the hash chain |
| POST | `/api/query` | one-shot upload and question (kept from the first prototype) |

## Layout

```
backend/app/
  core/       preflight, raster_io (decimation, reprojection, co-registration), parser,
              sufficiency, planner, executor (parallel DAG, session memory), controller
  tools/      spectral_index, sar_backscatter, threshold, delta_change, fusion_agreement,
              coregistration, grounding, georef, rendering, bands
  vlm/        client (mock / openai_compat + LoRA / anthropic), prompt_builder, retrieval,
              adjudicator
  ledger/     hash-chained writer + SQLite store
  report/     PDF
  samples/    synthetic scene generator with ground truth
backend/eval/     routing benchmark, run_eval.py, RESULTS.md
backend/tests/    51 pytest tests
training/         LoRA data prep + fine-tuning for Qwen2.5-VL (written, not run here)
frontend/src/     React: session panel, conversation, inspector (evidence / pipeline / ledger)
```

## Known limitations

- **No real-imagery evaluation yet.** Accuracy is demonstrated on synthetic scenes only. The next step
  is BigEarthNet-MM, CDVQA and LEVIR-CD, plus ISRO's Cartosat/RISAT evaluation set.
- **LoRA adapters are not trained.** The scripts exist, and the serving path and adapter routing are
  implemented, but the default narrator is the template one.
- **Classical tools, not segmentation networks.** Index thresholds (Otsu clamped to literature ranges)
  are transparent and CPU-only, but miss what band ratios can't separate. Built-up versus bare soil
  is split by NDBI/BSI dominance, a known confusion in the literature.
- **Confidence is a heuristic.** It's a transparent combination of defensible signals, not a
  statistically calibrated probability.
- **Decimation loses small features.** Very large scenes are analysed at a coarser GSD, which is
  reported in the manifest and the ledger. Tiled full-resolution analysis is future work.
- **Routing is rule-based,** which keeps it auditable but can miss unusual phrasing (see the misses
  listed in RESULTS.md).
