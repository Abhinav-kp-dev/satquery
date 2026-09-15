# SatQuery AI

A prototype vision-language assistant for satellite imagery. Upload a
GeoTIFF (or a co-registered pair — optical+SAR, or two dates of the same
place), ask a question in plain English, and get back an answer grounded
in numbers a deterministic tool actually measured — not a guess from
pixels — plus the full execution trace behind it.

Built for the ISRO/SIH "SatQuery AI" problem statement: a chat interface
for remote sensing imagery, backed by an agentic controller that decides
for itself which tools a query needs, refuses to answer questions its
evidence can't support, and shows its work.

## What makes this "agentic," concretely

A query goes through six stages, every one of them logged to the Evidence
Ledger and shown in the UI's trace inspector:

```
preflight (rasterio metadata)
      -> sufficiency check ("can this evidence answer this question?")
      -> task routing (deterministic input-gating + keyword classification)
      -> deterministic tool execution (NDWI/NDBI/NDVI, SAR backscatter,
         delta-change, cross-sensor IoU -- pure NumPy, no model)
      -> narration (a VLM turns the numbers into a sentence)
      -> adjudication (every number the narration states is checked
         against the tool output; the measurement wins on conflict)
```

Nothing upstream of narration depends on a model at all. `preflight`,
`sufficiency`, `routing`, and every tool in `app/tools/` are plain Python
and NumPy — they will give the same answer every time, on any machine,
with no GPU.

### The core differentiator: evidence-sufficiency reasoning

A single image can measure what's *present* ("41% of this scene is
water") but not what *changed* ("this area is flooded" presupposes a
baseline the system was never given). Rather than let a language model
guess at that baseline, `app/core/sufficiency.py` recognizes the mismatch
deterministically and scopes the answer down to what the evidence
actually supports — then says plainly what a second image would add.
Try it: upload one optical scene and ask *"is this area flooded?"*

## Why no fine-tuning — and why that's a real design choice, not a gap

The problem statement is explicit that a generic LLM/VLM with no
remote-sensing adaptation doesn't satisfy the brief. This prototype's
answer to that is **retrieval-augmented in-context learning**, not
gradient fine-tuning: `app/data/example_pool.json` holds hand-curated
question/answer exemplars per task family (VQA, captioning, grounding,
change, fusion, and — the hardest to find in any public dataset — scoped
"insufficient evidence" answers). `app/vlm/retrieval.py` retrieves the
closest-matching exemplars by TF-IDF similarity and shows them to the
narration model in-context alongside a domain system prompt and the
measured evidence, via `app/vlm/prompt_builder.py`.

This is weaker than LoRA fine-tuning on BigEarthNet/VRSBench/CDVQA, and
that trade-off is a compute/time constraint, not a claim that it's
equivalent — see `app/vlm/prompt_builder.py`'s docstring. Swapping in a
fine-tuned adapter later changes exactly one component
(`app/vlm/client.py`); nothing upstream or downstream of it needs to
change, because the tool layer and controller never depended on model
weights in the first place.

The `<ANSWER>`/`<CLAIM>` tag format the narration model is asked to
follow (see `OUTPUT_FORMAT_INSTRUCTION` in `prompt_builder.py`) is what
makes `app/vlm/adjudicator.py`'s job regex-simple instead of
extraction-from-free-text: every number the model states is checked
against the tool evidence, and a mismatch is flagged in the trace rather
than silently trusted.

## Architecture

```
backend/app/
  core/
    preflight.py      GeoTIFF -> InputManifest (modality, GSD, bands, CRS)
    sufficiency.py     evidence-sufficiency verdict (proceed / scope_down / request_input)
    classifier.py       two-layer router: input-gated legal task set -> keyword classification
    registry.py, data/registry.yaml   task -> tool plan, as an actual data structure
    controller.py       orchestrates the full pipeline, builds the QueryTrace
  tools/
    spectral_index.py   NDWI / NDVI / NDBI, Otsu-thresholded
    sar_backscatter.py  VV/VH dB thresholding, water + built-up (double-bounce)
    delta_change.py     T2-minus-T1 mask arithmetic -> hectares, direction
    fusion_agreement.py IoU between independently-derived optical/SAR masks
    rendering.py, region.py   GeoTIFF -> 8-bit composite/overlay, mask centroid -> region phrase
  vlm/
    retrieval.py, prompt_builder.py   retrieval-augmented in-context prompting
    client.py            pluggable narration backend (mock, offline / Anthropic)
    adjudicator.py        <ANSWER>/<CLAIM> parsing + evidence-ledger cross-check
  ledger/store.py        SQLite Evidence Ledger (every trace, persisted)
  utils/confidence.py     adjudication agreement + cross-sensor IoU -> confidence band
  main.py                 FastAPI routes

frontend/src/
  components/            UploadZone, QueryPanel, ImageViewer, AnswerCard,
                          StatusBadges, ConfidenceBadge, TracePanel, HistoryRail
  api/client.ts, types.ts
  App.tsx
```

## Running it

### Backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Runs fully offline by default (`SATQUERY_VLM_PROVIDER=mock`). To narrate
with Claude instead, copy `backend/.env.example` to `backend/.env`, set
`SATQUERY_VLM_PROVIDER=anthropic` and `SATQUERY_ANTHROPIC_API_KEY`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Opens on `http://localhost:5173`, proxying `/api` and `/renders` to the
backend on port 8000.

## Input configurations

| Configuration | Legal tasks | Example question |
|---|---|---|
| Single image | VQA, captioning, grounding | "Describe the land cover here." |
| Cross-modal pair (optical + SAR) | Fusion, VQA, grounding | "Use both images to find water and built-up regions." |
| Bi-temporal pair (two dates, same modality) | Change, VQA | "What changed between these two dates, and where?" |

`preflight.py` infers which configuration was uploaded from band count
and dimensions alone (SAR: 1-2 bands; optical: 3+ bands) — mismatched
dimensions or an unreadable file are rejected before anything downstream
runs.

## What's intentionally out of scope for this prototype

- **Model fine-tuning** — see "Why no fine-tuning" above.
- **Coregistration** — inputs are assumed pre-aligned (true of the actual
  ISRO evaluation set, which is pre-georeferenced Cartosat-2S/RISAT).
- **Tiling for very large scenes** — capped at 4096px per side.
- **Multi-user auth / cloud storage** — local filesystem + SQLite, as
  befits a prototype.
