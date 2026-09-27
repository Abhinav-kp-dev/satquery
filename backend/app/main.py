from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import RENDER_DIR, REPORT_DIR, SAMPLE_DIR, UPLOAD_DIR, settings
from app.core.controller import InputRejected, SessionNotFound, create_session, load_session, run_query
from app.ledger.store import get_trace, ledger_for_query, list_sessions, list_traces, verify_chain
from app.models.schemas import QueryResponse, QueryTrace, SampleInfo, SessionInfo
from app.report.pdf import build_pdf
from app.samples.generate import generate as generate_samples
from app.tools.grounding import regions_to_geojson

@asynccontextmanager
async def lifespan(_: FastAPI):
    generate_samples(SAMPLE_DIR)  # no-op once the demo scenes exist
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/renders", StaticFiles(directory=RENDER_DIR), name="renders")


class QueryBody(BaseModel):
    question: str
    stress_test: bool = False  # mock narrator misstates one figure, to demonstrate adjudication


async def _save_uploads(files: list[UploadFile]) -> list[tuple[str, str]]:
    saved = []
    for f in files:
        if f.size and f.size > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"'{f.filename}' exceeds the {settings.max_upload_mb} MB upload cap.")
        suffix = Path(f.filename or "upload.tif").suffix or ".tif"
        dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:12]}{suffix}"
        with dest.open("wb") as out:
            while chunk := await f.read(1 << 20):
                out.write(chunk)
        saved.append((str(dest), f.filename or dest.name))
    return saved


def _trace_or_404(query_id: str) -> QueryTrace:
    t = get_trace(query_id)
    if t is None:
        raise HTTPException(404, "No trace found for that query id.")
    return t


# ------------------------------------------------------------------ meta


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "vlm_provider": settings.vlm_provider,
        "vlm_model": settings.vlm_model if settings.vlm_provider == "openai_compat" else (settings.anthropic_model if settings.vlm_provider == "anthropic" else "template"),
        "analysis_max_dim": settings.analysis_max_dim,
    }


@app.get("/api/samples", response_model=list[SampleInfo])
def samples():
    data = json.loads((SAMPLE_DIR / "samples.json").read_text())
    return [SampleInfo(**{k: s[k] for k in SampleInfo.model_fields}) for s in data]


# ------------------------------------------------------------------ sessions


@app.post("/api/sessions", response_model=SessionInfo)
async def new_session(image_a: UploadFile = File(...), image_b: UploadFile | None = File(None)):
    saved = await _save_uploads([image_a] + ([image_b] if image_b is not None else []))
    try:
        return create_session(saved)
    except InputRejected as exc:
        raise HTTPException(422, exc.reason) from exc


@app.post("/api/sessions/sample/{sample_id}", response_model=SessionInfo)
def new_sample_session(sample_id: str):
    data = {s["sample_id"]: s for s in json.loads((SAMPLE_DIR / "samples.json").read_text())}
    if sample_id not in data:
        raise HTTPException(404, "Unknown sample.")
    return create_session([(str(SAMPLE_DIR / f), f) for f in data[sample_id]["files"]], sample_id=sample_id)


@app.get("/api/sessions", response_model=list[SessionInfo])
def sessions(limit: int = 20):
    return list_sessions(limit)


@app.get("/api/sessions/{session_id}")
def session(session_id: str):
    try:
        info, _ = load_session(session_id)
    except SessionNotFound as exc:
        raise HTTPException(404, "Unknown session.") from exc
    return {"session": info, "traces": list_traces(limit=100, session_id=session_id)}


@app.post("/api/sessions/{session_id}/query", response_model=QueryTrace)
def session_query(session_id: str, body: QueryBody):
    if not body.question.strip():
        raise HTTPException(422, "Ask a question.")
    try:
        return run_query(session_id, body.question.strip(), perturb=body.stress_test)
    except SessionNotFound as exc:
        raise HTTPException(404, "Unknown session.") from exc


# One-shot endpoint kept for scripts and the original client: upload + ask in one call.
@app.post("/api/query", response_model=QueryResponse)
async def query(question: str = Form(...), image_a: UploadFile = File(...), image_b: UploadFile | None = File(None)):
    saved = await _save_uploads([image_a] + ([image_b] if image_b is not None else []))
    try:
        info = create_session(saved)
    except InputRejected as exc:
        raise HTTPException(422, exc.reason) from exc
    trace = run_query(info.session_id, question)
    return QueryResponse(trace=trace, render_urls=trace.render_urls, overlay_url=trace.overlay_url)


# ------------------------------------------------------------------ traces, ledger, reports


@app.get("/api/trace/{query_id}", response_model=QueryTrace)
def trace(query_id: str):
    return _trace_or_404(query_id)


@app.get("/api/history", response_model=list[QueryTrace])
def history(limit: int = 20):
    return list_traces(limit)


@app.get("/api/ledger/verify")
def ledger_verify():
    return verify_chain()


@app.get("/api/ledger/{query_id}")
def ledger(query_id: str):
    return ledger_for_query(query_id)


@app.get("/api/report/{query_id}.pdf")
def report_pdf(query_id: str):
    t = _trace_or_404(query_id)
    path = build_pdf(t, REPORT_DIR / f"{query_id}.pdf")
    return FileResponse(path, media_type="application/pdf", filename=f"satquery_{query_id}.pdf")


@app.get("/api/report/{query_id}.json")
def report_json(query_id: str):
    t = _trace_or_404(query_id)
    return JSONResponse(t.model_dump(mode="json"), headers={"Content-Disposition": f'attachment; filename="satquery_{query_id}.json"'})


@app.get("/api/report/{query_id}.geojson")
def report_geojson(query_id: str):
    t = _trace_or_404(query_id)
    fc = regions_to_geojson([r.model_dump() for r in t.regions], {"query_id": query_id, "question": t.question})
    return JSONResponse(fc, media_type="application/geo+json",
                        headers={"Content-Disposition": f'attachment; filename="satquery_{query_id}.geojson"'})
