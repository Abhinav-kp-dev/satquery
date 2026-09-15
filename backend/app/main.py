from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import RENDER_DIR, UPLOAD_DIR, settings
from app.core.controller import InputRejected, run_query
from app.ledger.store import get_trace, list_traces
from app.models.schemas import QueryResponse, QueryTrace

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/renders", StaticFiles(directory=RENDER_DIR), name="renders")


@app.get("/api/health")
def health():
    return {"status": "ok", "vlm_provider": settings.vlm_provider}


@app.post("/api/query", response_model=QueryResponse)
async def query(
    question: str = Form(...),
    image_a: UploadFile = File(...),
    image_b: UploadFile | None = File(None),
):
    files = [image_a] + ([image_b] if image_b is not None else [])
    for f in files:
        if f.size and f.size > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"'{f.filename}' exceeds the {settings.max_upload_mb}MB prototype cap.")

    saved: list[tuple[str, str]] = []
    for f in files:
        suffix = Path(f.filename or "upload.tif").suffix or ".tif"
        dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:12]}{suffix}"
        contents = await f.read()
        dest.write_bytes(contents)
        saved.append((str(dest), f.filename or dest.name))

    try:
        trace, render_urls, overlay_url = run_query(saved, question, max_dim=settings.max_image_dim)
    except InputRejected as exc:
        raise HTTPException(422, exc.reason) from exc

    return QueryResponse(trace=trace, render_urls=render_urls, overlay_url=overlay_url)


@app.get("/api/trace/{query_id}", response_model=QueryTrace)
def trace(query_id: str):
    t = get_trace(query_id)
    if t is None:
        raise HTTPException(404, "No trace found for that query id.")
    return t


@app.get("/api/history", response_model=list[QueryTrace])
def history(limit: int = 20):
    return list_traces(limit)
