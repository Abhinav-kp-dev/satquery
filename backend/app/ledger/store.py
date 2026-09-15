"""SQLite-backed Evidence Ledger. Every query's full trace is persisted
here immediately after adjudication -- this is the auditable execution
record the problem statement asks to be evaluated on, and it is what the
UI's trace inspector and PDF/JSON export read from.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager

from app.config import LEDGER_DB
from app.models.schemas import QueryTrace

_SCHEMA = """
CREATE TABLE IF NOT EXISTS query_traces (
    query_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    question TEXT NOT NULL,
    task TEXT,
    config TEXT NOT NULL,
    confidence REAL,
    trace_json TEXT NOT NULL
);
"""


@contextmanager
def _connect():
    conn = sqlite3.connect(LEDGER_DB)
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_trace(trace: QueryTrace) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO query_traces "
            "(query_id, created_at, question, task, config, confidence, trace_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                trace.query_id,
                trace.created_at,
                trace.question,
                trace.task_classified.value if trace.task_classified else None,
                trace.config.value,
                trace.confidence.overall,
                trace.model_dump_json(),
            ),
        )


def get_trace(query_id: str) -> QueryTrace | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT trace_json FROM query_traces WHERE query_id = ?", (query_id,)
        ).fetchone()
    if row is None:
        return None
    return QueryTrace.model_validate_json(row[0])


def list_traces(limit: int = 50) -> list[QueryTrace]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT trace_json FROM query_traces ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [QueryTrace.model_validate_json(r[0]) for r in rows]
