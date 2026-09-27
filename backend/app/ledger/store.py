"""SQLite persistence for the Evidence Ledger, query traces and sessions.

`commit_ledger` is the only writer of ledger rows and is append-only:
rows are inserted, never updated or deleted. `verify_chain` recomputes
every hash from the stored fields, so any after-the-fact edit is detected.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager

from app.config import LEDGER_DB
from app.ledger.ledger import GENESIS_HASH, entry_digest
from app.models.schemas import LedgerEntry, QueryTrace, SessionInfo

_SCHEMA = """
CREATE TABLE IF NOT EXISTS query_traces (
    query_id TEXT PRIMARY KEY,
    session_id TEXT,
    created_at TEXT NOT NULL,
    question TEXT NOT NULL,
    task TEXT,
    config TEXT NOT NULL,
    confidence REAL,
    trace_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ledger (
    seq INTEGER PRIMARY KEY,
    query_id TEXT NOT NULL,
    entry_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    kind TEXT NOT NULL,
    key TEXT NOT NULL,
    value_json TEXT,
    unit TEXT,
    source TEXT,
    reused_from TEXT,
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ledger_query ON ledger(query_id);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    paths_json TEXT NOT NULL,
    info_json TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS ledger_no_update BEFORE UPDATE ON ledger
BEGIN SELECT RAISE(ABORT, 'evidence ledger is append-only'); END;
CREATE TRIGGER IF NOT EXISTS ledger_no_delete BEFORE DELETE ON ledger
BEGIN SELECT RAISE(ABORT, 'evidence ledger is append-only'); END;
"""

_commit_lock = threading.Lock()


@contextmanager
def _connect():
    conn = sqlite3.connect(LEDGER_DB, timeout=10)
    try:
        conn.executescript(_SCHEMA)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(query_traces)")}
        if "session_id" not in cols:  # databases created by the first prototype
            conn.execute("ALTER TABLE query_traces ADD COLUMN session_id TEXT")
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------- ledger


def commit_ledger(entries: list[LedgerEntry]) -> list[LedgerEntry]:
    """Assigns global seq numbers and chains hashes, then appends."""
    with _commit_lock, _connect() as conn:
        row = conn.execute("SELECT seq, hash FROM ledger ORDER BY seq DESC LIMIT 1").fetchone()
        seq, prev = (row[0], row[1]) if row else (0, GENESIS_HASH)
        committed = []
        for e in entries:
            seq += 1
            e = e.model_copy(update={"seq": seq})
            h = entry_digest(e, prev)
            e = e.model_copy(update={"prev_hash": prev, "hash": h})
            conn.execute(
                "INSERT INTO ledger (seq, query_id, entry_id, stage, kind, key, value_json, unit, source, reused_from, prev_hash, hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (e.seq, e.query_id, e.entry_id, e.stage, e.kind, e.key, json.dumps(e.value, default=str), e.unit, e.source, e.reused_from, prev, h),
            )
            committed.append(e)
            prev = h
    return committed


def _row_to_entry(r) -> LedgerEntry:
    return LedgerEntry(
        seq=r[0], query_id=r[1], entry_id=r[2], stage=r[3], kind=r[4], key=r[5],
        value=json.loads(r[6]) if r[6] is not None else None, unit=r[7], source=r[8] or "",
        reused_from=r[9], prev_hash=r[10], hash=r[11],
    )


def ledger_for_query(query_id: str) -> list[LedgerEntry]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM ledger WHERE query_id = ? ORDER BY seq", (query_id,)).fetchall()
    return [_row_to_entry(r) for r in rows]


def verify_chain() -> dict:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM ledger ORDER BY seq").fetchall()
    prev = GENESIS_HASH
    for r in rows:
        e = _row_to_entry(r)
        if e.prev_hash != prev or entry_digest(e, prev) != e.hash:
            return {"valid": False, "entries": len(rows), "first_bad_seq": e.seq, "query_id": e.query_id, "entry_id": e.entry_id}
        prev = e.hash
    return {"valid": True, "entries": len(rows), "head": prev}


# ---------------------------------------------------------------- traces


def save_trace(trace: QueryTrace) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO query_traces "
            "(query_id, session_id, created_at, question, task, config, confidence, trace_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                trace.query_id,
                trace.session_id,
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
        row = conn.execute("SELECT trace_json FROM query_traces WHERE query_id = ?", (query_id,)).fetchone()
    return QueryTrace.model_validate_json(row[0]) if row else None


def list_traces(limit: int = 50, session_id: str | None = None) -> list[QueryTrace]:
    with _connect() as conn:
        if session_id:
            rows = conn.execute(
                "SELECT trace_json FROM query_traces WHERE session_id = ? ORDER BY created_at ASC LIMIT ?", (session_id, limit)
            ).fetchall()
        else:
            rows = conn.execute("SELECT trace_json FROM query_traces ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [QueryTrace.model_validate_json(r[0]) for r in rows]


# ---------------------------------------------------------------- sessions


def save_session(info: SessionInfo, paths: list[str]) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO sessions (session_id, created_at, paths_json, info_json) VALUES (?, ?, ?, ?)",
            (info.session_id, info.created_at, json.dumps(paths), info.model_dump_json()),
        )


def get_session(session_id: str) -> tuple[SessionInfo, list[str]] | None:
    with _connect() as conn:
        row = conn.execute("SELECT info_json, paths_json FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if row is None:
        return None
    return SessionInfo.model_validate_json(row[0]), json.loads(row[1])


def list_sessions(limit: int = 20) -> list[SessionInfo]:
    with _connect() as conn:
        rows = conn.execute("SELECT info_json FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [SessionInfo.model_validate_json(r[0]) for r in rows]
