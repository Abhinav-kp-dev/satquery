"""The Evidence Ledger: one append-only structure, four roles.

1. Agent memory      -- follow-up turns in a session reuse earlier entries
                        instead of recomputing (see `reused_from`).
2. Generation grounding -- the narrator is shown measurement entries by id
                        ("[E7] water_fraction = 0.412") and must cite them.
3. Confidence substrate -- calibration reads quality/agreement entries.
4. Audit trail       -- entries are SHA-256 hash-chained across all queries;
                        editing any stored entry breaks every later hash.

During a query, entries accumulate in a `LedgerWriter` with query-local ids
(E1, E2, ...). On commit they receive a global sequence number and a hash
linking them to the previous entry in the chain.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from app.models.schemas import LedgerEntry

GENESIS_HASH = "0" * 64


def _jsonable(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):  # numpy scalar
        return _jsonable(value.item())
    return value


def entry_digest(entry: LedgerEntry, prev_hash: str) -> str:
    payload = {
        "seq": entry.seq,
        "query_id": entry.query_id,
        "entry_id": entry.entry_id,
        "stage": entry.stage,
        "kind": entry.kind,
        "key": entry.key,
        "value": entry.value,
        "unit": entry.unit,
        "source": entry.source,
        "reused_from": entry.reused_from,
        "prev_hash": prev_hash,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class LedgerWriter:
    def __init__(self, query_id: str):
        self.query_id = query_id
        self.entries: list[LedgerEntry] = []
        self._by_key: dict[str, LedgerEntry] = {}

    def append(
        self,
        stage: str,
        kind: str,
        key: str,
        value: Any = None,
        unit: str | None = None,
        source: str = "",
        reused_from: str | None = None,
    ) -> str:
        entry = LedgerEntry(
            entry_id=f"E{len(self.entries) + 1}",
            query_id=self.query_id,
            stage=stage,
            kind=kind,
            key=key,
            value=_jsonable(value),
            unit=unit,
            source=source,
            reused_from=reused_from,
        )
        self.entries.append(entry)
        if kind == "measurement":
            self._by_key[key] = entry
        return entry.entry_id

    def measurement(self, key: str) -> LedgerEntry | None:
        return self._by_key.get(key)

    def measurements(self) -> dict[str, LedgerEntry]:
        return dict(self._by_key)

    def numeric_measurements(self) -> dict[str, LedgerEntry]:
        return {k: e for k, e in self._by_key.items() if isinstance(e.value, (int, float)) and not isinstance(e.value, bool)}
