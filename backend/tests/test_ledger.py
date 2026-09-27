import sqlite3

import pytest

from app.config import LEDGER_DB
from app.ledger.ledger import LedgerWriter
from app.ledger.store import commit_ledger, verify_chain


def test_chain_links_and_detects_tampering():
    w = LedgerWriter("qtest1")
    w.append("execute", "measurement", "water_fraction", 0.41, "fraction", "spectral_index#0")
    w.append("execute", "measurement", "water_area_ha", 1078.5, "ha", "spectral_index#0")
    committed = commit_ledger(w.entries)
    assert committed[1].prev_hash == committed[0].hash
    assert verify_chain()["valid"]

    conn = sqlite3.connect(LEDGER_DB)
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("UPDATE ledger SET value_json = '0.9' WHERE seq = ?", (committed[0].seq,))
    # A determined attacker can drop the trigger -- the hash chain still catches it.
    conn.execute("DROP TRIGGER ledger_no_update")
    conn.execute("UPDATE ledger SET value_json = '0.9' WHERE seq = ?", (committed[0].seq,))
    conn.commit()
    conn.close()
    result = verify_chain()
    assert not result["valid"] and result["first_bad_seq"] == committed[0].seq

    # restore for the rest of the session (verify_chain re-created the trigger)
    conn = sqlite3.connect(LEDGER_DB)
    conn.execute("DROP TRIGGER ledger_no_update")
    conn.execute("UPDATE ledger SET value_json = '0.41' WHERE seq = ?", (committed[0].seq,))
    conn.commit()
    conn.close()
    assert verify_chain()["valid"]
