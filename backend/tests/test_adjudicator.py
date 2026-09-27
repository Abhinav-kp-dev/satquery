from app.models.schemas import LedgerEntry
from app.vlm.adjudicator import adjudicate


def ledger(**vals):
    out = {}
    for i, (k, (v, unit)) in enumerate(vals.items(), start=1):
        out[k] = LedgerEntry(entry_id=f"E{i}", query_id="q", stage="execute", kind="measurement", key=k, value=v, unit=unit, source="t")
    return out


L = ledger(
    water_fraction=(0.412, "fraction"), water_area_ha=(1078.5, "ha"),
    built_up_fraction=(0.07, "fraction"), built_up_area_ha=(182.8, "ha"), iou=(0.81, "iou"),
)


def test_consistent_claims_cite_ledger_entries():
    a = adjudicate('<ANSWER>Water covers <CLAIM value="41.2" unit="percent">41.2%</CLAIM>.</ANSWER>', L)
    assert a.status == "consistent"
    assert a.claims[0].ledger_entry_id == "E1" and a.answer == "Water covers 41.2%."


def test_measurement_wins_on_conflict():
    a = adjudicate('<ANSWER>Water covers <CLAIM value="65" unit="percent">about 65%</CLAIM> of the scene.</ANSWER>', L)
    assert a.status == "corrected"
    assert "41.2%" in a.answer and "65" not in a.answer
    assert a.claims[0].corrected and a.claims[0].ledger_value == 41.2


def test_untagged_numbers_are_checked_too():
    a = adjudicate("<ANSWER>Roughly 90% of the scene is water, about 1,078 ha.</ANSWER>", L)
    by_unit = {c.unit: c for c in a.claims}
    assert by_unit["percent"].corrected and by_unit["percent"].source == "untagged"
    assert not by_unit["ha"].corrected
    assert "41.2%" in a.answer


def test_units_are_not_mixed():
    # 41.2 ha must not be accepted just because 41.2% exists.
    a = adjudicate('<ANSWER>Water: <CLAIM value="41.2" unit="hectares">41.2 ha</CLAIM>.</ANSWER>', L)
    assert a.claims[0].corrected and "1,078.5 ha" in a.answer


def test_class_context_does_not_leak_between_clauses():
    text = '<ANSWER>Cover: water (<CLAIM value="41.2" unit="percent">41.2%</CLAIM>), built-up area (<CLAIM value="7.0" unit="percent">7.0%</CLAIM>).</ANSWER>'
    a = adjudicate(text, L)
    assert a.status == "consistent", [c.model_dump() for c in a.claims]
    assert a.claims[1].matched_ledger_key == "built_up_fraction"


def test_unsupported_figure_is_withheld():
    a = adjudicate('<ANSWER>There are <CLAIM value="12" unit="dB">12 dB</CLAIM> of something.</ANSWER>', L)
    assert a.status == "corrected" and "withheld" in a.answer


def test_no_numbers_is_unverified():
    assert adjudicate("<ANSWER>A rural scene.</ANSWER>", L).status == "unverified"
