import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import write_tif


@pytest.fixture(scope="module")
def client(samples_dir):
    with TestClient(app) as c:
        yield c


def sample_session(client, sample_id):
    r = client.post(f"/api/sessions/sample/{sample_id}")
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


def ask(client, sid, q, **kw):
    r = client.post(f"/api/sessions/{sid}/query", json={"question": q, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def test_samples_listed(client):
    ids = {s["sample_id"] for s in client.get("/api/samples").json()}
    assert {"flood_single", "flood_pair", "optical_sar", "mining_pair", "sar_single"} <= ids


def test_flooded_single_image_is_scoped(client):
    t = ask(client, sample_session(client, "flood_single"), "Is this area flooded?")
    assert t["sufficiency"] == "scope_down" and "pre-event" in t["answer"]
    assert [s["name"] for s in t["stages"]] == ["parse", "assess", "plan", "execute", "adjudicate", "calibrate", "report"]
    assert t["confidence"]["overall"] <= 0.74


def test_flood_change_matches_ground_truth(client, samples_dir):
    import json

    truth = json.loads((samples_dir / "ground_truth.json").read_text())["changes"]["flood_pair"]
    t = ask(client, sample_session(client, "flood_pair"), "How much did the water extent change between the two dates?")
    gained = next(c["output"]["water_gained_ha"] for c in t["tool_calls"] if c["tool"] == "delta_change")
    assert abs(gained - truth["gained_ha"]) / truth["gained_ha"] < 0.05
    assert t["adjudication_status"] == "consistent"


def test_mining_pair_corrects_injected_shift(client):
    t = ask(client, sample_session(client, "mining_pair"), "Has any new mining or excavation appeared?")
    assert t["task_classified"] == "change" and t["spec"]["target"] == "bare_soil"
    info = client.get(f"/api/sessions/{t['session_id']}").json()["session"]
    assert info["manifest"]["alignment"]["shift_corrected"] is True


def test_fusion_confidence_uses_cross_sensor_agreement(client):
    t = ask(client, sample_session(client, "optical_sar"), "Use both sensors to map the water. Do they agree?")
    assert "cross_sensor_agreement" in t["confidence"]["components"]
    assert {l["name"] for l in t["layers"]} == {"agreement", "optical_only", "sar_only"}


def test_follow_up_reuses_session_memory(client):
    sid = sample_session(client, "flood_single")
    ask(client, sid, "How much water is there?")
    t = ask(client, sid, "and where is it?")
    assert t["spec"]["follow_up"] and t["spec"]["target"] == "water"
    assert any(c["reused_from"] for c in t["tool_calls"])
    assert any(e["reused_from"] for e in t["ledger"])


def test_stress_test_hallucination_is_corrected(client):
    t = ask(client, sample_session(client, "flood_single"), "How much of the scene is water?", stress_test=True)
    assert t["adjudication_status"] == "corrected"
    wrong = [c for c in t["claims"] if c["corrected"]][0]
    assert f"{wrong['ledger_value']:.1f}" in t["answer"]
    assert str(wrong["value"]) not in t["answer"]


def test_change_question_on_single_image_requests_input(client):
    t = ask(client, sample_session(client, "sar_single"), "What changed since last year?")
    assert t["sufficiency"] == "request_input" and t["tool_calls"] == []
    assert t["confidence"]["band"] == "low"


def test_reports_and_ledger(client):
    t = ask(client, sample_session(client, "flood_single"), "Where are the largest water bodies? Give coordinates.")
    assert t["regions"] and t["regions"][0]["centroid_lonlat"]
    qid = t["query_id"]
    assert client.get(f"/api/report/{qid}.pdf").content[:4] == b"%PDF"
    gj = client.get(f"/api/report/{qid}.geojson").json()
    assert gj["type"] == "FeatureCollection" and gj["features"]
    assert client.get("/api/ledger/verify").json()["valid"]


def test_upload_pair_on_different_grids_is_reprojected(client, tmp_path):
    rng = np.random.default_rng(0)
    base = (rng.random((4, 200, 200)) * 3000 + 500).astype("uint16")
    a = write_tif(tmp_path / "a.tif", base, date="2024-01-01")
    # Same place, different grid: 20 m pixels, offset origin.
    b = write_tif(tmp_path / "b.tif", base[:, ::2, ::2].copy(), origin=(368_000.0, 2_900_000.0), gsd=20.0, date="2024-06-01")
    with open(a, "rb") as fa, open(b, "rb") as fb:
        r = client.post("/api/sessions", files={"image_a": ("a.tif", fa), "image_b": ("b.tif", fb)})
    assert r.status_code == 200, r.text
    m = r.json()["manifest"]
    assert m["config"] == "bi_temporal_pair" and m["alignment"]["method"] == "reprojected"


def test_non_overlapping_pair_rejected(client, tmp_path):
    img = (np.ones((4, 50, 50)) * 1000).astype("uint16")
    a = write_tif(tmp_path / "a.tif", img)
    b = write_tif(tmp_path / "b.tif", img, origin=(500_000.0, 2_900_000.0))
    with open(a, "rb") as fa, open(b, "rb") as fb:
        r = client.post("/api/sessions", files={"image_a": ("a.tif", fa), "image_b": ("b.tif", fb)})
    assert r.status_code == 422 and "overlap" in r.json()["detail"]


def test_large_scene_is_decimated_not_rejected(client, tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "analysis_max_dim", 256)
    img = np.ones((4, 1024, 1024), dtype="uint16") * 1000
    img[3] = 4000  # vegetated
    img[:, :256] = np.array([700, 600, 400, 200], dtype="uint16")[:, None, None]  # water strip
    p = write_tif(tmp_path / "big.tif", img)
    with open(p, "rb") as f:
        r = client.post("/api/sessions", files={"image_a": ("big.tif", f)})
    info = r.json()
    img_m = info["manifest"]["images"][0]
    assert img_m["analysis_width"] == 256 and img_m["analysis_gsd_m"] == 40.0
    t = ask(client, info["session_id"], "How many hectares of water?")
    water_ha = next(c["output"]["water_area_ha"] for c in t["tool_calls"] if c["tool"] == "spectral_index")
    assert abs(water_ha - 256 * 1024 * 0.01) / (256 * 1024 * 0.01) < 0.02  # area preserved under decimation


def test_one_shot_endpoint_still_works(client, samples_dir):
    with open(samples_dir / "s2_2024-07-08_post.tif", "rb") as f:
        r = client.post("/api/query", data={"question": "Describe the land cover"}, files={"image_a": ("x.tif", f)})
    assert r.status_code == 200 and r.json()["trace"]["task_classified"] == "caption"
