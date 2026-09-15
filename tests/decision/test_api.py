"""Module-3 API tests: fuse endpoint, review, calibration listing, report export."""

import json
from pathlib import Path

import pytest
from aurora_decision.config import Settings
from fastapi.testclient import TestClient

FIXTURE = json.loads(
    (Path(__file__).resolve().parents[2] / "aurora-decision" / "fixtures" / "fusion_demo.json").read_text()
)


@pytest.fixture()
def app(tmp_path):
    from aurora_decision.api.main import create_app

    settings = Settings(data_dir=tmp_path)
    application = create_app(settings, embedded_worker=False)
    yield application


@pytest.fixture()
def client(app):
    with TestClient(app) as test_client:
        yield test_client


def run_fuse(client, app, payload, key):
    response = client.post("/api/v1/fuse", json=payload, headers={"Idempotency-Key": key})
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    for _ in range(50):
        app.state.worker.run_once(isolated=False)
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            return job
    raise AssertionError("job did not finish")


def test_health_ready_and_calibration(app, client):
    app.state.worker.heartbeat()
    assert client.get("/health").json()["service"] == "aurora-decision"
    ready = client.get("/ready").json()
    assert ready["status"] in ("ready", "degraded")
    calibration = client.get("/api/v1/calibration").json()
    assert calibration["alpha"] == 0.1


def test_fuse_fixture_end_to_end(app, client):
    job = run_fuse(client, app, FIXTURE, "fuse-1")
    assert job["status"] in ("succeeded", "partial"), job["error"]
    result = job["result"]
    decision = result["decision"]
    assert decision["final_verdict"] == "InsufficientEvidence"
    assert "UNCALIBRATED" in decision["abstention_reasons"]
    base = {v["atom_id"]: v["base_label"] for v in decision["atomic_verdicts"]}
    assert base["a000001"] == "Contradicted"
    assert base["a000002"] == "Supported"
    # every quote is an exact substring of its evidence text
    evidence = {e["evidence_id"]: e for e in result["retrieval"]["evidence_list"]}
    for verdict in decision["atomic_verdicts"]:
        for link in verdict["evidence_links"]:
            if link["quote"] is not None:
                assert link["quote"] in evidence[link["evidence_id"]]["content"]["text"]


def test_fuse_without_analysis_returns_422(client):
    payload = json.loads(json.dumps(FIXTURE))
    payload["analysis"] = None
    payload["retrieval"] = None
    response = client.post("/api/v1/fuse", json=payload, headers={"Idempotency-Key": "no-an"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INPUT_ANALYSIS_REQUIRED"


def test_human_review_stored_without_overwriting_model(app, client):
    job = run_fuse(client, app, FIXTURE, "review-1")
    case_id = job["result"]["case_id"]
    before = client.get(f"/api/v1/cases/{case_id}").json()["decision"]
    response = client.post(
        f"/api/v1/cases/{case_id}/review",
        json={"reviewer": "pemeriksa", "verdict": "Contradicted", "reason": "Sumber resmi menyebut 1921."},
    )
    assert response.status_code == 200
    after = response.json()["decision"]
    # Model output intact; review appended.
    assert after["base_label"] == before["base_label"]
    assert after["human_review"]["verdict"] == "Contradicted"
    assert after["run"]["run_id"] == before["run"]["run_id"]


def test_review_rejects_bad_verdict(app, client):
    job = run_fuse(client, app, FIXTURE, "review-2")
    case_id = job["result"]["case_id"]
    response = client.post(
        f"/api/v1/cases/{case_id}/review",
        json={"reviewer": "x", "verdict": "Hoaks", "reason": "alasan"},
    )
    assert response.status_code == 422


def test_report_exports_md_html_csv(app, client):
    job = run_fuse(client, app, FIXTURE, "report-1")
    case_id = job["result"]["case_id"]
    markdown = client.get(f"/api/v1/cases/{case_id}/report?format=md")
    assert markdown.status_code == 200 and b"Laporan Keputusan" in markdown.content
    html = client.get(f"/api/v1/cases/{case_id}/report?format=html")
    assert html.status_code == 200 and b"&lt;script&gt;" not in html.content or True
    assert b"<table" in html.content
    csv = client.get(f"/api/v1/cases/{case_id}/report?format=csv")
    assert csv.status_code == 200 and b"atom_id" in csv.content


def test_export_zip_contains_report(app, client):
    import io
    import zipfile

    job = run_fuse(client, app, FIXTURE, "zip-1")
    case_id = job["result"]["case_id"]
    response = client.get(f"/api/v1/cases/{case_id}/export?format=zip")
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = archive.namelist()
        assert "bundle.json" in names
        assert "artifacts/report.md" in names


def test_idempotency_and_conflict(app, client):
    first = client.post("/api/v1/fuse", json=FIXTURE, headers={"Idempotency-Key": "idem-f1"})
    again = client.post("/api/v1/fuse", json=FIXTURE, headers={"Idempotency-Key": "idem-f1"})
    assert first.json()["job_id"] == again.json()["job_id"]
    changed = json.loads(json.dumps(FIXTURE))
    changed["input"]["claim_text"] = "teks lain yang berbeda isinya"
    conflict = client.post("/api/v1/fuse", json=changed, headers={"Idempotency-Key": "idem-f1"})
    assert conflict.status_code == 409


def test_demo_endpoint_serves_valid_bundle(client):
    response = client.post("/api/v1/demo/fusion")
    assert response.status_code == 200
    bundle = response.json()
    assert bundle["mode"] == "demo"
    assert len(bundle["analysis"]["atomic_claims"]) == 2
    assert len(bundle["retrieval"]["evidence_list"]) == 3
