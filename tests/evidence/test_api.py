"""Module-2 API integration tests against an isolated app instance."""

import io
import json
from pathlib import Path
from uuid import uuid4

import pytest
from aurora_evidence.config import Settings
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture()
def app(tmp_path):
    from aurora_evidence.api.main import create_app

    settings = Settings(
        data_dir=tmp_path,
        tavily_api_key="",
        gptzero_api_key="",
        hive_api_key="",
        corpus_path=str(
            Path(__file__).resolve().parents[2] / "aurora-evidence" / "fixtures" / "demo_corpus.jsonl"
        ),
    )
    application = create_app(settings, embedded_worker=False)
    yield application


@pytest.fixture()
def client(app):
    with TestClient(app) as test_client:
        yield test_client


def bundle(mode="live", caption="Presiden menetapkan 30 September sebagai hari libur nasional"):
    return {
        "schema_version": "1.0.0",
        "case_id": str(uuid4()),
        "claim_revision": 1,
        "mode": mode,
        "created_at": "2026-09-15T00:00:00+00:00",
        "input": {"claim_text": caption, "language": "id", "images": [], "as_of": None},
        "analysis": None,
        "retrieval": None,
        "decision": None,
        "warnings": [],
        "extensions": {},
    }


def run(client, app, payload, key):
    response = client.post("/api/v1/retrieve", json=payload, headers={"Idempotency-Key": key})
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    for _ in range(50):
        app.state.worker.run_once(isolated=False)
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            return job
    raise AssertionError("job did not finish")


def test_health_and_ready(app, client):
    app.state.worker.heartbeat()  # seed the heartbeat like a live worker
    assert client.get("/health").json()["service"] == "aurora-evidence"
    ready = client.get("/ready").json()
    assert ready["status"] in ("ready", "degraded")
    assert any(c["provider"] == "local-corpus" and c["status"] == "ok" for c in ready["capabilities"])


def test_retrieve_live_produces_contract_valid_bundle(app, client):
    job = run(client, app, bundle(), "e2e-live-1")
    assert job["status"] == "succeeded", job["error"]
    result = job["result"]
    assert result["retrieval"]["evidence_list"], "corpus should produce evidence"
    for evidence in result["retrieval"]["evidence_list"]:
        # excerpt must be an exact substring of content text
        assert evidence["content"]["excerpt"] in evidence["content"]["text"]
        assert evidence["provenance"]["duplicate_cluster_id"]
        assert evidence["provenance"]["independence_group_id"]
    # GPTZero unconfigured -> one unavailable text signal, honest status
    signals = result["retrieval"]["forensic_signals"]
    assert any(s["status"] == "unavailable" and s["error_code"] == "unconfigured" for s in signals)


def test_retrieve_demo_mode_uses_labeled_fixtures_only(app, client):
    job = run(client, app, bundle(mode="demo", caption="Bidang ini berwarna merah"), "e2e-demo-1")
    assert job["status"] in ("succeeded", "partial")
    result = job["result"]
    assert result["retrieval"]["run"]["mode"] == "demo"
    providers = {p["provider"] for p in result["retrieval"]["provider_status"]}
    assert "demo-fixtures" in providers


def test_as_of_excludes_future_evidence(app, client):
    payload = bundle()
    payload["input"]["as_of"] = "2026-09-01T00:00:00+00:00"
    job = run(client, app, payload, "e2e-asof-1")
    result = job["result"]
    for evidence in result["retrieval"]["evidence_list"]:
        if evidence["source"]["published_at"]:
            assert evidence["source"]["published_at"] <= "2026-09-01T00:00:00+00:00"


def test_idempotency_same_key_same_payload(client):
    payload = bundle()
    first = client.post("/api/v1/retrieve", json=payload, headers={"Idempotency-Key": "idem-1"})
    second = client.post("/api/v1/retrieve", json=payload, headers={"Idempotency-Key": "idem-1"})
    assert first.json()["job_id"] == second.json()["job_id"]
    payload["input"]["claim_text"] = "berubah total"
    conflict = client.post("/api/v1/retrieve", json=payload, headers={"Idempotency-Key": "idem-1"})
    assert conflict.status_code == 409


def test_wrong_schema_version_rejected(client):
    payload = bundle()
    payload["schema_version"] = "2.0.0"
    response = client.post("/api/v1/retrieve", json=payload, headers={"Idempotency-Key": "x1"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "SCHEMA_VERSION_UNSUPPORTED"


def test_media_upload_requires_real_image(client):
    content = io.BytesIO()
    Image.new("RGB", (32, 32), "#cc2222").save(content, "PNG")
    good = client.post("/api/v1/media", files={"images": ("a.png", content.getvalue(), "image/png")})
    assert good.status_code == 200
    ref = good.json()["images"][0]
    assert ref["asset_id"].startswith("asset_")
    bad = client.post("/api/v1/media", files={"images": ("b.png", b"not an image", "image/png")})
    assert bad.status_code == 422


def test_export_csv_and_zip(app, client):
    job = run(client, app, bundle(), "e2e-export-1")
    case_id = job["result"]["case_id"]
    csv = client.get(f"/api/v1/cases/{case_id}/export?format=csv")
    assert csv.status_code == 200 and b"evidence_id" in csv.content
    zip_ = client.get(f"/api/v1/cases/{case_id}/export?format=zip")
    assert zip_.status_code == 200 and zip_.headers["content-type"] == "application/zip"


def test_import_roundtrip(app, client):
    job = run(client, app, bundle(), "e2e-import-1")
    export = job["result"]
    response = client.post(
        "/api/v1/import", files={"file": ("bundle.json", json.dumps(export).encode(), "application/json")}
    )
    assert response.status_code == 200, response.text
    assert response.json()["bundle"]["case_id"] == export["case_id"]
