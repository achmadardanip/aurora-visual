import copy
import io
import json
import stat
import zipfile
from uuid import uuid4

import pytest
from app.api.main import create_app
from app.config import Settings
from app.models.db import Job
from conftest import png, run
from fastapi.testclient import TestClient
from PIL import Image


def test_upload_real_type_size_and_exif(client):
    for payload, ctype in [(b"not an image", "image/png"), (png(), "image/jpeg")]:
        r = client.post("/api/v1/media", files={"images": ("bad.png", payload, ctype)})
        assert r.status_code == 422
    image = Image.new("RGB", (24, 12), "red")
    exif = Image.Exif()
    exif[274] = 6
    stream = io.BytesIO()
    image.save(stream, "JPEG", exif=exif)
    response = client.post("/api/v1/media", files={"images": ("photo.jpg", stream.getvalue(), "image/jpeg")})
    assert response.status_code == 200, response.text
    ref = response.json()["images"][0]
    assert (ref["width"], ref["height"]) == (12, 24)
    from app.models.contract import sha

    assert ref["sha256"] == sha(stream.getvalue())
    assert client.get("/api/v1/media/" + ref["asset_id"]).content == stream.getvalue()


def test_upload_large(client, app):
    app.state.settings.max_upload = 20
    assert (
        client.post("/api/v1/media", files={"images": ("large.png", png(), "image/png")}).status_code == 413
    )


def test_multi_upload_batch_and_thumbnail(client):
    files = [
        ("images", ("one.png", png("red"), "image/png")),
        ("images", ("two.png", png("green", (80, 40)), "image/png")),
        ("images", ("three.png", png("blue"), "image/png")),
    ]
    response = client.post("/api/v1/media", files=files)
    assert response.status_code == 200, response.text
    refs = response.json()["images"]
    assert len(refs) == 3
    assert len({r["asset_id"] for r in refs}) == 3
    for ref in refs:
        thumb = client.get(f"/api/v1/media/{ref['asset_id']}/thumbnail")
        assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/png"
        with Image.open(io.BytesIO(thumb.content)) as thumbnail:
            assert max(thumbnail.size) <= 320


def test_multi_upload_count_limit(client, app):
    app.state.settings.max_images = 3
    files = [("images", (f"{i}.png", png(), "image/png")) for i in range(4)]
    assert client.post("/api/v1/media", files=files).status_code == 413


@pytest.mark.parametrize(
    "fixture,expected",
    [("supported", "Supported"), ("contradicted", "Contradicted"), ("unobservable", "Unobservable")],
)
def test_fixture_semantics(client, app, fixture, expected):
    b = client.post("/api/v1/demo/" + fixture).json()
    result, _ = run(client, app, b)
    assert all(a["visual_status"] == expected for a in result["analysis"]["visual_assessments"])
    assert all(a["inference_kind"] == "fixture" for a in result["analysis"]["visual_assessments"])
    if fixture == "unobservable":
        roles = {a["role"] for a in result["analysis"]["atomic_claims"]}
        assert {"time", "location", "actor"} <= roles


def test_origin_screen_stays_module_local_and_preserves_visual_semantics(client, app, demo):
    result, _ = run(client, app, demo)
    screening = result["extensions"]["aurora_visual"]["screening"]
    assert screening["version"] == "origin-screen-v2"
    assert screening["target"]["asset_id"] == result["input"]["images"][0]["asset_id"]
    assert screening["decision"]["does_not_affect_visual_assessment"] is True
    assert screening["decision"]["does_not_decide_claim_truth"] is True
    # Demo runs never use external detectors: reported as not selected, not as
    # a server configuration failure.
    assert all(
        detector["status"] == "not_selected" and "mode Live" in detector["message"]
        for detector in screening["detectors"]
    )
    assert result["retrieval"] is None and result["decision"] is None
    assert all(a["visual_status"] == "Supported" for a in result["analysis"]["visual_assessments"])


def test_ready_reports_origin_screen_and_unconfigured_mafindo(client, app):
    app.state.worker.heartbeat()
    response = client.get("/ready")
    assert response.status_code == 200
    capabilities = {item["provider"]: item for item in response.json()["capabilities"]}
    assert capabilities["origin-screen-v1"]["status"] == "ok"
    assert capabilities["mafindo-v1"]["status"] == "unconfigured"
    assert client.get("/api/v1/providers/mafindo/latest").json()["status"] == "unconfigured"


def test_live_real_pipeline_and_no_false_contradiction(client, app, demo):
    demo["mode"] = "live"
    demo["warnings"] = []
    result, _ = run(client, app, demo)
    assert result["analysis"]["visual_assessments"][0]["inference_kind"] == "heuristic"
    assert result["analysis"]["visual_assessments"][0]["probabilities"] is None
    unrelated = copy.deepcopy(demo)
    unrelated["case_id"] = str(uuid4())
    unrelated["input"]["claim_text"] = "Dua mahasiswa membawa tas merah di Monas pada 2026"
    out, _ = run(client, app, unrelated, "another")
    assert all(a["visual_status"] == "Unobservable" for a in out["analysis"]["visual_assessments"])


def test_idempotency_jcs_conflict_and_restart(client, app, demo):
    r = client.post("/api/v1/analyze", json=demo, headers={"Idempotency-Key": "same"})
    data = copy.deepcopy(demo)
    reordered = {k: data[k] for k in reversed(data)}
    r2 = client.post("/api/v1/analyze", json=reordered, headers={"Idempotency-Key": "same"})
    assert r.json()["job_id"] == r2.json()["job_id"]
    data["input"]["images"][0]["uri"] = "media/other.png"
    assert client.post("/api/v1/analyze", json=data, headers={"Idempotency-Key": "same"}).status_code == 409
    restarted = create_app(app.state.settings, embedded_worker=False)
    with TestClient(restarted) as c:
        r3 = c.post("/api/v1/analyze", json=demo, headers={"Idempotency-Key": "same"})
        assert r3.json()["job_id"] == r.json()["job_id"]
        restarted.state.worker.run_once(isolated=False)
        assert c.get("/api/v1/jobs/" + r.json()["job_id"]).json()["result"]
        assert c.get("/api/v1/cases").json()[0]["case_id"] == demo["case_id"]


def test_expired_lease_recovery(client, app, demo):
    r = client.post("/api/v1/analyze", json=demo, headers={"Idempotency-Key": "recovery"}).json()
    with app.state.session.begin() as s:
        job = s.get(Job, r["job_id"])
        job.status = "running"
        job.lease_until = 0
        job.attempts = 1
    app.state.worker.run_once(isolated=False)
    result = client.get("/api/v1/jobs/" + r["job_id"]).json()
    assert result["status"] in ("succeeded", "partial") and result["attempts"] == 2


def test_correction_stale_and_original_preserved(client, app, demo):
    result, _ = run(client, app, demo)
    atoms = copy.deepcopy(result["analysis"]["atomic_claims"])
    atoms[0]["statement"] = "Bidang ini berwarna biru"
    atoms[0]["object"] = "biru"
    data = {
        "atomic_claims": atoms,
        "expected_atom_set_id": result["analysis"]["atom_set_id"],
        "reason": "Warna klaim direvisi untuk audit",
    }
    url = "/api/v1/cases/" + demo["case_id"] + "/atoms"
    changed = client.patch(url, json=data)
    assert changed.status_code == 200, changed.text
    b = changed.json()
    assert b["input"] == result["input"] and b["claim_revision"] == result["claim_revision"]
    assert (
        b["analysis"]["atom_set_id"] != result["analysis"]["atom_set_id"]
        and not b["analysis"]["visual_assessments"]
    )
    assert client.patch(url, json=data).status_code == 409
    assert (
        client.post("/api/v1/analyze", json=result, headers={"Idempotency-Key": "old-atoms"}).status_code
        == 409
    )
    rerun, _ = run(client, app, b, "corrected")
    assert rerun["analysis"]["visual_assessments"][0]["visual_status"] == "Contradicted"


def test_revision_and_queued_stale_result(client, app, demo):
    job = client.post("/api/v1/analyze", json=demo, headers={"Idempotency-Key": "old"}).json()
    newer = copy.deepcopy(demo)
    newer["claim_revision"] = 2
    newer["input"]["claim_text"] = "Bidang ini berwarna biru"
    response = client.post("/api/v1/analyze", json=newer, headers={"Idempotency-Key": "new"})
    assert response.status_code == 202
    app.state.worker.run_once(isolated=False)
    assert client.get("/api/v1/jobs/" + job["job_id"]).json()["error"]["code"] == "STALE_RESULT"
    app.state.worker.run_once(isolated=False)
    assert client.get("/api/v1/cases/" + demo["case_id"]).json()["claim_revision"] == 2


def test_export_import_portable(client, app, demo, tmp_path):
    result, _ = run(client, app, demo)
    url = "/api/v1/cases/" + demo["case_id"] + "/export?format="
    for fmt in ("json", "csv", "overlay"):
        assert client.get(url + fmt).status_code == 200
    data = client.get(url + "zip").content
    receiver = create_app(Settings(data_dir=tmp_path / "receiver"), embedded_worker=False)
    with TestClient(receiver) as other:
        imported = other.post("/api/v1/import", files={"file": ("case.zip", data, "application/zip")})
        assert imported.status_code == 200, imported.text
        b = imported.json()["bundle"]
        assert len(imported.json()["assets_available"]) == len(b["input"]["images"])
        assert b["analysis"]["atom_set_id"] == result["analysis"]["atom_set_id"]
        r, _ = run(other, receiver, b, "imported-analysis")
        assert [m["sha256"] for m in r["input"]["images"]] == [m["sha256"] for m in demo["input"]["images"]]


@pytest.mark.parametrize("attack", ["traversal", "absolute", "symlink", "duplicate", "bomb", "hash"])
def test_unsafe_import(client, demo, attack):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("bundle.json", json.dumps(demo))
        if attack == "traversal":
            z.writestr("../secret", "x")
        if attack == "absolute":
            z.writestr("/tmp/x", "x")
        if attack == "symlink":
            info = zipfile.ZipInfo("media/link")
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(info, "/tmp")
        if attack == "duplicate":
            z.writestr("bundle.json", "{}")
        if attack == "bomb":
            z.writestr("media/bomb", "0" * 2_000_000)
        if attack == "hash":
            z.writestr(demo["input"]["images"][0]["uri"], b"wrong")
    assert (
        client.post(
            "/api/v1/import", files={"file": ("bad.zip", stream.getvalue(), "application/zip")}
        ).status_code
        == 422
    )


def test_json_import_without_media(client, demo):
    demo["case_id"] = str(uuid4())
    demo["input"]["images"][0]["sha256"] = "f" * 64
    demo["input"]["images"][0]["asset_id"] = "asset_" + "f" * 64
    response = client.post(
        "/api/v1/import", files={"file": ("bundle.json", json.dumps(demo).encode(), "application/json")}
    )
    assert response.status_code == 200 and response.json()["assets_available"] == []
    assert (
        client.post("/api/v1/analyze", json=demo, headers={"Idempotency-Key": "missing"}).status_code == 422
    )


def test_missing_model_failure(client, app, demo, monkeypatch):
    monkeypatch.delenv("AURORA_OPENCLIP_PRETRAINED", raising=False)
    demo["mode"] = "live"
    demo["extensions"] = {"aurora_visual": {"options": {"backbone": "openclip"}}}
    r = client.post("/api/v1/analyze", json=demo, headers={"Idempotency-Key": "missing-model"}).json()
    app.state.worker.run_once(isolated=False)
    job = client.get("/api/v1/jobs/" + r["job_id"]).json()
    assert job["status"] == "failed" and job["error"]["code"] == "MODEL_UNAVAILABLE"


def test_cancel_and_readiness(client, app, demo):
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    app.state.worker.heartbeat()
    assert client.get("/ready").status_code == 200
    j = client.post("/api/v1/analyze", json=demo, headers={"Idempotency-Key": "cancel"}).json()
    assert client.post("/api/v1/jobs/" + j["job_id"] + "/cancel").json()["status"] == "failed"


def test_auth_origin_host_and_security_headers(tmp_path, monkeypatch):
    monkeypatch.setenv("AURORA_CORS", "https://aurora.example.org")
    settings = Settings(
        data_dir=tmp_path / "secure",
        public=True,
        token="long-secret-token-for-production-test",
        allowed_hosts=["aurora.example.org"],
    )
    secure = create_app(settings, embedded_worker=False)
    auth = {"Authorization": "Bearer " + settings.token, "Host": "aurora.example.org"}
    with TestClient(secure) as client:
        assert client.get("/api/v1/cases", headers={"Host": "aurora.example.org"}).status_code == 401
        unauthorized = client.get("/api/v1/cases", headers={"Host": "aurora.example.org"})
        assert unauthorized.headers["cache-control"] == "no-store"
        assert unauthorized.headers["strict-transport-security"].startswith("max-age=")
        response = client.get("/api/v1/cases", headers=auth)
        assert response.status_code == 200
        assert response.headers["strict-transport-security"].startswith("max-age=")
        assert response.headers["referrer-policy"] == "no-referrer"
        assert response.headers["x-frame-options"] == "DENY"
        assert client.get("/api/v1/cases", headers={**auth, "Host": "evil.example"}).status_code == 400
        assert client.get("/api/v1/providers/mafindo/latest", headers=auth).status_code == 404
        secure.state.worker.heartbeat()
        ready = client.get("/ready", headers=auth)
        capabilities = {item["provider"]: item for item in ready.json()["capabilities"]}
        assert ready.json()["status"] == "degraded"
        assert capabilities["mafindo-v1"]["status"] == "disabled"
        assert (
            client.post(
                "/api/v1/demo/supported",
                headers={**auth, "Origin": "https://evil.example"},
            ).status_code
            == 403
        )


def test_public_configuration_rejects_weak_auth_and_wildcards(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="at least 32"):
        Settings(
            data_dir=tmp_path / "weak",
            public=True,
            token="short",
            allowed_hosts=["aurora.example.org"],
        ).prepare()
    with pytest.raises(ValueError, match="explicit host"):
        Settings(data_dir=tmp_path / "wild-host", allowed_hosts=["*.example.org"]).prepare()
    with pytest.raises(ValueError, match="deployment host"):
        Settings(
            data_dir=tmp_path / "local-host",
            public=True,
            token="test-token-with-at-least-32-characters",
            allowed_hosts=["LOCALHOST."],
        ).prepare()
    monkeypatch.setenv("AURORA_CORS", "*")
    with pytest.raises(ValueError, match="explicit origins"):
        create_app(Settings(data_dir=tmp_path / "wild-cors"), embedded_worker=False)


def test_chunked_body_limit(client):
    content = iter([b'{"payload":"', b"x" * (2 * 1024**2), b'"}'])
    response = client.post(
        "/api/v1/analyze",
        content=content,
        headers={"Content-Type": "application/json", "Idempotency-Key": "chunked"},
    )
    assert response.status_code == 413


def test_preflight_for_authenticated_frontend(tmp_path, monkeypatch):
    monkeypatch.setenv("AURORA_CORS", "http://127.0.0.1:5171")
    secure = create_app(
        Settings(
            data_dir=tmp_path / "cors",
            public=True,
            token="test-token-with-at-least-32-characters",
            allowed_hosts=["aurora.test"],
        ),
        embedded_worker=False,
    )
    with TestClient(secure, base_url="http://aurora.test") as client:
        response = client.options(
            "/api/v1/cases",
            headers={
                "Origin": "http://127.0.0.1:5171",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "authorization",
            },
        )
        assert response.status_code == 200


def test_idempotency_survives_asset_unavailability(client, app, demo):
    response = client.post("/api/v1/analyze", json=demo, headers={"Idempotency-Key": "media-removed"})
    path, _, _ = app.state.media.resolve(demo["input"]["images"][0]["asset_id"])
    path.unlink()
    replay = client.post("/api/v1/analyze", json=demo, headers={"Idempotency-Key": "media-removed"})
    assert replay.status_code == 202 and replay.json()["job_id"] == response.json()["job_id"]
