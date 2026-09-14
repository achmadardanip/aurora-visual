"""SynthID Detector adapter: gateway contract, conservative mapping, and wiring.

SynthID Detector has no public documented API, so the adapter targets an
operator-configured gateway and is verified only against mocked transports.
These tests pin the strict response contract, the never-an-authenticity-proof
semantics of negative verdicts, config policy, and pipeline wiring.
"""

import io
import json
from datetime import datetime, timezone
from uuid import uuid4

import httpx
import pytest
from app.config import Settings
from app.models.contract import AuroraBundle
from app.models.db import Base, database
from app.services.media import MediaService
from app.services.pipeline import analyze
from aurora_visual.synthid import (
    PROVIDER,
    SynthIDClient,
    SynthIDError,
    normalize_detection,
    stage1_watermark,
    watermark_for_error,
)
from conftest import run
from PIL import Image


def test_unconfigured_client_never_sends_a_request():
    client = SynthIDClient("", "")
    with pytest.raises(SynthIDError) as excinfo:
        client.detect_watermark(_tmp_png())
    assert excinfo.value.code == "unconfigured"
    entry = watermark_for_error("unconfigured")
    assert entry["provider"] == "unconfigured"
    assert entry["assessment"] == "not_assessed"


def _tmp_png():
    import tempfile
    from pathlib import Path

    path = Path(tempfile.mkstemp(suffix=".png")[1])
    Image.new("RGB", (32, 24), "#2255aa").save(path)
    return path


def _handler(payload, seen=None):
    def handle(request):
        if seen is not None:
            seen.append(request)
        return httpx.Response(200, json=payload)

    return handle


def test_detected_verdict_is_normalized_and_never_leaks_the_key():
    seen = []
    client = SynthIDClient(
        "https://gateway.example.test/detect",
        "secret-key",
        transport=httpx.MockTransport(
            _handler({"synthid_detected": True, "confidence": 0.93, "model": "synthid/v1"}, seen)
        ),
    )
    result = client.detect_watermark(_tmp_png())
    assert result["synthid_detected"] is True
    assert result["confidence"] == 0.93
    assert result["model"] == "synthid/v1"
    assert seen[0].headers["Authorization"] == "Bearer secret-key"
    entry = stage1_watermark(result)
    assert entry["provider"] == PROVIDER
    assert entry["status"] == "ok"
    assert entry["assessment"] == "synthid_watermark_detected"
    assert entry["score"] == 0.93
    assert "secret-key" not in json.dumps(entry)


def test_negative_verdict_is_not_authenticity_evidence():
    entry = stage1_watermark({"synthid_detected": False, "confidence": 0.88, "model": None, "message": None})
    assert entry["status"] == "inconclusive"
    assert entry["assessment"] == "no_synthid_watermark"
    assert "bukan bukti asal kamera" in entry["message"].lower()


def test_null_verdict_maps_to_uncertain():
    entry = stage1_watermark(
        {"synthid_detected": None, "confidence": None, "model": None, "message": "queue"}
    )
    assert entry["status"] == "inconclusive"
    assert entry["assessment"] == "uncertain"


@pytest.mark.parametrize(
    "payload",
    [
        {"confidence": 0.5},
        {"synthid_detected": "yes", "confidence": None},
        {"synthid_detected": True, "confidence": 1.5},
        {"synthid_detected": True, "confidence": "high"},
        {"synthid_detected": True, "extra": 1},
        {"synthid_detected": True, "model": ["synthid"]},
    ],
)
def test_malformed_payloads_are_rejected(payload):
    with pytest.raises(SynthIDError, match="malformed_response"):
        normalize_detection(payload)


def test_bounded_strings_in_verdict():
    result = normalize_detection({"synthid_detected": True, "model": "m" * 500, "message": "x" * 1000})
    assert len(result["model"]) == 120
    assert len(result["message"]) == 300


@pytest.mark.parametrize(
    ("status_code", "code"),
    [(429, "rate_limited"), (500, "http_error_500"), (403, "http_error_403")],
)
def test_http_failures_map_to_error_codes(status_code, code):
    client = SynthIDClient(
        "https://gateway.example.test/detect",
        "secret-key",
        transport=httpx.MockTransport(lambda request: httpx.Response(status_code, text="down")),
    )
    with pytest.raises(SynthIDError) as excinfo:
        client.detect_watermark(_tmp_png())
    assert excinfo.value.code == code
    entry = watermark_for_error(code)
    assert entry["status"] in {"rate_limited", "failed"}
    assert entry["assessment"] == "not_assessed"


def test_oversized_response_is_rejected():
    client = SynthIDClient(
        "https://gateway.example.test/detect",
        "secret-key",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"x" * 1_000_001)),
    )
    with pytest.raises(SynthIDError) as excinfo:
        client.detect_watermark(_tmp_png())
    assert excinfo.value.code in {"response_too_large", "malformed_response"}


def test_redirect_is_rejected():
    client = SynthIDClient(
        "https://gateway.example.test/detect",
        "secret-key",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(307, headers={"Location": "https://elsewhere.test/"})
        ),
    )
    with pytest.raises(SynthIDError) as excinfo:
        client.detect_watermark(_tmp_png())
    assert excinfo.value.code == "redirect_rejected"


def test_prepare_requires_gateway_when_synthid_enabled(tmp_path):
    with pytest.raises(ValueError, match="synthid_endpoint"):
        Settings(data_dir=tmp_path, synthid_enabled=True, synthid_endpoint="", synthid_api_key="").prepare()
    Settings(
        data_dir=tmp_path,
        synthid_enabled=True,
        synthid_endpoint="https://gateway.example.test/detect",
        synthid_api_key="key",
    ).prepare()
    with pytest.raises(ValueError, match="URL http"):
        Settings(
            data_dir=tmp_path,
            synthid_enabled=True,
            synthid_endpoint="ftp://gateway.example.test",
            synthid_api_key="key",
        ).prepare()


def test_put_settings_cannot_enable_synthid_without_gateway(client):
    response = client.put("/api/v1/settings", json={"synthid_enabled": True})
    assert response.status_code == 422
    assert "synthid" in response.json()["error"]["message"]
    caps = {c["provider"]: c["status"] for c in client.get("/ready").json()["capabilities"]}
    assert caps["synthid-detector"] == "unconfigured"
    response = client.put(
        "/api/v1/settings",
        json={
            "synthid_enabled": True,
            "synthid_endpoint": "https://gateway.example.test/detect",
            "synthid_api_key": "gateway-key",
        },
    )
    assert response.status_code == 200, response.text
    caps = {c["provider"]: c["status"] for c in client.get("/ready").json()["capabilities"]}
    assert caps["synthid-detector"] == "ok"
    values = client.get("/api/v1/settings").json()["values"]
    assert values["synthid_api_key"] == "__CONFIGURED__"


class StubSynthIDClient:
    """Deterministic gateway stand-in returning a positive watermark verdict."""

    def __init__(self, endpoint, api_key, timeout=45.0):
        assert endpoint and api_key

    @property
    def configured(self):
        return True

    def detect_watermark(self, path):
        assert path.is_file()
        return {
            "synthid_detected": True,
            "confidence": 0.97,
            "model": "synthid-fixture/v1",
            "message": None,
            "duration_ms": 12.5,
        }


class FailingSynthIDClient(StubSynthIDClient):
    def detect_watermark(self, path):
        raise SynthIDError("network_error")


def live_bundle(ref, options):
    return AuroraBundle(
        schema_version="1.0.0",
        case_id=str(uuid4()),
        claim_revision=1,
        mode="live",
        created_at=datetime.now(timezone.utc),
        input={"claim_text": "Bidang ini berwarna biru", "language": "id", "images": [ref], "as_of": None},
        analysis=None,
        retrieval=None,
        decision=None,
        warnings=[],
        extensions={"aurora_visual": {"options": options}},
    )


def pipeline_fixture(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        synthid_enabled=True,
        synthid_endpoint="https://gateway.example.test/detect",
        synthid_api_key="gateway-key",
    )
    for name in ("media", "derived", "artifacts", "cache"):
        (settings.data_dir / name).mkdir(parents=True, exist_ok=True)
    engine, session = database(tmp_path)
    Base.metadata.create_all(engine)
    media = MediaService(settings, session)
    stream = io.BytesIO()
    Image.new("RGB", (64, 48), "#2244cc").save(stream, "PNG")
    ref = media.upload(stream.getvalue(), "local")
    return settings, media, ref


def test_opted_in_synthid_run_reports_watermark_verdict(tmp_path, monkeypatch):
    settings, media, ref = pipeline_fixture(tmp_path)
    monkeypatch.setattr("app.services.pipeline.SynthIDClient", StubSynthIDClient)
    result = analyze(live_bundle(ref, {"synthid_detector": True}), str(uuid4()), settings, media)
    synthid = result.extensions["aurora_visual"]["synthid"]
    assert synthid["stage1"] == {"provider": PROVIDER, "status": "ok"}
    assert synthid["egress"] == {"original_media_stage1": True}
    watermark = result.extensions["aurora_visual"]["screening"]["provenance"]["watermark"]
    assert watermark["assessment"] == "synthid_watermark_detected"
    assert watermark["score"] == 0.97
    # A positive watermark verdict is an origin signal only: the run still
    # completes without HIVE_/SYNTHID_ failure warnings.
    codes = [w.code for w in result.analysis.run.warnings]
    assert not any(code.startswith("SYNTHID_") for code in codes)
    assert result.analysis.run.status == "completed"
    assert result.analysis.visual_assessments


def test_opted_in_synthid_failure_marks_run_partial(tmp_path, monkeypatch):
    settings, media, ref = pipeline_fixture(tmp_path)
    monkeypatch.setattr("app.services.pipeline.SynthIDClient", FailingSynthIDClient)
    result = analyze(live_bundle(ref, {"synthid_detector": True}), str(uuid4()), settings, media)
    watermark = result.extensions["aurora_visual"]["screening"]["provenance"]["watermark"]
    assert watermark["status"] == "failed"
    assert watermark["assessment"] == "not_assessed"
    assert result.extensions["aurora_visual"]["synthid"]["stage1"]["status"] == "network_error"
    codes = [w.code for w in result.analysis.run.warnings]
    assert "SYNTHID_NETWORK_ERROR" in codes
    assert result.analysis.run.status == "partial"


def test_opt_in_without_gateway_configuration_warns(tmp_path, monkeypatch):
    settings, media, ref = pipeline_fixture(tmp_path)
    settings.synthid_enabled = False
    result = analyze(live_bundle(ref, {"synthid_detector": True}), str(uuid4()), settings, media)
    watermark = result.extensions["aurora_visual"]["screening"]["provenance"]["watermark"]
    assert watermark["status"] == "unconfigured"
    assert watermark["provider"] == "unconfigured"
    assert result.extensions["aurora_visual"]["synthid"]["stage1"]["status"] == "unconfigured"
    codes = [w.code for w in result.analysis.run.warnings]
    assert "SYNTHID_UNCONFIGURED" in codes
    assert result.analysis.run.status == "partial"


def test_without_opt_in_no_synthid_extension_and_local_watermark_slot(tmp_path, monkeypatch):
    settings, media, ref = pipeline_fixture(tmp_path)
    requested = []

    class TrackingClient(StubSynthIDClient):
        def detect_watermark(self, path):
            requested.append(path)
            return super().detect_watermark(path)

    monkeypatch.setattr("app.services.pipeline.SynthIDClient", TrackingClient)
    result = analyze(live_bundle(ref, {}), str(uuid4()), settings, media)
    assert requested == []
    assert result.extensions["aurora_visual"]["synthid"] is None
    assert (
        result.extensions["aurora_visual"]["screening"]["provenance"]["watermark"]["status"] == "unavailable"
    )
    assert result.analysis.run.status == "completed"


def test_demo_mode_never_uploads_to_the_gateway(tmp_path, monkeypatch):
    settings, media, ref = pipeline_fixture(tmp_path)
    requested = []

    class TrackingClient(StubSynthIDClient):
        def detect_watermark(self, path):
            requested.append(path)
            return super().detect_watermark(path)

    monkeypatch.setattr("app.services.pipeline.SynthIDClient", TrackingClient)
    bundle = live_bundle(ref, {"synthid_detector": True})
    demo_bundle = bundle.model_copy(update={"mode": "demo"})
    result = analyze(demo_bundle, str(uuid4()), settings, media)
    assert requested == []
    assert result.extensions["aurora_visual"]["synthid"] is None


def test_analyze_accepts_synthid_option_in_live_mode(client, app, demo):
    demo["mode"] = "live"
    demo["warnings"] = []
    demo["extensions"] = {"aurora_visual": {"options": {"synthid_detector": True}}}
    result, _ = run(client, app, demo)
    extension = result["extensions"]["aurora_visual"]
    # Unconfigured gateway on the test app: recorded, warned, run partial.
    assert extension["synthid"]["stage1"]["status"] == "unconfigured"
    assert extension["synthid"]["egress"] == {"original_media_stage1": True}
    assert extension["screening"]["provenance"]["watermark"]["status"] == "unconfigured"
    codes = [w["code"] for w in result["analysis"]["run"]["warnings"]]
    assert "SYNTHID_UNCONFIGURED" in codes


def test_analyze_rejects_synthid_option_in_demo_mode(client, app, demo):
    demo["extensions"] = {"aurora_visual": {"options": {"synthid_detector": True}}}
    response = client.post("/api/v1/analyze", json=demo, headers={"Idempotency-Key": "synthid-demo"})
    assert response.status_code == 422


def test_analyze_rejects_non_boolean_synthid_option(client, app, demo):
    demo["mode"] = "live"
    demo["extensions"] = {"aurora_visual": {"options": {"synthid_detector": "yes"}}}
    response = client.post("/api/v1/analyze", json=demo, headers={"Idempotency-Key": "synthid-bad"})
    assert response.status_code == 422
