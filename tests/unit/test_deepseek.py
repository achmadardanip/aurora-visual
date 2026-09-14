"""DeepSeek Flash provider: client contract, config policy, and pipeline wiring.

DeepSeek Flash (V4.1) is documented at api-docs.deepseek.com as vision-capable
with JSON output; there is no API key on this machine, so the provider is
verified through mocked HTTP transports plus policy and pipeline tests only.
"""

import io
import json
from datetime import datetime, timezone
from uuid import uuid4

import httpx
import pytest
from app.api.main import create_app
from app.config import Settings
from app.models.contract import AuroraBundle
from app.models.db import Base, database
from app.services.media import MediaService
from app.services.pipeline import analyze
from aurora_visual.deepseek import (
    DeepSeekAtomizer,
    DeepSeekClient,
    DeepSeekError,
)
from aurora_visual.deepseek import test_connection as deepseek_test_connection
from PIL import Image

CAPTION = "Bidang ini berwarna merah"


def atom_payload():
    return {
        "atoms": [
            {
                "atom_id": "a000001",
                "statement": "Bidang ini berwarna merah",
                "role": "attribute",
                "subject": "Bidang",
                "predicate": "berwarna",
                "object": "merah",
                "qualifiers": {
                    "negated": False,
                    "quantity": None,
                    "time": None,
                    "location": None,
                },
                "spans": [{"start": 0, "end": 24}],
                "depends_on": [],
                "check_worthiness": 0.8,
                "parser_confidence": None,
            }
        ]
    }


def observation_payload():
    return {
        "observations": [
            {
                "atom_id": "a000001",
                "visibility": "visible",
                "relation": "supports",
                "observation": "Bidang terlihat berwarna merah",
                "alternative": None,
                "confidence": 0.9,
                "bbox": [0.1, 0.1, 0.5, 0.5],
            }
        ]
    }


def _chat_response(content):
    return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(content)}}]})


def _tmp_png():
    import tempfile
    from pathlib import Path

    path = Path(tempfile.mkstemp(suffix=".png")[1])
    Image.new("RGB", (32, 24), "#cc2222").save(path)
    return path


def test_unconfigured_client_never_requests():
    client = DeepSeekClient("")
    with pytest.raises(DeepSeekError) as excinfo:
        client.parse_atoms(CAPTION, "id")
    assert excinfo.value.code == "unconfigured"


def test_parse_atoms_uses_json_mode_without_thinking():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.read())
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        return _chat_response(atom_payload())

    client = DeepSeekClient(
        "test-key",
        base_url="https://api.deepseek.com/",
        model="deepseek-flash",
        transport=httpx.MockTransport(handler),
    )
    atoms = DeepSeekAtomizer(client).parse(CAPTION, "id").atoms
    assert seen["url"] == "https://api.deepseek.com/chat/completions"
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["model"] == "deepseek-flash"
    assert seen["body"]["response_format"] == {"type": "json_object"}
    # Default: the optional thinking field is omitted for gateway compatibility.
    assert "thinking" not in seen["body"]
    assert seen["body"]["temperature"] == 0
    assert "json" in seen["body"]["messages"][0]["content"].lower()
    assert atoms[0].statement == "Bidang ini berwarna merah"


def test_thinking_field_is_opt_in():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.read())
        return _chat_response(atom_payload())

    client = DeepSeekClient(
        "test-key",
        transport=httpx.MockTransport(handler),
        disable_thinking=True,
    )
    DeepSeekAtomizer(client).parse(CAPTION, "id")
    assert seen["body"]["thinking"] == {"type": "disabled"}


def test_gateway_error_body_with_http_200_is_reported():
    client = DeepSeekClient(
        "key",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"error": {"message": "bad_response_status_code", "type": "openai_error"}}
            )
        ),
    )
    with pytest.raises(DeepSeekError) as excinfo:
        client.parse_atoms(CAPTION, "id")
    assert excinfo.value.code == "provider_error"
    report = deepseek_test_connection(client)
    assert report["status"] == "failed"
    assert report["checks"][0]["status"] == "provider_error"


def test_observe_sends_inline_image_with_detail():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.read())
        return _chat_response(observation_payload())

    client = DeepSeekClient(
        "test-key",
        transport=httpx.MockTransport(handler),
    )
    raw = client.observe(_tmp_png(), "image/png", CAPTION, [])
    content = seen["body"]["messages"][0]["content"]
    image_part = next(part for part in content if part["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert image_part["image_url"]["detail"] == "original"
    assert raw["observations"][0]["relation"] == "supports"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"choices": [{"message": {"content": ""}}]}),
        httpx.Response(200, json={"choices": [{"message": {"content": "[1, 2]"}}]}),
        httpx.Response(200, text="not json"),
    ],
)
def test_malformed_responses_are_rejected(response):
    client = DeepSeekClient(
        "test-key",
        transport=httpx.MockTransport(lambda request: response),
        retries=0,
    )
    with pytest.raises(DeepSeekError) as excinfo:
        client.parse_atoms(CAPTION, "id")
    assert excinfo.value.code == "malformed_response"


@pytest.mark.parametrize(
    ("status_code", "code"),
    [(429, "rate_limited"), (500, "http_error"), (404, "http_error")],
)
def test_http_failures_map_to_error_codes(status_code, code):
    client = DeepSeekClient(
        "test-key",
        transport=httpx.MockTransport(lambda request: httpx.Response(status_code, text="down")),
        retries=0,
    )
    with pytest.raises(DeepSeekError) as excinfo:
        client.parse_atoms(CAPTION, "id")
    assert excinfo.value.code == code


def test_redirect_is_rejected():
    client = DeepSeekClient(
        "test-key",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(307, headers={"Location": "https://elsewhere.test/"})
        ),
    )
    with pytest.raises(DeepSeekError) as excinfo:
        client.parse_atoms(CAPTION, "id")
    assert excinfo.value.code == "redirect_rejected"


def test_prepare_requires_api_key_when_deepseek_enabled(tmp_path):
    with pytest.raises(ValueError, match="deepseek_api_key"):
        Settings(data_dir=tmp_path, deepseek_enabled=True, deepseek_api_key="").prepare()
    Settings(
        data_dir=tmp_path,
        deepseek_enabled=True,
        deepseek_api_key="key",
        deepseek_base_url="https://api.deepseek.com",
    ).prepare()


def test_base_url_must_be_https(tmp_path):
    with pytest.raises(ValueError, match="https"):
        Settings(
            data_dir=tmp_path,
            deepseek_enabled=True,
            deepseek_api_key="key",
            deepseek_base_url="http://api.deepseek.com",
        ).prepare()


def test_put_settings_cannot_enable_deepseek_without_key(client):
    response = client.put("/api/v1/settings", json={"deepseek_enabled": True})
    assert response.status_code == 422
    assert "deepseek_api_key" in response.json()["error"]["message"]
    caps = {item["provider"]: item["status"] for item in client.get("/ready").json()["capabilities"]}
    assert caps["deepseek-flash"] == "unconfigured"
    response = client.put(
        "/api/v1/settings",
        json={"deepseek_enabled": True, "deepseek_api_key": "server-key"},
    )
    assert response.status_code == 200, response.text
    caps = {item["provider"]: item["status"] for item in client.get("/ready").json()["capabilities"]}
    assert caps["deepseek-flash"] == "ok"
    values = client.get("/api/v1/settings").json()["values"]
    assert values["deepseek_api_key"] == "__CONFIGURED__"


class StubDeepSeekClient:
    """Deterministic stand-in returning valid atoms and observations."""

    def __init__(
        self,
        api_key,
        base_url="https://api.deepseek.com",
        model="deepseek-flash",
        timeout=90,
        disable_thinking=False,
    ):
        assert api_key
        self.model = model

    def parse_atoms(self, caption, language):
        return atom_payload()

    def observe(self, path, media_type, caption, atoms):
        assert path.is_file()
        return observation_payload()


class FailingDeepSeekClient(StubDeepSeekClient):
    def parse_atoms(self, caption, language):
        raise DeepSeekError("network_error")

    def observe(self, path, media_type, caption, atoms):
        raise DeepSeekError("network_error")


def live_bundle(ref, options):
    return AuroraBundle(
        schema_version="1.0.0",
        case_id=str(uuid4()),
        claim_revision=1,
        mode="live",
        created_at=datetime.now(timezone.utc),
        input={"claim_text": CAPTION, "language": "id", "images": [ref], "as_of": None},
        analysis=None,
        retrieval=None,
        decision=None,
        warnings=[],
        extensions={"aurora_visual": {"options": options}},
    )


def pipeline_fixture(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        deepseek_enabled=True,
        deepseek_api_key="server-key",
        deepseek_base_url="https://api.deepseek.com",
        hive_enabled=False,
        hive_v3_secret="",
        mafindo_api_key="",
    )
    for name in ("media", "derived", "artifacts", "cache", "models"):
        (settings.data_dir / name).mkdir(parents=True, exist_ok=True)
    engine, session = database(tmp_path)
    Base.metadata.create_all(engine)
    media = MediaService(settings, session)
    stream = io.BytesIO()
    Image.new("RGB", (64, 48), "#cc2222").save(stream, "PNG")
    ref = media.upload(stream.getvalue(), "local")
    return settings, media, ref


def test_deepseek_run_uses_provider_for_stages_2_and_3(tmp_path, monkeypatch):
    settings, media, ref = pipeline_fixture(tmp_path)
    monkeypatch.setattr("app.services.pipeline.DeepSeekClient", StubDeepSeekClient)
    result = analyze(
        live_bundle(ref, {"provider": "deepseek", "parser": "deepseek-vlm"}),
        str(uuid4()),
        settings,
        media,
    )
    extension = result.extensions["aurora_visual"]["deepseek"]
    assert extension["stage2"]["status"] == "ok" and extension["stage2"]["atom_count"] == 1
    assert extension["stage3"]["status"] == "ok"
    assert extension["egress"] == {"caption_stage2": True, "normalized_preview_stage3": True}
    assert result.analysis.atomic_claims[0].parser_confidence is None
    codes = [warning.code for warning in result.analysis.run.warnings]
    assert "DEEPSEEK_ATOMIZER_REVIEW" in codes
    assert not any(code.startswith("DEEPSEEK_") and code != "DEEPSEEK_ATOMIZER_REVIEW" for code in codes)
    assert result.analysis.run.status == "completed"
    # Explicit VLM support fuses to Supported with region-bounded evidence.
    assessment = result.analysis.visual_assessments[0]
    assert assessment.visual_status == "Supported"
    assert assessment.supporting_regions


def test_deepseek_failure_falls_back_to_rules_and_partial(tmp_path, monkeypatch):
    settings, media, ref = pipeline_fixture(tmp_path)
    monkeypatch.setattr("app.services.pipeline.DeepSeekClient", FailingDeepSeekClient)
    result = analyze(
        live_bundle(ref, {"provider": "deepseek", "parser": "deepseek-vlm"}),
        str(uuid4()),
        settings,
        media,
    )
    extension = result.extensions["aurora_visual"]["deepseek"]
    assert extension["stage2"]["status"] == "failed_fallback_rules"
    assert extension["stage3"]["status"] == "network_error"
    codes = [warning.code for warning in result.analysis.run.warnings]
    assert "DEEPSEEK_NETWORK_ERROR" in codes
    assert result.analysis.run.status == "partial"
    assert result.analysis.atomic_claims  # rules fallback still produced atoms


def test_deepseek_provider_requires_api_key(tmp_path, monkeypatch):
    settings, media, ref = pipeline_fixture(tmp_path)
    settings.deepseek_api_key = ""
    with pytest.raises(ValueError, match="DEEPSEEK_KEY_REQUIRED"):
        analyze(
            live_bundle(ref, {"provider": "deepseek", "parser": "deepseek-vlm"}),
            str(uuid4()),
            settings,
            media,
        )


def test_deepseek_parser_without_provider_is_rejected(tmp_path, monkeypatch):
    settings, media, ref = pipeline_fixture(tmp_path)
    settings.deepseek_enabled = False
    with pytest.raises(ValueError, match="DEEPSEEK_PROVIDER_REQUIRED"):
        analyze(
            live_bundle(ref, {"provider": "local", "parser": "deepseek-vlm"}),
            str(uuid4()),
            settings,
            media,
        )


def test_api_rejects_deepseek_parser_without_provider(client, app, demo):
    demo["mode"] = "live"
    demo["extensions"] = {"aurora_visual": {"options": {"provider": "local", "parser": "deepseek-vlm"}}}
    response = client.post("/api/v1/analyze", json=demo, headers={"Idempotency-Key": "ds-bad"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "DEEPSEEK_PROVIDER_REQUIRED"


def test_api_rejects_deepseek_provider_in_demo_mode(client, app, demo):
    demo["extensions"] = {"aurora_visual": {"options": {"provider": "deepseek", "parser": "deepseek-vlm"}}}
    response = client.post("/api/v1/analyze", json=demo, headers={"Idempotency-Key": "ds-demo"})
    assert response.status_code == 422


def test_api_rejects_disabled_deepseek_provider(client, app, demo):
    demo["mode"] = "live"
    demo["extensions"] = {"aurora_visual": {"options": {"provider": "deepseek", "parser": "deepseek-vlm"}}}
    response = client.post("/api/v1/analyze", json=demo, headers={"Idempotency-Key": "ds-disabled"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "DEEPSEEK_DISABLED"


def test_connection_probe_reports_ok_without_leaking_key():
    calls = []

    def handler(request):
        body = json.loads(request.read())
        calls.append(body)
        content = json.dumps({"ok": True, "color": "red", "purpose": "deepseek-connection-test"})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    client = DeepSeekClient("secret-key", transport=httpx.MockTransport(handler))
    report = deepseek_test_connection(client, vision=True)
    assert report["status"] == "ok"
    assert report["base_url_host"] == "api.deepseek.com"
    assert [check["name"] for check in report["checks"]] == ["chat-json", "vision-json"]
    assert all(check["status"] == "ok" for check in report["checks"])
    assert report["checks"][1]["observed_color"] == "red"
    assert "secret-key" not in json.dumps(report)
    # Second call carries the vision image part.
    image_part = next(part for part in calls[1]["messages"][0]["content"] if part["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


def test_connection_probe_unconfigured_never_requests():
    calls = []
    client = DeepSeekClient("", transport=httpx.MockTransport(lambda request: calls.append(request)))
    report = deepseek_test_connection(client)
    assert report["status"] == "unconfigured"
    assert calls == []
    assert report["checks"] == []


def test_connection_probe_reports_auth_error():
    client = DeepSeekClient(
        "bad-key",
        transport=httpx.MockTransport(lambda request: httpx.Response(401, json={"error": "bad key"})),
    )
    report = deepseek_test_connection(client)
    assert report["status"] == "failed"
    assert report["checks"][0]["name"] == "chat-json"
    assert report["checks"][0]["status"] == "auth_error"
    assert report["checks"][0]["http_status"] == 401
    assert "bad-key" not in json.dumps(report)


def test_connection_probe_partial_when_vision_fails():
    seen = {"count": 0}

    def handler(request):
        seen["count"] += 1
        if seen["count"] == 1:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": json.dumps({"ok": True})}}]},
            )
        return httpx.Response(404, json={"error": "vision unsupported"})

    client = DeepSeekClient("secret-key", transport=httpx.MockTransport(handler))
    report = deepseek_test_connection(client, vision=True)
    assert report["status"] == "partial"
    assert report["checks"][0]["status"] == "ok"
    assert report["checks"][1]["status"] == "http_error"
    assert report["checks"][1]["http_status"] == 404


def test_api_deepseek_test_unconfigured_is_reported(client):
    response = client.post("/api/v1/providers/deepseek/test", json={"vision": True})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "unconfigured"
    assert payload["checks"] == []


def test_api_deepseek_test_rejects_unknown_fields(client):
    response = client.post("/api/v1/providers/deepseek/test", json={"vision": True, "key": "x"})
    assert response.status_code == 422


def test_api_deepseek_test_uses_probe(client, app, monkeypatch):
    calls = {}

    def fake_probe(client_instance, vision=True):
        calls["vision"] = vision
        return {"provider": "deepseek-flash", "status": "ok", "checks": [], "message": "ok"}

    monkeypatch.setattr("app.api.main.test_deepseek_connection", fake_probe)
    response = client.post("/api/v1/providers/deepseek/test", json={"vision": False})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert calls["vision"] is False


def test_api_deepseek_test_disabled_in_public_mode(tmp_path):
    from fastapi.testclient import TestClient

    public_settings = Settings(
        data_dir=tmp_path,
        public=True,
        token="x" * 32,
        allowed_hosts=["example.test"],
    )
    app = create_app(public_settings, embedded_worker=False)
    with TestClient(app, base_url="http://example.test") as public_client:
        response = public_client.post(
            "/api/v1/providers/deepseek/test",
            json={"vision": True},
            headers={"Authorization": "Bearer " + "x" * 32},
        )
    assert response.status_code == 404


def test_cli_deepseek_test_reports_status(tmp_path, monkeypatch):
    from aurora_visual import cli

    monkeypatch.setenv("AURORA_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("AURORA_DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        ["aurora", "deepseek-test", "--output", str(tmp_path / "deepseek-test.json")],
    )
    cli.main()
    payload = json.loads((tmp_path / "deepseek-test.json").read_text())
    assert payload["status"] == "unconfigured"


def test_parse_atoms_repairs_estimated_spans():
    payload = atom_payload()
    payload["atoms"][0]["spans"] = [{"start": 0, "end": 5}]  # model-estimated, wrong
    client = DeepSeekClient(
        "key",
        transport=httpx.MockTransport(lambda request: _chat_response(payload)),
    )
    atoms = DeepSeekAtomizer(client).parse(CAPTION, "id").atoms
    assert atoms[0].spans[0].start == 0
    assert atoms[0].spans[0].end == len(CAPTION)


def test_parse_atoms_rejects_invalid_payloads():
    payload = atom_payload()
    payload["atoms"][0]["subject"] = "entitas yang tidak ada di caption"
    client = DeepSeekClient(
        "key",
        transport=httpx.MockTransport(lambda request: _chat_response(payload)),
    )
    with pytest.raises(ValueError, match="grounded"):
        DeepSeekAtomizer(client).parse(CAPTION, "id")


def test_chat_tolerates_markdown_fenced_json():
    fenced = "```json\n" + json.dumps(atom_payload()) + "\n```"
    client = DeepSeekClient(
        "key",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"choices": [{"message": {"content": fenced}}]})
        ),
    )
    atoms = DeepSeekAtomizer(client).parse(CAPTION, "id").atoms
    assert atoms[0].atom_id == "a000001"


def test_chat_detects_web_panel_instead_of_api():
    client = DeepSeekClient(
        "key",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, content=b"<!doctype html><html></html>", headers={"content-type": "text/html"}
            )
        ),
    )
    with pytest.raises(DeepSeekError) as excinfo:
        client.parse_atoms(CAPTION, "id")
    assert excinfo.value.code == "endpoint_not_api"
    report = deepseek_test_connection(client)
    assert report["status"] == "failed"
    assert report["checks"][0]["status"] == "endpoint_not_api"
    assert "/v1" in report["message"]


class StubHiveV3:
    """Stage-1-only Hive stand-in: benign detection, no VLM calls."""

    def __init__(self, secret, timeout=45.0):
        assert secret

    def detect_ai_generated(self, path, media_type):
        assert path.is_file()
        return {
            "model": "hive/ai-generated-and-deepfake-content-detection",
            "frames": 1,
            "classes": [
                {"label": "ai_generated", "score": 0.02},
                {"label": "not_ai_generated", "score": 0.98},
                {"label": "deepfake", "score": 0.01},
                {"label": "none", "score": 0.99},
            ],
            "metadata": {},
        }

    def parse_atoms(self, caption, language):
        raise AssertionError("Hive atomizer must not run when DeepSeek is selected")

    def observe(self, path, media_type, caption, atoms):
        raise AssertionError("Hive VLM must not run when DeepSeek is selected")


def test_hive_and_deepseek_together_split_stages(tmp_path, monkeypatch):
    settings, media, ref = pipeline_fixture(tmp_path)
    settings.hive_enabled = True
    settings.hive_v3_secret = "v3-secret"
    monkeypatch.setattr("app.services.pipeline.DeepSeekClient", StubDeepSeekClient)
    monkeypatch.setattr("app.services.pipeline.HiveV3Client", StubHiveV3)
    result = analyze(
        live_bundle(ref, {"hive": True, "deepseek": True, "parser": "deepseek-vlm"}),
        str(uuid4()),
        settings,
        media,
    )
    extension = result.extensions["aurora_visual"]
    hive = extension["hive"]
    deepseek = extension["deepseek"]
    # Hive keeps stage-1 detection on original bytes only.
    assert hive["egress"] == {
        "original_media_stage1": True,
        "normalized_preview_stage3": False,
        "caption_stage2_3": False,
    }
    assert hive["stage1"]["provider"] == "hive-v3" and hive["stage1"]["status"] == "ok"
    assert hive["stage2"] is None and hive["stage3"] is None
    detectors = extension["screening"]["detectors"]
    assert {detector["provider"] for detector in detectors} == {"hive-v3"}
    # DeepSeek owns stages 2-3.
    assert deepseek["stage2"]["status"] == "ok" and deepseek["stage2"]["atom_count"] == 1
    assert deepseek["stage3"]["status"] == "ok"
    codes = [warning.code for warning in result.analysis.run.warnings]
    assert "DEEPSEEK_ATOMIZER_REVIEW" in codes
    assert "HIVE_ATOMIZER_REVIEW" not in codes
    assert not any(code.startswith(("HIVE_", "DEEPSEEK_")) and not code.endswith("_REVIEW") for code in codes)
    assert result.analysis.run.status == "completed"


def test_api_rejects_hive_and_provider_mixed(client, app, demo):
    demo["mode"] = "live"
    demo["extensions"] = {"aurora_visual": {"options": {"hive": True, "provider": "hive"}}}
    response = client.post("/api/v1/analyze", json=demo, headers={"Idempotency-Key": "mix-1"})
    assert response.status_code == 422


def test_api_rejects_provider_flags_in_demo_mode(client, app, demo):
    demo["extensions"] = {"aurora_visual": {"options": {"hive": True, "deepseek": True}}}
    response = client.post("/api/v1/analyze", json=demo, headers={"Idempotency-Key": "dual-demo"})
    assert response.status_code == 422


def test_api_rejects_hive_vlm_parser_without_hive_flag(client, app, demo):
    demo["mode"] = "live"
    demo["extensions"] = {"aurora_visual": {"options": {"hive": False, "parser": "hive-vlm"}}}
    response = client.post("/api/v1/analyze", json=demo, headers={"Idempotency-Key": "hive-no-flag"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "HIVE_PROVIDER_REQUIRED"
