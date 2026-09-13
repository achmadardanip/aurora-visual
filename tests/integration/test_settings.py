import json

from app.api.main import create_app
from app.config import Settings
from conftest import run
from fastapi.testclient import TestClient


def test_get_settings_masks_secrets(client):
    response = client.get("/api/v1/settings")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["fields"]["llm_url"]["kind"] == "str"
    assert body["fields"]["hive_enabled"]["kind"] == "bool"
    assert body["fields"]["mafindo_api_key"]["secret"] is True
    assert body["values"]["ocr_lang"] == "ind+eng"
    # Env-only keys are reported for transparency but never writable.
    assert body["env_only"]["public"] is False


def test_put_settings_partial_update_applies_live(client, app):
    response = client.put(
        "/api/v1/settings",
        json={"llm_url": "http://127.0.0.1:11434", "llm_model": "llama3.1"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["applied"] is True
    assert app.state.settings.llm_model == "llama3.1"
    assert app.state.settings.llm_url == "http://127.0.0.1:11434"
    # /ready reflects runtime configuration without contacting the provider.
    ready = client.get("/ready").json()
    ollama = next(c for c in ready["capabilities"] if c["provider"] == "ollama")
    assert ollama["status"] == "ok"


def test_put_settings_secret_masked_sentinel_kept(client, app):
    client.put("/api/v1/settings", json={"mafindo_api_key": "secret-value-123"})
    # The masked sentinel must not overwrite the stored secret.
    client.put(
        "/api/v1/settings",
        json={"mafindo_api_key": "__CONFIGURED__", "mafindo_timeout": 20.0},
    )
    assert app.state.settings.mafindo_api_key == "secret-value-123"
    assert app.state.settings.mafindo_timeout == 20.0
    values = client.get("/api/v1/settings").json()["values"]
    assert values["mafindo_api_key"] == "__CONFIGURED__"


def test_put_settings_validation_errors(client):
    for payload, fragment in [
        ({"max_images": 99}, "max_images"),
        ({"backbone": "resnet"}, "backbone"),
        ({"llm_url": "ftp://x"}, "llm_url"),
        ({"llm_allowed_origins": "http://*"}, "llm_allowed_origins"),
        ({"max_images": "eight"}, "max_images"),
        ({"unknown_key": 1}, "unknown_key"),
        ({}, "Kiriman"),
    ]:
        response = client.put("/api/v1/settings", json=payload)
        assert response.status_code == 422, payload
        assert fragment in response.json()["error"]["message"]
    # Empty string clears a secret; the stored overlay is updated.
    client.put("/api/v1/settings", json={"mafindo_api_key": ""})
    assert client.get("/api/v1/settings").json()["values"]["mafindo_api_key"] == ""


def test_settings_persist_across_restart(tmp_path):
    settings = Settings(data_dir=tmp_path, hive_v3_secret="")
    first = create_app(settings, embedded_worker=False)
    with TestClient(first) as client:
        client.put("/api/v1/settings", json={"openclip_model": "ViT-L-14", "job_timeout": 240})
    # A fresh app on the same data dir (e.g. subprocess worker) re-applies the overlay.
    second = create_app(Settings(data_dir=tmp_path), embedded_worker=False)
    assert second.state.settings.openclip_model == "ViT-L-14"
    assert second.state.settings.job_timeout == 240
    # Persisted file contains the overlay and no env-only keys.
    data = json.loads((tmp_path / "settings.json").read_text())
    assert data["openclip_model"] == "ViT-L-14"
    assert "data_dir" not in data and "token" not in data
    mode = (tmp_path / "settings.json").stat().st_mode
    assert mode & 0o077 == 0  # owner-only permissions


def test_settings_disabled_on_public(tmp_path):
    app = create_app(
        Settings(data_dir=tmp_path, public=True, token="x" * 40, allowed_hosts=["aurora.example"]),
        embedded_worker=False,
    )
    with TestClient(app) as client:
        headers = {"Authorization": "Bearer " + "x" * 40, "Host": "aurora.example"}
        for method in ("get", "put"):
            response = getattr(client, method)("/api/v1/settings", headers=headers)
            assert response.status_code == 404
            assert response.json()["error"]["code"] == "SETTINGS_DISABLED"


def test_settings_affect_next_analysis(client, app):
    bundle = client.post("/api/v1/demo/supported").json()
    client.put("/api/v1/settings", json={"ocr_lang": "eng"})
    result, _ = run(client, app, bundle)
    assert result["analysis"]["run"]["status"] == "completed"
