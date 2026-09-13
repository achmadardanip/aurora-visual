import io
import os

import pytest
from app.api.main import create_app
from app.config import Settings
from fastapi.testclient import TestClient
from PIL import Image

# The suite must be hermetic: importing app modules loads the developer's
# .env once (load_dotenv), so strip provider/pipeline values afterwards.
# Every Settings() instance then uses code defaults unless a test (or a
# monkeypatch) sets values explicitly.
for _name in list(os.environ):
    if _name.startswith(("AURORA_", "VITE_")):
        del os.environ[_name]


@pytest.fixture
def app(tmp_path):
    return create_app(
        Settings(
            data_dir=tmp_path,
            mafindo_api_key="",
            mafindo_timeout=12,
            hive_enabled=False,
            hive_v3_secret="",
            hive_v2_shared_key="",
            hive_v2_origin_key="",
            hive_v2_ocr_key="",
            hive_v2_object_key="",
            hive_v2_scene_key="",
            hive_v2_people_key="",
            hive_v2_logo_key="",
            hive_v2_celebrity_key="",
            hive_v2_translation_key="",
        ),
        embedded_worker=False,
    )


@pytest.fixture
def client(app):
    with TestClient(app) as client:
        yield client


@pytest.fixture
def demo(client):
    return client.post("/api/v1/demo/supported").json()


def png(color="blue", size=(64, 48)):
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, "PNG")
    return stream.getvalue()


def run(client, app, bundle, key="test-job"):
    response = client.post("/api/v1/analyze", json=bundle, headers={"Idempotency-Key": key})
    assert response.status_code == 202, response.text
    app.state.worker.run_once(isolated=False)
    result = client.get("/api/v1/jobs/" + response.json()["job_id"]).json()
    assert result["status"] in ("succeeded", "partial"), result
    return result["result"], response.json()["job_id"]
