"""Hive policy: V3 secret mandatory, V2 project keys optional.

Covers config validation, runtime settings updates, overlay loading, and the
analysis pipeline running a full Hive pass with only V3 configured.
"""

import io
import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from app.api.main import create_app
from app.config import Settings
from app.models.contract import AuroraBundle
from app.models.db import Base, database
from app.services.media import MediaService
from app.services.pipeline import analyze
from aurora_visual.atomization.parser import RuleAtomizer
from PIL import Image


def test_prepare_requires_v3_secret_when_hive_enabled(tmp_path):
    with pytest.raises(ValueError, match="hive_v3_secret"):
        Settings(data_dir=tmp_path, hive_enabled=True, hive_v3_secret="").prepare()
    # V2 keys are never required for a valid Hive configuration.
    Settings(data_dir=tmp_path, hive_enabled=True, hive_v3_secret="secret").prepare()


def test_put_settings_cannot_enable_hive_without_v3_secret(client):
    response = client.put("/api/v1/settings", json={"hive_enabled": True})
    assert response.status_code == 422
    assert "hive_v3_secret" in response.json()["error"]["message"]
    caps = {
        c["provider"]: c["status"]
        for c in client.get("/ready").json()["capabilities"]
        if c["provider"].startswith("hive-")
    }
    assert caps["hive-v3-vlm"] == "unconfigured"
    assert caps["hive-v2-origin"] == "optional"


def test_put_settings_allows_v3_only_and_protects_the_secret(client):
    response = client.put("/api/v1/settings", json={"hive_enabled": True, "hive_v3_secret": "v3-secret"})
    assert response.status_code == 200, response.text
    caps = {
        c["provider"]: c["status"]
        for c in client.get("/ready").json()["capabilities"]
        if c["provider"].startswith("hive-")
    }
    assert caps["hive-v3-vlm"] == "ok"
    assert all(status == "optional" for provider, status in caps.items() if provider != "hive-v3-vlm")
    # Clearing the mandatory secret while Hive stays enabled is rejected.
    response = client.put("/api/v1/settings", json={"hive_v3_secret": ""})
    assert response.status_code == 422
    assert "hive_v3_secret" in response.json()["error"]["message"]


def test_overlay_enabling_hive_without_v3_secret_is_ignored(tmp_path):
    (tmp_path / "settings.json").write_text(json.dumps({"hive_enabled": True, "hive_v3_secret": ""}))
    app = create_app(Settings(data_dir=tmp_path), embedded_worker=False)
    assert app.state.settings.hive_enabled is False


class StubV3Client:
    """Deterministic V3 stand-in: rules-quality atoms, no observations."""

    def __init__(self, secret, timeout=45.0):
        assert secret == "v3-secret"

    def parse_atoms(self, caption, language):
        atoms = RuleAtomizer().parse(caption, language).atoms
        return {"atoms": [atom.model_dump() for atom in atoms]}

    def observe(self, path, media_type, caption, atoms):
        assert path.is_file()
        return {"observations": []}


def live_bundle(ref, options):
    return AuroraBundle(
        schema_version="1.0.0",
        case_id=str(uuid4()),
        claim_revision=1,
        mode="live",
        created_at=datetime.now(timezone.utc),
        input={"claim_text": "Bidang ini berwarna merah", "language": "id", "images": [ref], "as_of": None},
        analysis=None,
        retrieval=None,
        decision=None,
        warnings=[],
        extensions={"aurora_visual": {"options": options}},
    )


def pipeline_fixture(tmp_path, hive_v3_secret="v3-secret"):
    settings = Settings(
        data_dir=tmp_path,
        hive_enabled=True,
        hive_v3_secret=hive_v3_secret,
        mafindo_api_key="",
    )
    for name in ("media", "derived", "artifacts", "cache"):
        (settings.data_dir / name).mkdir(parents=True, exist_ok=True)
    engine, session = database(tmp_path)
    Base.metadata.create_all(engine)
    media = MediaService(settings, session)
    stream = io.BytesIO()
    Image.new("RGB", (64, 48), "#e02222").save(stream, "PNG")
    ref = media.upload(stream.getvalue(), "local")
    return settings, media, ref


def test_v3_only_hive_run_completes_without_v2(tmp_path, monkeypatch):
    settings, media, ref = pipeline_fixture(tmp_path)
    monkeypatch.setattr("app.services.pipeline.HiveV3Client", StubV3Client)
    result = analyze(
        live_bundle(ref, {"provider": "hive", "parser": "hive-vlm"}), str(uuid4()), settings, media
    )
    hive = result.extensions["aurora_visual"]["hive"]
    assert hive["stage2"]["provider"] == "hive-v3-vlm" and hive["stage2"]["status"] == "ok"
    assert hive["stage3"]["vlm"]["status"] == "ok"
    assert hive["stage1"]["status"] == "unconfigured" and hive["stage1"]["optional"] is True
    assert hive["stage3"]["status"] == "unconfigured_or_unsupported"
    # Without a V2 origin key the original bytes never leave the server.
    assert hive["egress"]["original_media_stage1"] is False
    assert hive["egress"]["normalized_preview_stage3"] is True
    assert hive["egress"]["caption_stage2_3"] is True
    assert len(hive["provider_status"]) == 7
    assert all(entry["status"] == "unconfigured" for entry in hive["provider_status"])
    # Unconfigured V2 capabilities are optional: no HIVE_ failure warnings and
    # the run completes (only the atomizer review advisory remains).
    codes = [w.code for w in result.analysis.run.warnings]
    assert not any(code.startswith("HIVE_") and code != "HIVE_ATOMIZER_REVIEW" for code in codes)
    assert result.analysis.run.status == "completed"
    assert result.analysis.visual_assessments


def test_hive_run_without_v3_secret_fails_fast(tmp_path, monkeypatch):
    settings, media, ref = pipeline_fixture(tmp_path, hive_v3_secret="")
    with pytest.raises(ValueError, match="HIVE_V3_REQUIRED"):
        analyze(live_bundle(ref, {"provider": "hive", "parser": "hive-vlm"}), str(uuid4()), settings, media)


class FailingLLMParser:
    def __init__(self, url, model, allowed_origins):
        pass

    def parse(self, caption, language):
        raise ValueError("LLM entity not grounded in caption")


def test_llm_parser_failure_falls_back_to_rules(tmp_path, monkeypatch):
    settings, media, ref = pipeline_fixture(tmp_path)
    monkeypatch.setattr("app.services.pipeline.StructuredLLMAtomizer", FailingLLMParser)
    result = analyze(live_bundle(ref, {"provider": "local", "parser": "llm"}), str(uuid4()), settings, media)
    codes = [w.code for w in result.analysis.run.warnings]
    assert "LLM_FALLBACK_RULES" in codes
    assert result.analysis.run.status == "partial"
    assert result.analysis.atomic_claims
