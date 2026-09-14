"""Pretrained capability models: registry status, downloads, segmentation regions.

Downloads are exercised through an injected transport (torchvision source) and a
monkeypatched snapshot_download (HF source); the real weights are fetched live
by the operator via `aurora models --download` and are not bundled here.
"""

import json
from pathlib import Path

import httpx
import pytest
from aurora_visual import pretrained
from aurora_visual.vision.features import extract, segmentation_regions
from aurora_visual.vision.segmentation import (
    WEIGHTS_FILENAME,
    InstanceSegmenter,
    weights_available,
    weights_path,
)
from PIL import Image


class StubProposer:
    def __init__(self, detections):
        self.detections = detections

    def propose(self, image, top_k=16, min_score=0.5):
        return self.detections[:top_k]


def _tmp_image(size=(100, 80), color="#3355cc"):
    import tempfile

    path = Path(tempfile.mkstemp(suffix=".png")[1])
    Image.new("RGB", size, color).save(path)
    return path


def test_registry_covers_capabilities_and_excludes_three_state_head():
    assert set(pretrained.REGISTRY) == {
        "segmentation",
        "segmentation-advanced",
        "geolocation",
        "identity",
    }
    assert "three-state-head" in pretrained.MODEL_FREE_NOTES
    assert "trained" in pretrained.MODEL_FREE_NOTES["three-state-head"]
    for entry in pretrained.REGISTRY.values():
        assert entry.license and entry.note and entry.model_id


def test_status_reports_missing_then_downloaded(tmp_path):
    report = pretrained.status(tmp_path)
    assert all(not item["available"] for item in report["capabilities"])
    assert report["model_free"]["relations-attributes"]

    payload = b"weights-bytes" * 100_000

    def handler(request):
        return httpx.Response(200, content=payload, headers={"content-length": str(len(payload))})

    manifest = pretrained.download(tmp_path, "segmentation", transport=httpx.MockTransport(handler))
    assert manifest["license"].startswith("torchvision BSD-3-Clause")
    assert manifest["inference"] == "ready"
    stored = tmp_path / "models" / "segmentation" / WEIGHTS_FILENAME
    assert stored.is_file() and stored.stat().st_size == len(payload)
    report = pretrained.status(tmp_path)
    segmentation = next(item for item in report["capabilities"] if item["capability"] == "segmentation")
    assert segmentation["available"] is True
    assert segmentation["files"][0]["bytes"] == len(payload)
    assert pretrained.load_manifest(tmp_path, "segmentation")["model_id"].startswith("torchvision/")


def test_download_rejects_unknown_capability(tmp_path):
    with pytest.raises(ValueError, match="Unknown capability"):
        pretrained.download(tmp_path, "three-state-head")


def test_hf_download_uses_allow_patterns(tmp_path, monkeypatch):
    seen = {}

    def fake_snapshot_download(repo_id, allow_patterns=None, local_dir=None):
        seen["repo"] = repo_id
        seen["patterns"] = allow_patterns
        target = Path(local_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.safetensors").write_bytes(b"hf-weights")
        return str(target)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    manifest = pretrained.download(tmp_path, "segmentation-advanced")
    assert seen["repo"] == "facebook/sam2.1-hiera-tiny"
    assert "*.pt" in seen["patterns"]
    assert manifest["inference"] == "not_wired"
    assert any(item["name"] == "model.safetensors" for item in manifest["files"])


def test_weights_available_requires_real_size(tmp_path):
    class _Settings:
        segmentation_weights = ""
        data_dir = tmp_path

    assert weights_available(_Settings()) is False
    target = weights_path(_Settings())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"tiny")
    assert weights_available(_Settings()) is False
    target.write_bytes(b"x" * 2_000_000)
    assert weights_available(_Settings()) is True


def test_segmentation_regions_uses_proposals_and_normalizes_boxes():
    image = Image.new("RGB", (200, 100), "white")
    proposer = StubProposer([{"label": "car", "score": 0.91, "bbox": [0.1, 0.2, 0.6, 0.9]}])
    regions, crops = segmentation_regions(
        image, "asset_" + "a" * 64, "a" * 32, top_k=8, start=0, proposer=proposer
    )
    assert len(regions) == 1
    assert regions[0].bbox == (0.1, 0.2, 0.6, 0.9)
    assert regions[0].score == 0.91
    assert regions[0].description.startswith("Mask R-CNN: car")
    assert crops[0].width == 100 and crops[0].height == 70


def test_segmentation_regions_falls_back_to_grid_without_detections():
    image = Image.new("RGB", (60, 40), "white")
    regions, crops = segmentation_regions(
        image, "asset_" + "b" * 64, "b" * 32, top_k=4, start=0, proposer=StubProposer([])
    )
    assert len(regions) == 4
    assert all("Grid" in region.description for region in regions)


def test_extract_marks_maskrcnn_preprocessing_only_for_segmentation():
    import tempfile

    image = Image.new("RGB", (80, 60), "#2244aa")
    media = type("M", (), {"asset_id": "asset_" + "c" * 64, "sha256": "c" * 64})()
    atoms = []
    proposer = StubProposer([{"label": "dog", "score": 0.8, "bbox": [0.0, 0.0, 0.5, 0.5]}])
    with tempfile.TemporaryDirectory() as cache:
        grid = extract(image, media, atoms, "c" * 32, Path(cache), region_method="grid")
        assert grid["config"]["preprocessing"] == "exif-rgb-covering-grid-global-caption-v3"
        segmented = extract(
            image,
            media,
            atoms,
            "c" * 32,
            Path(cache),
            region_method="segmentation",
            region_proposer=proposer,
        )
        assert segmented["config"]["preprocessing"] == "exif-rgb-covering-maskrcnn-global-caption-v3"
        assert segmented["regions"][0].description.startswith("Mask R-CNN")


def test_region_method_option_requires_downloaded_weights(client, app, demo):
    demo["mode"] = "live"
    demo["extensions"] = {
        "aurora_visual": {
            "options": {"provider": "local", "region_method": "segmentation"},
        }
    }
    response = client.post("/api/v1/analyze", json=demo, headers={"Idempotency-Key": "seg-missing"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "SEGMENTATION_UNCONFIGURED"


def test_region_method_option_rejected_in_demo_mode(client, app, demo):
    demo["extensions"] = {"aurora_visual": {"options": {"region_method": "segmentation"}}}
    response = client.post("/api/v1/analyze", json=demo, headers={"Idempotency-Key": "seg-demo"})
    assert response.status_code == 422


def test_ready_reports_segmentation_unconfigured_without_weights(client, app):
    app.state.worker.heartbeat()
    caps = {item["provider"]: item["status"] for item in client.get("/ready").json()["capabilities"]}
    assert caps["torchvision-maskrcnn"] == "unconfigured"


def test_instance_segmenter_load_is_cached_per_path(tmp_path, monkeypatch):
    class _Settings:
        segmentation_weights = ""
        data_dir = tmp_path

    created = []

    def fake_init(self, settings=None):
        created.append(settings)

    monkeypatch.setattr(InstanceSegmenter, "__init__", fake_init)
    first = InstanceSegmenter.load(_Settings())
    second = InstanceSegmenter.load(_Settings())
    assert first is second and len(created) == 1


def test_models_cli_reports_status(tmp_path, monkeypatch, capsys):
    from aurora_visual import cli

    monkeypatch.setenv("AURORA_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        "sys.argv",
        ["aurora", "models", "--status", "--output", str(tmp_path / "models.json")],
    )
    cli.main()
    payload = json.loads((tmp_path / "models.json").read_text())
    assert payload["downloads"] == []
    assert len(payload["status"]["capabilities"]) == 4
