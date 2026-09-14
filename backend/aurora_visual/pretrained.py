"""Pretrained model registry: download, license notes, and local status.

Capability models are frozen pretrained checkpoints (except the project's own
three-state S/C/U head, which is trained and excluded here). Downloads are
explicit (``aurora models --download …``), land in ``{data_dir}/models``, and
are recorded in a manifest with source revision, file hashes, and license
notes. Nothing is downloaded implicitly at analysis time.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.models.contract import sha

MODEL_REGISTRY_VERSION = "pretrained-models-v1"


@dataclass
class ModelEntry:
    capability: str
    model_id: str
    source: str  # "torchvision" | "hf"
    license: str
    note: str
    url: str | None = None
    repo: str | None = None
    allow_patterns: list[str] = field(default_factory=list)
    inference: str = "not_wired"  # "ready" | "not_wired"


REGISTRY: dict[str, ModelEntry] = {
    "segmentation": ModelEntry(
        capability="segmentation",
        model_id="torchvision/maskrcnn-resnet50-fpn-coco",
        source="torchvision",
        url="https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth",
        license=(
            "torchvision BSD-3-Clause; weights trained on COCO (annotations CC-BY 4.0; "
            "image licenses remain with original owners)"
        ),
        note=(
            "Frozen instance segmentation used for region proposals and object counting "
            "via aurora_visual.vision.segmentation.InstanceSegmenter."
        ),
        inference="ready",
    ),
    "segmentation-advanced": ModelEntry(
        capability="segmentation-advanced",
        model_id="facebook/sam2.1-hiera-tiny",
        source="hf",
        repo="facebook/sam2.1-hiera-tiny",
        allow_patterns=["*.pt", "*.yaml"],
        license="Apache-2.0",
        note=(
            "Optional advanced promptable segmenter (SAM 2.1 tiny). Registered for download; "
            "inference adapter not wired (requires the sam2 or transformers runtime)."
        ),
        inference="not_wired",
    ),
    "geolocation": ModelEntry(
        capability="geolocation",
        model_id="osv5m/baseline",
        source="hf",
        repo="osv5m/baseline",
        allow_patterns=["config.json", "pytorch_model.bin", "README.md"],
        license="MIT",
        note=(
            "OSV-5M street-view geolocation baseline (Plonk). Registered for download; "
            "inference adapter not wired (requires the osv5m runtime)."
        ),
        inference="not_wired",
    ),
    "identity": ModelEntry(
        capability="identity",
        model_id="immich-app/buffalo_l (InsightFace)",
        source="hf",
        repo="immich-app/buffalo_l",
        allow_patterns=["detection/model.onnx", "recognition/model.onnx", "README.md"],
        license=(
            "InsightFace pretrained models are for non-commercial research use only; "
            "identity workflows also require consent under applicable law (UU PDP)"
        ),
        note=(
            "Face detection + recognition ONNX pack. Registered for download; inference "
            "adapter not wired (requires onnxruntime) and identity claims stay Unobservable "
            "without explicit evidence."
        ),
        inference="not_wired",
    ),
}

# Capabilities with no standalone downloadable checkpoint:
MODEL_FREE_NOTES = {
    "relations-attributes": (
        "Uses the project's configured OpenCLIP backbone (already integrated) with "
        "VL-CheckList-style probes; no separate checkpoint."
    ),
    "photo-date-validation": (
        "No established pretrained checkpoint; combination of EXIF metadata inspection "
        "(local) and DeepSeek/Hive VLM reasoning over visible cues, evaluated against a "
        "human-annotated corpus."
    ),
    "three-state-head": (
        "Excluded by design: the S/C/U head is the project's own trained component "
        "(aurora train/tune), not a pretrained model."
    ),
}


def models_dir(data_dir) -> Path:
    path = Path(data_dir) / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _manifest_path(data_dir, entry: ModelEntry) -> Path:
    return models_dir(data_dir) / f"{entry.capability}.manifest.json"


def _hash_file(path: Path) -> str:
    return sha(path.read_bytes())


def download(data_dir, capability: str, progress=lambda message: None, transport=None) -> dict:
    """Download one capability checkpoint; returns a manifest summary."""
    entry = REGISTRY.get(capability)
    if entry is None:
        raise ValueError(f"Unknown capability: {capability}")
    target = models_dir(data_dir) / capability
    target.mkdir(parents=True, exist_ok=True)
    files = []
    if entry.source == "torchvision":
        destination = target / Path(entry.url or "").name
        if not destination.is_file():
            import httpx

            progress(f"downloading {entry.model_id}")
            with httpx.Client(timeout=600, follow_redirects=True, transport=transport) as client:
                with client.stream("GET", entry.url) as response:
                    response.raise_for_status()
                    with destination.open("wb") as sink:
                        for chunk in response.iter_bytes(1 << 20):
                            sink.write(chunk)
        files.append(
            {"name": destination.name, "sha256": _hash_file(destination), "bytes": destination.stat().st_size}
        )
    else:
        from huggingface_hub import snapshot_download

        progress(f"downloading {entry.repo}")
        snapshot_download(
            repo_id=entry.repo,
            allow_patterns=entry.allow_patterns or None,
            local_dir=target,
        )
        for path in sorted(target.rglob("*")):
            if path.is_file() and not path.name.endswith(".manifest.json"):
                if ".cache" in path.relative_to(target).parts:
                    continue  # HF download bookkeeping, not model content
                files.append(
                    {
                        "name": str(path.relative_to(target)),
                        "sha256": _hash_file(path),
                        "bytes": path.stat().st_size,
                    }
                )
    manifest = {
        "version": MODEL_REGISTRY_VERSION,
        "capability": entry.capability,
        "model_id": entry.model_id,
        "source": entry.source,
        "repo": entry.repo,
        "url": entry.url,
        "license": entry.license,
        "note": entry.note,
        "inference": entry.inference,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    _manifest_path(data_dir, entry).write_text(json.dumps(manifest, indent=2))
    return manifest


def load_manifest(data_dir, capability: str) -> dict | None:
    entry = REGISTRY.get(capability)
    if entry is None:
        raise ValueError(f"Unknown capability: {capability}")
    path = _manifest_path(data_dir, entry)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def status(data_dir) -> dict:
    """Per-capability availability, hashes, and license notes (never downloads)."""
    result = {"version": MODEL_REGISTRY_VERSION, "capabilities": []}
    for capability, entry in REGISTRY.items():
        manifest = load_manifest(data_dir, capability)
        available = bool(manifest) and all(
            (models_dir(data_dir) / capability / item["name"]).is_file() for item in manifest["files"]
        )
        result["capabilities"].append(
            {
                "capability": capability,
                "model_id": entry.model_id,
                "source": entry.source,
                "license": entry.license,
                "inference": entry.inference,
                "available": available,
                "files": manifest["files"] if manifest else [],
                "downloaded_at": manifest["downloaded_at"] if manifest else None,
            }
        )
    result["model_free"] = MODEL_FREE_NOTES
    return result
