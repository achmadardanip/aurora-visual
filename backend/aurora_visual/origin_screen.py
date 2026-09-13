"""Local origin/provenance screening that never decides claim truth."""

import json
from datetime import datetime, timezone
from pathlib import Path

import c2pa
from PIL import ExifTags, Image

GENERATOR_MARKERS = (
    "midjourney",
    "dall-e",
    "stable diffusion",
    "comfyui",
    "automatic1111",
    "adobe firefly",
    "runway",
    "ideogram",
    "leonardo.ai",
    "bing image creator",
    "google imagen",
    "openai",
)
C2PA_MARKERS = (b"c2pa", b"jumbf", b"content credentials")
# Digital source types (IPTC NewsCodes CV) that state the media itself was
# produced by a trained algorithm; only these flip the origin decision, and
# only when the manifest signature validates.
TRAINED_ALGORITHMIC_SOURCES = {
    "trainedAlgorithmicMedia",
    "compositeWithTrainedAlgorithmicMedia",
}
_MAX_C2PA_REPORT_BYTES = 4_000_000


def _metadata(path: Path):
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            fields = sorted({ExifTags.TAGS.get(key, str(key)) for key in exif})
            software = str(
                exif.get(305, "") or image.info.get("Software", "") or image.info.get("software", "")
            ).casefold()
            info_fields = sorted(key for key in image.info if key.lower() not in {"icc_profile", "exif"})
    except (OSError, ValueError):
        return {
            "status": "unavailable",
            "field_names": [],
            "info_field_names": [],
            "camera_metadata_present": False,
            "capture_time_present": False,
            "software_classification": "unavailable",
        }
    return {
        "status": "observed",
        "field_names": fields,
        "info_field_names": info_fields,
        "camera_metadata_present": bool({"Make", "Model"} & set(fields)),
        "capture_time_present": bool({"DateTime", "DateTimeOriginal", "DateTimeDigitized"} & set(fields)),
        "software_classification": (
            "known_generative_marker"
            if any(marker in software for marker in GENERATOR_MARKERS)
            else "software_present_unclassified"
            if software
            else "not_present"
        ),
    }


def _digital_source_types(manifest: dict) -> list[str]:
    """Extract IPTC digitalSourceType tokens from action assertions."""
    found = []
    for assertion in manifest.get("assertions", []):
        if not isinstance(assertion, dict) or not str(assertion.get("label", "")).startswith("c2pa.actions"):
            continue
        data = assertion.get("data") if isinstance(assertion.get("data"), dict) else {}
        for action in data.get("actions", []):
            if not isinstance(action, dict):
                continue
            source = action.get("digitalSourceType")
            if isinstance(source, str) and source:
                token = source.rsplit("/", 1)[-1]
                if token and token not in found:
                    found.append(token)
    return found[:10]


def _claim_generator(manifest: dict) -> str | None:
    infos = manifest.get("claim_generator_info")
    if isinstance(infos, list) and infos and isinstance(infos[0], dict):
        name = str(infos[0].get("name", ""))[:120]
        version = str(infos[0].get("version", ""))[:60]
        return " ".join(part for part in (name, version) if part) or None
    return None


def _signer(manifest: dict) -> dict | None:
    info = manifest.get("signature_info")
    if not isinstance(info, dict):
        return None
    return {
        "alg": str(info.get("alg", ""))[:40] or None,
        "issuer": str(info.get("issuer", ""))[:120] or None,
    }


def _load_trust_anchors(value: str) -> tuple[str | None, bool]:
    """Resolve optional extra trust anchors: a PEM file path or inline PEM."""
    value = value.strip()
    if not value:
        return None, True
    if "BEGIN CERTIFICATE" in value:
        return value, True
    path = Path(value)
    if path.is_file():
        try:
            pem = path.read_text()
        except OSError:
            return None, False
        return (pem, True) if "BEGIN CERTIFICATE" in pem else (None, False)
    return None, False


def _read_manifest_store(path: Path, media_type: str, trust_anchors: str | None):
    """Parse and validate the C2PA manifest store from original bytes.

    Returns (store, status, validation_status) where status distinguishes a
    missing manifest from an unreadable one. Validation runs against the SDK's
    default trust store plus optional operator anchors; it never fetches
    remote manifests.
    """
    try:
        if trust_anchors:
            settings = c2pa.Settings.from_dict({"trust": {"trust_anchors": trust_anchors}})
            with c2pa.Context(settings) as ctx, open(path, "rb") as stream:
                with c2pa.Reader(media_type, stream, context=ctx) as reader:
                    report = reader.json()
        else:
            with open(path, "rb") as stream:
                with c2pa.Reader(media_type, stream) as reader:
                    report = reader.json()
    except c2pa.C2paError.ManifestNotFound:
        return None, "not_detected", []
    except c2pa.C2paError as exc:
        reason = str(exc)[:200]
        return None, "unavailable", [{"code": "reader_error", "explanation": reason}]
    if len(report) > _MAX_C2PA_REPORT_BYTES:
        return None, "unavailable", [{"code": "report_too_large", "explanation": None}]
    try:
        store = json.loads(report)
    except ValueError:
        return None, "unavailable", [{"code": "reader_error", "explanation": "malformed manifest JSON"}]
    if not isinstance(store, dict) or not isinstance(store.get("manifests", {}), dict):
        return None, "unavailable", [{"code": "reader_error", "explanation": "unexpected manifest store"}]
    raw = store.get("validation_status")
    codes = []
    if isinstance(raw, list):
        for item in raw[:20]:
            if isinstance(item, dict) and item.get("code"):
                codes.append(
                    {"code": str(item["code"])[:80], "explanation": _bounded(item.get("explanation"))}
                )
    return store, "manifest_present", codes


def _bounded(value, limit=200):
    return str(value)[:limit] if value is not None else None


def _provenance(path: Path, media_type: str, trust_anchors_setting: str):
    raw = path.read_bytes()
    markers = [marker.decode("ascii") for marker in C2PA_MARKERS if marker in raw.lower()]
    anchors, anchors_ok = _load_trust_anchors(trust_anchors_setting)
    store, status, codes = _read_manifest_store(path, media_type, anchors)
    if not anchors_ok:
        return {
            "c2pa": {
                "status": "unavailable",
                "verification": "not_verified",
                "validation_status": [{"code": "trust_anchors_misconfigured", "explanation": None}],
                "claim_generator": None,
                "signer": None,
                "digital_source_types": [],
                "manifest_count": 0,
                "marker_presence": markers,
                "message": "Konfigurasi trust anchor C2PA tidak valid; manifest tidak divalidasi.",
            },
            "watermark": {
                "status": "unavailable",
                "message": "Tidak ada detektor watermark tak terlihat yang dikonfigurasi.",
            },
        }
    if status == "not_detected":
        return {
            "c2pa": {
                "status": "not_detected",
                "verification": "not_applicable",
                "validation_status": [],
                "claim_generator": None,
                "signer": None,
                "digital_source_types": [],
                "manifest_count": 0,
                "marker_presence": markers,
                "message": (
                    "Penanda byte C2PA/JUMBF ditemukan tanpa manifest yang dapat diurai."
                    if markers
                    else "Tidak ada manifest C2PA pada byte unggahan."
                ),
            },
            "watermark": {
                "status": "unavailable",
                "message": "Tidak ada detektor watermark tak terlihat yang dikonfigurasi.",
            },
        }
    if status == "unavailable":
        return {
            "c2pa": {
                "status": "unavailable",
                "verification": "not_verified",
                "validation_status": codes,
                "claim_generator": None,
                "signer": None,
                "digital_source_types": [],
                "manifest_count": 0,
                "marker_presence": markers,
                "message": "Manifest C2PA tidak dapat dibaca/divalidasi pada byte unggahan.",
            },
            "watermark": {
                "status": "unavailable",
                "message": "Tidak ada detektor watermark tak terlihat yang dikonfigurasi.",
            },
        }
    manifests = store.get("manifests", {})
    active = manifests.get(store.get("active_manifest"), {})
    active = active if isinstance(active, dict) else {}
    signature_valid = not codes
    return {
        "c2pa": {
            "status": "verified" if signature_valid else "present_unverified",
            "verification": "signature_validated" if signature_valid else "not_verified",
            "validation_status": codes,
            "claim_generator": _claim_generator(active),
            "signer": _signer(active),
            "digital_source_types": _digital_source_types(active),
            "manifest_count": len(manifests),
            "marker_presence": markers,
            "message": (
                "Manifest C2PA ditemukan dan tanda tangan tervalidasi terhadap trust store c2pa-python; "
                "bukan audit rantai kepercayaan penuh dan tidak membuktikan kebenaran caption."
                if signature_valid
                else "Manifest C2PA ditemukan tetapi tidak lolos validasi tanda tangan/integritas; "
                "klaim provenance di dalamnya tidak dipercaya."
            ),
        },
        "watermark": {
            "status": "unavailable",
            "message": "Tidak ada detektor watermark tak terlihat yang dikonfigurasi.",
        },
    }


def _decision(metadata, provenance, mode):
    c2pa = provenance["c2pa"]
    if c2pa["status"] == "verified" and set(c2pa["digital_source_types"]) & TRAINED_ALGORITHMIC_SOURCES:
        return {
            "label": "likely_ai_generated",
            "rationale": (
                "Manifest C2PA tervalidasi menyatakan digitalSourceType media algoritmik terlatih "
                "(AI generatif)."
            ),
        }
    if metadata["software_classification"] == "known_generative_marker":
        return {
            "label": "likely_ai_generated",
            "rationale": "Metadata perangkat lunak berisi indikator alat generatif yang dikenal.",
        }
    if metadata["camera_metadata_present"] and c2pa["status"] not in {"verified", "present_unverified"}:
        return {
            "label": "no_strong_ai_signal",
            "rationale": "Metadata kamera tersedia, tanpa indikator generatif kuat pada pemeriksaan lokal.",
        }
    if mode == "demo":
        return {
            "label": "inconclusive",
            "rationale": "Fixture demo tidak diperlakukan sebagai hasil detektor asal media.",
        }
    return {
        "label": "inconclusive",
        "rationale": "Pemeriksaan metadata/provenance lokal belum cukup untuk menyimpulkan asal media.",
    }


def screen_image(path, media, mode, c2pa_trust_anchors: str = ""):
    """Return a bounded, privacy-preserving screen for one uploaded asset."""
    path = Path(path)
    metadata = _metadata(path)
    provenance = _provenance(path, media.media_type, c2pa_trust_anchors)
    decision = _decision(metadata, provenance, mode)
    return {
        "version": "origin-screen-v2",
        "target": {"asset_id": media.asset_id, "sha256": media.sha256},
        "decision": {
            **decision,
            "does_not_affect_visual_assessment": True,
            "does_not_decide_claim_truth": True,
        },
        "metadata": metadata,
        "provenance": provenance,
        "detectors": [
            {
                "task": "ai_generation_detection",
                "provider": "unconfigured",
                "status": "unavailable",
                "assessment": "not_assessed",
                "message": "Tidak ada model deteksi gambar AI tervalidasi yang dikonfigurasi.",
            },
            {
                "task": "deepfake_detection",
                "provider": "unconfigured",
                "status": "unavailable",
                "assessment": "not_assessed",
                "message": "Tidak ada detektor manipulasi wajah/deepfake yang dikonfigurasi.",
            },
        ],
        "limitations": [
            "Metadata dapat dihapus atau diubah dan bukan bukti asal media.",
            "Validasi C2PA memeriksa tanda tangan manifest terhadap trust store bawaan c2pa-python; "
            "rantai kepercayaan penuh, tanda waktu, dan kebijakan signer tidak diaudit.",
            "Tidak adanya sinyal AI bukan bukti bahwa gambar berasal dari kamera.",
            "Hasil screening tidak mengubah status visual atau verdict faktual.",
        ],
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }
