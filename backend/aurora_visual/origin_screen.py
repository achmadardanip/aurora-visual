"""Local origin/provenance screening that never decides claim truth."""

from datetime import datetime, timezone
from pathlib import Path

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


def _provenance(raw: bytes):
    markers = [marker.decode("ascii") for marker in C2PA_MARKERS if marker in raw.lower()]
    c2pa_present = bool(markers)
    return {
        "c2pa": {
            "status": "marker_present" if c2pa_present else "not_detected",
            "verification": "not_verified",
            "message": (
                "Penanda C2PA/JUMBF ditemukan, tetapi manifest tidak diverifikasi secara kriptografis."
                if c2pa_present
                else "Tidak ada penanda C2PA/JUMBF yang ditemukan pada byte unggahan."
            ),
        },
        "watermark": {
            "status": "unavailable",
            "message": "Tidak ada detektor watermark tak terlihat yang dikonfigurasi.",
        },
    }


def _decision(metadata, provenance, mode):
    if metadata["software_classification"] == "known_generative_marker":
        return {
            "label": "likely_ai_generated",
            "rationale": "Metadata perangkat lunak berisi indikator alat generatif yang dikenal.",
        }
    if metadata["camera_metadata_present"] and provenance["c2pa"]["status"] != "marker_present":
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


def screen_image(path, media, mode):
    """Return a bounded, privacy-preserving screen for one uploaded asset."""
    path = Path(path)
    raw = path.read_bytes()
    metadata = _metadata(path)
    provenance = _provenance(raw)
    decision = _decision(metadata, provenance, mode)
    return {
        "version": "origin-screen-v1",
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
            "Marker C2PA tidak diverifikasi tanpa validator manifest dan rantai kepercayaan.",
            "Tidak adanya sinyal AI bukan bukti bahwa gambar berasal dari kamera.",
            "Hasil screening tidak mengubah status visual atau verdict faktual.",
        ],
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }
