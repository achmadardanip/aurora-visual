"""Server-side SynthID Detector adapter for stage-1 invisible watermark screening.

SynthID Detector is currently published by Google DeepMind as an early-access
verification portal; there is no publicly documented API. This adapter therefore
targets an operator-configured gateway endpoint (for example an authorized
partner proxy of the portal) and reports ``unconfigured`` unless both endpoint
and API key are set. It is exercised through mocked HTTP transports only; live
verification has not been performed. Detector output is a probabilistic
observation: it never decides claim truth, and a negative result is never
evidence that an image came from a camera.
"""

import math
import mimetypes
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import httpx
from app.models.contract import Warning, strict_json

PROVIDER = "synthid-detector"
_MAX_RESPONSE = 1_000_000
_MAX_TEXT = 300


class SynthIDError(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _text(value: Any, limit: int = _MAX_TEXT) -> str:
    if not isinstance(value, str):
        return ""
    return value[:limit]


def _score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) and 0 <= result <= 1 else None


def _read_response(response: httpx.Response) -> bytes:
    declared = response.headers.get("content-length")
    if declared:
        try:
            if int(declared) > _MAX_RESPONSE:
                raise SynthIDError("response_too_large")
        except ValueError:
            raise SynthIDError("malformed_response") from None
    data = bytearray()
    for chunk in response.iter_bytes():
        if len(data) + len(chunk) > _MAX_RESPONSE:
            raise SynthIDError("response_too_large")
        data.extend(chunk)
    return bytes(data)


def normalize_detection(payload: Any) -> dict:
    """Normalize the gateway verdict: strict shape, bounded values.

    The contract is {"synthid_detected": bool|null, "confidence": number|null,
    "model": string|null, "message": string|null}; unknown fields are rejected
    so an ambiguous response can never be interpreted as a detection verdict.
    """
    if not isinstance(payload, dict) or set(payload) - {
        "synthid_detected",
        "confidence",
        "model",
        "message",
    }:
        raise SynthIDError("malformed_response")
    if "synthid_detected" not in payload:
        raise SynthIDError("malformed_response")
    detected = payload["synthid_detected"]
    if detected is not None and not isinstance(detected, bool):
        raise SynthIDError("malformed_response")
    confidence = _score(payload.get("confidence"))
    if payload.get("confidence") is not None and confidence is None:
        raise SynthIDError("malformed_response")
    for field in ("model", "message"):
        if payload.get(field) is not None and not isinstance(payload[field], str):
            raise SynthIDError("malformed_response")
    model = _text(payload.get("model"), 120) or None
    message = _text(payload.get("message")) or None
    return {
        "synthid_detected": detected,
        "confidence": confidence,
        "model": model,
        "message": message,
    }


@dataclass
class SynthIDClient:
    endpoint: str
    api_key: str
    timeout: float = 45
    transport: httpx.BaseTransport | None = None

    @property
    def configured(self):
        return bool(self.endpoint and self.api_key)

    def detect_watermark(self, path) -> dict:
        """Upload original bytes to the configured gateway and return the verdict."""
        if not self.configured:
            raise SynthIDError("unconfigured")
        media_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        started = perf_counter()
        try:
            with (
                httpx.Client(
                    timeout=self.timeout,
                    follow_redirects=False,
                    transport=self.transport,
                ) as client,
                open(path, "rb") as media,
            ):
                with client.stream(
                    "POST",
                    self.endpoint,
                    headers={"Authorization": "Bearer " + self.api_key, "Accept": "application/json"},
                    files={"media": (path.name, media, media_type)},
                ) as response:
                    if response.is_redirect:
                        raise SynthIDError("redirect_rejected")
                    if response.status_code == 429:
                        raise SynthIDError("rate_limited")
                    if response.status_code >= 400:
                        raise SynthIDError("http_error")
                    content = _read_response(response)
            payload = strict_json(content)
        except SynthIDError:
            raise
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise SynthIDError(
                "timeout" if isinstance(exc, httpx.TimeoutException) else "network_error"
            ) from None
        except (OSError, ValueError, TypeError):
            raise SynthIDError("malformed_response") from None
        normalized = normalize_detection(payload)
        normalized["duration_ms"] = round((perf_counter() - started) * 1000, 3)
        return normalized


def stage1_watermark(detected: dict) -> dict:
    """Map a normalized verdict to the screening watermark slot.

    A positive result only states that a SynthID watermark (Google AI output)
    was reported by the gateway; a negative result only states that no SynthID
    watermark was found — most generators do not watermark, so it is never
    treated as authenticity evidence.
    """
    model = detected.get("model")
    entry = {
        "task": "invisible_watermark_detection",
        "provider": PROVIDER,
        "status": "inconclusive",
        "assessment": "not_assessed",
        "score": detected.get("confidence"),
        "model": model,
        "calibration": "provider_claimed_not_locally_validated",
        "message": "Verdikt watermark belum dapat disimpulkan.",
    }
    if detected["synthid_detected"] is True:
        return {
            **entry,
            "status": "ok",
            "assessment": "synthid_watermark_detected",
            "message": (
                "Watermark SynthID dilaporkan terdeteksi: indikasi konten dihasilkan/diedit Google AI. "
                "Sinyal probabilistik provider; bukan bukti kebenaran caption."
            ),
        }
    if detected["synthid_detected"] is False:
        return {
            **entry,
            "status": "inconclusive",
            "assessment": "no_synthid_watermark",
            "message": (
                "Watermark SynthID tidak terdeteksi. Bukan bukti asal kamera/non-AI karena tidak semua "
                "generator memakai SynthID."
            ),
        }
    return {
        **entry,
        "assessment": "uncertain",
        "message": detected.get("message") or "Provider tidak memberikan verdikt biner.",
    }


def watermark_for_error(code: str) -> dict:
    status = {
        "unconfigured": "unconfigured",
        "rate_limited": "rate_limited",
        "response_too_large": "failed",
        "timeout": "failed",
        "network_error": "failed",
    }.get(code, "failed")
    return {
        "task": "invisible_watermark_detection",
        "provider": PROVIDER if status != "unconfigured" else "unconfigured",
        "status": status,
        "assessment": "not_assessed",
        "score": None,
        "model": None,
        "error_code": code,
        "message": "Detektor watermark tidak menghasilkan verdikt; ini bukan hasil negatif.",
    }


def warning_for_error(code: str, component: str = "synthid_stage1") -> Warning:
    return Warning(
        code="SYNTHID_" + re.sub(r"[^A-Z0-9]+", "_", code.upper()).strip("_"),
        message="Provider SynthID Detector tidak tersedia untuk screening watermark; jalur lokal tetap digunakan.",
        component=component,
    )
