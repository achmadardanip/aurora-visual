"""GPTZero text AI-detection adapter (documented API: https://gptzero.me/docs).

Score semantics per GPTZero docs: fully_generated_prob is the provider's own
probability-like score, NOT a locally validated probability — calibration_status
stays provider_claimed. Indonesian support is not documented; signals for id
carry an explicit limitation. Min length per docs ~150 characters (shorter
input maps to inconclusive input_too_short, never "human-written").
"""

import httpx

from aurora_evidence.core.providers.interfaces import TextAIDetector

MIN_CHARS = 150


class GPTZeroTextDetector(TextAIDetector):
    name = "gptzero"
    capability = "ai_generation_detection_text"
    ENDPOINT = "https://api.gptzero.me/v2/predict/text"

    def status(self, settings) -> str:
        return "ok" if settings.gptzero_api_key else "unconfigured"

    def detect(self, text: str, language: str, settings) -> dict:
        if not settings.gptzero_api_key:
            return {
                "status": "unavailable",
                "error_code": "unconfigured",
                "limitations": ["API key GPTZero belum diatur di server"],
            }
        stripped = text.strip()
        if len(stripped) < MIN_CHARS:
            return {
                "status": "inconclusive",
                "error_code": "input_too_short",
                "limitations": [f"Teks kurang dari {MIN_CHARS} karakter; tidak dapat dinilai"],
            }
        try:
            with httpx.Client(timeout=settings.provider_timeout) as client:
                response = client.post(
                    self.ENDPOINT,
                    headers={"x-api-key": settings.gptzero_api_key},
                    json={"document": stripped},
                )
        except httpx.HTTPError as exc:
            code = "timeout" if isinstance(exc, httpx.TimeoutException) else "network_error"
            return {"status": "failed", "error_code": code, "limitations": []}
        if response.status_code in (401, 403):
            return {"status": "unavailable", "error_code": "auth_error", "limitations": []}
        if response.status_code == 429:
            return {"status": "failed", "error_code": "rate_limited", "limitations": []}
        if response.status_code >= 400:
            return {
                "status": "failed",
                "error_code": f"http_error_{response.status_code}",
                "limitations": [],
            }
        try:
            documents = response.json().get("documents", [])
            result = documents[0] if documents else response.json()
            raw = float(result.get("fully_generated_prob"))
        except (ValueError, TypeError, KeyError, IndexError):
            return {"status": "failed", "error_code": "malformed_response", "limitations": []}
        limitations = ["Skor provider; bukan probabilitas terkalibrasi lokal"]
        if not language.startswith("en"):
            limitations.append(f"Dukungan bahasa {language} tidak terdokumentasi resmi")
        return {
            "status": "ok",
            "raw_score": raw,
            "raw_scale": {"min": 0.0, "max": 1.0, "higher_means_ai": True},
            "raw_label": "generated" if raw >= 0.5 else "human",
            "model_version": "gptzero-v2",
            "limitations": limitations,
        }
