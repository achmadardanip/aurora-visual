"""Hive V3 image AI/deepfake detection adapter for module 2's forensic branch.

Uses the same documented model endpoint as module 1 (stage-1 detection):
model hive/ai-generated-and-deepfake-content-detection on original bytes.
Raw scores keep the provider scale [0,1]; ai_generated_score is a min-max
mapping of that scale, explicitly not a calibrated probability.
"""

import base64

import httpx

from aurora_evidence.core.providers.interfaces import ImageAIDetector

V3_ENDPOINT = "https://api.thehive.ai/api/v3/task/async"
V3_RESULT_ENDPOINT = "https://api.thehive.ai/api/v3/task/{task_id}"
MODEL = "hive/ai-generated-and-deepfake-content-detection"


class HiveImageDetector(ImageAIDetector):
    name = "hive-v3"
    capability = "ai_generation_detection_image"

    def status(self, settings) -> str:
        return "ok" if settings.hive_api_key else "unconfigured"

    def detect(self, image_bytes: bytes, media_type: str, settings) -> dict:
        if not settings.hive_api_key:
            return {
                "status": "unavailable",
                "error_code": "unconfigured",
                "limitations": ["API key Hive belum diatur di server"],
            }
        data_uri = f"data:{media_type};base64," + base64.b64encode(image_bytes).decode("ascii")
        try:
            with httpx.Client(timeout=settings.provider_timeout) as client:
                submit = client.post(
                    V3_ENDPOINT,
                    headers={"Authorization": f"Bearer {settings.hive_api_key}"},
                    json={"model_name": MODEL, "input": {"source": data_uri}},
                )
                if submit.status_code in (401, 403):
                    return {"status": "unavailable", "error_code": "auth_error", "limitations": []}
                if submit.status_code == 429:
                    return {"status": "failed", "error_code": "rate_limited", "limitations": []}
                if submit.status_code >= 400:
                    return {
                        "status": "failed",
                        "error_code": f"http_error_{submit.status_code}",
                        "limitations": [],
                    }
                task_id = submit.json().get("task_id")
                # Async polling with bounded attempts.
                for _ in range(settings.hive_poll_attempts):
                    poll = client.get(
                        V3_RESULT_ENDPOINT.format(task_id=task_id),
                        headers={"Authorization": f"Bearer {settings.hive_api_key}"},
                    )
                    if poll.status_code >= 400:
                        return {
                            "status": "failed",
                            "error_code": f"http_error_{poll.status_code}",
                            "limitations": [],
                        }
                    payload = poll.json()
                    status = payload.get("status")
                    if status == "finished":
                        return self._normalize(payload)
                    if status in ("failed", "rejected"):
                        return {
                            "status": "failed",
                            "error_code": "provider_task_failed",
                            "limitations": [],
                        }
                return {"status": "failed", "error_code": "poll_timeout", "limitations": []}
        except httpx.HTTPError as exc:
            code = "timeout" if isinstance(exc, httpx.TimeoutException) else "network_error"
            return {"status": "failed", "error_code": code, "limitations": []}

    @staticmethod
    def _normalize(payload: dict) -> dict:
        try:
            classes = payload["output"]["classes"]
        except (KeyError, TypeError):
            return {"status": "failed", "error_code": "malformed_response", "limitations": []}
        scores = {}
        for entry in classes:
            label, value = entry.get("class"), entry.get("score")
            if isinstance(value, (int, float)):
                scores[label] = float(value)
        ai = max(scores.get("ai_generated", 0.0), scores.get("deepfake", 0.0))
        return {
            "status": "ok",
            "raw_score": ai,
            "raw_scale": {"min": 0.0, "max": 1.0, "higher_means_ai": True},
            "raw_label": "ai_generated" if scores.get("ai_generated", 0) >= 0.5 else "not_ai",
            "model_version": MODEL,
            "limitations": [
                "Skor provider; bukan probabilitas terkalibrasi lokal",
                "Bukan indikasi kebenaran klaim; konten AI bisa saja klaimnya benar",
            ],
        }
