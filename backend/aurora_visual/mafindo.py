"""Read-only MAFINDO V1 diagnostic client; never produces AURORA evidence or verdicts."""

from dataclasses import dataclass
from math import isfinite
from time import perf_counter
from urllib.parse import quote

import httpx

BASE_URL = "https://yudistira.turnbackhoax.id/Antihoax"
SEARCH_FIELDS = {"title", "source_link", "content", "tags"}
_MAX_STATUS_NUMBER = 1_000_000_000_000


def redact_url(value: str, key: str = ""):
    """Remove the credential-bearing final path segment before surfacing a URL."""
    if key:
        return value.replace("/" + quote(key, safe="") + "/", "/[REDACTED]/").replace(
            "/" + quote(key, safe=""), "/[REDACTED]"
        )
    parts = value.rstrip("/").split("/")
    return "/".join([*parts[:-1], "[REDACTED]"])


def _safe_text(value, api_key: str, limit=500):
    """Bound provider text and redact raw or URL-encoded credential echoes."""
    text = "" if value is None else str(value)
    if api_key:
        secret = api_key
        variants = {secret}
        for _ in range(2):
            secret = quote(secret, safe="")
            variants.add(secret)
        for variant in sorted(variants, key=len, reverse=True):
            text = text.replace(variant, "[REDACTED]")
    return text[:limit]


def _safe_scalar(value, api_key: str):
    """Return a bounded JSON scalar, never a provider-controlled nested object."""
    if isinstance(value, str):
        return _safe_text(value, api_key, limit=200)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value if abs(value) <= _MAX_STATUS_NUMBER else None
    if isinstance(value, float):
        return value if isfinite(value) and abs(value) <= _MAX_STATUS_NUMBER else None
    return None


@dataclass
class MafindoClient:
    api_key: str
    timeout: float = 12
    transport: httpx.BaseTransport | None = None

    @property
    def configured(self):
        return bool(self.api_key)

    def _request(self, *segments):
        if not self.configured:
            return {
                "provider": "mafindo-v1",
                "capability": "fact-check-provider-diagnostic",
                "status": "unconfigured",
                "message": "AURORA_MAFINDO_API_KEY belum dikonfigurasi pada server.",
                "results": [],
            }
        url = "/".join(
            [BASE_URL, *(quote(str(segment), safe="") for segment in segments), quote(self.api_key, safe="")]
        )
        started = perf_counter()
        try:
            with httpx.Client(
                timeout=self.timeout,
                follow_redirects=False,
                transport=self.transport,
            ) as client:
                response = client.get(url, headers={"Accept": "application/json"})
                if len(response.content) > 2_000_000:
                    raise ValueError("response too large")
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError, TypeError):
            return {
                "provider": "mafindo-v1",
                "capability": "fact-check-provider-diagnostic",
                "status": "failed",
                "message": "Pemeriksaan provider MAFINDO gagal; credential dan URL internal disembunyikan.",
                "duration_ms": round((perf_counter() - started) * 1000, 3),
                "results": [],
            }
        rows = data if isinstance(data, list) else data.get("data", []) if isinstance(data, dict) else []
        if not isinstance(rows, list):
            rows = []
        return {
            "provider": "mafindo-v1",
            "capability": "fact-check-provider-diagnostic",
            "status": "ok",
            "message": "Respons provider dinormalisasi untuk uji kompatibilitas; bukan bukti AURORA.",
            "duration_ms": round((perf_counter() - started) * 1000, 3),
            "results": [
                {
                    "id": _safe_text(row.get("id"), self.api_key, limit=100),
                    "title": _safe_text(row.get("title"), self.api_key),
                    "classification": _safe_text(row.get("classification"), self.api_key, limit=200) or None,
                    "provider_status": _safe_scalar(row.get("status"), self.api_key),
                    "published_at": _safe_text(row.get("tanggal"), self.api_key, limit=100) or None,
                }
                for row in rows[:20]
                if isinstance(row, dict)
            ],
        }

    def latest(self, limit=1):
        if not isinstance(limit, int) or not 1 <= limit <= 20:
            raise ValueError("limit must be 1–20")
        return self._request("latest", limit)

    def search(self, field: str, value: str):
        if field not in SEARCH_FIELDS or not value.strip() or len(value) > 500:
            raise ValueError("invalid MAFINDO search")
        return self._request(field, value)
