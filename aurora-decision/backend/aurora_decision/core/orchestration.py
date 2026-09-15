"""Upstream orchestration: analyze (module 1) -> retrieve (module 2) -> fuse.

Typed HTTP clients with allowlisted upstream URLs, media relay preserving
asset_id/sha256, deadline polling, bounded retries, and per-stage idempotency.
Never sends local filesystem paths between services.
"""

import json
import time

import httpx

from aurora_decision.contract import sha


class UpstreamError(Exception):
    def __init__(self, code, message, retryable=False):
        self.code, self.message, self.retryable = code, message, retryable
        super().__init__(message)


class UpstreamClient:
    """Minimal typed client for an AURORA service (analyze or retrieve)."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        if not base_url.startswith(("http://127.0.0.1", "http://localhost", "https://")):
            # Allowlist-style guard: operator-configured internal URLs only.
            raise UpstreamError("UPSTREAM_URL_FORBIDDEN", f"URL upstream tidak diizinkan: {base_url}")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.idempotency_prefix = "aurora-decision"

    def idempotency_key(self, endpoint: str, bundle: dict) -> str:
        """Deterministic per (endpoint, case, revision, payload hash) so retries
        return the same upstream job instead of re-submitting."""
        digest = sha(json.dumps(bundle, sort_keys=True).encode())[:24]
        return f"{self.idempotency_prefix}-{endpoint.strip('/').replace('/', '-')}-{digest}"

    def health(self) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(self.base_url + "/health")
            if response.status_code != 200:
                raise UpstreamError("UPSTREAM_UNHEALTHY", f"Upstream {self.base_url} tidak sehat.")
            return response.json()

    def upload_media(self, content: bytes, media_type: str) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                self.base_url + "/api/v1/media",
                files={"images": ("upload", content, media_type)},
            )
            if response.status_code >= 400:
                raise UpstreamError("MEDIA_UPLOAD_FAILED", f"Unggah media gagal: HTTP {response.status_code}")
            images = response.json()["images"]
            return images[0] if images else {}

    def submit(self, endpoint: str, bundle: dict) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                self.base_url + endpoint,
                json=bundle,
                headers={"Idempotency-Key": self.idempotency_key(endpoint, bundle)},
            )
            if response.status_code >= 400:
                body = response.json().get("error", {})
                raise UpstreamError(
                    body.get("code", "UPSTREAM_REJECTED"),
                    body.get("message", f"Upstream menolak bundle: HTTP {response.status_code}"),
                )
            return response.json()

    def poll(self, job_id: str, deadline_seconds: float) -> dict:
        """Poll until terminal status or deadline; returns the job payload."""
        deadline = time.monotonic() + deadline_seconds
        last = None
        with httpx.Client(timeout=self.timeout) as client:
            while time.monotonic() < deadline:
                response = client.get(self.base_url + f"/api/v1/jobs/{job_id}")
                if response.status_code != 200:
                    raise UpstreamError("JOB_POLL_FAILED", f"Polling job gagal: HTTP {response.status_code}")
                last = response.json()
                if last["status"] not in ("queued", "running"):
                    return last
                time.sleep(0.5)
        raise UpstreamError("UPSTREAM_TIMEOUT", f"Upstream job {job_id} melewati batas waktu.", True)


def run_pipeline(
    bundle_dict: dict,
    visual_client: UpstreamClient,
    evidence_client: UpstreamClient,
    media_loader=None,
    deadline_seconds: float = 240.0,
    progress=lambda _: None,
) -> dict:
    """Full pipeline: relay media, analyze, retrieve; fusion runs locally.

    media_loader(asset_id) -> (bytes, media_type) relays original bytes to
    upstream services. The bundle keeps its original MediaRefs (same
    asset_id/sha256); the relayed upload only makes the bytes resolvable
    upstream, it does not replace the references.
    """
    bundle = dict(bundle_dict)
    if media_loader:
        for image in bundle.get("input", {}).get("images", []):
            try:
                content, media_type = media_loader(image["asset_id"])
            except Exception:
                progress(
                    f"Relay media {image['asset_id'][:16]} gagal; layanan upstream mungkin tidak dapat memprosesnya"
                )
                continue
            visual_client.upload_media(content, media_type)
            evidence_client.upload_media(content, media_type)
    progress("Menjalankan analisis visual (layanan 1)")
    visual_client.health()
    job = visual_client.submit("/api/v1/analyze", bundle)
    result = visual_client.poll(job["job_id"], deadline_seconds)
    if result["status"] == "failed":
        raise UpstreamError("UPSTREAM_ANALYZE_FAILED", "Analisis visual gagal di layanan 1.")
    bundle = result["result"]
    progress("Menjalankan pencarian bukti (layanan 2)")
    evidence_client.health()
    job = evidence_client.submit("/api/v1/retrieve", bundle)
    result = evidence_client.poll(job["job_id"], deadline_seconds)
    if result["status"] == "failed":
        progress("Layanan 2 gagal; melanjutkan dengan jalur visual-saja")
    else:
        bundle = result["result"]
    return bundle
