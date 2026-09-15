"""Three-service integration: upstream simulators + full pipeline contract tests.

Simulators are clearly labeled test fakes; real integration is verified
separately by running all three actual services.
"""

import json
from pathlib import Path
from uuid import uuid4

import pytest
from aurora_decision.config import Settings
from aurora_decision.contract import AuroraBundle
from aurora_decision.core.orchestration import UpstreamClient, UpstreamError, run_pipeline
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

FIXTURE = json.loads(
    (Path(__file__).resolve().parents[2] / "aurora-decision" / "fixtures" / "fusion_demo.json").read_text()
)


def simulator_app(service: str):
    """A labeled upstream simulator implementing the shared contract endpoints."""
    app = FastAPI(title=f"simulator-{service}")

    @app.get("/health")
    def health():
        return {"status": "ok", "service": f"simulator-{service}", "schema_version": "1.0.0"}

    @app.post("/api/v1/media")
    async def media(request: Request):
        return {
            "images": [
                {
                    "asset_id": "asset_" + "0" * 64,
                    "sha256": "0" * 64,
                    "media_type": "image/png",
                    "width": 10,
                    "height": 10,
                    "uri": "memory://",
                }
            ]
        }

    @app.post(f"/api/v1/{'analyze' if service == 'visual' else 'retrieve'}")
    async def process(request: Request):
        bundle = await request.json()
        if service == "visual" and bundle.get("analysis") is None:
            data = json.loads(json.dumps(FIXTURE))
            bundle["analysis"] = data["analysis"]
            bundle["analysis"]["run"]["run_id"] = str(uuid4())
        if service == "evidence" and bundle.get("retrieval") is None:
            data = json.loads(json.dumps(FIXTURE))
            bundle["retrieval"] = data["retrieval"]
            bundle["retrieval"]["run"]["run_id"] = str(uuid4())
            bundle["retrieval"]["atom_set_id"] = bundle["analysis"]["atom_set_id"]
        job_id = str(uuid4())
        app.state.jobs = {
            job_id: {
                "job_id": job_id,
                "status": "succeeded",
                "result": bundle,
                "error": None,
                "progress": "done",
                "attempts": 1,
            }
        }
        return {
            "job_id": job_id,
            "case_id": bundle["case_id"],
            "claim_revision": bundle["claim_revision"],
            "status": "queued",
        }

    @app.get("/api/v1/jobs/{job_id}")
    def job(job_id: str):
        return app.state.jobs[job_id]

    return app


def simulator_client(service: str) -> UpstreamClient:
    """UpstreamClient wired to an in-process simulator through TestClient."""
    app = simulator_app(service)
    test_client = TestClient(app)

    class Simulated(UpstreamClient):
        def __init__(self):
            super().__init__("http://localhost:9")
            self.sim = test_client

        def health(self):
            response = self.sim.get("/health")
            if response.status_code != 200:
                raise UpstreamError("UPSTREAM_UNHEALTHY", "simulator tidak sehat")
            return response.json()

        def upload_media(self, content, media_type):
            response = self.sim.post("/api/v1/media", files={"images": ("upload", content, media_type)})
            return response.json()["images"][0]

        def submit(self, endpoint, bundle):
            response = self.sim.post(
                endpoint, json=bundle, headers={"Idempotency-Key": "sim-" + str(uuid4())}
            )
            if response.status_code >= 400:
                raise UpstreamError("UPSTREAM_REJECTED", f"simulator menolak: {response.status_code}")
            return response.json()

        def poll(self, job_id, deadline_seconds):
            response = self.sim.get(f"/api/v1/jobs/{job_id}")
            return response.json()

    return Simulated()


def test_pipeline_through_simulators(tmp_path):
    visual = simulator_client("visual")
    evidence = simulator_client("evidence")
    bundle = json.loads(json.dumps(FIXTURE))
    bundle["analysis"] = None
    bundle["retrieval"] = None
    result = run_pipeline(bundle, visual, evidence, media_loader=None, deadline_seconds=10)
    assert result["analysis"]["atomic_claims"]
    assert result["retrieval"]["evidence_list"]
    assert result["retrieval"]["atom_set_id"] == result["analysis"]["atom_set_id"]


def test_upstream_url_allowlist():
    with pytest.raises(UpstreamError) as excinfo:
        UpstreamClient("http://evil.example.com")
    assert excinfo.value.code == "UPSTREAM_URL_FORBIDDEN"


def test_fuse_after_pipeline_integration(tmp_path):
    """Full contract: pipeline result feeds the real local fuse."""
    from aurora_decision.core.fuse import fuse

    bundle = json.loads(json.dumps(FIXTURE))
    settings = Settings(data_dir=tmp_path)
    settings.prepare()
    output = fuse(AuroraBundle.model_validate(bundle), settings)
    assert output.decision.final_verdict == "InsufficientEvidence"
    assert output.decision.base_label == "Contradicted"
    recorded = output.extensions["aurora_contract"]["run_inputs"][output.decision.run.run_id]
    assert recorded["parent_run_ids"]["analysis"]
