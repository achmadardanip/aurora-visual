"""Clean HTTP smoke: new temp DB, subprocess worker, real upload, all exports, restart."""

import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import httpx
from app.models.contract import AuroraBundle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "handoff"
OUT.mkdir(parents=True, exist_ok=True)
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
with tempfile.TemporaryDirectory(prefix="aurora-smoke-") as temp:
    env = {
        **os.environ,
        "AURORA_DATA_DIR": temp,
        "AURORA_PUBLIC": "false",
        "AURORA_MAFINDO_API_KEY": "",
        "AURORA_MAFINDO_TIMEOUT_SECONDS": "12",
    }
    logfile = (ROOT / "artifacts/reports/http-smoke-server.log").open("w")

    def launch():
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.api.main:app", "--host", "127.0.0.1", "--port", str(port)],
            env=env,
            cwd=ROOT,
            stdout=logfile,
            stderr=logfile,
        )
        for _ in range(150):
            try:
                if httpx.get(f"http://127.0.0.1:{port}/ready").status_code == 200:
                    return process
            except httpx.ConnectError:
                pass
            if process.poll() is not None:
                raise RuntimeError("Smoke server failed")
            time.sleep(0.1)
        process.terminate()
        raise RuntimeError("Server startup timeout")

    process = launch()
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30) as client:
            results = []
            for fixture in ("supported", "contradicted", "unobservable"):
                b = client.post("/api/v1/demo/" + fixture).json()
                key = "http-smoke-" + fixture
                response = client.post("/api/v1/analyze", json=b, headers={"Idempotency-Key": key})
                response.raise_for_status()
                job = response.json()["job_id"]
                for _ in range(150):
                    state = client.get("/api/v1/jobs/" + job).json()
                    if state["status"] not in ("queued", "running"):
                        break
                    time.sleep(0.1)
                assert state["status"] in ("succeeded", "partial"), state
                AuroraBundle.model_validate(state["result"])
                for fmt in ("json", "zip", "csv", "overlay"):
                    exported = client.get(f"/api/v1/cases/{b['case_id']}/export", params={"format": fmt})
                    exported.raise_for_status()
                    (OUT / f"{fixture}.{'png' if fmt == 'overlay' else fmt}").write_bytes(exported.content)
                results.append(
                    {
                        "fixture": fixture,
                        "case_id": b["case_id"],
                        "job_id": job,
                        "status": state["status"],
                        "labels": [
                            v["visual_status"] for v in state["result"]["analysis"]["visual_assessments"]
                        ],
                    }
                )
            image = Image.new("RGB", (80, 60), "red")
            blob = io.BytesIO()
            image.save(blob, "PNG")
            ref = client.post(
                "/api/v1/media", files={"image": ("live.png", blob.getvalue(), "image/png")}
            ).json()
            live = {
                **b,
                "case_id": str(uuid4()),
                "mode": "live",
                "warnings": [],
                "extensions": {},
                "input": {**b["input"], "image": ref, "claim_text": "Bidang ini berwarna merah"},
            }
            response = client.post("/api/v1/analyze", json=live, headers={"Idempotency-Key": "live-smoke"})
            response.raise_for_status()
            job = response.json()["job_id"]
            for _ in range(150):
                state = client.get("/api/v1/jobs/" + job).json()
                if state["status"] not in ("queued", "running"):
                    break
                time.sleep(0.1)
            assert state["result"]["analysis"]["visual_assessments"][0]["inference_kind"] == "heuristic"
            (OUT / "live-local.json").write_text(json.dumps(state["result"], indent=2))
            process.terminate()
            process.wait(timeout=8)
            process = launch()
            assert client.get("/api/v1/jobs/" + job).json()["result"] == state["result"]
            replay = client.post("/api/v1/analyze", json=live, headers={"Idempotency-Key": "live-smoke"})
            replay.raise_for_status()
            assert replay.json()["job_id"] == job
            results.append(
                {
                    "live_local": "verified",
                    "restart_persistence": "verified",
                    "idempotency_after_restart": "verified",
                }
            )
            report = {
                "clean_temporary_database": True,
                "http": True,
                "isolated_worker": True,
                "results": results,
                "research_evaluated": False,
            }
            (ROOT / "artifacts/reports/http-smoke.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2))
    finally:
        process.terminate()
        process.wait(timeout=10)
        logfile.close()
