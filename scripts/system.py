"""Run all three AURORA services as one coordinated system.

Services:
  - aurora-visual   (module 1): API :8101, UI :5171  — analysis
  - aurora-evidence (module 2): API :8102, UI :5172  — retrieval
  - aurora-decision (module 3): API :8103, UI :5173  — fusion + orchestrator

The decision service is preconfigured to call the other two, so its
"Pipeline penuh" button runs caption+image through all three modules.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]

SERVICES = [
    {
        "name": "aurora-visual",
        "api_port": 8101,
        "ui_port": 5171,
        "api_app": "app.api.main:app",
        "backend": ROOT / "backend",
        "frontend": ROOT / "frontend",
    },
    {
        "name": "aurora-evidence",
        "api_port": 8102,
        "ui_port": 5172,
        "api_app": "aurora_evidence.api.main:app",
        "backend": ROOT / "aurora-evidence" / "backend",
        "frontend": ROOT / "aurora-evidence" / "frontend",
    },
    {
        "name": "aurora-decision",
        "api_port": 8103,
        "ui_port": 5173,
        "api_app": "aurora_decision.api.main:app",
        "backend": ROOT / "aurora-decision" / "backend",
        "frontend": ROOT / "aurora-decision" / "frontend",
    },
]


def wait_ready(url, process, timeout=90):
    deadline = time.monotonic() + timeout
    with httpx.Client(timeout=2) as client:
        while time.monotonic() < deadline:
            if process is not None and process.poll() is not None:
                raise RuntimeError("Process exited early; see log above")
            try:
                if client.get(url).status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(0.3)
    raise RuntimeError(f"Service did not become ready: {url}")


def main():
    children = []
    try:
        for service in SERVICES:
            env = {
                **os.environ,
                "AURORA_HOST": "127.0.0.1",
                "AURORA_PORT": str(service["api_port"]),
                "AURORA_FRONTEND_PORT": str(service["ui_port"]),
                "AURORA_API_INTERNAL_URL": f"http://127.0.0.1:{service['api_port']}",
                "AURORA_BROWSER_URL": f"http://127.0.0.1:{service['ui_port']}",
                "AURORA_CORS": (
                    f"http://localhost:{service['ui_port']},http://127.0.0.1:{service['ui_port']}"
                ),
            }
            if service["name"] == "aurora-evidence":
                env["AURORA_EVIDENCE_DATA_DIR"] = str(ROOT / "var-evidence")
            if service["name"] == "aurora-decision":
                env["AURORA_DECISION_DATA_DIR"] = str(ROOT / "var-decision")
                env["AURORA_VISUAL_URL"] = "http://127.0.0.1:8101"
                env["AURORA_EVIDENCE_URL"] = "http://127.0.0.1:8102"
            print(
                f"[system] starting {service['name']} (API :{service['api_port']}, UI :{service['ui_port']})"
            )
            children.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        service["api_app"],
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(service["api_port"]),
                    ],
                    cwd=service["backend"],
                    env=env,
                )
            )
            children.append(
                subprocess.Popen(
                    ["npm", "run", "dev", "--", "--port", str(service["ui_port"])],
                    cwd=service["frontend"],
                    env=env,
                )
            )
        for service in SERVICES:
            wait_ready(f"http://127.0.0.1:{service['api_port']}/health", None)
        print()
        print("=" * 62)
        print("AURORA system is running:")
        print()
        print("  Module 1 · Visual    UI: http://127.0.0.1:5171  (API :8101)")
        print("  Module 2 · Evidence  UI: http://127.0.0.1:5172  (API :8102)")
        print("  Module 3 · Decision  UI: http://127.0.0.1:5173  (API :8103)")
        print()
        print("  Start at the Decision UI (5173) for the full three-module")
        print("  pipeline, or use each UI independently. Ctrl+C stops all.")
        print("=" * 62)
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
        while all(p.poll() is None for p in children):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[system] stopping all services...")
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        print("[system] stopped.")


if __name__ == "__main__":
    main()
