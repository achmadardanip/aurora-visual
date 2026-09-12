"""Run browser acceptance tests against an isolated app and disposable database."""

import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    api_port, ui_port = free_port(), free_port()
    while ui_port == api_port:
        ui_port = free_port()
    ui_url = f"http://127.0.0.1:{ui_port}"
    reports = ROOT / "artifacts/reports"
    reports.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aurora-browser-") as data_dir:
        env = {
            **os.environ,
            "AURORA_DATA_DIR": data_dir,
            "AURORA_PUBLIC": "false",
            "AURORA_HOST": "127.0.0.1",
            "AURORA_PORT": str(api_port),
            "AURORA_FRONTEND_PORT": str(ui_port),
            "AURORA_API_INTERNAL_URL": f"http://127.0.0.1:{api_port}",
            "AURORA_CORS": ui_url,
            "AURORA_BROWSER_URL": ui_url,
            "AURORA_BACKBONE": "local-color-v1",
            "AURORA_OPENCLIP_PRETRAINED": "",
            "AURORA_CHECKPOINT": "",
            "AURORA_LLM_URL": "",
            "AURORA_MAFINDO_API_KEY": "",
            "AURORA_MAFINDO_TIMEOUT_SECONDS": "12",
            "VITE_API_URL": "",
        }
        with (reports / "browser-server.log").open("w") as log:
            app = subprocess.Popen(
                [sys.executable, "scripts/dev.py"],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 45
                with httpx.Client(timeout=2) as client:
                    while time.monotonic() < deadline:
                        if app.poll() is not None:
                            raise RuntimeError("Browser app exited; see artifacts/reports/browser-server.log")
                        try:
                            if client.get(ui_url + "/ready").status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        time.sleep(0.2)
                    else:
                        raise RuntimeError("Browser app startup timed out")
                result = subprocess.run(
                    ["npm", "run", "test:e2e"], cwd=ROOT / "frontend", env=env, check=False
                )
                return result.returncode
            finally:
                # Own process group only: also stop npm's Vite child on failure/interruption.
                try:
                    os.killpg(app.pid, signal.SIGTERM)
                    app.wait(timeout=12)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    os.killpg(app.pid, signal.SIGKILL)
                    app.wait(timeout=3)


if __name__ == "__main__":
    raise SystemExit(main())
