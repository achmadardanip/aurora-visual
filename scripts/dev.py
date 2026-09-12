"""Run API supervisor and Vite with coordinated shutdown."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
root = Path(__file__).resolve().parents[1]
host = os.getenv("AURORA_HOST", "127.0.0.1")
if host not in ("127.0.0.1", "localhost", "::1") and os.getenv("AURORA_PUBLIC", "false").lower() != "true":
    raise SystemExit("AURORA_PUBLIC=true and a strong API token are required for a non-localhost bind")
children = []
try:
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=root, check=True)
    children.append(
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.api.main:app",
                "--host",
                host,
                "--port",
                os.getenv("AURORA_PORT", "8101"),
            ],
            cwd=root,
        )
    )
    children.append(
        subprocess.Popen(
            ["npm", "run", "dev", "--", "--port", os.getenv("AURORA_FRONTEND_PORT", "5171")],
            cwd=root / "frontend",
        )
    )

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    while all(p.poll() is None for p in children):
        time.sleep(0.5)
except KeyboardInterrupt:
    pass
finally:
    for child in children:
        if child.poll() is None:
            child.terminate()
    for child in children:
        try:
            child.wait(timeout=8)
        except subprocess.TimeoutExpired:
            child.kill()
