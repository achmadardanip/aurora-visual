"""Module 3 settings: fusion/calibration config + upstream service URLs."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load only this module's dedicated env file (if present); the shared .env of
# other AURORA services must not leak provider keys into this app's Settings.
load_dotenv(Path(__file__).resolve().parents[2] / ".env.decision", override=False)

UI_FIELDS: dict[str, tuple[str, bool]] = {
    "alpha": ("float", False),
    "head_checkpoint": ("str", False),
    "visual_url": ("str", False),
    "evidence_url": ("str", False),
    "upstream_deadline": ("float", False),
}

UI_RANGES = {
    "alpha": (0.01, 0.5),
    "upstream_deadline": (30, 900),
}


def normalize_overlay(values: dict) -> dict:
    result = {}
    for name, value in values.items():
        if name not in UI_FIELDS:
            raise ValueError(f"Kolom konfigurasi tidak dikenal: {name}")
        kind, _ = UI_FIELDS[name]
        if kind == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} harus angka")
            result[name] = float(value)
        elif not isinstance(value, str):
            raise ValueError(f"{name} harus teks")
        else:
            result[name] = value.strip()
    for name, (low, high) in UI_RANGES.items():
        if name in result and not low <= result[name] <= high:
            raise ValueError(f"{name} harus antara {low} dan {high}")
    for url_name in ("visual_url", "evidence_url"):
        url = result.get(url_name)
        if url and not url.startswith(("http://", "https://")):
            raise ValueError(f"{url_name} harus URL http(s)")
    return result


@dataclass
class Settings:
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("AURORA_DECISION_DATA_DIR", "var-decision")).resolve()
    )
    public: bool = field(default_factory=lambda: os.getenv("AURORA_PUBLIC", "false").lower() == "true")
    token: str = field(default_factory=lambda: os.getenv("AURORA_API_TOKEN", ""))
    max_upload: int = field(default_factory=lambda: int(os.getenv("AURORA_MAX_UPLOAD_MB", "10")) * 1024**2)
    max_images: int = field(default_factory=lambda: int(os.getenv("AURORA_MAX_IMAGES", "8")))
    job_timeout: int = field(default_factory=lambda: int(os.getenv("AURORA_JOB_TIMEOUT", "300")))
    max_pending: int = field(default_factory=lambda: int(os.getenv("AURORA_MAX_PENDING", "20")))
    alpha: float = field(default_factory=lambda: float(os.getenv("AURORA_DECISION_ALPHA", "0.1")))
    head_checkpoint: str = field(default_factory=lambda: os.getenv("AURORA_DECISION_HEAD", ""))
    calibration_id: str = field(default_factory=lambda: os.getenv("AURORA_DECISION_CALIBRATION_ID", ""))
    visual_url: str = field(default_factory=lambda: os.getenv("AURORA_VISUAL_URL", "http://127.0.0.1:8101"))
    evidence_url: str = field(
        default_factory=lambda: os.getenv("AURORA_EVIDENCE_URL", "http://127.0.0.1:8102")
    )
    upstream_deadline: float = field(
        default_factory=lambda: float(os.getenv("AURORA_UPSTREAM_DEADLINE", "240"))
    )
    allowed_hosts: list[str] = field(
        default_factory=lambda: [
            item.strip()
            for item in os.getenv("AURORA_ALLOWED_HOSTS", "127.0.0.1,localhost,testserver").split(",")
            if item.strip()
        ]
    )

    def overlay_values(self) -> dict:
        return {name: getattr(self, name) for name in UI_FIELDS}

    def apply_overlay(self, overlay: dict):
        for name, value in overlay.items():
            setattr(self, name, value)

    def prepare(self):
        if self.public and len(self.token) < 32:
            raise ValueError("Public mode requires AURORA_API_TOKEN with at least 32 characters")
        if not self.allowed_hosts or any("*" in host for host in self.allowed_hosts):
            raise ValueError("AURORA_ALLOWED_HOSTS must contain explicit host names")
        normalize_overlay(self.overlay_values())
        for name in ("media", "derived", "artifacts", "calibration"):
            (self.data_dir / name).mkdir(parents=True, exist_ok=True)


class ServiceError(Exception):
    def __init__(self, code, message, status=422, retryable=False):
        self.code, self.message, self.status, self.retryable = code, message, status, retryable
        super().__init__(message)
