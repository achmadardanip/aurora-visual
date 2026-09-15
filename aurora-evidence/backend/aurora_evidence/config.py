"""Module 2 settings: env-driven provider configuration with UI overlay."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load only this module's dedicated env file (if present); the shared .env of
# other AURORA services must not leak provider keys into this app's Settings.
load_dotenv(Path(__file__).resolve().parents[2] / ".env.evidence", override=False)

UI_FIELDS: dict[str, tuple[str, bool]] = {
    "corpus_path": ("str", False),
    "corpus_version": ("str", False),
    "budget_seconds": ("float", False),
    "max_evidence": ("int", False),
    "web_hits": ("int", False),
    "corpus_hits": ("int", False),
    "max_search_queries": ("int", False),
    "web_fetch": ("bool", False),
    "tavily_api_key": ("str", True),
    "gptzero_api_key": ("str", True),
    "hive_api_key": ("str", True),
    "provider_timeout": ("float", False),
    "mode_profile": ("str", False),
}

UI_RANGES = {
    "budget_seconds": (5, 120),
    "max_evidence": (1, 100),
    "web_hits": (1, 20),
    "corpus_hits": (1, 20),
    "max_search_queries": (1, 8),
    "provider_timeout": (1, 60),
}


def normalize_overlay(values: dict) -> dict:
    result = {}
    for name, value in values.items():
        if name not in UI_FIELDS:
            raise ValueError(f"Kolom konfigurasi tidak dikenal: {name}")
        kind, _ = UI_FIELDS[name]
        if kind == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"{name} harus boolean")
            result[name] = value
        elif kind == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} harus bilangan bulat")
            result[name] = value
        elif kind == "float":
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
    if result.get("mode_profile") not in (None, "local", "full"):
        raise ValueError("mode_profile harus local atau full")
    return result


@dataclass
class Settings:
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("AURORA_EVIDENCE_DATA_DIR", "var-evidence")).resolve()
    )
    public: bool = field(default_factory=lambda: os.getenv("AURORA_PUBLIC", "false").lower() == "true")
    token: str = field(default_factory=lambda: os.getenv("AURORA_API_TOKEN", ""))
    max_upload: int = field(default_factory=lambda: int(os.getenv("AURORA_MAX_UPLOAD_MB", "10")) * 1024**2)
    max_images: int = field(default_factory=lambda: int(os.getenv("AURORA_MAX_IMAGES", "8")))
    job_timeout: int = field(default_factory=lambda: int(os.getenv("AURORA_JOB_TIMEOUT", "120")))
    max_pending: int = field(default_factory=lambda: int(os.getenv("AURORA_MAX_PENDING", "20")))
    corpus_path: str = field(
        default_factory=lambda: os.getenv(
            "AURORA_EVIDENCE_CORPUS",
            str(Path(__file__).resolve().parents[2] / "fixtures" / "demo_corpus.jsonl"),
        )
    )
    corpus_version: str = field(
        default_factory=lambda: os.getenv("AURORA_EVIDENCE_CORPUS_VERSION", "demo-sintetis-v1")
    )
    budget_seconds: float = field(default_factory=lambda: float(os.getenv("AURORA_EVIDENCE_BUDGET", "45")))
    max_evidence: int = field(default_factory=lambda: int(os.getenv("AURORA_EVIDENCE_MAX", "30")))
    web_hits: int = field(default_factory=lambda: int(os.getenv("AURORA_EVIDENCE_WEB_HITS", "8")))
    corpus_hits: int = field(default_factory=lambda: int(os.getenv("AURORA_EVIDENCE_CORPUS_HITS", "8")))
    max_search_queries: int = field(
        default_factory=lambda: int(os.getenv("AURORA_EVIDENCE_MAX_QUERIES", "3"))
    )
    image_matches: int = field(default_factory=lambda: int(os.getenv("AURORA_EVIDENCE_IMAGE_MATCHES", "5")))
    web_fetch: bool = field(
        default_factory=lambda: os.getenv("AURORA_EVIDENCE_WEB_FETCH", "false").lower() == "true"
    )
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    gptzero_api_key: str = field(default_factory=lambda: os.getenv("GPTZERO_API_KEY", ""))
    hive_api_key: str = field(default_factory=lambda: os.getenv("HIVE_API_KEY", ""))
    provider_timeout: float = field(
        default_factory=lambda: float(os.getenv("AURORA_EVIDENCE_PROVIDER_TIMEOUT", "20"))
    )
    hive_poll_attempts: int = field(
        default_factory=lambda: int(os.getenv("AURORA_EVIDENCE_HIVE_POLLS", "10"))
    )
    fetch_max_bytes: int = 2_000_000
    fetch_max_chars: int = 50_000
    mode_profile: str = field(default_factory=lambda: os.getenv("AURORA_EVIDENCE_PROFILE", "full"))
    allowed_hosts: list[str] = field(
        default_factory=lambda: [
            item.strip()
            for item in os.getenv("AURORA_ALLOWED_HOSTS", "127.0.0.1,localhost,testserver").split(",")
            if item.strip()
        ]
    )

    def overlay_values(self) -> dict:
        values = {}
        for name in UI_FIELDS:
            values[name] = getattr(self, name)
        return values

    def apply_overlay(self, overlay: dict):
        for name, value in overlay.items():
            setattr(self, name, value)

    def prepare(self):
        if self.public and len(self.token) < 32:
            raise ValueError("Public mode requires AURORA_API_TOKEN with at least 32 characters")
        if not self.allowed_hosts or any("*" in host for host in self.allowed_hosts):
            raise ValueError("AURORA_ALLOWED_HOSTS must contain explicit host names")
        normalize_overlay(self.overlay_values())
        for name in ("media", "derived", "artifacts", "cache"):
            (self.data_dir / name).mkdir(parents=True, exist_ok=True)


class ServiceError(Exception):
    def __init__(self, code, message, status=422, retryable=False):
        self.code, self.message, self.status, self.retryable = code, message, status, retryable
        super().__init__(message)
