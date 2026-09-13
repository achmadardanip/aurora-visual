import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("AURORA_DATA_DIR", "var")).resolve())
    public: bool = field(default_factory=lambda: os.getenv("AURORA_PUBLIC", "false").lower() == "true")
    token: str = field(default_factory=lambda: os.getenv("AURORA_API_TOKEN", ""))
    max_upload: int = field(default_factory=lambda: int(os.getenv("AURORA_MAX_UPLOAD_MB", "10")) * 1024**2)
    job_timeout: int = field(default_factory=lambda: int(os.getenv("AURORA_JOB_TIMEOUT", "180")))
    max_pending: int = field(default_factory=lambda: int(os.getenv("AURORA_MAX_PENDING", "20")))
    backbone: str = field(default_factory=lambda: os.getenv("AURORA_BACKBONE", "local-color-v1"))
    ocr_lang: str = field(default_factory=lambda: os.getenv("AURORA_OCR_LANG", "ind+eng"))
    mafindo_api_key: str = field(default_factory=lambda: os.getenv("AURORA_MAFINDO_API_KEY", ""))
    mafindo_timeout: float = field(
        default_factory=lambda: float(os.getenv("AURORA_MAFINDO_TIMEOUT_SECONDS", "12"))
    )
    allowed_hosts: list[str] = field(
        default_factory=lambda: [
            item.strip()
            for item in os.getenv("AURORA_ALLOWED_HOSTS", "127.0.0.1,localhost,testserver").split(",")
            if item.strip()
        ]
    )

    def prepare(self):
        if self.public and len(self.token) < 32:
            raise ValueError("Public mode requires AURORA_API_TOKEN with at least 32 characters")
        if not self.allowed_hosts or any("*" in host for host in self.allowed_hosts):
            raise ValueError("AURORA_ALLOWED_HOSTS must contain explicit host names")
        if self.public and any(
            host.rstrip(".").lower() in {"127.0.0.1", "localhost", "testserver"}
            for host in self.allowed_hosts
        ):
            raise ValueError("Public mode requires deployment host names in AURORA_ALLOWED_HOSTS")
        if not 1 <= self.mafindo_timeout <= 60:
            raise ValueError("AURORA_MAFINDO_TIMEOUT_SECONDS must be between 1 and 60")
        for name in ("media", "derived", "artifacts", "cache"):
            (self.data_dir / name).mkdir(parents=True, exist_ok=True)


class ServiceError(Exception):
    def __init__(self, code: str, message: str, status: int = 422, retryable: bool = False):
        self.code, self.message, self.status, self.retryable = code, message, status, retryable
        super().__init__(message)
