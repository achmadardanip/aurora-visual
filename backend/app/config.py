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
    hive_enabled: bool = field(
        default_factory=lambda: os.getenv("AURORA_HIVE_ENABLED", "false").lower() == "true"
    )
    hive_timeout: float = field(default_factory=lambda: float(os.getenv("AURORA_HIVE_TIMEOUT_SECONDS", "45")))
    hive_v3_secret: str = field(default_factory=lambda: os.getenv("AURORA_HIVE_V3_SECRET_KEY", ""))
    hive_v2_shared_key: str = field(default_factory=lambda: os.getenv("AURORA_HIVE_V2_SHARED_API_KEY", ""))
    hive_v2_origin_key: str = field(default_factory=lambda: os.getenv("AURORA_HIVE_V2_ORIGIN_API_KEY", ""))
    hive_v2_ocr_key: str = field(default_factory=lambda: os.getenv("AURORA_HIVE_V2_OCR_API_KEY", ""))
    hive_v2_object_key: str = field(default_factory=lambda: os.getenv("AURORA_HIVE_V2_OBJECT_API_KEY", ""))
    hive_v2_scene_key: str = field(default_factory=lambda: os.getenv("AURORA_HIVE_V2_SCENE_API_KEY", ""))
    hive_v2_people_key: str = field(default_factory=lambda: os.getenv("AURORA_HIVE_V2_PEOPLE_API_KEY", ""))
    hive_v2_logo_key: str = field(default_factory=lambda: os.getenv("AURORA_HIVE_V2_LOGO_API_KEY", ""))
    hive_v2_celebrity_key: str = field(
        default_factory=lambda: os.getenv("AURORA_HIVE_V2_CELEBRITY_API_KEY", "")
    )
    hive_v2_translation_key: str = field(
        default_factory=lambda: os.getenv("AURORA_HIVE_V2_TRANSLATION_API_KEY", "")
    )
    allowed_hosts: list[str] = field(
        default_factory=lambda: [
            item.strip()
            for item in os.getenv("AURORA_ALLOWED_HOSTS", "127.0.0.1,localhost,testserver").split(",")
            if item.strip()
        ]
    )

    def hive_v2_key(self, capability: str) -> str:
        value = getattr(self, f"hive_v2_{capability}_key", "")
        return value or self.hive_v2_shared_key

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
        if not 1 <= self.hive_timeout <= 120:
            raise ValueError("AURORA_HIVE_TIMEOUT_SECONDS must be between 1 and 120")
        for name in ("media", "derived", "artifacts", "cache"):
            (self.data_dir / name).mkdir(parents=True, exist_ok=True)


class ServiceError(Exception):
    def __init__(self, code: str, message: str, status: int = 422, retryable: bool = False):
        self.code, self.message, self.status, self.retryable = code, message, status, retryable
        super().__init__(message)
