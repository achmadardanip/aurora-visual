import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

load_dotenv()

# UI-configurable fields: name -> (kind, secret). Secrets are masked in API
# responses and persisted server-side only.
UI_FIELDS: dict[str, tuple[str, bool]] = {
    "backbone": ("str", False),
    "ocr_lang": ("str", False),
    "max_upload_mb": ("int", False),
    "max_images": ("int", False),
    "job_timeout": ("int", False),
    "max_pending": ("int", False),
    "openclip_model": ("str", False),
    "openclip_pretrained": ("str", False),
    "llm_url": ("str", False),
    "llm_model": ("str", False),
    "llm_allowed_origins": ("str", False),
    "checkpoint": ("str", False),
    "mafindo_api_key": ("str", True),
    "mafindo_timeout": ("float", False),
    "hive_enabled": ("bool", False),
    "hive_timeout": ("float", False),
    "hive_v3_secret": ("str", True),
    "hive_v2_shared_key": ("str", True),
    "hive_v2_origin_key": ("str", True),
    "hive_v2_ocr_key": ("str", True),
    "hive_v2_object_key": ("str", True),
    "hive_v2_scene_key": ("str", True),
    "hive_v2_people_key": ("str", True),
    "hive_v2_logo_key": ("str", True),
    "hive_v2_celebrity_key": ("str", True),
    "hive_v2_translation_key": ("str", True),
}
UI_RANGES = {
    "max_upload_mb": (1, 50),
    "max_images": (1, 16),
    "job_timeout": (30, 1800),
    "max_pending": (1, 100),
    "mafindo_timeout": (1, 60),
    "hive_timeout": (1, 120),
}


def normalize_overlay(values: dict) -> dict:
    """Validate and normalize a UI overlay (partial or full); raises ValueError."""
    result = {}
    for name, value in values.items():
        if name not in UI_FIELDS:
            raise ValueError(f"Kolom konfigurasi tidak dikenal: {name}")
        kind, _ = UI_FIELDS[name]
        if kind == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"{name} harus boolean (true/false)")
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
    if result.get("backbone") not in (None, "local-color-v1", "openclip"):
        raise ValueError("backbone harus local-color-v1 atau openclip")
    url = result.get("llm_url")
    if url:
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or parsed.username or not parsed.netloc:
            raise ValueError("llm_url harus URL http(s) lengkap tanpa kredensial")
    for origin in (o.strip() for o in result.get("llm_allowed_origins", "").split(",")):
        if origin and ("*" in origin or "@" in origin):
            raise ValueError("llm_allowed_origins harus origin eksplisit tanpa wildcard/kredensial")
    return result


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("AURORA_DATA_DIR", "var")).resolve())
    public: bool = field(default_factory=lambda: os.getenv("AURORA_PUBLIC", "false").lower() == "true")
    token: str = field(default_factory=lambda: os.getenv("AURORA_API_TOKEN", ""))
    max_upload: int = field(default_factory=lambda: int(os.getenv("AURORA_MAX_UPLOAD_MB", "5")) * 1024**2)
    max_images: int = field(default_factory=lambda: int(os.getenv("AURORA_MAX_IMAGES", "8")))
    job_timeout: int = field(default_factory=lambda: int(os.getenv("AURORA_JOB_TIMEOUT", "180")))
    max_pending: int = field(default_factory=lambda: int(os.getenv("AURORA_MAX_PENDING", "20")))
    backbone: str = field(default_factory=lambda: os.getenv("AURORA_BACKBONE", "local-color-v1"))
    openclip_model: str = field(default_factory=lambda: os.getenv("AURORA_OPENCLIP_MODEL", "ViT-B-32"))
    openclip_pretrained: str = field(default_factory=lambda: os.getenv("AURORA_OPENCLIP_PRETRAINED", ""))
    llm_url: str = field(default_factory=lambda: os.getenv("AURORA_LLM_URL", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("AURORA_LLM_MODEL", ""))
    llm_allowed_origins: list[str] = field(
        default_factory=lambda: [
            item.strip()
            for item in os.getenv("AURORA_LLM_ALLOWED_ORIGINS", "http://127.0.0.1:11434").split(",")
            if item.strip()
        ]
    )
    checkpoint: str = field(default_factory=lambda: os.getenv("AURORA_CHECKPOINT", ""))
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

    def overlay_values(self) -> dict:
        """Current values of all UI-configurable fields (secrets included)."""
        values = {}
        for name in UI_FIELDS:
            if name == "max_upload_mb":
                values[name] = self.max_upload // 1024**2
            elif name == "llm_allowed_origins":
                values[name] = ",".join(self.llm_allowed_origins)
            else:
                values[name] = getattr(self, name)
        return values

    def apply_overlay(self, overlay: dict):
        """Apply a validated UI overlay (see normalize_overlay) onto this instance."""
        for name, value in overlay.items():
            if name == "max_upload_mb":
                self.max_upload = int(value) * 1024**2
            elif name == "llm_allowed_origins":
                self.llm_allowed_origins = [item.strip() for item in value.split(",") if item.strip()]
            else:
                setattr(self, name, value)

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
        # Range/type validation for every UI field (covers env-provided values too).
        normalize_overlay(self.overlay_values())
        for name in ("media", "derived", "artifacts", "cache"):
            (self.data_dir / name).mkdir(parents=True, exist_ok=True)


class ServiceError(Exception):
    def __init__(self, code: str, message: str, status: int = 422, retryable: bool = False):
        self.code, self.message, self.status, self.retryable = code, message, status, retryable
        super().__init__(message)
