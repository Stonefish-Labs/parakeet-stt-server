from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    return int(value)


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    cors_origins: list[str]
    api_key: str
    model_id: str
    device: str | None
    max_concurrent: int
    warmup_on_load: bool
    max_file_bytes: int
    temp_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        cors_raw = os.getenv("CORS_ORIGINS", "*")
        device = os.getenv("PARAKEET_STT_DEVICE", "auto")
        temp_default = Path(tempfile.gettempdir()) / "parakeet-stt"
        max_file_mb = env_int("PARAKEET_STT_MAX_FILE_MB", 100)
        return cls(
            host=os.getenv("HOST", "127.0.0.1"),
            port=env_int("PORT", 5092),
            cors_origins=[item.strip() for item in cors_raw.split(",") if item.strip()],
            api_key=os.getenv("PARAKEET_STT_API_KEY", ""),
            model_id=os.getenv("PARAKEET_STT_MODEL", "nvidia/parakeet-tdt-0.6b-v3"),
            device=None if device == "auto" else device,
            max_concurrent=env_int("PARAKEET_STT_MAX_CONCURRENT", 1),
            warmup_on_load=env_bool("PARAKEET_STT_WARMUP_ON_LOAD", True),
            max_file_bytes=max_file_mb * 1024 * 1024,
            temp_dir=Path(os.getenv("PARAKEET_STT_TEMP_DIR", str(temp_default))),
        )

