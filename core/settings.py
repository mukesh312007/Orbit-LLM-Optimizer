from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    database_url: str | None = os.getenv("DATABASE_URL")
    llama_server_url: str = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:9931")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "180"))
    energy_meter_reading_path: str | None = os.getenv("ENERGY_METER_READING_PATH") or os.getenv("ORBIT_ENERGY_METER_READING_PATH")
    retain_live_response_text: bool = _enabled("ORBIT_RETAIN_LIVE_RESPONSE_TEXT")
    safety_reserve_mb: float = float(os.getenv("ORBIT_SAFETY_RESERVE_MB", "512.0"))


settings = Settings()
