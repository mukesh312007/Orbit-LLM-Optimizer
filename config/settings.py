"""Environment-backed application settings."""
from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    database_url: str | None = os.getenv("DATABASE_URL")
    llama_server_url: str = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:9931")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    request_timeout_seconds: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
    max_generation_tokens: int = int(os.getenv("MAX_GENERATION_TOKENS", "256"))


settings = Settings()
