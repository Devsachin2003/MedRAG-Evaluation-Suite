"""Runtime configuration shared by API, retrieval and evaluation."""

import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


DATA_DIR = Path(
    os.getenv("MEDRAG_DATA_DIR", str(Path(__file__).resolve().parents[1] / "data"))
)
MAX_UPLOAD_BYTES = positive_int("MAX_UPLOAD_BYTES", 10 * 1024 * 1024)
MAX_TEXT_CHARS = positive_int("MAX_TEXT_CHARS", 2_000_000)
MAX_PDF_PAGES = positive_int("MAX_PDF_PAGES", 500)
MAX_CORPUS_CHUNKS = positive_int("MAX_CORPUS_CHUNKS", 20_000)
MAX_CONCURRENT_REQUESTS = positive_int("MAX_CONCURRENT_REQUESTS", 4)
LLM_TIMEOUT_SECONDS = positive_int("LLM_TIMEOUT_SECONDS", 60)
OUTPUT_DIR = DATA_DIR / "evaluations"
PRODUCTION = os.getenv("APP_ENV", "development") == "production"
API_KEY = os.getenv("MEDRAG_API_KEY", "")
CORS_ORIGINS = [
    s.strip()
    for s in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if s.strip()
]


def validate_production() -> None:
    if PRODUCTION and not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", API_KEY):
        raise RuntimeError(
            "Production requires MEDRAG_API_KEY of 32-128 URL-safe characters"
        )
    if PRODUCTION and not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("Production requires GROQ_API_KEY")
