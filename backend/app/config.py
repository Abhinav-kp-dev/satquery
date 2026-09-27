import os
from pathlib import Path

from pydantic_settings import BaseSettings

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_ROOT / "app" / "data"
STORAGE_DIR = Path(os.environ.get("SATQUERY_STORAGE_DIR", BACKEND_ROOT / "storage"))
UPLOAD_DIR = STORAGE_DIR / "uploads"
RENDER_DIR = STORAGE_DIR / "renders"
SAMPLE_DIR = STORAGE_DIR / "samples"
REPORT_DIR = STORAGE_DIR / "reports"
LEDGER_DB = STORAGE_DIR / "ledger.db"


class Settings(BaseSettings):
    app_name: str = "SatQuery AI"
    max_upload_mb: int = 500
    # Scenes larger than this (either side) are analysed on a decimated grid.
    analysis_max_dim: int = 1536
    max_native_dim: int = 60_000

    # Narration backend: "mock" (offline, default) | "openai_compat" (self-hosted
    # open-weight VLM, e.g. Qwen2.5-VL on vLLM with per-task LoRA adapters) |
    # "anthropic" (Claude).
    vlm_provider: str = "mock"
    vlm_base_url: str = "http://127.0.0.1:8001"
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    vlm_api_key: str | None = None
    vlm_use_adapters: bool = True
    vlm_timeout_s: float = 120.0
    # Any narration failure falls back to the offline narrator so a query
    # still returns its measured evidence.
    vlm_fallback_to_mock: bool = True

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5"

    few_shot_k: int = 3

    class Config:
        env_prefix = "SATQUERY_"
        env_file = ".env"


settings = Settings()

for d in (STORAGE_DIR, UPLOAD_DIR, RENDER_DIR, SAMPLE_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)
