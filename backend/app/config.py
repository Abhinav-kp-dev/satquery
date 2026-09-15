from pathlib import Path

from pydantic_settings import BaseSettings

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_ROOT / "app" / "data"
STORAGE_DIR = BACKEND_ROOT / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
RENDER_DIR = STORAGE_DIR / "renders"
LEDGER_DB = STORAGE_DIR / "ledger.db"


class Settings(BaseSettings):
    app_name: str = "SatQuery AI"
    max_upload_mb: int = 60
    max_image_dim: int = 4096

    # VLM narration backend: "mock" needs no credentials and runs fully offline;
    # "anthropic" calls Claude with the rendered composite + tool evidence.
    vlm_provider: str = "mock"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5"

    few_shot_k: int = 3

    class Config:
        env_prefix = "SATQUERY_"
        env_file = ".env"


settings = Settings()

for d in (STORAGE_DIR, UPLOAD_DIR, RENDER_DIR):
    d.mkdir(parents=True, exist_ok=True)
