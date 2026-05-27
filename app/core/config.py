from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Multimodal RAG API"
    app_env: str = "local"
    api_prefix: str = "/api"

    gemini_api_key: Optional[str] = Field(default=None, validation_alias="GEMINI_API_KEY")
    gemini_model: str = "gemini-3.5-flash"
    gemini_embedding_model: str = "gemini-embedding-2"
    gemini_thinking_level: str = "low"
    enable_gemini_embeddings: bool = True
    enable_gemini_generation: bool = True
    enable_gemini_vision: bool = True

    data_dir: Path = Path("data")
    storage_dir: Path = Path("storage")
    sqlite_path: Path = Path("data/rag.db")
    upload_dir: Path = Path("storage/uploads")

    chunk_size: int = 900
    chunk_overlap: int = 150
    fallback_embedding_dimensions: int = 384
    max_context_chars: int = 12000
    default_top_k: int = 5

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
