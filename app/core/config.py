from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = "Multimodal RAG API"
    app_env: str = "local"
    api_prefix: str = "/api"

    llm_provider: str = Field(default="gemini", validation_alias="LLM_PROVIDER")
    embedding_provider: str = Field(default="gemini", validation_alias="EMBEDDING_PROVIDER")
    enable_llm_generation: bool = True
    enable_embeddings: bool = True

    openai_api_key: Optional[str] = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    gemini_api_key: Optional[str] = Field(default=None, validation_alias="GEMINI_API_KEY")
    gemini_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"
    gemini_thinking_level: str = "low"
    enable_gemini_embeddings: bool = True
    enable_gemini_generation: bool = True
    enable_gemini_vision: bool = True

    qwen_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("QWEN_API_KEY", "DASHSCOPE_API_KEY"),
    )
    qwen_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen-plus"
    qwen_embedding_model: str = "text-embedding-v4"
    qwen_embedding_dimensions: Optional[int] = 1024

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

    def with_runtime_models(
        self,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> "Settings":
        updates: dict[str, object] = {}
        active_llm_provider = _clean_provider(llm_provider or self.llm_provider)
        active_embedding_provider = _clean_provider(embedding_provider or self.embedding_provider)

        if llm_provider:
            updates["llm_provider"] = active_llm_provider
        if embedding_provider:
            updates["embedding_provider"] = active_embedding_provider
        if llm_model:
            updates[_model_field(active_llm_provider)] = llm_model.strip()
        if embedding_model and active_embedding_provider != "local":
            updates[_embedding_model_field(active_embedding_provider)] = embedding_model.strip()

        return self.model_copy(update=updates)


def _clean_provider(provider: str) -> str:
    return provider.strip().lower().replace("-", "_")


def _model_field(provider: str) -> str:
    if provider == "openai":
        return "openai_model"
    if provider == "qwen":
        return "qwen_model"
    return "gemini_model"


def _embedding_model_field(provider: str) -> str:
    if provider == "openai":
        return "openai_embedding_model"
    if provider == "qwen":
        return "qwen_embedding_model"
    return "gemini_embedding_model"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
