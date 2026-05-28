from typing import Optional

from openai import OpenAI

from app.core.config import Settings

LOCAL_PROVIDER = "local"
GEMINI_PROVIDER = "gemini"
OPENAI_PROVIDER = "openai"
QWEN_PROVIDER = "qwen"


def normalize_provider(provider: Optional[str]) -> str:
    return (provider or LOCAL_PROVIDER).strip().lower().replace("-", "_")


def configured_providers(settings: Settings) -> dict[str, bool]:
    return {
        GEMINI_PROVIDER: bool(settings.gemini_api_key),
        OPENAI_PROVIDER: bool(settings.openai_api_key),
        QWEN_PROVIDER: bool(settings.qwen_api_key),
        LOCAL_PROVIDER: True,
    }


def provider_configured(settings: Settings, provider: Optional[str]) -> bool:
    return configured_providers(settings).get(normalize_provider(provider), False)


def active_llm_model(settings: Settings, provider: Optional[str] = None) -> str:
    provider_name = normalize_provider(provider or settings.llm_provider)
    if provider_name == OPENAI_PROVIDER:
        return settings.openai_model
    if provider_name == QWEN_PROVIDER:
        return settings.qwen_model
    return settings.gemini_model


def active_embedding_model(settings: Settings, provider: Optional[str] = None) -> str:
    provider_name = normalize_provider(provider or settings.embedding_provider)
    if provider_name == LOCAL_PROVIDER:
        return "hash-embeddings"
    if provider_name == OPENAI_PROVIDER:
        return settings.openai_embedding_model
    if provider_name == QWEN_PROVIDER:
        return settings.qwen_embedding_model
    return settings.gemini_embedding_model


def model_options() -> dict[str, list[str]]:
    return {
        GEMINI_PROVIDER: ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash"],
        OPENAI_PROVIDER: ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
        QWEN_PROVIDER: ["qwen-plus", "qwen-max", "qwen-turbo"],
    }


def embedding_model_options() -> dict[str, list[str]]:
    return {
        GEMINI_PROVIDER: ["gemini-embedding-001"],
        OPENAI_PROVIDER: ["text-embedding-3-small", "text-embedding-3-large"],
        QWEN_PROVIDER: ["text-embedding-v4", "text-embedding-v3"],
        LOCAL_PROVIDER: ["hash-embeddings"],
    }


def openai_compatible_client(settings: Settings, provider: Optional[str]) -> Optional[OpenAI]:
    provider_name = normalize_provider(provider)
    if provider_name == OPENAI_PROVIDER and settings.openai_api_key:
        return OpenAI(api_key=settings.openai_api_key)
    if provider_name == QWEN_PROVIDER and settings.qwen_api_key:
        return OpenAI(api_key=settings.qwen_api_key, base_url=settings.qwen_base_url)
    return None
