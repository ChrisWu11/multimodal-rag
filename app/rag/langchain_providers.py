import importlib.util
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.config import Settings


LOCAL_PROVIDER = "local"
GEMINI_PROVIDER = "gemini"
OPENAI_PROVIDER = "openai"
QWEN_PROVIDER = "qwen"
SENTENCE_TRANSFORMERS_PROVIDER = "sentence_transformers"


def normalize_provider(provider: Optional[str]) -> str:
    return (provider or LOCAL_PROVIDER).strip().lower().replace("-", "_")


def configured_providers(settings: Settings) -> dict[str, bool]:
    return {
        GEMINI_PROVIDER: bool(settings.gemini_api_key),
        OPENAI_PROVIDER: bool(settings.openai_api_key),
        QWEN_PROVIDER: bool(settings.qwen_api_key),
        SENTENCE_TRANSFORMERS_PROVIDER: _package_available("sentence_transformers"),
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
    if provider_name == SENTENCE_TRANSFORMERS_PROVIDER:
        return settings.sentence_transformer_model
    if provider_name == OPENAI_PROVIDER:
        return settings.openai_embedding_model
    if provider_name == QWEN_PROVIDER:
        return settings.qwen_embedding_model
    return settings.gemini_embedding_model


def model_options() -> dict[str, list[str]]:
    return {
        GEMINI_PROVIDER: [
            "gemini-3.5-flash",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-1.5-flash",
        ],
        OPENAI_PROVIDER: ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
        QWEN_PROVIDER: ["qwen-plus", "qwen-max", "qwen-turbo"],
    }


def embedding_model_options() -> dict[str, list[str]]:
    return {
        GEMINI_PROVIDER: ["gemini-embedding-001"],
        OPENAI_PROVIDER: ["text-embedding-3-small", "text-embedding-3-large"],
        QWEN_PROVIDER: ["text-embedding-v4", "text-embedding-v3"],
        SENTENCE_TRANSFORMERS_PROVIDER: [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
        ],
        LOCAL_PROVIDER: ["hash-embeddings"],
    }


def provider_generation_enabled(settings: Settings, provider: Optional[str]) -> bool:
    provider_name = normalize_provider(provider)
    if not settings.enable_llm_generation:
        return False
    if provider_name == GEMINI_PROVIDER:
        return settings.enable_gemini_generation
    return provider_name in {OPENAI_PROVIDER, QWEN_PROVIDER}


def provider_embedding_enabled(settings: Settings, provider: Optional[str]) -> bool:
    provider_name = normalize_provider(provider)
    if not settings.enable_embeddings:
        return False
    if provider_name == GEMINI_PROVIDER:
        return settings.enable_gemini_embeddings
    if provider_name == SENTENCE_TRANSFORMERS_PROVIDER:
        return settings.enable_sentence_transformer_embeddings
    return provider_name in {OPENAI_PROVIDER, QWEN_PROVIDER, LOCAL_PROVIDER}


def build_chat_model(settings: Settings) -> Optional[BaseChatModel]:
    provider = normalize_provider(settings.llm_provider)
    if not provider_generation_enabled(settings, provider):
        return None

    if provider == GEMINI_PROVIDER and settings.gemini_api_key:
        kwargs = {
            "model": settings.gemini_model,
            "api_key": settings.gemini_api_key,
            "temperature": 0,
            "request_timeout": settings.llm_request_timeout_seconds,
            "retries": settings.llm_max_retries,
        }
        if _gemini_supports_thinking_level(settings.gemini_model, settings.gemini_thinking_level):
            kwargs["thinking_level"] = settings.gemini_thinking_level
        return ChatGoogleGenerativeAI(**kwargs)
    if provider == OPENAI_PROVIDER and settings.openai_api_key:
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
            timeout=settings.llm_request_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    if provider == QWEN_PROVIDER and settings.qwen_api_key:
        return ChatOpenAI(
            model=settings.qwen_model,
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
            temperature=0,
            use_responses_api=False,
            timeout=settings.llm_request_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    return None


def build_embeddings(settings: Settings, local_embeddings: Embeddings) -> Embeddings:
    provider = normalize_provider(settings.embedding_provider)
    if not provider_embedding_enabled(settings, provider):
        return local_embeddings

    if provider == GEMINI_PROVIDER and settings.gemini_api_key:
        return GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            api_key=settings.gemini_api_key,
        )
    if provider == OPENAI_PROVIDER and settings.openai_api_key:
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )
    if provider == QWEN_PROVIDER and settings.qwen_api_key:
        return OpenAIEmbeddings(
            model=settings.qwen_embedding_model,
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
            dimensions=settings.qwen_embedding_dimensions,
            tiktoken_enabled=False,
            check_embedding_ctx_length=False,
        )
    return local_embeddings


def _package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _gemini_supports_thinking_level(model: str, thinking_level: Optional[str]) -> bool:
    if not thinking_level:
        return False
    normalized = model.strip().lower()
    return normalized.startswith("gemini-3")
