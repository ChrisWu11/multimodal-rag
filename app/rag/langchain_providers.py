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
    return provider_name in {OPENAI_PROVIDER, QWEN_PROVIDER, LOCAL_PROVIDER}


def build_chat_model(settings: Settings) -> Optional[BaseChatModel]:
    provider = normalize_provider(settings.llm_provider)
    if not provider_generation_enabled(settings, provider):
        return None

    if provider == GEMINI_PROVIDER and settings.gemini_api_key:
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            api_key=settings.gemini_api_key,
            temperature=0,
            thinking_level=settings.gemini_thinking_level,
        )
    if provider == OPENAI_PROVIDER and settings.openai_api_key:
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
    if provider == QWEN_PROVIDER and settings.qwen_api_key:
        return ChatOpenAI(
            model=settings.qwen_model,
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
            temperature=0,
            use_responses_api=False,
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
