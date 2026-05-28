import hashlib
import math
import re
from typing import List, Optional

from google import genai

from app.core.config import Settings
from app.rag.model_providers import (
    GEMINI_PROVIDER,
    LOCAL_PROVIDER,
    active_embedding_model,
    normalize_provider,
    openai_compatible_client,
)

TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


class EmbeddingProvider:
    def embed(self, text: str) -> List[float]:
        raise NotImplementedError

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(text) for text in texts]


class HashEmbeddingProvider(EmbeddingProvider):
    """Deterministic local embedding fallback for development and tests."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        tokens = TOKEN_RE.findall(text.lower())
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[-1] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.fallback = HashEmbeddingProvider(settings.fallback_embedding_dimensions)

    def embed(self, text: str) -> List[float]:
        if not self.settings.gemini_api_key or not self.settings.enable_gemini_embeddings:
            return self.fallback.embed(text)

        response = self.client.models.embed_content(
            model=self.settings.gemini_embedding_model,
            contents=f"task: question answering | query: {text}",
        )
        return list(response.embeddings[0].values)


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.provider = normalize_provider(settings.embedding_provider)
        self.client = openai_compatible_client(settings, self.provider)
        self.fallback = HashEmbeddingProvider(settings.fallback_embedding_dimensions)

    def embed(self, text: str) -> List[float]:
        if not self.client:
            return self.fallback.embed(text)
        kwargs = {
            "model": active_embedding_model(self.settings),
            "input": text,
        }
        if self.provider == "qwen" and self.settings.qwen_embedding_dimensions:
            kwargs["dimensions"] = self.settings.qwen_embedding_dimensions
        response = self.client.embeddings.create(**kwargs)
        return list(response.data[0].embedding)


def get_embedding_provider(settings: Settings, force_local: Optional[bool] = None) -> EmbeddingProvider:
    provider = normalize_provider(settings.embedding_provider)
    if force_local is True:
        return HashEmbeddingProvider(settings.fallback_embedding_dimensions)
    if provider == LOCAL_PROVIDER:
        return HashEmbeddingProvider(settings.fallback_embedding_dimensions)
    if not settings.enable_embeddings:
        return HashEmbeddingProvider(settings.fallback_embedding_dimensions)
    if provider == GEMINI_PROVIDER and settings.gemini_api_key and settings.enable_gemini_embeddings:
        return GeminiEmbeddingProvider(settings)
    if provider in {"openai", "qwen"}:
        return OpenAICompatibleEmbeddingProvider(settings)
    return HashEmbeddingProvider(settings.fallback_embedding_dimensions)


def cosine_similarity(left: List[float], right: List[float]) -> float:
    if not left or not right:
        return 0.0
    length = min(len(left), len(right))
    dot = sum(left[i] * right[i] for i in range(length))
    left_norm = math.sqrt(sum(left[i] * left[i] for i in range(length)))
    right_norm = math.sqrt(sum(right[i] * right[i] for i in range(length)))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
