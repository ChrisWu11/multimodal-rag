import hashlib
import math
import re
from typing import List, Optional

from langchain_core.embeddings import Embeddings

from app.core.config import Settings
from app.rag.langchain_providers import build_embeddings

TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


class EmbeddingProvider:
    def embed(self, text: str) -> List[float]:
        raise NotImplementedError

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(text) for text in texts]


class HashEmbeddings(Embeddings):
    """Deterministic local embedding fallback for development and tests."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed_query(self, text: str) -> List[float]:
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

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]


class HashEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimensions: int = 384) -> None:
        self.embeddings = HashEmbeddings(dimensions=dimensions)

    def embed(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)


class LangChainEmbeddingProvider(EmbeddingProvider):
    def __init__(self, embeddings: Embeddings) -> None:
        self.embeddings = embeddings

    def embed(self, text: str) -> List[float]:
        return list(self.embeddings.embed_query(text))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [list(vector) for vector in self.embeddings.embed_documents(texts)]


def get_embedding_provider(settings: Settings, force_local: Optional[bool] = None) -> EmbeddingProvider:
    local = HashEmbeddings(settings.fallback_embedding_dimensions)
    if force_local is True:
        return LangChainEmbeddingProvider(local)
    return LangChainEmbeddingProvider(build_embeddings(settings, local))


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
