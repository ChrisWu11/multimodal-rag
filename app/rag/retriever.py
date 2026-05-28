import json
import re
from typing import List, Optional, Set

from app.models.schemas import EvidenceItem
from app.rag.embeddings import EmbeddingProvider, cosine_similarity
from app.rag.langchain_providers import normalize_provider
from app.rag.storage import RagStore

TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


class HybridRetriever:
    def __init__(
        self,
        store: RagStore,
        embeddings: EmbeddingProvider,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        self.store = store
        self.embeddings = embeddings
        self.embedding_provider = normalize_provider(embedding_provider) if embedding_provider else None
        self.embedding_model = embedding_model

    def retrieve(self, query: str, top_k: int = 5, modality: Optional[str] = None) -> List[EvidenceItem]:
        query_embedding = self.embeddings.embed(query)
        query_tokens = _tokens(query)
        scored = []

        for row in self.store.iter_chunks(modality=modality):
            if not self._embedding_matches(row):
                continue
            chunk_embedding = json.loads(row["embedding_json"])
            vector_score = cosine_similarity(query_embedding, chunk_embedding)
            keyword_score = _keyword_overlap(query_tokens, _tokens(row["content"]))
            combined = 0.78 * vector_score + 0.22 * keyword_score
            scored.append((combined, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [self.store.row_to_evidence(row, score) for score, row in scored[:top_k]]

    def _embedding_matches(self, row) -> bool:
        if not self.embedding_provider and not self.embedding_model:
            return True
        metadata = json.loads(row["chunk_metadata_json"] or "{}")
        stored_provider = metadata.get("embedding_provider")
        stored_model = metadata.get("embedding_model")
        if not stored_provider and not stored_model:
            return True
        if self.embedding_provider and stored_provider != self.embedding_provider:
            return False
        if self.embedding_model and stored_model != self.embedding_model:
            return False
        return True


def _tokens(text: str) -> Set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def _keyword_overlap(query_tokens: Set[str], content_tokens: Set[str]) -> float:
    if not query_tokens or not content_tokens:
        return 0.0
    return len(query_tokens & content_tokens) / len(query_tokens)
