import json
import re
from typing import List, Optional, Set

from app.models.schemas import EvidenceItem
from app.rag.embeddings import EmbeddingProvider, cosine_similarity
from app.rag.storage import RagStore

TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


class HybridRetriever:
    def __init__(self, store: RagStore, embeddings: EmbeddingProvider) -> None:
        self.store = store
        self.embeddings = embeddings

    def retrieve(self, query: str, top_k: int = 5, modality: Optional[str] = None) -> List[EvidenceItem]:
        query_embedding = self.embeddings.embed(query)
        query_tokens = _tokens(query)
        scored = []

        for row in self.store.iter_chunks(modality=modality):
            chunk_embedding = json.loads(row["embedding_json"])
            vector_score = cosine_similarity(query_embedding, chunk_embedding)
            keyword_score = _keyword_overlap(query_tokens, _tokens(row["content"]))
            combined = 0.78 * vector_score + 0.22 * keyword_score
            scored.append((combined, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [self.store.row_to_evidence(row, score) for score, row in scored[:top_k]]


def _tokens(text: str) -> Set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def _keyword_overlap(query_tokens: Set[str], content_tokens: Set[str]) -> float:
    if not query_tokens or not content_tokens:
        return 0.0
    return len(query_tokens & content_tokens) / len(query_tokens)
