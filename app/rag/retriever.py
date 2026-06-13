import json
import re
from collections import defaultdict
from typing import List, Optional, Set

from app.core.config import Settings
from app.models.schemas import EvidenceItem
from app.rag.embeddings import EmbeddingProvider, cosine_similarity
from app.rag.langchain_providers import normalize_provider
from app.rag.reranker import CrossEncoderReranker
from app.rag.storage import RagStore

TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


class HybridRetriever:
    def __init__(
        self,
        store: RagStore,
        embeddings: EmbeddingProvider,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.store = store
        self.embeddings = embeddings
        self.embedding_provider = normalize_provider(embedding_provider) if embedding_provider else None
        self.embedding_model = embedding_model
        self.settings = settings
        self.candidate_k = settings.retrieval_candidate_k if settings else 30
        self.fusion = normalize_provider(settings.retrieval_fusion) if settings else "rrf"
        self.vector_weight = settings.retrieval_vector_weight if settings else 0.78
        self.keyword_weight = settings.retrieval_keyword_weight if settings else 0.22
        self.rrf_k = settings.retrieval_rrf_k if settings else 60
        self._reranker = (
            CrossEncoderReranker(
                model_name=settings.reranker_model,
                device=settings.reranker_device,
                batch_size=settings.reranker_batch_size,
                max_chars=settings.reranker_max_chars,
                max_length=settings.reranker_max_length,
            )
            if settings
            else None
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        modality: Optional[str] = None,
        use_reranker: Optional[bool] = None,
    ) -> List[EvidenceItem]:
        query_embedding = self.embeddings.embed(query)
        query_tokens = _tokens(query)
        candidates = []

        for row in self.store.iter_chunks(modality=modality):
            if not self._embedding_matches(row):
                continue
            chunk_embedding = json.loads(row["embedding_json"])
            vector_score = cosine_similarity(query_embedding, chunk_embedding)
            keyword_score = _keyword_overlap(query_tokens, _tokens(row["content"]))
            candidates.append(
                {
                    "row": row,
                    "vector_score": vector_score,
                    "keyword_score": keyword_score,
                    "weighted_score": self.vector_weight * vector_score
                    + self.keyword_weight * keyword_score,
                }
            )

        if not candidates:
            return []

        scored = self._score_candidates(candidates)
        pool_size = max(top_k, self.candidate_k)
        evidence = [
            self._candidate_to_evidence(candidate)
            for candidate in sorted(scored, key=lambda item: item["score"], reverse=True)[:pool_size]
        ]

        should_rerank = self.settings.enable_reranker if self.settings else False
        if use_reranker is not None:
            should_rerank = use_reranker
        if should_rerank and self._reranker:
            try:
                return self._reranker.rerank(query=query, candidates=evidence, top_k=top_k)
            except RuntimeError as exc:
                for item in evidence:
                    item.metadata["reranker_error"] = str(exc)
        return evidence[:top_k]

    def _score_candidates(self, candidates: list[dict]) -> list[dict]:
        if self.fusion != "rrf":
            for candidate in candidates:
                candidate["score"] = candidate["weighted_score"]
                candidate["retrieval_method"] = "hybrid_weighted"
            return candidates

        scores: defaultdict[int, float] = defaultdict(float)
        vector_ranked = sorted(
            enumerate(candidates),
            key=lambda item: item[1]["vector_score"],
            reverse=True,
        )
        keyword_ranked = [
            item
            for item in sorted(
                enumerate(candidates),
                key=lambda item: item[1]["keyword_score"],
                reverse=True,
            )
            if item[1]["keyword_score"] > 0
        ]
        for rank, (index, _candidate) in enumerate(vector_ranked, start=1):
            scores[index] += 1.0 / (self.rrf_k + rank)
        for rank, (index, _candidate) in enumerate(keyword_ranked, start=1):
            scores[index] += 1.0 / (self.rrf_k + rank)

        for index, candidate in enumerate(candidates):
            candidate["score"] = scores[index]
            candidate["retrieval_method"] = "hybrid_rrf"
        return candidates

    def _candidate_to_evidence(self, candidate: dict) -> EvidenceItem:
        evidence = self.store.row_to_evidence(candidate["row"], candidate["score"])
        metadata = dict(evidence.metadata)
        metadata.update(
            {
                "vector_score": round(float(candidate["vector_score"]), 4),
                "keyword_score": round(float(candidate["keyword_score"]), 4),
                "weighted_score": round(float(candidate["weighted_score"]), 4),
                "retrieval_method": candidate["retrieval_method"],
            }
        )
        return evidence.model_copy(update={"metadata": metadata})

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
