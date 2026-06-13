from functools import lru_cache
from typing import List, Optional

from app.models.schemas import EvidenceItem


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        batch_size: int = 16,
        max_chars: int = 900,
        max_length: int = 384,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_chars = max_chars
        self.max_length = max_length

    def rerank(self, query: str, candidates: List[EvidenceItem], top_k: int) -> List[EvidenceItem]:
        if not candidates:
            return []
        model = _load_cross_encoder(self.model_name, self.device, self.max_length)
        pairs = [(query, _candidate_text(item, self.max_chars)) for item in candidates]
        scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        ranked = sorted(
            enumerate(scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        results: list[EvidenceItem] = []
        for rank, (index, score) in enumerate(ranked[:top_k], start=1):
            item = candidates[index]
            metadata = dict(item.metadata)
            metadata.update(
                {
                    "pre_rerank_score": item.score,
                    "rerank_rank": rank,
                    "reranker_model": self.model_name,
                    "retrieval_method": "hybrid_with_cross_encoder_rerank",
                }
            )
            results.append(item.model_copy(update={"score": round(float(score), 4), "metadata": metadata}))
        return results


@lru_cache(maxsize=4)
def _load_cross_encoder(model_name: str, device: Optional[str], max_length: int):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Install requirements or disable reranking."
        ) from exc
    return CrossEncoder(model_name, device=device, max_length=max_length)


def _candidate_text(item: EvidenceItem, max_chars: int) -> str:
    metadata = item.metadata
    parts = [
        item.title,
        str(metadata.get("section", "")),
        item.content,
    ]
    text = " ".join(" ".join(parts).split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0]
