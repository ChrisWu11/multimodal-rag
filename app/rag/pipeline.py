from typing import Optional

from app.core.config import Settings
from app.models.schemas import ChatResponse, SearchResponse
from app.rag.generator import RagAnswerGenerator
from app.rag.retriever import HybridRetriever
from app.services.image_analysis import ImageAnalyzer
from app.services.safety import safety_notice


class RagPipeline:
    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        generator: RagAnswerGenerator,
        image_analyzer: ImageAnalyzer,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.generator = generator
        self.image_analyzer = image_analyzer

    def search(
        self,
        query: str,
        top_k: int = 5,
        modality: Optional[str] = None,
        use_reranker: Optional[bool] = None,
    ) -> SearchResponse:
        return SearchResponse(
            query=query,
            results=self.retriever.retrieve(
                query=query,
                top_k=top_k,
                modality=modality,
                use_reranker=use_reranker,
            ),
        )

    def answer(
        self,
        question: str,
        top_k: int = 5,
        modality: Optional[str] = None,
        use_llm: bool = True,
        image_filename: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        use_reranker: Optional[bool] = None,
    ) -> ChatResponse:
        visual_summary = None
        retrieval_query = question

        if image_filename and image_bytes:
            visual_summary, _metadata = self.image_analyzer.summarize(
                image_filename,
                image_bytes,
                modality or "unknown",
            )
            retrieval_query = f"{question}\n\nVisual summary:\n{visual_summary}"

        evidence = self.retriever.retrieve(
            query=retrieval_query,
            top_k=top_k,
            modality=modality,
            use_reranker=use_reranker,
        )
        answer, used_llm, model, provider = self.generator.generate(
            question=question,
            evidence=evidence,
            visual_summary=visual_summary,
            use_llm=use_llm,
        )
        return ChatResponse(
            answer=answer,
            evidence=evidence,
            safety_notice=safety_notice(),
            visual_summary=visual_summary,
            used_llm=used_llm,
            provider=provider,
            model=model,
        )
