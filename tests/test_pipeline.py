from pathlib import Path

from app.core.config import Settings
from app.rag.embeddings import HashEmbeddingProvider
from app.rag.generator import RagAnswerGenerator
from app.rag.ingestion import DocumentIngestor
from app.rag.pipeline import RagPipeline
from app.rag.retriever import HybridRetriever
from app.rag.storage import RagStore
from app.services.image_analysis import ImageAnalyzer


def build_test_pipeline(tmp_path: Path) -> tuple[DocumentIngestor, RagPipeline]:
    settings = Settings(
        gemini_api_key=None,
        sqlite_path=tmp_path / "rag.db",
        data_dir=tmp_path,
        storage_dir=tmp_path / "storage",
        upload_dir=tmp_path / "storage" / "uploads",
        fallback_embedding_dimensions=64,
    )
    settings.ensure_directories()
    store = RagStore(settings.sqlite_path)
    store.init_db()
    embeddings = HashEmbeddingProvider(64)
    image_analyzer = ImageAnalyzer(settings)
    ingestor = DocumentIngestor(settings, store, embeddings, image_analyzer)
    retriever = HybridRetriever(store, embeddings)
    generator = RagAnswerGenerator(settings)
    return ingestor, RagPipeline(settings, retriever, generator, image_analyzer)


def test_ingest_and_answer_without_llm(tmp_path: Path) -> None:
    ingestor, pipeline = build_test_pipeline(tmp_path)
    response = ingestor.ingest_text(
        title="Thermal note",
        text="Thermal imaging can identify surface temperature asymmetry in research workflows.",
        modality="thermal",
    )
    assert response.chunks_indexed == 1

    answer = pipeline.answer("What can thermal imaging identify?", modality="thermal", use_llm=False)
    assert answer.evidence
    assert answer.used_llm is False
    assert answer.provider == "local"
    assert "Thermal note" in answer.answer
