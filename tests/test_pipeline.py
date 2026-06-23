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


def test_image_question_is_used_for_visual_retrieval(tmp_path: Path) -> None:
    class RecordingImageAnalyzer:
        def __init__(self) -> None:
            self.question = None
            self.modality = None

        def summarize(self, filename, data, modality="unknown", question=None):
            self.question = question
            self.modality = modality
            return "Observable focal bright region. Retrieval terms: ultrasound tissue heating.", {}

    class RecordingRetriever:
        def __init__(self) -> None:
            self.query = None
            self.modality = "not-called"

        def retrieve(self, query, top_k, modality=None, use_reranker=None):
            self.query = query
            self.modality = modality
            return []

    class RecordingGenerator:
        def __init__(self) -> None:
            self.visual_summary = None

        def generate(self, question, evidence, visual_summary=None, use_llm=True):
            self.visual_summary = visual_summary
            return "No evidence available.", False, None, "local"

    settings = Settings(
        gemini_api_key=None,
        sqlite_path=tmp_path / "rag.db",
        data_dir=tmp_path,
        storage_dir=tmp_path / "storage",
        upload_dir=tmp_path / "storage" / "uploads",
    )
    analyzer = RecordingImageAnalyzer()
    retriever = RecordingRetriever()
    generator = RecordingGenerator()
    pipeline = RagPipeline(settings, retriever, generator, analyzer)

    response = pipeline.answer(
        question="What visible feature may relate to heating?",
        image_filename="sample.png",
        image_bytes=b"image-bytes",
        image_modality="ultrasound",
        use_llm=False,
    )

    assert analyzer.question == "What visible feature may relate to heating?"
    assert analyzer.modality == "ultrasound"
    assert "Observable focal bright region" in retriever.query
    assert retriever.modality is None
    assert generator.visual_summary == response.visual_summary
