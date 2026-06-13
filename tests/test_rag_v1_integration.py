from pathlib import Path

from app.models.schemas import EvidenceItem
from app.rag.generator import _format_context
from app.rag.storage import RagStore


def test_store_can_preserve_source_ids(tmp_path: Path) -> None:
    store = RagStore(tmp_path / "rag.db")
    store.init_db()

    store.add_document(
        title="Paper",
        source_type="paper",
        modality="text",
        source_path=None,
        metadata={"source_tag": "rag_v1"},
        chunks=[
            {
                "id": "doc_0001_c0001",
                "chunk_index": 1,
                "content": "Focused ultrasound can deposit heat in target tissue.",
                "embedding": [1.0, 0.0],
                "metadata": {"source_chunk_id": "doc_0001_c0001"},
            }
        ],
        document_id="doc_0001",
    )

    rows = store.iter_chunks()
    assert rows[0]["document_id"] == "doc_0001"
    assert rows[0]["chunk_id"] == "doc_0001_c0001"


def test_context_includes_citation_metadata() -> None:
    item = EvidenceItem(
        chunk_id="doc_0001_c0001",
        document_id="doc_0001",
        title="Thermal ablation paper",
        content="Temperature monitoring is required during thermal ablation.",
        score=0.91,
        modality="text",
        metadata={
            "year": "2025",
            "doi": "10.123/example",
            "page_start": 4,
            "page_end": 5,
            "section": "Monitoring",
            "source_chunk_id": "doc_0001_c0001",
            "retrieval_method": "hybrid_rrf",
        },
    )

    context = _format_context([item], max_chars=2000)

    assert "10.123/example" in context
    assert "pp. 4-5" in context
    assert "doc_0001_c0001" in context
    assert "hybrid_rrf" in context
