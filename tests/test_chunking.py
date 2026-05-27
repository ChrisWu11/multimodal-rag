from app.rag.chunking import chunk_text


def test_chunk_text_returns_chunks() -> None:
    chunks = chunk_text("A first paragraph.\n\nA second paragraph.", chunk_size=30, overlap=5)
    assert chunks
    assert chunks[0].index == 0
    assert "first" in chunks[0].content
