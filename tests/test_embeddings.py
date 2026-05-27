from app.rag.embeddings import HashEmbeddingProvider, cosine_similarity


def test_hash_embedding_is_deterministic() -> None:
    provider = HashEmbeddingProvider(dimensions=32)
    left = provider.embed("thermal imaging asymmetry")
    right = provider.embed("thermal imaging asymmetry")
    assert left == right
    assert cosine_similarity(left, right) > 0.99
