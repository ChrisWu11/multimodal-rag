import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.rag.embeddings import EmbeddingProvider, get_embedding_provider  # noqa: E402
from app.rag.langchain_providers import active_embedding_model, normalize_provider  # noqa: E402
from app.rag.storage import RagStore  # noqa: E402


DEFAULT_CHUNKS = ROOT / "data" / "rag_v1" / "chunks.jsonl"
DEFAULT_SOURCE_TAG = "rag_v1_ultrasound_heat_papers"
DEFAULT_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DOCUMENT_METADATA_KEYS = [
    "doc_id",
    "title",
    "authors",
    "year",
    "date",
    "journal",
    "doi",
    "pmid",
    "pmcid",
    "source",
    "source_api",
    "source_query",
    "source_queries",
    "pdf_url",
    "landing_url",
    "license",
    "relevance_score",
    "relevance_reasons",
    "page_count",
    "downloaded_pdf_path",
]

CHUNK_METADATA_KEYS = [
    "chunk_id",
    "doc_id",
    "chunk_index",
    "authors",
    "year",
    "doi",
    "pmid",
    "pmcid",
    "source",
    "access_mode",
    "section",
    "page_start",
    "page_end",
    "word_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import the RAG v1 paper chunks into this app's SQLite RAG store."
    )
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--documents", type=Path, default=None)
    parser.add_argument("--vectors", type=Path, default=None)
    parser.add_argument("--embedding-provider", default="sentence_transformers")
    parser.add_argument("--embedding-model", default=DEFAULT_ST_MODEL)
    parser.add_argument("--modality", default="text")
    parser.add_argument("--source-tag", default=DEFAULT_SOURCE_TAG)
    parser.add_argument("--source-type", default="paper")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reset-source", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    chunks_path = _resolve_path(args.chunks)
    documents_path = _resolve_path(args.documents) if args.documents else None
    vectors_path = _resolve_path(args.vectors) if args.vectors else None

    chunks = load_jsonl(chunks_path)
    if args.limit > 0:
        chunks = chunks[: args.limit]
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")

    document_lookup = (
        {
            str(row.get("doc_id")): row
            for row in load_jsonl(documents_path)
            if row.get("doc_id")
        }
        if documents_path
        else {}
    )

    settings = get_settings().with_runtime_models(
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
    )
    settings.ensure_directories()
    store = RagStore(settings.sqlite_path)
    store.init_db()

    reset_count = reset_source_documents(store, args.source_tag) if args.reset_source else 0

    embedding_provider = normalize_provider(args.embedding_provider)
    embedding_model = args.embedding_model or active_embedding_model(settings, embedding_provider)
    embeddings = load_embeddings(chunks, vectors_path, settings, args.batch_size)
    grouped = group_by_doc(chunks, embeddings)

    added_docs = 0
    skipped_docs = 0
    indexed_chunks = 0
    for doc_index, (source_doc_id, rows) in enumerate(grouped.items(), start=1):
        if document_exists(store, source_doc_id):
            skipped_docs += 1
            continue
        first = rows[0]["chunk"]
        document_meta = build_document_metadata(
            first=first,
            document_row=document_lookup.get(source_doc_id, {}),
            source_tag=args.source_tag,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
        )
        chunk_rows = [
            {
                "id": row["chunk"].get("chunk_id"),
                "chunk_index": int(row["chunk"].get("chunk_index") or index),
                "content": str(row["chunk"].get("text") or ""),
                "embedding": row["embedding"],
                "metadata": build_chunk_metadata(
                    row["chunk"],
                    source_tag=args.source_tag,
                    embedding_provider=embedding_provider,
                    embedding_model=embedding_model,
                ),
            }
            for index, row in enumerate(rows, start=1)
        ]
        store.add_document(
            title=str(first.get("title") or source_doc_id),
            source_type=args.source_type,
            modality=args.modality,
            source_path=source_path_for(document_meta),
            metadata=document_meta,
            chunks=chunk_rows,
            document_id=source_doc_id,
        )
        added_docs += 1
        indexed_chunks += len(chunk_rows)
        if doc_index % 25 == 0:
            print(
                f"Imported {added_docs} documents / {indexed_chunks} chunks...",
                file=sys.stderr,
                flush=True,
            )

    summary = {
        "sqlite_path": str(settings.sqlite_path),
        "chunks_path": str(chunks_path),
        "vectors_path": str(vectors_path) if vectors_path else None,
        "source_tag": args.source_tag,
        "reset_count": reset_count,
        "documents_added": added_docs,
        "documents_skipped": skipped_docs,
        "chunks_indexed": indexed_chunks,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _resolve_path(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    return path if path.is_absolute() else ROOT / path


def load_jsonl(path: Optional[Path]) -> list[dict[str, Any]]:
    if path is None:
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_embeddings(
    chunks: list[dict[str, Any]],
    vectors_path: Optional[Path],
    settings,
    batch_size: int,
) -> list[list[float]]:
    if vectors_path:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required to import precomputed embeddings.") from exc
        vectors = np.load(vectors_path)
        if vectors.shape[0] < len(chunks):
            raise ValueError(f"Vector count {vectors.shape[0]} is smaller than chunk count {len(chunks)}")
        return [list(map(float, vector)) for vector in vectors[: len(chunks)]]

    provider = get_embedding_provider(settings)
    texts = [embedding_text(chunk) for chunk in chunks]
    return embed_in_batches(provider, texts, batch_size)


def embed_in_batches(
    provider: EmbeddingProvider,
    texts: list[str],
    batch_size: int,
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(provider.embed_documents(batch))
        print(f"Embedded {len(vectors)}/{len(texts)} chunks...", file=sys.stderr, flush=True)
    return vectors


def embedding_text(chunk: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(chunk.get("title") or ""),
            str(chunk.get("section") or ""),
            str(chunk.get("text") or ""),
        ]
    )


def group_by_doc(
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk, embedding in zip(chunks, embeddings):
        source_doc_id = str(chunk.get("doc_id") or "unknown_doc")
        grouped[source_doc_id].append({"chunk": chunk, "embedding": embedding})
    return grouped


def build_document_metadata(
    first: dict[str, Any],
    document_row: dict[str, Any],
    source_tag: str,
    embedding_provider: str,
    embedding_model: str,
) -> dict[str, Any]:
    metadata = {
        key: document_row.get(key, first.get(key))
        for key in DOCUMENT_METADATA_KEYS
        if document_row.get(key, first.get(key)) not in (None, "")
    }
    metadata.update(
        {
            "source_tag": source_tag,
            "source_doc_id": str(first.get("doc_id") or ""),
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
        }
    )
    return metadata


def build_chunk_metadata(
    chunk: dict[str, Any],
    source_tag: str,
    embedding_provider: str,
    embedding_model: str,
) -> dict[str, Any]:
    metadata = {
        key: chunk.get(key)
        for key in CHUNK_METADATA_KEYS
        if chunk.get(key) not in (None, "")
    }
    metadata.update(
        {
            "source_tag": source_tag,
            "source_doc_id": str(chunk.get("doc_id") or ""),
            "source_chunk_id": str(chunk.get("chunk_id") or ""),
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "char_count": len(str(chunk.get("text") or "")),
        }
    )
    return metadata


def source_path_for(metadata: dict[str, Any]) -> Optional[str]:
    for key in ("pdf_url", "landing_url", "downloaded_pdf_path"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def document_exists(store: RagStore, document_id: str) -> bool:
    with store.connect() as conn:
        row = conn.execute("SELECT 1 FROM documents WHERE id = ?", (document_id,)).fetchone()
    return row is not None


def reset_source_documents(store: RagStore, source_tag: str) -> int:
    deleted = 0
    for document in store.list_documents():
        if document.metadata.get("source_tag") == source_tag:
            if store.delete_document(document.document_id):
                deleted += 1
    return deleted


if __name__ == "__main__":
    raise SystemExit(main())
