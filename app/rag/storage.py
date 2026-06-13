import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app.models.schemas import DocumentSummary, EvidenceItem


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RagStore:
    def __init__(self, sqlite_path: Path) -> None:
        self.sqlite_path = sqlite_path
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    modality TEXT NOT NULL,
                    source_path TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_modality ON documents(modality)")

    def add_document(
        self,
        title: str,
        source_type: str,
        modality: str,
        source_path: Optional[str],
        metadata: Dict[str, Any],
        chunks: Iterable[Dict[str, Any]],
        document_id: Optional[str] = None,
    ) -> str:
        document_id = document_id or str(uuid.uuid4())
        created_at = utc_now_iso()
        chunk_rows = list(chunks)
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (id, title, source_type, modality, source_path, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    title,
                    source_type,
                    modality,
                    source_path,
                    json.dumps(metadata, ensure_ascii=False),
                    created_at,
                ),
            )
            for row in chunk_rows:
                conn.execute(
                    """
                    INSERT INTO chunks
                    (id, document_id, chunk_index, content, embedding_json, metadata_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(row.get("id") or row.get("chunk_id") or uuid.uuid4()),
                        document_id,
                        int(row["chunk_index"]),
                        row["content"],
                        json.dumps(row["embedding"]),
                        json.dumps(row.get("metadata", {}), ensure_ascii=False),
                        created_at,
                    ),
                )
        return document_id

    def list_documents(self) -> List[DocumentSummary]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT d.*, COUNT(c.id) AS chunks_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                GROUP BY d.id
                ORDER BY d.created_at DESC
                """
            ).fetchall()
        return [self._row_to_document_summary(row) for row in rows]

    def delete_document(self, document_id: str) -> bool:
        with self.connect() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            return cursor.rowcount > 0

    def iter_chunks(self, modality: Optional[str] = None) -> List[sqlite3.Row]:
        query = """
            SELECT
                c.id AS chunk_id,
                c.document_id,
                c.chunk_index,
                c.content,
                c.embedding_json,
                c.metadata_json AS chunk_metadata_json,
                d.title,
                d.modality,
                d.source_path,
                d.metadata_json AS document_metadata_json
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
        """
        params: tuple[Any, ...] = ()
        if modality:
            query += " WHERE d.modality = ?"
            params = (modality,)
        with self.connect() as conn:
            return conn.execute(query, params).fetchall()

    def row_to_evidence(self, row: sqlite3.Row, score: float) -> EvidenceItem:
        metadata = json.loads(row["document_metadata_json"] or "{}")
        chunk_metadata = json.loads(row["chunk_metadata_json"] or "{}")
        metadata.update(chunk_metadata)
        return EvidenceItem(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            title=row["title"],
            content=row["content"],
            score=round(float(score), 4),
            source_path=row["source_path"],
            modality=row["modality"],
            metadata=metadata,
        )

    @staticmethod
    def _row_to_document_summary(row: sqlite3.Row) -> DocumentSummary:
        return DocumentSummary(
            document_id=row["id"],
            title=row["title"],
            modality=row["modality"],
            source_type=row["source_type"],
            source_path=row["source_path"],
            chunks_count=int(row["chunks_count"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            metadata=json.loads(row["metadata_json"] or "{}"),
        )
