from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import Settings
from app.models.schemas import IngestResponse
from app.rag.chunking import chunk_text
from app.rag.embeddings import EmbeddingProvider
from app.rag.storage import RagStore
from app.services.file_extraction import extract_text, source_type_for_filename
from app.services.image_analysis import ImageAnalyzer


class DocumentIngestor:
    def __init__(
        self,
        settings: Settings,
        store: RagStore,
        embeddings: EmbeddingProvider,
        image_analyzer: ImageAnalyzer,
    ) -> None:
        self.settings = settings
        self.store = store
        self.embeddings = embeddings
        self.image_analyzer = image_analyzer

    def ingest_text(
        self,
        title: str,
        text: str,
        modality: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
        source_path: Optional[str] = None,
        source_type: str = "text",
    ) -> IngestResponse:
        chunks = chunk_text(text, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            raise ValueError("No text content could be indexed.")

        chunk_rows = []
        for chunk in chunks:
            chunk_rows.append(
                {
                    "chunk_index": chunk.index,
                    "content": chunk.content,
                    "embedding": self.embeddings.embed(chunk.content),
                    "metadata": {"char_count": len(chunk.content)},
                }
            )

        document_id = self.store.add_document(
            title=title,
            source_type=source_type,
            modality=modality,
            source_path=source_path,
            metadata=metadata or {},
            chunks=chunk_rows,
        )
        return IngestResponse(
            document_id=document_id,
            title=title,
            chunks_indexed=len(chunk_rows),
            source_path=source_path,
            modality=modality,
        )

    def ingest_file(
        self,
        filename: str,
        data: bytes,
        title: Optional[str] = None,
        modality: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IngestResponse:
        saved_path = self._save_upload(filename, data)
        source_type = source_type_for_filename(filename)
        merged_metadata = dict(metadata or {})
        merged_metadata["original_filename"] = filename

        if source_type == "image":
            summary, image_metadata = self.image_analyzer.summarize(filename, data, modality)
            merged_metadata.update(image_metadata)
            text = summary
        else:
            text, source_type = extract_text(filename, data)

        return self.ingest_text(
            title=title or Path(filename).stem,
            text=text,
            modality=modality,
            metadata=merged_metadata,
            source_path=str(saved_path),
            source_type=source_type,
        )

    def _save_upload(self, filename: str, data: bytes) -> Path:
        safe_name = Path(filename).name.replace(" ", "_")
        target = self.settings.upload_dir / safe_name
        if target.exists():
            stem = target.stem
            suffix = target.suffix
            counter = 2
            while target.exists():
                target = self.settings.upload_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        target.write_bytes(data)
        return target
