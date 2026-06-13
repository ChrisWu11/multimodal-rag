# Roadmap

## Phase 1: Local RAG MVP

- Ingest sample documents
- Search and answer with citations
- Upload images and turn them into searchable descriptions
- Test with Postman or the built-in debug UI

## Phase 2: Real Dataset Integration

- Confirm data schema with the team
- Add batch ingestion jobs
- Add metadata filters
- Add de-identification checks
- Add duplicate detection

## Phase 3: Better Retrieval

- Move vector storage from SQLite to Qdrant or pgvector
- Add corpus-scale lexical retrieval with SQLite FTS/BM25 if the corpus grows beyond the current in-memory scan
- Tune RRF/weighted fusion with the labelled evaluation set
- Extend the optional CrossEncoder reranker with model comparison and latency measurements
- Add modality-specific retrievers

## Phase 4: Multimodal Expansion

- Add calibrated thermal metadata extraction
- Add DICOM support with `pydicom`
- Add ultrasound video frame sampling
- Add image embeddings for similar-case retrieval

## Phase 5: Fine-Tuning Decision

Fine-tune only after there is a stable dataset and an evaluation set. Use fine-tuning for output format, classification, or consistent task behavior. Keep factual domain knowledge in RAG.
