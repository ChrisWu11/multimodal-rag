# Architecture

The MVP is a local-first RAG backend. It is intentionally small so it can be validated before the team confirms the final ultrasound and thermal-imaging datasets.

## Runtime Components

```text
Client/Postman
  |
  v
FastAPI
  |
  +-- Ingestion service
  |     +-- text/PDF extraction
  |     +-- image metadata + optional OpenAI vision summary
  |     +-- chunking
  |     +-- embeddings
  |
  +-- SQLite RAG store
  |     +-- documents table
  |     +-- chunks table with embedding JSON
  |
  +-- Hybrid retriever
  |     +-- cosine vector similarity
  |     +-- keyword overlap
  |
  +-- Answer generator
        +-- OpenAI Responses API when configured
        +-- extractive fallback when not configured
```

## Why SQLite First

SQLite keeps the first version easy to run on a laptop and easy to test. The storage boundary is isolated in `app/rag/storage.py`, so the next step can replace it with Qdrant or pgvector without changing the API layer.

## Data Model

`documents`

- `id`
- `title`
- `source_type`: text, pdf, image, unknown
- `modality`: text, ultrasound, thermal, unknown
- `source_path`
- `metadata_json`
- `created_at`

`chunks`

- `id`
- `document_id`
- `chunk_index`
- `content`
- `embedding_json`
- `metadata_json`
- `created_at`

## Multimodal Strategy

For images, the MVP indexes a text representation:

- local image metadata
- local intensity summary
- optional OpenAI vision description

This is enough for the first RAG prototype. When actual image datasets arrive, add:

- radiometric thermal metadata parsing
- DICOM/ultrasound video frame extraction
- image embedding model
- similarity search over image vectors
- dataset-specific annotation schema

## OpenAI Use

- Embeddings: `OPENAI_EMBEDDING_MODEL`
- Generation: `OPENAI_CHAT_MODEL`
- Vision summary: same chat model with image input

If `OPENAI_API_KEY` is missing, the system falls back to deterministic hash embeddings and a conservative extractive answer. This keeps local development and tests stable.
