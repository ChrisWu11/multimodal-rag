# Architecture

The MVP is a local-first RAG backend. This branch introduces LangChain as the model, embedding, text splitting, and prompt chaining layer while keeping storage and retrieval explicit.

## Runtime Components

```text
Client/Postman
  |
  v
FastAPI
  |
  +-- Ingestion service
  |     +-- text/PDF extraction
  |     +-- image metadata + optional LangChain multimodal summary
  |     +-- LangChain text splitter
  |     +-- LangChain embedding provider
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
        +-- LangChain chat model when configured
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
- optional LangChain multimodal vision description

This is enough for the first RAG prototype. When actual image datasets arrive, add:

- radiometric thermal metadata parsing
- DICOM/ultrasound video frame extraction
- image embedding model
- similarity search over image vectors
- dataset-specific annotation schema

## Provider Use

Choose providers with:

- `LLM_PROVIDER`: `gemini`, `openai`, or `qwen`
- `EMBEDDING_PROVIDER`: `gemini`, `openai`, `qwen`, or local fallback

Provider-specific variables:

- Gemini: `GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_EMBEDDING_MODEL`
- OpenAI: `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_EMBEDDING_MODEL`
- Qwen: `QWEN_API_KEY`, `QWEN_BASE_URL`, `QWEN_MODEL`, `QWEN_EMBEDDING_MODEL`

If the chosen provider API key is missing, the system falls back to deterministic hash embeddings and a conservative extractive answer. This keeps local development and tests stable.
