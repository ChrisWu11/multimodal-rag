# Multimodal RAG

FastAPI backend for an ultrasound and thermal-imaging oriented multimodal RAG project. The current scope is deliberately simple: ingest text/PDF/image files, create searchable chunks, retrieve evidence, and answer through Gemini, OpenAI, or Qwen when an API key is configured.

The app also works without external model keys by using a deterministic local embedding fallback and an extractive answer fallback. That makes it easy to test with Postman before real data arrives.

## What Is Implemented

- Text ingestion from raw text, `.txt`, `.md`, `.csv`, `.json`, and `.pdf`
- Image ingestion for `.png`, `.jpg`, `.jpeg`, `.webp`, and `.gif`
- Basic ultrasound/thermal image metadata extraction with Pillow
- Optional Gemini/OpenAI/Qwen generation
- Optional Gemini/OpenAI/Qwen embeddings with local fallback
- Optional vision summary for uploaded images
- SQLite-backed local vector store
- Hybrid retrieval: vector similarity plus keyword overlap
- RAG chat endpoint with evidence citations
- Minimal browser debug UI at `/`
- CLI scripts for ingestion and querying
- Unit tests for chunking, embedding, storage, retrieval, and API health

## Quick Start

```bash
cd /Users/apple/Desktop/uob/Final-Project/multimodal-rag
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Choose providers in `.env` when you want real generation and external embeddings:

```bash
LLM_PROVIDER=gemini
EMBEDDING_PROVIDER=gemini
GEMINI_API_KEY=...
```

The debug UI also lets you switch `LLM_PROVIDER`, chat model, `EMBEDDING_PROVIDER`, and embedding model for a single ingest/chat request without editing `.env`. Use the same embedding provider/model for ingestion and chat when comparing retrieval quality.

Run the API:

```bash
uvicorn app.main:app --reload
```

Open the debug UI:

```text
http://127.0.0.1:8000
```

## Postman Flow

Health:

```http
GET http://127.0.0.1:8000/api/health
```

Ingest text:

```http
POST http://127.0.0.1:8000/api/ingest/text
Content-Type: application/json

{
  "title": "Thermal imaging notes",
  "text": "Thermal imaging can highlight surface temperature asymmetry...",
  "modality": "thermal",
  "metadata": {"source": "manual-test"}
}
```

Ask:

```http
POST http://127.0.0.1:8000/api/chat
Content-Type: application/json

{
  "question": "How can thermal imaging help with screening?",
  "modality": "thermal",
  "top_k": 5,
  "llm_provider": "gemini",
  "embedding_provider": "gemini",
  "use_llm": true
}
```

Upload file:

```http
POST http://127.0.0.1:8000/api/ingest/file
form-data:
  file: sample.pdf
  modality: ultrasound
  title: Ultrasound sample report
```

Ask with image:

```http
POST http://127.0.0.1:8000/api/chat-with-image
form-data:
  image: thermal.jpg
  question: What visual cues should I pay attention to?
  modality: thermal
```

## Project Layout

```text
app/
  api/          FastAPI routes
  core/         settings and app wiring
  models/       Pydantic request/response schemas
  rag/          chunking, embeddings, retrieval, generation, storage
  services/     file and image extraction services
  static/       minimal debug UI
scripts/        local CLI helpers
tests/          pytest coverage
docs/           architecture and data notes
data/           local SQLite DB and sample docs
storage/        uploaded file copies
```

## Safety Scope

This project is for research and education. It should not provide definitive diagnosis or treatment advice. Answers should be grounded in retrieved evidence and clearly state uncertainty when the evidence is insufficient.
