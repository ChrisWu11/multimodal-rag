# Multimodal RAG

FastAPI backend for an ultrasound and thermal-imaging oriented multimodal RAG project. This branch uses LangChain for text splitting, embedding providers, chat models, and prompt chaining while keeping the FastAPI API surface simple.

The app works without external API keys by using a deterministic local embedding fallback and an extractive answer fallback. That makes it easy to test with Postman before real data arrives.

## What Is Implemented

- Text ingestion from raw text, `.txt`, `.md`, `.csv`, `.json`, and `.pdf`
- Image ingestion for `.png`, `.jpg`, `.jpeg`, `.webp`, and `.gif`
- Basic ultrasound/thermal image metadata extraction with Pillow
- LangChain text splitting and provider adapters
- Optional Gemini, OpenAI, or Qwen model generation
- Optional Gemini, OpenAI, or Qwen embeddings with local fallback
- Optional LangChain multimodal image summary for uploaded images
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

Choose providers in `.env`:

```bash
LLM_PROVIDER=gemini
EMBEDDING_PROVIDER=gemini
GEMINI_API_KEY=...
```

Provider examples:

```bash
# Gemini
LLM_PROVIDER=gemini
EMBEDDING_PROVIDER=gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001

# OpenAI
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Qwen / DashScope OpenAI-compatible API
LLM_PROVIDER=qwen
EMBEDDING_PROVIDER=qwen
QWEN_API_KEY=...
QWEN_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-plus
QWEN_EMBEDDING_MODEL=text-embedding-v4
```

Run the API:

```bash
uvicorn app.main:app --reload
```

Or use the quick debug launcher:

```bash
./scripts/dev_server.sh
```

On macOS, you can also double-click:

```text
start-debug.command
```

Open the debug UI:

```text
http://127.0.0.1:8000
```

The debug UI includes runtime model controls. You can switch `LLM_PROVIDER`, chat model,
`EMBEDDING_PROVIDER`, and embedding model for a single ingest/chat request without editing `.env`.
Use the same embedding provider/model for ingestion and chat when comparing retrieval quality.

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
