# Multimodal RAG

FastAPI backend for an ultrasound and thermal-imaging oriented multimodal RAG project. This branch uses LangChain for text splitting, embedding providers, chat models, and prompt chaining while keeping the FastAPI API surface simple.

The app works without external API keys by using a deterministic local embedding fallback and an extractive answer fallback. That makes it easy to test with Postman before real data arrives.

The reviewed report, LaTeX source, and privacy-safe evaluation summaries are available in [`docs/final-report/`](docs/final-report/README.md).

## What Is Implemented

- Text ingestion from raw text, `.txt`, `.md`, `.csv`, `.json`, and `.pdf`
- Image ingestion for `.png`, `.jpg`, `.jpeg`, `.webp`, and `.gif`
- Basic ultrasound/thermal image metadata extraction with Pillow
- LangChain text splitting and provider adapters
- Optional Gemini, OpenAI, or Qwen model generation
- Optional Gemini, OpenAI, or Qwen embeddings with local fallback
- Optional local SentenceTransformers embeddings for offline scientific-paper retrieval
- Question-aware Gemini Vision summaries for uploaded images
- Image-grounded chat with separate visual observations and literature citations
- SQLite-backed local vector store
- Hybrid retrieval: vector similarity plus keyword overlap with RRF/weighted fusion
- Optional CrossEncoder reranking over the retrieved candidate pool
- RAG chat endpoint with evidence citations
- React demo chat UI at `/`
- Browser debug UI at `/debug`
- CLI scripts for ingestion and querying
- Unit tests for chunking, embedding, storage, retrieval, and API health

## Quick Start

```bash
cd multimodal-rag
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
cd frontend
npm install
npm run build
cd ..
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

# Local SentenceTransformers retrieval + Gemini answer generation
LLM_PROVIDER=gemini
EMBEDDING_PROVIDER=sentence_transformers
GEMINI_API_KEY=...
SENTENCE_TRANSFORMER_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENABLE_RERANKER=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
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
http://127.0.0.1:8000/debug
```

Open the final demo chat UI:

```text
http://127.0.0.1:8000
```

The demo UI is built with Vite + React and is served by FastAPI after `npm run build`.
The debug UI keeps the lower-level runtime model controls for ingestion, search, and API checks.
Use the same embedding provider/model for ingestion and chat when comparing retrieval quality.

In the demo composer, attach a PNG, JPEG, or WebP image (maximum 10 MB), select its
type, and ask a question. The image is converted into a question-aware visual summary,
which expands the literature retrieval query. The answer keeps visual observations
labelled as `[Image]` and reserves numbered citations such as `[1]` for retrieved papers.

## Import The RAG V1 Paper Corpus

The final demo can reuse the curated ultrasound/thermal paper chunks produced during the first RAG route. If the precomputed SentenceTransformers vectors are available, import them directly:

```bash
export RAG_V1_DIR=/path/to/private/rag_v1
python scripts/ingest_rag_v1_chunks.py \
  --chunks "$RAG_V1_DIR/chunks.jsonl" \
  --documents "$RAG_V1_DIR/documents.jsonl" \
  --vectors "$RAG_V1_DIR/st_all_minilm_l6_v2_embeddings.npy" \
  --embedding-provider sentence_transformers \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --reset-source
```

This preserves the original paper `doc_id`, `chunk_id`, DOI, year, section, and page metadata in `data/rag.db`. In the debug UI, choose `SentenceTransformers` for embeddings and keep Gemini as the LLM provider to run the demo as: local scientific retrieval + Gemini grounded answer generation.

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
  question: What features are visible, and what does the literature say about them?
  image_modality: thermal
  top_k: 5
  use_llm: true
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
frontend/       Vite + React demo chat UI
scripts/        local CLI helpers
tests/          pytest coverage
docs/           architecture and data notes
data/           local SQLite DB and sample docs
storage/        uploaded file copies
```

## Safety Scope

This project is for research and education. It should not provide definitive diagnosis or treatment advice. Answers should be grounded in retrieved evidence and clearly state uncertainty when the evidence is insufficient.
