from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.core.config import get_settings
from app.core.container import AppContainer, get_container
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentSummary,
    HealthResponse,
    IngestResponse,
    IngestTextRequest,
    SearchRequest,
    SearchResponse,
)
from app.rag.langchain_providers import configured_providers, provider_configured
from app.services.file_extraction import ExtractionError, parse_metadata_json

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        environment=settings.app_env,
        llm_provider=settings.llm_provider,
        embedding_provider=settings.embedding_provider,
        provider_configured=provider_configured(settings, settings.llm_provider),
        configured_providers=configured_providers(settings),
        gemini_configured=bool(settings.gemini_api_key),
    )


@router.get("/documents", response_model=list[DocumentSummary])
def list_documents(container: AppContainer = Depends(get_container)) -> list[DocumentSummary]:
    return container.store.list_documents()


@router.delete("/documents/{document_id}")
def delete_document(document_id: str, container: AppContainer = Depends(get_container)) -> dict:
    deleted = container.store.delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"deleted": True, "document_id": document_id}


@router.post("/ingest/text", response_model=IngestResponse)
def ingest_text(
    request: IngestTextRequest,
    container: AppContainer = Depends(get_container),
) -> IngestResponse:
    try:
        return container.ingestor.ingest_text(
            title=request.title,
            text=request.text,
            modality=request.modality,
            metadata=request.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    title: Optional[str] = Form(default=None),
    modality: str = Form(default="unknown"),
    metadata_json: Optional[str] = Form(default=None),
    container: AppContainer = Depends(get_container),
) -> IngestResponse:
    try:
        metadata = parse_metadata_json(metadata_json)
        data = await file.read()
        return container.ingestor.ingest_file(
            filename=file.filename or "uploaded-file",
            data=data,
            title=title,
            modality=modality,
            metadata=metadata,
        )
    except (ExtractionError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    container: AppContainer = Depends(get_container),
) -> SearchResponse:
    return container.pipeline.search(
        query=request.query,
        top_k=request.top_k,
        modality=request.modality,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    container: AppContainer = Depends(get_container),
) -> ChatResponse:
    return container.pipeline.answer(
        question=request.question,
        top_k=request.top_k,
        modality=request.modality,
        use_llm=request.use_llm,
    )


@router.post("/chat-with-image", response_model=ChatResponse)
async def chat_with_image(
    image: UploadFile = File(...),
    question: str = Form(...),
    top_k: int = Form(default=5),
    modality: Optional[str] = Form(default=None),
    use_llm: bool = Form(default=True),
    container: AppContainer = Depends(get_container),
) -> ChatResponse:
    data = await image.read()
    return container.pipeline.answer(
        question=question,
        top_k=top_k,
        modality=modality,
        use_llm=use_llm,
        image_filename=image.filename or "uploaded-image",
        image_bytes=data,
    )
