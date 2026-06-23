from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.core.container import AppContainer, get_container
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentSummary,
    HealthResponse,
    IngestResponse,
    IngestTextRequest,
    ModelConfigResponse,
    SearchRequest,
    SearchResponse,
)
from app.rag.langchain_providers import (
    active_embedding_model,
    active_llm_model,
    configured_providers,
    embedding_model_options,
    model_options,
    provider_configured,
)
from app.services.file_extraction import ExtractionError, parse_metadata_json
from app.services.image_analysis import ImageValidationError, validate_image_upload

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


@router.get("/model-config", response_model=ModelConfigResponse)
def model_config() -> ModelConfigResponse:
    settings = get_settings()
    return ModelConfigResponse(
        llm_provider=settings.llm_provider,
        embedding_provider=settings.embedding_provider,
        llm_model=active_llm_model(settings),
        embedding_model=active_embedding_model(settings),
        configured_providers=configured_providers(settings),
        model_options=model_options(),
        embedding_model_options=embedding_model_options(),
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
    settings = get_settings().with_runtime_models(
        embedding_provider=request.embedding_provider,
        embedding_model=request.embedding_model,
    )
    try:
        return container.create_ingestor(settings).ingest_text(
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
    embedding_provider: Optional[str] = Form(default=None),
    embedding_model: Optional[str] = Form(default=None),
    container: AppContainer = Depends(get_container),
) -> IngestResponse:
    settings = get_settings().with_runtime_models(
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
    )
    try:
        metadata = parse_metadata_json(metadata_json)
        data = await file.read()
        return container.create_ingestor(settings).ingest_file(
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
    settings = get_settings().with_runtime_models(
        embedding_provider=request.embedding_provider,
        embedding_model=request.embedding_model,
    )
    return container.create_pipeline(settings).search(
        query=request.query,
        top_k=request.top_k,
        modality=request.modality,
        use_reranker=request.use_reranker,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    container: AppContainer = Depends(get_container),
) -> ChatResponse:
    settings = get_settings().with_runtime_models(
        llm_provider=request.llm_provider,
        llm_model=request.llm_model,
        embedding_provider=request.embedding_provider,
        embedding_model=request.embedding_model,
    )
    return container.create_pipeline(settings).answer(
        question=request.question,
        top_k=request.top_k,
        modality=request.modality,
        use_llm=request.use_llm,
        use_reranker=request.use_reranker,
    )


@router.post("/chat-with-image", response_model=ChatResponse)
async def chat_with_image(
    image: UploadFile = File(...),
    question: str = Form(...),
    top_k: int = Form(default=5),
    modality: Optional[str] = Form(default=None),
    image_modality: str = Form(default="unknown"),
    use_llm: bool = Form(default=True),
    llm_provider: Optional[str] = Form(default=None),
    llm_model: Optional[str] = Form(default=None),
    embedding_provider: Optional[str] = Form(default=None),
    embedding_model: Optional[str] = Form(default=None),
    use_reranker: Optional[bool] = Form(default=None),
    container: AppContainer = Depends(get_container),
) -> ChatResponse:
    settings = get_settings().with_runtime_models(
        llm_provider=llm_provider,
        llm_model=llm_model,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
    )
    clean_question = question.strip()
    if not clean_question:
        raise HTTPException(status_code=400, detail="Question is required.")
    if top_k < 1 or top_k > 20:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 20.")
    clean_image_modality = image_modality.strip().lower()
    if clean_image_modality not in {"thermal", "ultrasound", "scientific_figure", "unknown"}:
        raise HTTPException(status_code=400, detail="Unsupported image modality.")

    filename = image.filename or "uploaded-image"
    data = await image.read(settings.max_image_upload_bytes + 1)
    try:
        validate_image_upload(
            filename=filename,
            data=data,
            content_type=image.content_type,
            max_bytes=settings.max_image_upload_bytes,
            max_pixels=settings.max_image_pixels,
        )
    except ImageValidationError as exc:
        status_code = 413 if len(data) > settings.max_image_upload_bytes else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    pipeline = container.create_pipeline(settings)
    return await run_in_threadpool(
        pipeline.answer,
        question=clean_question,
        top_k=top_k,
        modality=modality,
        image_modality=clean_image_modality,
        use_llm=use_llm,
        image_filename=filename,
        image_bytes=data,
        use_reranker=use_reranker,
    )
