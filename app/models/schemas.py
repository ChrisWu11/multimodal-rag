from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app: str
    environment: str
    openai_configured: bool


class EvidenceItem(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    content: str
    score: float
    source_path: Optional[str] = None
    modality: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestTextRequest(BaseModel):
    title: str
    text: str
    modality: str = "text"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    document_id: str
    title: str
    chunks_indexed: int
    source_path: Optional[str] = None
    modality: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    modality: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    results: List[EvidenceItem]


class ChatRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)
    modality: Optional[str] = None
    use_openai: bool = True


class ChatResponse(BaseModel):
    answer: str
    evidence: List[EvidenceItem]
    safety_notice: str
    visual_summary: Optional[str] = None
    used_openai: bool
    model: Optional[str] = None


class DocumentSummary(BaseModel):
    document_id: str
    title: str
    modality: str
    source_type: str
    source_path: Optional[str] = None
    chunks_count: int
    created_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
