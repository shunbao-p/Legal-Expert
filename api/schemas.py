from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    chat_id: str = Field(..., min_length=1, description="Chat session id")
    question: str = Field(..., min_length=1, description="User question text")
    explain_routing: bool = Field(False, description="Whether to expose routing explanation metadata")
    active_file_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional active file ids override (None means using current session scope)",
    )


class ChatSessionResponse(BaseModel):
    chat_id: str


class SessionFileDTO(BaseModel):
    file_id: str
    file_name: str
    modality: str
    status: str
    size_bytes: int
    uploaded_at: str
    active: bool = True
    parsed_chunks: int = 0
    error: str = ""


class UploadFileResponse(BaseModel):
    file: SessionFileDTO


class SessionFilesResponse(BaseModel):
    chat_id: str
    files: List[SessionFileDTO]


class DeleteFileResponse(BaseModel):
    chat_id: str
    file_id: str
    deleted: bool


class AnalysisDTO(BaseModel):
    strategy: str
    query_complexity: float
    relationship_intensity: float
    confidence: float
    reasoning_required: bool
    reasoning: str


class EvidenceStateDTO(BaseModel):
    mode: str
    reason: str
    top_rerank_score: float = 0.0
    top_must_hit_count: int = 0


class DocumentDTO(BaseModel):
    display_title: str
    law_name: str = ""
    article_id: str = ""
    article_title: str = ""
    snippet: str = ""
    score: float = 0.0
    search_type: str = ""
    route_strategy: str = ""
    search_source: str = ""
    route_fallback: str = ""


class ChatResponse(BaseModel):
    answer: str
    analysis: AnalysisDTO
    evidence: EvidenceStateDTO
    documents: List[DocumentDTO]
    elapsed_seconds: float
    route_fallback: str = ""
    routing_explanation: str = ""


class HealthResponse(BaseModel):
    status: str
    initialized: bool
    system_ready: bool
    startup_error: str = ""
