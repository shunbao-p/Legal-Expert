from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question text")
    explain_routing: bool = Field(False, description="Whether to expose routing explanation metadata")


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
