from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import ChatRequest, ChatResponse, HealthResponse
from api.service import RAGDemoService

logger = logging.getLogger(__name__)

app = FastAPI(title="Legal GraphRAG Demo API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = RAGDemoService()


@app.on_event("startup")
def on_startup() -> None:
    try:
        service.startup()
    except Exception:
        logger.exception("FastAPI startup could not fully initialize the RAG service")


@app.on_event("shutdown")
def on_shutdown() -> None:
    service.shutdown()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(**service.health())


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        response = service.chat(payload.question, explain_routing=payload.explain_routing)
        return ChatResponse(**response)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Chat endpoint failed")
        raise HTTPException(status_code=500, detail=f"问答接口执行失败: {exc}") from exc
