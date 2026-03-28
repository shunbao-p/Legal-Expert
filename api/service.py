from __future__ import annotations

import logging
import threading
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from main import AdvancedGraphRAGSystem

logger = logging.getLogger(__name__)


class RAGDemoService:
    """Thin service wrapper that exposes the existing GraphRAG system to FastAPI."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._system: Optional["AdvancedGraphRAGSystem"] = None
        self._initialized = False
        self._startup_error = ""

    @property
    def startup_error(self) -> str:
        return self._startup_error

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def system_ready(self) -> bool:
        return bool(self._system and self._system.system_ready)

    def startup(self) -> None:
        with self._lock:
            if self._initialized and self._system is not None:
                return

            try:
                from main import AdvancedGraphRAGSystem

                system = AdvancedGraphRAGSystem()
                system.initialize_system()
                system.build_knowledge_base()
                self._system = system
                self._initialized = True
                self._startup_error = ""
                logger.info("RAG demo service startup complete")
            except Exception as exc:
                self._startup_error = str(exc)
                logger.exception("RAG demo service startup failed")
                raise

    def shutdown(self) -> None:
        with self._lock:
            if self._system is not None:
                self._system._cleanup()
            self._system = None
            self._initialized = False

    def health(self) -> dict:
        if self.system_ready:
            status = "ready"
        elif self._startup_error:
            status = "failed"
        else:
            status = "starting"
        return {
            "status": status,
            "initialized": self._initialized,
            "system_ready": self.system_ready,
            "startup_error": self._startup_error,
        }

    def chat(self, question: str, explain_routing: bool = False) -> dict:
        if not self._initialized or self._system is None:
            self.startup()
        assert self._system is not None
        return self._system.ask_question_payload(question, explain_routing=explain_routing)
