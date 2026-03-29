# =============================================================
# 文件介绍：智能查询路由器（IntelligentQueryRouter）
# 目标：法律问题自动路由到传统检索、图RAG检索或组合检索。
# =============================================================
"""
法律智能查询路由器
"""

import json
import logging
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document
from .text_safety import sanitize_query_text

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    HYBRID_TRADITIONAL = "hybrid_traditional"
    GRAPH_RAG = "graph_rag"
    COMBINED = "combined"


@dataclass
class QueryAnalysis:
    query_complexity: float
    relationship_intensity: float
    reasoning_required: bool
    entity_count: int
    recommended_strategy: SearchStrategy
    confidence: float
    reasoning: str


class IntelligentQueryRouter:
    """法律场景智能查询路由。"""

    def __init__(
        self,
        traditional_retrieval,
        graph_rag_retrieval,
        llm_client,
        config,
        llm_dispatcher: Optional[object] = None,
    ):
        self.traditional_retrieval = traditional_retrieval
        self.graph_rag_retrieval = graph_rag_retrieval
        self.llm_client = llm_client
        self.config = config
        self.llm_dispatcher = llm_dispatcher
        self.route_stats = {
            "traditional_count": 0,
            "graph_rag_count": 0,
            "combined_count": 0,
            "total_queries": 0,
        }

    def _assist_chat_completion(self, messages: List[Dict[str, str]], max_tokens: int = 500):
        if self.llm_dispatcher is not None:
            response, provider, model = self.llm_dispatcher.create_chat_completion(
                role="assist",
                messages=messages,
                temperature=0.1,
                max_tokens=max_tokens,
            )
            logger.info("辅助调用通道(路由): provider=%s model=%s", provider, model)
            return response
        return self.llm_client.chat.completions.create(
            model=self.config.llm_model,
            messages=messages,
            temperature=0.1,
            max_tokens=max_tokens,
        )

    def analyze_query(self, query: str) -> QueryAnalysis:
        query = sanitize_query_text(query)
        if not query:
            return self._rule_based_analysis("")

        prompt = f"""
        你是法律RAG路由分析器。请评估以下查询并返回JSON。

        查询: {query}

        评估维度:
        1) query_complexity 0-1:
           - 0.0-0.3: 简单法条定位（如“劳动合同法第39条是什么”）
           - 0.4-0.7: 中等复杂（如“试用期辞退需要满足什么条件”）
           - 0.8-1.0: 复杂推理（如“未签合同+违法辞退会触发哪些责任链条”）
        2) relationship_intensity 0-1:
           - 是否涉及引用链、多法规关联、程序链条
        3) reasoning_required: 是否需要多跳解释
        4) entity_count: 识别到的法规/条文/场景实体数量
        5) recommended_strategy:
           - hybrid_traditional / graph_rag / combined

        返回:
        {{
          "query_complexity": 0.6,
          "relationship_intensity": 0.7,
          "reasoning_required": true,
          "entity_count": 2,
          "recommended_strategy": "graph_rag",
          "confidence": 0.82,
          "reasoning": "涉及条文关联和责任链"
        }}
        """
        try:
            response = self._assist_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            data = self._safe_json_loads(response.choices[0].message.content.strip())
            return QueryAnalysis(
                query_complexity=float(data.get("query_complexity", 0.5)),
                relationship_intensity=float(data.get("relationship_intensity", 0.5)),
                reasoning_required=bool(data.get("reasoning_required", False)),
                entity_count=int(data.get("entity_count", 1)),
                recommended_strategy=SearchStrategy(data.get("recommended_strategy", "hybrid_traditional")),
                confidence=float(data.get("confidence", 0.6)),
                reasoning=str(data.get("reasoning", "LLM分析")),
            )
        except Exception as e:
            logger.error("路由分析失败，使用规则降级: %s", e)
            return self._rule_based_analysis(query)

    def _safe_json_loads(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise

    def _rule_based_analysis(self, query: str) -> QueryAnalysis:
        query = sanitize_query_text(query)

        complexity_keywords = ["是否", "适用", "依据", "责任", "后果", "引用", "关联", "区别"]
        relation_keywords = ["根据", "参照", "同时违反", "连带", "程序", "条件", "路径", "链条"]

        query_complexity = min(
            sum(1 for kw in complexity_keywords if kw in query) / len(complexity_keywords),
            1.0,
        )
        relationship_intensity = min(
            sum(1 for kw in relation_keywords if kw in query) / len(relation_keywords),
            1.0,
        )
        reasoning_required = query_complexity > 0.35 or relationship_intensity > 0.35

        if query_complexity > 0.65 and relationship_intensity > 0.45:
            strategy = SearchStrategy.COMBINED
        elif reasoning_required:
            strategy = SearchStrategy.GRAPH_RAG
        else:
            strategy = SearchStrategy.HYBRID_TRADITIONAL

        return QueryAnalysis(
            query_complexity=query_complexity,
            relationship_intensity=relationship_intensity,
            reasoning_required=reasoning_required,
            entity_count=max(1, len(query.split())),
            recommended_strategy=strategy,
            confidence=0.65,
            reasoning="规则降级分析",
        )

    def format_routing_explanation(self, query: str, analysis: QueryAnalysis) -> str:
        query = sanitize_query_text(query)
        return f"""
查询路由分析报告

查询：{query}
复杂度：{analysis.query_complexity:.2f}
关系密集度：{analysis.relationship_intensity:.2f}
推理需求：{'是' if analysis.reasoning_required else '否'}
实体数量：{analysis.entity_count}
推荐策略：{analysis.recommended_strategy.value}
置信度：{analysis.confidence:.2f}
理由：{analysis.reasoning}
""".strip()

    def route_query(
        self,
        query: str,
        top_k: int = 5,
        analysis: Optional[QueryAnalysis] = None,
        retrieval_scope: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Document], QueryAnalysis]:
        query = sanitize_query_text(query)
        if not query:
            analysis = self._rule_based_analysis("")
            return [], analysis

        if analysis is None:
            analysis = self.analyze_query(query)
        self._update_route_stats(analysis.recommended_strategy)

        try:
            if analysis.recommended_strategy == SearchStrategy.HYBRID_TRADITIONAL:
                documents = self.traditional_retrieval.hybrid_search(query, top_k, retrieval_scope=retrieval_scope)
            elif analysis.recommended_strategy == SearchStrategy.GRAPH_RAG:
                documents = self.graph_rag_retrieval.graph_rag_search(query, top_k)
            else:
                documents = self._combined_search(query, top_k, retrieval_scope=retrieval_scope)

            # 空结果降级：图检索或组合检索返回空时自动回退传统检索
            if (
                not documents
                and analysis.recommended_strategy in {SearchStrategy.GRAPH_RAG, SearchStrategy.COMBINED}
            ):
                graph_empty_reason = getattr(self.graph_rag_retrieval, "last_empty_reason", "empty_result")
                grounding_stats = getattr(self.graph_rag_retrieval, "last_grounding_stats", {}) or {}
                logger.warning(
                    "路由策略 %s 返回空结果，自动降级到传统检索: reason=%s",
                    analysis.recommended_strategy.value,
                    graph_empty_reason,
                )
                documents = self.traditional_retrieval.hybrid_search(query, top_k, retrieval_scope=retrieval_scope)
                for doc in documents:
                    doc.metadata["route_fallback"] = "empty_result_to_traditional"
                    doc.metadata["graph_empty_reason"] = graph_empty_reason
                    doc.metadata["graph_grounding_candidates"] = grounding_stats.get("source_candidates", [])
                    doc.metadata["graph_grounding_hit_count"] = grounding_stats.get("source_hit_count", 0)

            return self._post_process_results(documents, analysis), analysis
        except Exception as e:
            logger.error("查询路由失败，降级到传统检索: %s", e)
            return self.traditional_retrieval.hybrid_search(query, top_k, retrieval_scope=retrieval_scope), analysis

    def _combined_search(
        self,
        query: str,
        top_k: int,
        retrieval_scope: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        query = sanitize_query_text(query)
        if not query:
            return []

        traditional_k = max(1, top_k // 2)
        graph_k = top_k - traditional_k

        traditional_docs = self.traditional_retrieval.hybrid_search(
            query,
            traditional_k,
            retrieval_scope=retrieval_scope,
        )
        graph_docs = self.graph_rag_retrieval.graph_rag_search(query, graph_k)

        combined: List[Document] = []
        seen = set()
        max_len = max(len(traditional_docs), len(graph_docs))
        for i in range(max_len):
            if i < len(graph_docs):
                doc = graph_docs[i]
                key = hash(doc.page_content[:120])
                if key not in seen:
                    seen.add(key)
                    doc.metadata["search_source"] = "graph_rag"
                    combined.append(doc)
            if i < len(traditional_docs):
                doc = traditional_docs[i]
                key = hash(doc.page_content[:120])
                if key not in seen:
                    seen.add(key)
                    doc.metadata["search_source"] = "traditional"
                    combined.append(doc)
        return combined[:top_k]

    def _post_process_results(self, documents: List[Document], analysis: QueryAnalysis) -> List[Document]:
        for doc in documents:
            doc.metadata.update(
                {
                    "route_strategy": analysis.recommended_strategy.value,
                    "query_complexity": analysis.query_complexity,
                    "route_confidence": analysis.confidence,
                }
            )
        return documents

    def _update_route_stats(self, strategy: SearchStrategy):
        self.route_stats["total_queries"] += 1
        if strategy == SearchStrategy.HYBRID_TRADITIONAL:
            self.route_stats["traditional_count"] += 1
        elif strategy == SearchStrategy.GRAPH_RAG:
            self.route_stats["graph_rag_count"] += 1
        elif strategy == SearchStrategy.COMBINED:
            self.route_stats["combined_count"] += 1

    def get_route_statistics(self) -> Dict[str, Any]:
        total = self.route_stats["total_queries"]
        if total == 0:
            return self.route_stats
        return {
            **self.route_stats,
            "traditional_ratio": self.route_stats["traditional_count"] / total,
            "graph_rag_ratio": self.route_stats["graph_rag_count"] / total,
            "combined_ratio": self.route_stats["combined_count"] / total,
        }

    def explain_routing_decision(self, query: str) -> str:
        query = sanitize_query_text(query)
        analysis = self.analyze_query(query)
        return self.format_routing_explanation(query, analysis)
