# =============================================================
# 文件介绍：混合检索模块（HybridRetrievalModule）
# 目标：法律场景下结合实体级/主题级/向量检索，进行稳定混合召回。
# =============================================================
"""
法律混合检索模块
"""

import json
import logging
import re
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from neo4j import GraphDatabase

from .graph_indexing import GraphIndexingModule
from .text_safety import sanitize_query_text

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    content: str
    node_id: str
    node_type: str
    relevance_score: float
    retrieval_level: str
    metadata: Dict[str, Any]


class HybridRetrievalModule:
    """法律场景混合检索模块。"""

    def __init__(self, config, milvus_module, data_module, llm_client, llm_dispatcher: Optional[object] = None):
        self.config = config
        self.milvus_module = milvus_module
        self.data_module = data_module
        self.llm_client = llm_client
        self.llm_dispatcher = llm_dispatcher
        self.driver = None
        self.bm25_retriever = None
        self.graph_indexing = GraphIndexingModule(config, llm_client, llm_dispatcher=llm_dispatcher)
        self.graph_indexed = False

    def _assist_chat_completion(self, messages: List[Dict[str, str]], max_tokens: int = 500):
        if self.llm_dispatcher is not None:
            response, provider, model = self.llm_dispatcher.create_chat_completion(
                role="assist",
                messages=messages,
                temperature=0.1,
                max_tokens=max_tokens,
            )
            logger.info("辅助调用通道(关键词): provider=%s model=%s", provider, model)
            return response
        return self.llm_client.chat.completions.create(
            model=self.config.llm_model,
            messages=messages,
            temperature=0.1,
            max_tokens=max_tokens,
        )

    def initialize(self, chunks: List[Document]):
        """初始化检索系统。"""
        logger.info("初始化混合检索模块...")
        self.driver = GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_user, self.config.neo4j_password),
        )

        if chunks:
            self.bm25_retriever = BM25Retriever.from_documents(chunks)
            logger.info("BM25检索器初始化完成，文档数量: %s", len(chunks))
        self._build_graph_index()

    def _build_graph_index(self):
        if self.graph_indexed:
            return
        logger.info("开始构建法律图索引...")
        try:
            self.graph_indexing.create_entity_key_values(
                self.data_module.law_documents,
                self.data_module.articles,
                self.data_module.compliance_steps,
                self.data_module.risk_scenarios,
            )
            relationships = self._extract_relationships_from_graph()
            self.graph_indexing.create_relation_key_values(relationships)
            self.graph_indexing.deduplicate_entities_and_relations()
            self.graph_indexed = True
            logger.info("法律图索引构建完成: %s", self.graph_indexing.get_statistics())
        except Exception as e:
            logger.error("构建图索引失败: %s", e)

    def _extract_relationships_from_graph(self) -> List[Tuple[str, str, str]]:
        relationships: List[Tuple[str, str, str]] = []
        relation_types = getattr(self.config, "graph_relation_types", [])
        relation_filter = ""
        params: Dict[str, Any] = {}
        if relation_types:
            relation_filter = "AND type(r) IN $relation_types"
            params["relation_types"] = relation_types

        query = f"""
        MATCH (source)-[r]->(target)
        WHERE (source:LawDocument OR source:Article OR source:RiskScenario OR source:ComplianceStep)
          AND (target:LawDocument OR target:Article OR target:RiskScenario OR target:ComplianceStep)
          {relation_filter}
        RETURN
            COALESCE(toString(source.nodeId), elementId(source)) AS source_id,
            type(r) AS relation_type,
            COALESCE(toString(target.nodeId), elementId(target)) AS target_id
        LIMIT 3000
        """

        try:
            with self.driver.session(database=self.config.neo4j_database) as session:
                for record in session.run(query, params):
                    relationships.append(
                        (record["source_id"], record["relation_type"], record["target_id"])
                    )
        except Exception as e:
            logger.error("提取法律图关系失败: %s", e)
        return relationships

    def extract_query_keywords(self, query: str) -> Tuple[List[str], List[str]]:
        """提取法律查询关键词（实体级 + 主题级）。"""
        query = sanitize_query_text(query)
        if not query:
            return [], []

        prompt = f"""
        你是法律检索助手。请从下述问题提取两类关键词，并返回 JSON。

        问题：{query}

        1) entity_keywords：法规名、条文号、主体、行为、具体制度名
        2) topic_keywords：劳动争议、合同履行、数据合规、责任承担、程序义务等主题词

        返回格式：
        {{
          "entity_keywords": ["关键词1", "关键词2"],
          "topic_keywords": ["主题1", "主题2"]
        }}
        """
        try:
            response = self._assist_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            result = self._safe_json_loads(response.choices[0].message.content.strip())
            entity_keywords = [k.strip() for k in result.get("entity_keywords", []) if k and k.strip()]
            topic_keywords = [k.strip() for k in result.get("topic_keywords", []) if k and k.strip()]
            if entity_keywords or topic_keywords:
                return entity_keywords, topic_keywords
        except Exception as e:
            logger.error("关键词提取失败，使用规则降级: %s", e)
        return self._rule_based_keywords(query)

    def _rule_based_keywords(self, query: str) -> Tuple[List[str], List[str]]:
        query = sanitize_query_text(query)
        if not query:
            return [], []

        entity_keywords: List[str] = []
        topic_keywords: List[str] = []

        article_hits = re.findall(r"第[0-9一二三四五六七八九十百千]+条", query)
        entity_keywords.extend(article_hits)

        legal_terms = ["劳动合同法", "民法典", "个人信息保护法", "劳动争议", "试用期", "辞退", "赔偿", "社保"]
        for term in legal_terms:
            if term in query:
                if term in {"劳动争议", "试用期", "辞退", "赔偿", "社保"}:
                    topic_keywords.append(term)
                else:
                    entity_keywords.append(term)

        if not entity_keywords:
            entity_keywords = [query[:12]]
        if not topic_keywords:
            topic_keywords = [query[:12]]
        return entity_keywords[:5], topic_keywords[:5]

    def _safe_json_loads(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise

    def entity_level_retrieval(self, entity_keywords: List[str], top_k: int = 5) -> List[RetrievalResult]:
        """实体级检索：优先基于法律图索引。"""
        results: List[RetrievalResult] = []
        for keyword in entity_keywords:
            for entity in self.graph_indexing.search_entities(keyword, limit=top_k):
                node_id = entity.metadata["node_id"]
                neighbors = self._get_node_neighbors(node_id, max_neighbors=3)
                content = entity.value_content
                if neighbors:
                    content += f"\n相关节点: {', '.join(neighbors)}"
                results.append(
                    RetrievalResult(
                        content=content,
                        node_id=node_id,
                        node_type=entity.entity_type,
                        relevance_score=0.92,
                        retrieval_level="entity",
                        metadata={
                            "entity_name": entity.entity_name,
                            "entity_type": entity.entity_type,
                            "matched_keyword": keyword,
                            "index_keys": entity.index_keys,
                        },
                    )
                )

        if len(results) < top_k:
            results.extend(self._neo4j_entity_level_search(entity_keywords, top_k - len(results)))

        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return self._dedup_results(results)[:top_k]

    def _neo4j_entity_level_search(self, keywords: List[str], limit: int) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []
        index_name = getattr(self.config, "neo4j_legal_fulltext_index", "legal_fulltext_idx")
        query = """
        UNWIND $keywords AS keyword
        CALL db.index.fulltext.queryNodes($index_name, keyword) YIELD node, score
        RETURN
            COALESCE(toString(node.nodeId), elementId(node)) AS node_id,
            labels(node) AS labels,
            COALESCE(node.name, node.title, node.articleId, '') AS display_name,
            COALESCE(node.articleId, '') AS article_id,
            COALESCE(node.title, '') AS article_title,
            COALESCE(node.content, node.description, '') AS content,
            score
        ORDER BY score DESC
        LIMIT $limit
        """
        try:
            with self.driver.session(database=self.config.neo4j_database) as session:
                for record in session.run(
                    query, {"keywords": keywords, "limit": limit, "index_name": index_name}
                ):
                    labels = record["labels"] or []
                    node_type = labels[0] if labels else "LegalNode"
                    content = f"名称: {record['display_name']}"
                    if record["article_id"]:
                        content += f"\n条文编号: {record['article_id']}"
                    if record["article_title"]:
                        content += f"\n条文标题: {record['article_title']}"
                    if record["content"]:
                        content += f"\n内容: {record['content'][:400]}"
                    results.append(
                        RetrievalResult(
                            content=content,
                            node_id=record["node_id"],
                            node_type=node_type,
                            relevance_score=float(record["score"]) * 0.8,
                            retrieval_level="entity",
                            metadata={
                                "law_name": record["display_name"] if node_type == "LawDocument" else "",
                                "article_id": record["article_id"],
                                "article_title": record["article_title"],
                                "source": "neo4j_fulltext",
                            },
                        )
                    )
        except Exception as e:
            logger.error("Neo4j实体检索失败: %s", e)
        return results

    def topic_level_retrieval(self, topic_keywords: List[str], top_k: int = 5) -> List[RetrievalResult]:
        """主题级检索：优先关系索引，再降级 Neo4j 查询。"""
        results: List[RetrievalResult] = []
        for keyword in topic_keywords:
            relations = self.graph_indexing.search_relations(keyword, limit=top_k)
            for relation in relations:
                source = self.graph_indexing.entity_kv_store.get(relation.source_entity)
                target = self.graph_indexing.entity_kv_store.get(relation.target_entity)
                if not source or not target:
                    continue
                content = (
                    f"主题: {keyword}\n"
                    f"{relation.value_content}\n"
                    f"关系说明: {source.entity_name} 与 {target.entity_name}"
                )
                results.append(
                    RetrievalResult(
                        content=content,
                        node_id=relation.source_entity,
                        node_type=source.entity_type,
                        relevance_score=0.9,
                        retrieval_level="topic",
                        metadata={
                            "relation_type": relation.relation_type,
                            "law_name": source.entity_name if source.entity_type == "LawDocument" else "",
                            "article_id": source.metadata.get("properties", {}).get("articleId", ""),
                            "matched_keyword": keyword,
                            "source": "graph_relation",
                        },
                    )
                )

        if len(results) < top_k:
            results.extend(self._neo4j_topic_level_search(topic_keywords, top_k - len(results)))

        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return self._dedup_results(results)[:top_k]

    def _neo4j_topic_level_search(self, keywords: List[str], limit: int) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []
        query = """
        UNWIND $keywords AS keyword
        MATCH (a:Article)-[:BELONGS_TO_DOMAIN]->(d:LegalDomain)
        OPTIONAL MATCH (law:LawDocument)-[:HAS_ARTICLE]->(a)
        WHERE d.name CONTAINS keyword
           OR COALESCE(a.content, '') CONTAINS keyword
           OR COALESCE(a.title, '') CONTAINS keyword
        RETURN
            COALESCE(toString(a.nodeId), elementId(a)) AS node_id,
            COALESCE(law.name, '') AS law_name,
            COALESCE(a.articleId, '') AS article_id,
            COALESCE(a.title, '') AS article_title,
            COALESCE(d.name, '') AS legal_domain,
            COALESCE(a.content, '') AS content
        LIMIT $limit
        """
        try:
            with self.driver.session(database=self.config.neo4j_database) as session:
                for record in session.run(query, {"keywords": keywords, "limit": limit}):
                    text = (
                        f"法规: {record['law_name']}\n"
                        f"条文: {record['article_id']} {record['article_title']}\n"
                        f"领域: {record['legal_domain']}\n"
                        f"内容: {record['content'][:360]}"
                    )
                    results.append(
                        RetrievalResult(
                            content=text,
                            node_id=record["node_id"],
                            node_type="Article",
                            relevance_score=0.78,
                            retrieval_level="topic",
                            metadata={
                                "law_name": record["law_name"],
                                "article_id": record["article_id"],
                                "article_title": record["article_title"],
                                "legal_domain": record["legal_domain"],
                                "source": "neo4j_topic",
                            },
                        )
                    )
        except Exception as e:
            logger.error("Neo4j主题检索失败: %s", e)
        return results

    def dual_level_retrieval(self, query: str, top_k: int = 5) -> List[Document]:
        query = sanitize_query_text(query)
        if not query:
            return []

        entity_keywords, topic_keywords = self.extract_query_keywords(query)
        entity_results = self.entity_level_retrieval(entity_keywords, top_k)
        topic_results = self.topic_level_retrieval(topic_keywords, top_k)

        all_results = self._dedup_results(entity_results + topic_results)
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)

        documents: List[Document] = []
        for result in all_results[:top_k]:
            documents.append(
                Document(
                    page_content=result.content,
                    metadata={
                        "node_id": result.node_id,
                        "node_type": result.node_type,
                        "retrieval_level": result.retrieval_level,
                        "relevance_score": result.relevance_score,
                        "search_type": "dual_level",
                        **result.metadata,
                    },
                )
            )
        return documents

    def vector_search_enhanced(self, query: str, top_k: int = 5) -> List[Document]:
        query = sanitize_query_text(query)
        if not query:
            return []

        try:
            vector_docs = self.milvus_module.similarity_search(query, k=top_k * 2)
            enhanced_docs: List[Document] = []
            for result in vector_docs:
                content = result.get("text", "")
                metadata = result.get("metadata", {})
                node_id = metadata.get("node_id", "")
                if node_id:
                    neighbors = self._get_node_neighbors(node_id, max_neighbors=2)
                    if neighbors:
                        content += f"\n关联节点: {', '.join(neighbors)}"

                law_name = metadata.get("law_name") or metadata.get("recipe_name", "")
                doc = Document(
                    page_content=content,
                    metadata={
                        **metadata,
                        "law_name": law_name,
                        "article_id": metadata.get("article_id", ""),
                        "article_title": metadata.get("article_title", ""),
                        "search_type": "vector_enhanced",
                        "score": result.get("score", 0.0),
                    },
                )
                enhanced_docs.append(doc)
            return enhanced_docs[:top_k]
        except Exception as e:
            logger.error("向量增强检索失败: %s", e)
            return []

    def _get_node_neighbors(self, node_id: str, max_neighbors: int = 3) -> List[str]:
        relation_types = getattr(self.config, "graph_relation_types", [])
        relation_filter = ""
        params: Dict[str, Any] = {"node_id": str(node_id), "limit": max_neighbors}
        if relation_types:
            relation_filter = "AND type(r) IN $relation_types"
            params["relation_types"] = relation_types

        query = f"""
        MATCH (n)-[r]-(neighbor)
        WHERE COALESCE(toString(n.nodeId), elementId(n)) = $node_id
          {relation_filter}
        RETURN COALESCE(neighbor.name, neighbor.title, neighbor.articleId, '') AS name
        LIMIT $limit
        """
        try:
            with self.driver.session(database=self.config.neo4j_database) as session:
                return [record["name"] for record in session.run(query, params) if record["name"]]
        except Exception as e:
            logger.error("获取邻居节点失败: %s", e)
            return []

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Document]:
        """Round-robin 混合实体/主题/向量结果。"""
        query = sanitize_query_text(query)
        if not query:
            return []

        dual_docs = self.dual_level_retrieval(query, top_k)
        vector_docs = self.vector_search_enhanced(query, top_k)

        merged_docs: List[Document] = []
        seen_doc_ids = set()
        max_len = max(len(dual_docs), len(vector_docs))
        for i in range(max_len):
            if i < len(dual_docs):
                doc = dual_docs[i]
                doc_id = doc.metadata.get("node_id", hash(doc.page_content))
                if doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    doc.metadata["search_method"] = "dual_level"
                    doc.metadata["final_score"] = doc.metadata.get("relevance_score", 0.0)
                    merged_docs.append(doc)

            if i < len(vector_docs):
                doc = vector_docs[i]
                doc_id = doc.metadata.get("node_id", hash(doc.page_content))
                if doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    vector_score = doc.metadata.get("score", 0.0)
                    similarity_score = max(0.0, 1.0 - vector_score) if vector_score <= 1.0 else 0.0
                    doc.metadata["search_method"] = "vector_enhanced"
                    doc.metadata["final_score"] = similarity_score
                    merged_docs.append(doc)

        return merged_docs[:top_k]

    def _dedup_results(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        deduped: List[RetrievalResult] = []
        seen = set()
        for result in results:
            key = (result.node_id, result.node_type)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(result)
        return deduped

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭")
