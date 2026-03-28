# =============================================================
# 文件介绍：图 RAG 检索模块（GraphRAGRetrieval）
# 目标：法律场景多跳检索、关系链解释与路径推理。
# =============================================================
"""
法律图RAG检索模块
"""

import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document
from neo4j import GraphDatabase
from .text_safety import sanitize_query_text

logger = logging.getLogger(__name__)


class QueryType(Enum):
    ENTITY_RELATION = "entity_relation"
    MULTI_HOP = "multi_hop"
    SUBGRAPH = "subgraph"
    PATH_FINDING = "path_finding"
    CLUSTERING = "clustering"


@dataclass
class GraphQuery:
    query_type: QueryType
    source_entities: List[str]
    target_entities: List[str]
    relation_types: List[str]
    max_depth: int = 2
    max_nodes: int = 50
    constraints: Dict[str, Any] = None


@dataclass
class GraphPath:
    nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    path_length: int
    relevance_score: float
    path_type: str


@dataclass
class KnowledgeSubgraph:
    central_nodes: List[Dict[str, Any]]
    connected_nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    graph_metrics: Dict[str, float]
    reasoning_chains: List[str]


class GraphRAGRetrieval:
    """法律图RAG检索系统。"""

    def __init__(self, config, llm_client, llm_dispatcher: Optional[object] = None):
        self.config = config
        self.llm_client = llm_client
        self.llm_dispatcher = llm_dispatcher
        self.driver = None
        self.entity_cache: Dict[str, Dict[str, Any]] = {}
        self.relation_cache: Dict[str, int] = {}

    def _assist_chat_completion(self, messages: List[Dict[str, str]], max_tokens: int = 800):
        if self.llm_dispatcher is not None:
            response, provider, model = self.llm_dispatcher.create_chat_completion(
                role="assist",
                messages=messages,
                temperature=0.1,
                max_tokens=max_tokens,
            )
            logger.info("辅助调用通道(图理解): provider=%s model=%s", provider, model)
            return response
        return self.llm_client.chat.completions.create(
            model=self.config.llm_model,
            messages=messages,
            temperature=0.1,
            max_tokens=max_tokens,
        )

    def initialize(self):
        logger.info("初始化法律图RAG检索系统...")
        try:
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password),
            )
            with self.driver.session(database=self.config.neo4j_database) as session:
                session.run("RETURN 1")
            self._build_graph_index()
            logger.info("法律图RAG检索系统初始化完成")
        except Exception as e:
            logger.error("初始化图RAG检索系统失败: %s", e)

    def _build_graph_index(self):
        try:
            with self.driver.session(database=self.config.neo4j_database) as session:
                entity_query = """
                MATCH (n)
                WHERE n:LawDocument OR n:Article OR n:RiskScenario OR n:ComplianceStep
                WITH n, COUNT { (n)--() } AS degree
                RETURN
                    COALESCE(toString(n.nodeId), elementId(n)) AS node_id,
                    labels(n) AS labels,
                    COALESCE(n.name, n.title, n.articleId, '') AS name,
                    COALESCE(n.category, '') AS category,
                    degree
                ORDER BY degree DESC
                LIMIT 2000
                """
                for record in session.run(entity_query):
                    self.entity_cache[record["node_id"]] = {
                        "labels": record["labels"],
                        "name": record["name"],
                        "category": record["category"],
                        "degree": record["degree"],
                    }

                relation_query = """
                MATCH ()-[r]->()
                RETURN type(r) AS rel_type, count(r) AS frequency
                ORDER BY frequency DESC
                """
                for record in session.run(relation_query):
                    self.relation_cache[record["rel_type"]] = record["frequency"]
        except Exception as e:
            logger.error("构建图索引失败: %s", e)

    def understand_graph_query(self, query: str) -> GraphQuery:
        query = sanitize_query_text(query)
        if not query:
            return self._rule_based_query("")

        prompt = f"""
        你是法律图谱查询规划器。请将问题映射到图查询参数。

        可用实体: LawDocument, Article, LegalDomain, RiskScenario, ComplianceStep
        可用关系: HAS_ARTICLE, BELONGS_TO_DOMAIN, CITES, RELATES_TO, APPLIES_TO, REQUIRES_STEP, PRECEDES

        问题: {query}

        返回JSON:
        {{
          "query_type": "multi_hop|entity_relation|subgraph|path_finding|clustering",
          "source_entities": ["实体1"],
          "target_entities": ["实体2"],
          "relation_types": ["CITES", "RELATES_TO"],
          "max_depth": 2,
          "constraints": {{}}
        }}
        """
        try:
            response = self._assist_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
            )
            data = self._safe_json_loads(response.choices[0].message.content.strip())
            return GraphQuery(
                query_type=QueryType(data.get("query_type", "multi_hop")),
                source_entities=data.get("source_entities", []),
                target_entities=data.get("target_entities", []),
                relation_types=data.get("relation_types", []) or getattr(self.config, "graph_relation_types", []),
                max_depth=max(1, min(int(data.get("max_depth", 2)), 4)),
                max_nodes=50,
                constraints=data.get("constraints", {}),
            )
        except Exception as e:
            logger.error("查询意图理解失败，使用规则降级: %s", e)
            return self._rule_based_query(query)

    def _rule_based_query(self, query: str) -> GraphQuery:
        query = sanitize_query_text(query)
        relation_types = getattr(self.config, "graph_relation_types", [])
        if any(x in query for x in ["路径", "链路", "从", "到"]):
            qtype = QueryType.PATH_FINDING
        elif any(x in query for x in ["关联", "引用", "关系"]):
            qtype = QueryType.ENTITY_RELATION
        elif any(x in query for x in ["涉及", "依据", "责任", "如何"]):
            qtype = QueryType.MULTI_HOP
        else:
            qtype = QueryType.SUBGRAPH
        return GraphQuery(
            query_type=qtype,
            source_entities=[query[:24]],
            target_entities=[],
            relation_types=relation_types,
            max_depth=min(getattr(self.config, "max_graph_depth", 2), 4),
            constraints={},
        )

    def _safe_json_loads(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise

    def multi_hop_traversal(self, graph_query: GraphQuery) -> List[GraphPath]:
        logger.info("执行多跳遍历: %s", graph_query.source_entities)
        paths: List[GraphPath] = []
        if not self.driver:
            return paths

        if graph_query.query_type == QueryType.ENTITY_RELATION:
            return self._find_entity_relations(graph_query)
        if graph_query.query_type == QueryType.PATH_FINDING:
            return self._find_shortest_paths(graph_query)

        max_depth = max(1, min(int(graph_query.max_depth), 4))
        relation_filter = ""
        params: Dict[str, Any] = {
            "source_entities": graph_query.source_entities,
        }
        if graph_query.relation_types:
            relation_filter = "AND ALL(rel IN relationships(path) WHERE type(rel) IN $relation_types)"
            params["relation_types"] = graph_query.relation_types

        target_filter = ""
        if graph_query.target_entities:
            target_filter = """
            AND ANY(kw IN $target_entities WHERE
                COALESCE(target.name, target.title, target.articleId, '') CONTAINS kw
            )
            """
            params["target_entities"] = graph_query.target_entities

        query = f"""
        UNWIND $source_entities AS source_name
        MATCH (source)
        WHERE COALESCE(source.name, source.title, source.articleId, '') CONTAINS source_name
        MATCH path = (source)-[*1..{max_depth}]-(target)
        WHERE source <> target
          {relation_filter}
          {target_filter}
        WITH path, length(path) AS path_len
        RETURN path, path_len, relationships(path) AS rels, nodes(path) AS path_nodes,
               (1.0 / toFloat(path_len)) AS relevance
        ORDER BY relevance DESC
        LIMIT 25
        """

        try:
            with self.driver.session(database=self.config.neo4j_database) as session:
                for record in session.run(query, params):
                    parsed = self._parse_neo4j_path(record, path_type="multi_hop")
                    if parsed:
                        paths.append(parsed)
        except Exception as e:
            logger.error("多跳遍历失败: %s", e)
        return paths

    def _find_entity_relations(self, graph_query: GraphQuery) -> List[GraphPath]:
        """补齐实体关系查询，禁止空占位实现。"""
        paths: List[GraphPath] = []
        relation_filter = ""
        params: Dict[str, Any] = {
            "source_entities": graph_query.source_entities,
        }
        if graph_query.relation_types:
            relation_filter = "AND type(r) IN $relation_types"
            params["relation_types"] = graph_query.relation_types

        query = f"""
        UNWIND $source_entities AS source_name
        MATCH (source)-[r]-(target)
        WHERE COALESCE(source.name, source.title, source.articleId, '') CONTAINS source_name
          AND source <> target
          {relation_filter}
        RETURN
          [source, target] AS path_nodes,
          [r] AS rels,
          1 AS path_len,
          0.95 AS relevance
        LIMIT 30
        """
        try:
            with self.driver.session(database=self.config.neo4j_database) as session:
                for record in session.run(query, params):
                    parsed = self._parse_neo4j_path(record, path_type="entity_relation")
                    if parsed:
                        paths.append(parsed)
        except Exception as e:
            logger.error("实体关系查询失败: %s", e)
        return paths

    def _find_shortest_paths(self, graph_query: GraphQuery) -> List[GraphPath]:
        """补齐最短路径查询，禁止空占位实现。"""
        paths: List[GraphPath] = []
        if not graph_query.target_entities:
            return self._find_entity_relations(graph_query)

        max_depth = max(2, min(int(graph_query.max_depth), 4))
        params = {
            "sources": graph_query.source_entities,
            "targets": graph_query.target_entities,
            "max_depth": max_depth,
        }
        relation_filter = ""
        relation_types = graph_query.relation_types or getattr(self.config, "graph_relation_types", [])
        if relation_types:
            params["relation_types"] = relation_types
            relation_filter = "AND ALL(rel IN relationships(path) WHERE type(rel) IN $relation_types)"

        query = f"""
        UNWIND $sources AS source_kw
        UNWIND $targets AS target_kw
        MATCH (source), (target)
        WHERE COALESCE(source.name, source.title, source.articleId, '') CONTAINS source_kw
          AND COALESCE(target.name, target.title, target.articleId, '') CONTAINS target_kw
          AND source <> target
        MATCH path = shortestPath((source)-[*..{max_depth}]-(target))
        WHERE length(path) <= $max_depth
          {relation_filter}
        RETURN path, length(path) AS path_len, relationships(path) AS rels, nodes(path) AS path_nodes,
               (1.0 / toFloat(length(path))) AS relevance
        LIMIT 20
        """
        try:
            with self.driver.session(database=self.config.neo4j_database) as session:
                for record in session.run(query, params):
                    parsed = self._parse_neo4j_path(record, path_type="path_finding")
                    if parsed:
                        paths.append(parsed)
        except Exception as e:
            logger.error("最短路径查询失败: %s", e)
        return paths

    def extract_knowledge_subgraph(self, graph_query: GraphQuery) -> KnowledgeSubgraph:
        if not self.driver:
            return self._fallback_subgraph_extraction()
        query = f"""
        UNWIND $source_entities AS source_kw
        MATCH (source)
        WHERE COALESCE(source.name, source.title, source.articleId, '') CONTAINS source_kw
        MATCH (source)-[r*1..{graph_query.max_depth}]-(neighbor)
        WITH source, collect(DISTINCT neighbor) AS neighbors, collect(DISTINCT r) AS rels
        RETURN
          source,
          neighbors[0..$max_nodes] AS nodes,
          rels[0..$max_nodes] AS rel_groups
        LIMIT 1
        """
        try:
            with self.driver.session(database=self.config.neo4j_database) as session:
                record = session.run(
                    query,
                    {
                        "source_entities": graph_query.source_entities,
                        "max_nodes": graph_query.max_nodes,
                    },
                ).single()
            if not record:
                return self._fallback_subgraph_extraction()

            central = [self._serialize_node(record["source"])]
            connected = [self._serialize_node(n) for n in record["nodes"]]

            rels: List[Dict[str, Any]] = []
            for group in record["rel_groups"]:
                for rel in group:
                    rels.append(
                        {
                            "type": self._relationship_type(rel),
                            "properties": dict(rel),
                        }
                    )
            node_count = len(connected)
            rel_count = len(rels)
            density = (
                float(rel_count) / (node_count * (node_count - 1) / 2)
                if node_count > 1
                else 0.0
            )
            return KnowledgeSubgraph(
                central_nodes=central,
                connected_nodes=connected,
                relationships=rels,
                graph_metrics={
                    "node_count": node_count,
                    "relationship_count": rel_count,
                    "density": density,
                },
                reasoning_chains=[],
            )
        except Exception as e:
            logger.error("子图提取失败: %s", e)
            return self._fallback_subgraph_extraction()

    def graph_structure_reasoning(self, subgraph: KnowledgeSubgraph, query: str) -> List[str]:
        if not subgraph.connected_nodes:
            return []
        return [
            f"围绕问题“{query}”共关联 {len(subgraph.connected_nodes)} 个节点。",
            f"关系边数量为 {len(subgraph.relationships)}，图密度 {subgraph.graph_metrics.get('density', 0.0):.3f}。",
        ]

    def graph_rag_search(self, query: str, top_k: int = 5) -> List[Document]:
        query = sanitize_query_text(query)
        if not query:
            return []

        if not self.driver:
            logger.warning("Neo4j连接未建立，返回空结果")
            return []
        graph_query = self.understand_graph_query(query)
        results: List[Document] = []

        try:
            if graph_query.query_type in {QueryType.MULTI_HOP, QueryType.ENTITY_RELATION, QueryType.PATH_FINDING}:
                paths = self.multi_hop_traversal(graph_query)
                results.extend(self._paths_to_documents(paths))
            else:
                subgraph = self.extract_knowledge_subgraph(graph_query)
                reasoning_chains = self.graph_structure_reasoning(subgraph, query)
                results.extend(self._subgraph_to_documents(subgraph, reasoning_chains))
            results = sorted(
                results,
                key=lambda x: x.metadata.get("relevance_score", 0.0),
                reverse=True,
            )
            return results[:top_k]
        except Exception as e:
            logger.error("图RAG检索失败: %s", e)
            return []

    def _parse_neo4j_path(self, record, path_type: str) -> Optional[GraphPath]:
        try:
            nodes = [self._serialize_node(node) for node in record["path_nodes"]]
            relationships = [
                {"type": self._relationship_type(rel), "properties": dict(rel)}
                for rel in record["rels"]
            ]
            return GraphPath(
                nodes=nodes,
                relationships=relationships,
                path_length=int(record["path_len"]),
                relevance_score=float(record["relevance"]),
                path_type=path_type,
            )
        except Exception as e:
            logger.error("路径解析失败: %s", e)
            return None

    def _serialize_node(self, node) -> Dict[str, Any]:
        return {
            "id": node.get("nodeId", ""),
            "name": node.get("name", node.get("title", node.get("articleId", ""))),
            "labels": list(node.labels),
            "properties": dict(node),
        }

    def _relationship_type(self, rel) -> str:
        rel_type = getattr(rel, "type", None) or getattr(rel, "_type", None)
        if isinstance(rel_type, str) and rel_type:
            return rel_type
        return "RELATED"

    def _paths_to_documents(self, paths: List[GraphPath]) -> List[Document]:
        docs: List[Document] = []
        for path in paths:
            path_desc = self._build_path_description(path)
            path_relations = [r["type"] for r in path.relationships]
            reasoning_path = " -> ".join([node.get("name", "") for node in path.nodes])
            docs.append(
                Document(
                    page_content=path_desc,
                    metadata={
                        "search_type": "graph_path",
                        "relevance_score": path.relevance_score,
                        "path_type": path.path_type,
                        "path_length": path.path_length,
                        "path_depth": path.path_length,
                        "path_relations": path_relations,
                        "reasoning_path": reasoning_path,
                        "law_name": path.nodes[0].get("name", "") if path.nodes else "",
                        "article_id": path.nodes[0].get("properties", {}).get("articleId", "") if path.nodes else "",
                    },
                )
            )
        return docs

    def _subgraph_to_documents(
        self, subgraph: KnowledgeSubgraph, reasoning_chains: List[str]
    ) -> List[Document]:
        center_name = subgraph.central_nodes[0].get("name", "未知中心实体") if subgraph.central_nodes else "未知中心实体"
        has_evidence = bool(subgraph.central_nodes and subgraph.connected_nodes and subgraph.relationships)
        relevance = 0.8 if has_evidence else 0.15
        content = (
            f"中心实体: {center_name}\n"
            f"节点数: {len(subgraph.connected_nodes)}\n"
            f"关系数: {len(subgraph.relationships)}\n"
            f"推理说明: {' | '.join(reasoning_chains)}"
        )
        if not has_evidence:
            content += "\n证据状态: 当前子图证据不足，建议回退传统检索或扩展检索范围。"
        return [
            Document(
                page_content=content,
                metadata={
                    "search_type": "knowledge_subgraph",
                    "relevance_score": relevance,
                    "reasoning_path": center_name,
                    "path_relations": [r.get("type", "") for r in subgraph.relationships[:8]],
                    "law_name": center_name,
                    "evidence_insufficient": not has_evidence,
                },
            )
        ]

    def _build_path_description(self, path: GraphPath) -> str:
        if not path.nodes:
            return "空路径"
        parts: List[str] = []
        for i, node in enumerate(path.nodes):
            parts.append(node.get("name", f"节点{i}"))
            if i < len(path.relationships):
                parts.append(f" --{path.relationships[i].get('type', 'RELATED')}--> ")
        return "".join(parts)

    def _fallback_subgraph_extraction(self) -> KnowledgeSubgraph:
        return KnowledgeSubgraph(
            central_nodes=[],
            connected_nodes=[],
            relationships=[],
            graph_metrics={},
            reasoning_chains=[],
        )

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("图RAG检索系统已关闭")
