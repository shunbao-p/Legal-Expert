# =============================================================
# 文件介绍：图 RAG 检索模块（GraphRAGRetrieval）
# 目标：法律场景多跳检索、关系链解释与路径推理。
# =============================================================
"""
法律图RAG检索模块
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
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
    source_node_ids: List[str] = field(default_factory=list)
    target_node_ids: List[str] = field(default_factory=list)
    max_depth: int = 2
    max_nodes: int = 50
    constraints: Dict[str, Any] = field(default_factory=dict)
    grounding_meta: Dict[str, Any] = field(default_factory=dict)


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

    GENERIC_LABEL_TOKENS = {
        "lawdocument",
        "article",
        "legaldomain",
        "riskscenario",
        "compliancestep",
        "law",
        "domain",
        "risk",
        "label",
        "法律领域",
        "法规",
        "法条",
        "条文",
        "领域",
        "风险场景",
        "合规步骤",
        "法律概念",
        "实体",
    }

    ENTITY_ALIAS_MAP = {
        "孩童": "未成年人",
        "未成年": "未成年人",
        "欠薪": "拖欠工资",
        "拖薪": "拖欠工资",
        "恶意伤人": "故意伤害",
    }

    def __init__(self, config, llm_client, llm_dispatcher: Optional[object] = None):
        self.config = config
        self.llm_client = llm_client
        self.llm_dispatcher = llm_dispatcher
        self.driver = None
        self.entity_cache: Dict[str, Dict[str, Any]] = {}
        self.relation_cache: Dict[str, int] = {}
        self.last_empty_reason = ""
        self.last_grounding_stats: Dict[str, Any] = {}

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
                source_node_ids=[],
                target_node_ids=[],
                max_depth=max(1, min(int(data.get("max_depth", 2)), 4)),
                max_nodes=50,
                constraints=data.get("constraints", {}),
                grounding_meta={},
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
            source_node_ids=[],
            target_node_ids=[],
            max_depth=min(getattr(self.config, "max_graph_depth", 2), 4),
            constraints={},
            grounding_meta={},
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

    def _normalize_entity_term(self, term: str) -> str:
        term = sanitize_query_text(term)
        term = term.strip().strip("“”\"'[]()（）")
        if term.startswith("《") and term.endswith("》") and len(term) > 2:
            term = term[1:-1].strip()
        return term

    def _is_generic_label_token(self, token: str) -> bool:
        normalized = self._normalize_entity_term(token)
        if not normalized:
            return True
        compact = normalized.replace("_", "").replace("-", "").replace(" ", "").lower()
        return compact in self.GENERIC_LABEL_TOKENS

    def _fallback_extract_entity_terms(self, query: str) -> List[str]:
        query = sanitize_query_text(query)
        if not query:
            return []

        terms: List[str] = []
        law_refs = re.findall(r"《([^》]{2,40})》", query)
        for law in law_refs:
            terms.append(law.strip())

        article_refs = re.findall(r"第[0-9一二三四五六七八九十百千万〇零两]+条", query)
        terms.extend(article_refs)

        for alias, canonical in self.ENTITY_ALIAS_MAP.items():
            if alias in query:
                terms.append(canonical)

        if not terms:
            terms.append(query[:24])
        return [x for x in terms if x]

    def _lookup_grounded_nodes(self, terms: List[str], per_term_limit: int = 3) -> List[Dict[str, Any]]:
        if not self.driver or not terms:
            return []

        merged_hits: Dict[str, Dict[str, Any]] = {}
        index_name = getattr(self.config, "neo4j_legal_fulltext_index", "legal_fulltext_idx")

        exact_query = """
        UNWIND $terms AS term
        MATCH (n)
        WHERE (n:LawDocument OR n:Article OR n:LegalDomain OR n:RiskScenario OR n:ComplianceStep)
          AND (
                COALESCE(n.name, n.title, n.articleId, '') = term
             OR COALESCE(n.articleId, '') = term
          )
        RETURN term,
               COALESCE(toString(n.nodeId), elementId(n)) AS node_id,
               COALESCE(n.name, n.title, n.articleId, '') AS name,
               2.0 AS score
        LIMIT 300
        """

        fulltext_query = """
        UNWIND $terms AS term
        CALL db.index.fulltext.queryNodes($index_name, term) YIELD node, score
        RETURN term,
               COALESCE(toString(node.nodeId), elementId(node)) AS node_id,
               COALESCE(node.name, node.title, node.articleId, '') AS name,
               score
        LIMIT 800
        """

        try:
            with self.driver.session(database=self.config.neo4j_database) as session:
                for record in session.run(exact_query, {"terms": terms}):
                    key = f"{record['term']}::{record['node_id']}"
                    merged_hits[key] = {
                        "term": record["term"],
                        "node_id": record["node_id"],
                        "name": record["name"],
                        "score": float(record["score"]),
                    }
        except Exception as e:
            logger.warning("实体精确匹配失败: %s", e)

        try:
            with self.driver.session(database=self.config.neo4j_database) as session:
                for record in session.run(
                    fulltext_query,
                    {"terms": terms, "index_name": index_name},
                ):
                    score = float(record["score"])
                    if score <= 0:
                        continue
                    key = f"{record['term']}::{record['node_id']}"
                    existing = merged_hits.get(key)
                    if existing is None or score > existing["score"]:
                        merged_hits[key] = {
                            "term": record["term"],
                            "node_id": record["node_id"],
                            "name": record["name"],
                            "score": score,
                        }
        except Exception as e:
            logger.warning("实体fulltext匹配失败（可能索引不存在）: %s", e)

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in merged_hits.values():
            grouped.setdefault(item["term"], []).append(item)

        final_hits: List[Dict[str, Any]] = []
        for term, term_hits in grouped.items():
            ranked = sorted(term_hits, key=lambda x: x["score"], reverse=True)[:per_term_limit]
            for hit in ranked:
                final_hits.append(hit)
        return final_hits

    def _ground_entities(
        self,
        query: str,
        raw_entities: List[str],
        allow_query_fallback: bool = True,
    ) -> Dict[str, Any]:
        terms: List[str] = []
        for term in raw_entities or []:
            normalized = self._normalize_entity_term(term)
            if not normalized:
                continue
            if self._is_generic_label_token(normalized):
                continue
            terms.append(normalized)
            alias = self.ENTITY_ALIAS_MAP.get(normalized)
            if alias:
                terms.append(alias)

        if not terms and allow_query_fallback:
            terms = self._fallback_extract_entity_terms(query)

        dedup_terms: List[str] = []
        seen_terms = set()
        for term in terms:
            normalized = self._normalize_entity_term(term)
            if not normalized:
                continue
            if normalized in seen_terms:
                continue
            seen_terms.add(normalized)
            dedup_terms.append(normalized)

        hits = self._lookup_grounded_nodes(dedup_terms, per_term_limit=3)
        source_node_ids: List[str] = []
        source_entities: List[str] = []
        seen_ids = set()
        seen_names = set()
        for hit in sorted(hits, key=lambda x: x["score"], reverse=True):
            node_id = hit["node_id"]
            name = hit["name"]
            if node_id and node_id not in seen_ids:
                seen_ids.add(node_id)
                source_node_ids.append(node_id)
            if name and name not in seen_names:
                seen_names.add(name)
                source_entities.append(name)

        return {
            "candidate_terms": dedup_terms,
            "hits": hits,
            "source_node_ids": source_node_ids,
            "source_entities": source_entities,
        }

    def _apply_entity_grounding(self, graph_query: GraphQuery, query: str) -> GraphQuery:
        source_ground = self._ground_entities(query, graph_query.source_entities, allow_query_fallback=True)
        if graph_query.target_entities:
            target_ground = self._ground_entities(
                query,
                graph_query.target_entities,
                allow_query_fallback=(graph_query.query_type == QueryType.PATH_FINDING),
            )
        else:
            target_ground = {
                "candidate_terms": [],
                "hits": [],
                "source_node_ids": [],
                "source_entities": [],
            }

        graph_query.source_node_ids = source_ground["source_node_ids"]
        graph_query.target_node_ids = target_ground["source_node_ids"]
        # 收紧闭环：仅允许“已落地命中”的实体名进入后续图检索，
        # 禁止将未命中候选词直接回填到 source/target 查询条件。
        graph_query.source_entities = source_ground["source_entities"]
        graph_query.target_entities = target_ground["source_entities"]
        graph_query.grounding_meta = {
            "source_candidates": source_ground["candidate_terms"],
            "source_hit_count": len(source_ground["source_node_ids"]),
            "target_candidates": target_ground["candidate_terms"],
            "target_hit_count": len(target_ground["source_node_ids"]),
        }
        return graph_query

    def multi_hop_traversal(self, graph_query: GraphQuery) -> List[GraphPath]:
        logger.info(
            "执行多跳遍历: source_entities=%s source_node_ids=%s",
            graph_query.source_entities,
            graph_query.source_node_ids,
        )
        paths: List[GraphPath] = []
        if not self.driver:
            return paths

        if graph_query.query_type == QueryType.ENTITY_RELATION:
            return self._find_entity_relations(graph_query)
        if graph_query.query_type == QueryType.PATH_FINDING:
            return self._find_shortest_paths(graph_query)

        max_depth = max(1, min(int(graph_query.max_depth), 4))
        relation_filter = ""
        params: Dict[str, Any] = {}
        if graph_query.relation_types:
            relation_filter = "AND ALL(rel IN relationships(path) WHERE type(rel) IN $relation_types)"
            params["relation_types"] = graph_query.relation_types

        source_match = ""
        if graph_query.source_node_ids:
            source_match = """
            UNWIND $source_node_ids AS source_id
            MATCH (source)
            WHERE COALESCE(toString(source.nodeId), elementId(source)) = source_id
            """
            params["source_node_ids"] = graph_query.source_node_ids
        else:
            source_match = """
            UNWIND $source_entities AS source_name
            MATCH (source)
            WHERE COALESCE(source.name, source.title, source.articleId, '') CONTAINS source_name
            """
            params["source_entities"] = graph_query.source_entities

        target_filter = ""
        if graph_query.target_node_ids:
            target_filter = """
            AND COALESCE(toString(target.nodeId), elementId(target)) IN $target_node_ids
            """
            params["target_node_ids"] = graph_query.target_node_ids
        elif graph_query.target_entities:
            target_filter = """
            AND ANY(kw IN $target_entities WHERE
                COALESCE(target.name, target.title, target.articleId, '') CONTAINS kw
            )
            """
            params["target_entities"] = graph_query.target_entities

        query = f"""
        {source_match}
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
        params: Dict[str, Any] = {}
        if graph_query.relation_types:
            relation_filter = "AND type(r) IN $relation_types"
            params["relation_types"] = graph_query.relation_types

        source_match = ""
        if graph_query.source_node_ids:
            source_match = """
            UNWIND $source_node_ids AS source_id
            MATCH (source)
            WHERE COALESCE(toString(source.nodeId), elementId(source)) = source_id
            """
            params["source_node_ids"] = graph_query.source_node_ids
        else:
            source_match = """
            UNWIND $source_entities AS source_name
            MATCH (source)
            WHERE COALESCE(source.name, source.title, source.articleId, '') CONTAINS source_name
            """
            params["source_entities"] = graph_query.source_entities

        query = f"""
        {source_match}
        MATCH (source)-[r]-(target)
        WHERE 1=1
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
        if not graph_query.target_entities and not graph_query.target_node_ids:
            logger.warning("PATH_FINDING 缺少目标实体，跳过最短路径查询")
            return []

        max_depth = max(2, min(int(graph_query.max_depth), 4))
        params = {"max_depth": max_depth}
        relation_filter = ""
        relation_types = graph_query.relation_types or getattr(self.config, "graph_relation_types", [])
        if relation_types:
            params["relation_types"] = relation_types
            relation_filter = "AND ALL(rel IN relationships(path) WHERE type(rel) IN $relation_types)"

        if graph_query.source_node_ids and graph_query.target_node_ids:
            params["source_node_ids"] = graph_query.source_node_ids
            params["target_node_ids"] = graph_query.target_node_ids
            query = f"""
            UNWIND $source_node_ids AS source_id
            UNWIND $target_node_ids AS target_id
            MATCH (source), (target)
            WHERE COALESCE(toString(source.nodeId), elementId(source)) = source_id
              AND COALESCE(toString(target.nodeId), elementId(target)) = target_id
              AND source <> target
            MATCH path = shortestPath((source)-[*..{max_depth}]-(target))
            WHERE length(path) <= $max_depth
              {relation_filter}
            RETURN path, length(path) AS path_len, relationships(path) AS rels, nodes(path) AS path_nodes,
                   (1.0 / toFloat(length(path))) AS relevance
            LIMIT 20
            """
        else:
            params["sources"] = graph_query.source_entities
            params["targets"] = graph_query.target_entities
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
        source_match = ""
        params: Dict[str, Any] = {"max_nodes": graph_query.max_nodes}
        if graph_query.source_node_ids:
            source_match = """
            UNWIND $source_node_ids AS source_id
            MATCH (source)
            WHERE COALESCE(toString(source.nodeId), elementId(source)) = source_id
            """
            params["source_node_ids"] = graph_query.source_node_ids
        else:
            source_match = """
            UNWIND $source_entities AS source_kw
            MATCH (source)
            WHERE COALESCE(source.name, source.title, source.articleId, '') CONTAINS source_kw
            """
            params["source_entities"] = graph_query.source_entities

        query = f"""
        {source_match}
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
                    params,
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
        self.last_empty_reason = ""
        self.last_grounding_stats = {}
        if not query:
            self.last_empty_reason = "empty_query"
            return []

        if not self.driver:
            logger.warning("Neo4j连接未建立，返回空结果")
            self.last_empty_reason = "no_driver"
            return []
        graph_query = self.understand_graph_query(query)
        graph_query = self._apply_entity_grounding(graph_query, query)
        self.last_grounding_stats = graph_query.grounding_meta

        if not graph_query.source_entities and not graph_query.source_node_ids:
            self.last_empty_reason = "no_grounded_nodes"
            logger.warning("GraphRAG实体落地失败，跳过图检索: %s", graph_query.grounding_meta)
            return []
        if graph_query.query_type == QueryType.PATH_FINDING and not (
            graph_query.target_entities or graph_query.target_node_ids
        ):
            self.last_empty_reason = "path_target_not_grounded"
            logger.warning("PATH_FINDING 目标实体未落地，回退传统检索: %s", graph_query.grounding_meta)
            return []

        results: List[Document] = []

        try:
            if graph_query.query_type in {QueryType.MULTI_HOP, QueryType.ENTITY_RELATION, QueryType.PATH_FINDING}:
                paths = self.multi_hop_traversal(graph_query)
                results.extend(self._paths_to_documents(paths))
            else:
                subgraph = self.extract_knowledge_subgraph(graph_query)
                reasoning_chains = self.graph_structure_reasoning(subgraph, query)
                results.extend(self._subgraph_to_documents(subgraph, reasoning_chains))
            if not results:
                self.last_empty_reason = "no_paths_found"
            results = sorted(
                results,
                key=lambda x: x.metadata.get("relevance_score", 0.0),
                reverse=True,
            )
            return results[:top_k]
        except Exception as e:
            logger.error("图RAG检索失败: %s", e)
            self.last_empty_reason = "graph_query_exception"
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
