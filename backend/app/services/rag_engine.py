"""
Graph RAG Engine - Qdrant Retrieval & Graph Context Expansion (Phase 2, Steps 6.1 & 6.2).

Responsible for:
1. Step 6.1 — Semantic Vector Retrieval:
   - Accepts user question.
   - Generates vector embedding using configured provider (Gemini, FastEmbed, or local fallback).
   - Searches Qdrant for top-k most relevant regulatory obligations.
   - Returns obligation IDs, similarity scores, and metadata.

2. Step 6.2 — Graph Context Expansion:
   - Takes obligation node IDs returned by Qdrant.
   - Queries Neo4j using those IDs to expand the graph around each obligation.
   - Traverses and retrieves:
     * RegulatoryFramework (e.g. SOC 2, GDPR)
     * RegulatoryVersion (e.g. 2017, 2016)
     * ControlCategory (CATEGORIZED_AS)
     * EvidenceArtifacts (SATISFIES, with coverage, confidence, reasoning, evidence_text)
     * Related obligations: DEPENDS_ON (outgoing & incoming dependencies)
     * Related obligations: SUPERSEDES (outgoing & incoming superseding relationships)
   - Limits traversal depth to avoid retrieving an unnecessarily large graph.
   - Preserves node IDs, clause numbers, relationship types, evidence IDs, and provenance for citations.
   - Builds a structured GraphRAGContext object ready for downstream LLM synthesis (Step 6.3).
   - Handles missing nodes and empty graph results safely.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.config import settings
from app.integrations.qdrant_client import (
    QdrantClient,
    qdrant_client as global_qdrant_client,
)

logger = logging.getLogger(__name__)

# Default configurations
DEFAULT_TOP_K = 5
DEFAULT_TRAVERSAL_DEPTH = 1
DEFAULT_COLLECTION_NAME = getattr(settings, "QDRANT_COLLECTION", "regulatory_obligations")


class RAGRetrievalError(Exception):
    """Exception raised when retrieval in the RAG pipeline fails."""
    pass


class RAGGraphExpansionError(Exception):
    """Exception raised when graph context expansion in Neo4j fails."""
    pass


# -----------------------------------------------------------------------------
# Step 6.1: Vector Retrieval Models
# -----------------------------------------------------------------------------


class RetrievedObligation(BaseModel):
    """
    Structured model representing a regulatory obligation retrieved from vector search.

    Preserves the exact Neo4j node ID for downstream graph traversal and provides
    access to similarity scores, clause identifiers, and metadata.
    """
    obligation_id: str = Field(
        ...,
        description="Unique obligation ID (corresponds directly to the Neo4j node ID)",
    )
    node_id: str = Field(
        ...,
        description="Neo4j node ID (alias for obligation_id)",
    )
    score: float = Field(
        0.0,
        description="Cosine similarity score (0.0 to 1.0)",
    )
    framework: Optional[str] = Field(
        None,
        description="Regulatory framework name (e.g. 'GDPR', 'SOC 2', 'ISO 27001')",
    )
    version: Optional[str] = Field(
        None,
        description="Regulatory version slug (e.g. '2016', '2017')",
    )
    clause: Optional[str] = Field(
        None,
        description="Clause or article identifier (e.g. 'Article 5(1)(e)', 'CC6.1')",
    )
    category: Optional[str] = Field(
        None,
        description="Control or requirement category (e.g. 'Storage Limitation', 'Access Control')",
    )
    title: Optional[str] = Field(
        None,
        description="Obligation title or summary header",
    )
    text: Optional[str] = Field(
        None,
        description="Obligation requirement text or description",
    )
    mandatory: Optional[bool] = Field(
        True,
        description="Whether the requirement is mandatory",
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Associated keywords and domain concepts",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional payload metadata from the vector store",
    )

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def normalize_retrieved_data(cls, data: Any) -> Any:
        """
        Normalize input dictionary from Qdrant search results or raw payloads.
        Ensures both obligation_id and node_id are set and aliases are resolved.
        """
        if isinstance(data, dict):
            mapped = dict(data)
            payload = mapped.get("payload") or {}

            # 1. Resolve obligation_id and node_id
            ob_id = (
                mapped.get("obligation_id")
                or mapped.get("node_id")
                or mapped.get("id")
                or payload.get("obligation_id")
                or payload.get("id")
            )
            if ob_id is not None:
                id_str = str(ob_id)
                mapped["obligation_id"] = id_str
                if not mapped.get("node_id"):
                    mapped["node_id"] = id_str

            # 2. Extract score
            if "score" in mapped and mapped["score"] is not None:
                try:
                    mapped["score"] = float(mapped["score"])
                except (ValueError, TypeError):
                    mapped["score"] = 0.0

            # 3. Fallback fields from payload if top-level fields are missing
            for field_name in (
                "framework",
                "version",
                "clause",
                "category",
                "title",
                "text",
                "mandatory",
                "keywords",
            ):
                if mapped.get(field_name) is None and field_name in payload:
                    mapped[field_name] = payload[field_name]

            # 4. Populate metadata payload
            if not mapped.get("metadata") and payload:
                mapped["metadata"] = payload
            elif not mapped.get("metadata"):
                mapped["metadata"] = {
                    k: v for k, v in mapped.items()
                    if k not in ("obligation_id", "node_id", "score", "metadata")
                }

            return mapped
        return data

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "obligation_id": self.obligation_id,
            "node_id": self.node_id,
            "score": self.score,
            "framework": self.framework,
            "version": self.version,
            "clause": self.clause,
            "category": self.category,
            "title": self.title,
            "text": self.text,
            "mandatory": self.mandatory,
            "keywords": self.keywords,
            "metadata": self.metadata,
        }

    def __getitem__(self, item: str) -> Any:
        """Allow subscripting like a dict for backward compatibility."""
        return getattr(self, item)


# -----------------------------------------------------------------------------
# Step 6.2: Graph Context Expansion Models
# -----------------------------------------------------------------------------


class ObligationGraphNode(BaseModel):
    """Represents a RegulatoryObligation node from Neo4j."""
    id: str = Field(..., description="Unique node ID in Neo4j")
    code: Optional[str] = Field(None, description="Obligation code (e.g. 'CC6.1', 'Article 5(1)(e)')")
    clause: Optional[str] = Field(None, description="Clause identifier")
    title: Optional[str] = Field(None, description="Obligation title or summary")
    description: Optional[str] = Field(None, description="Requirement or control text")
    category: Optional[str] = Field(None, description="Control or requirement category")
    mandatory: Optional[bool] = Field(None, description="Whether requirement is mandatory")
    keywords: List[str] = Field(default_factory=list, description="Associated keywords")
    source_text: Optional[str] = Field(None, description="Original source text excerpt")
    properties: Dict[str, Any] = Field(default_factory=dict, description="All raw node properties")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def populate_obligation_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            mapped = dict(data)
            ob_id = mapped.get("id") or mapped.get("obligation_id") or mapped.get("node_id")
            if ob_id is not None:
                mapped["id"] = str(ob_id)
            if not mapped.get("clause") and mapped.get("code"):
                mapped["clause"] = mapped["code"]
            if not mapped.get("code") and mapped.get("clause"):
                mapped["code"] = mapped["clause"]
            if "properties" not in mapped:
                mapped["properties"] = dict(data)
            return mapped
        return data


class FrameworkGraphNode(BaseModel):
    """Represents a RegulatoryFramework node in Neo4j."""
    id: str = Field(..., description="Framework unique node ID")
    name: str = Field(..., description="Regulatory framework name (e.g. 'SOC 2', 'GDPR')")
    description: Optional[str] = Field(None, description="Framework description")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def normalize_id(cls, data: Any) -> Any:
        if isinstance(data, dict) and "id" in data and data["id"] is not None:
            mapped = dict(data)
            mapped["id"] = str(mapped["id"])
            return mapped
        return data


class VersionGraphNode(BaseModel):
    """Represents a RegulatoryVersion node in Neo4j."""
    id: str = Field(..., description="Version unique node ID")
    version_slug: str = Field(..., description="Version revision slug (e.g. '2017', '2016')")
    framework_id: Optional[str] = Field(None, description="Parent framework node ID")
    publication_date: Optional[str] = Field(None, description="Publication date string")
    is_active: bool = Field(True, description="Whether this version is currently active")
    description: Optional[str] = Field(None, description="Version description")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def normalize_version(cls, data: Any) -> Any:
        if isinstance(data, dict):
            mapped = dict(data)
            if "id" in mapped and mapped["id"] is not None:
                mapped["id"] = str(mapped["id"])
            if "framework_id" in mapped and mapped["framework_id"] is not None:
                mapped["framework_id"] = str(mapped["framework_id"])
            if "publication_date" in mapped and mapped["publication_date"] is not None:
                mapped["publication_date"] = str(mapped["publication_date"])
            return mapped
        return data


class ControlCategoryGraphNode(BaseModel):
    """Represents a ControlCategory node in Neo4j."""
    id: str = Field(..., description="Control category unique node ID")
    name: str = Field(..., description="Control category name (e.g. 'Access Control')")
    code: Optional[str] = Field(None, description="Control category code (e.g. 'AC')")
    description: Optional[str] = Field(None, description="Category description")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def normalize_id(cls, data: Any) -> Any:
        if isinstance(data, dict) and "id" in data and data["id"] is not None:
            mapped = dict(data)
            mapped["id"] = str(mapped["id"])
            return mapped
        return data


class EvidenceGraphNode(BaseModel):
    """
    Represents an EvidenceArtifact node and its SATISFIES relationship to an obligation.
    Preserves evidence ID and details for auditor-grade citation provenance.
    """
    id: str = Field(..., description="Evidence artifact ID (for citation provenance)")
    name: str = Field(..., description="Evidence document or artifact filename")
    file_path: Optional[str] = Field(None, description="Storage file path")
    status: Optional[str] = Field(None, description="Status (e.g. 'approved', 'COMPLETED')")
    coverage: Optional[str] = Field(None, description="Coverage status: FULL, PARTIAL, or NONE")
    coverage_status: Optional[str] = Field(None, description="Alias for coverage")
    confidence: Optional[float] = Field(None, description="Confidence score (0.0 to 1.0)")
    reasoning: Optional[str] = Field(None, description="Auditor justification explaining coverage")
    evidence_text: Optional[str] = Field(None, description="Excerpt or supporting quote from evidence")
    similarity_score: Optional[float] = Field(None, description="Vector similarity score if available")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Raw relationship/node properties")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def normalize_evidence(cls, data: Any) -> Any:
        if isinstance(data, dict):
            mapped = dict(data)
            if "id" in mapped and mapped["id"] is not None:
                mapped["id"] = str(mapped["id"])
            if not mapped.get("coverage_status") and mapped.get("coverage"):
                mapped["coverage_status"] = mapped["coverage"]
            if not mapped.get("coverage") and mapped.get("coverage_status"):
                mapped["coverage"] = mapped["coverage_status"]
            return mapped
        return data


class RelatedObligationNode(BaseModel):
    """Represents a related obligation connected via DEPENDS_ON or SUPERSEDES."""
    id: str = Field(..., description="Target obligation node ID")
    code: Optional[str] = Field(None, description="Target obligation clause/code")
    title: Optional[str] = Field(None, description="Target obligation title")
    description: Optional[str] = Field(None, description="Target obligation requirement text")
    rel_type: str = Field(..., description="Relationship type ('DEPENDS_ON', 'SUPERSEDES', etc.)")
    direction: str = Field("OUTGOING", description="Relationship direction ('OUTGOING' or 'INCOMING')")
    details: Optional[str] = Field(None, description="Relationship description or reason")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def normalize_related(cls, data: Any) -> Any:
        if isinstance(data, dict):
            mapped = dict(data)
            if "id" in mapped and mapped["id"] is not None:
                mapped["id"] = str(mapped["id"])
            if not mapped.get("details"):
                mapped["details"] = mapped.get("rel_description") or mapped.get("reason")
            return mapped
        return data


class ExpandedObligationContext(BaseModel):
    """
    Complete expanded graph context around a single regulatory obligation.

    Includes:
    - Target RegulatoryObligation details
    - Connected RegulatoryVersion & RegulatoryFramework
    - ControlCategories (CATEGORIZED_AS)
    - EvidenceArtifacts (SATISFIES, with coverage status, reasoning, and evidence_text)
    - Dependent obligations (DEPENDS_ON)
    - Superseded / Superseding obligations (SUPERSEDES)
    - Semantic similarity score carried over from Qdrant vector retrieval
    """
    obligation: ObligationGraphNode = Field(..., description="The primary obligation node")
    framework: Optional[FrameworkGraphNode] = Field(None, description="Parent regulatory framework")
    version: Optional[VersionGraphNode] = Field(None, description="Parent regulatory version")
    categories: List[ControlCategoryGraphNode] = Field(default_factory=list, description="Control categories")
    evidence_artifacts: List[EvidenceGraphNode] = Field(default_factory=list, description="Linked audit evidence artifacts")
    dependencies: List[RelatedObligationNode] = Field(default_factory=list, description="Related dependencies (DEPENDS_ON)")
    supersedes: List[RelatedObligationNode] = Field(default_factory=list, description="Supersedes / superseded obligations")
    retrieval_score: Optional[float] = Field(None, description="Semantic similarity score from Qdrant")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    @property
    def node_id(self) -> str:
        """Alias for obligation ID."""
        return self.obligation.id

    @property
    def clause(self) -> Optional[str]:
        """Alias for obligation clause or code."""
        return self.obligation.clause or self.obligation.code


class GraphRAGContext(BaseModel):
    """
    Top-level structured graph context object for the Graph RAG engine.

    Collects all expanded obligation subgraphs and provides structured formatting
    and citation provenance ready to be passed to the LLM in Phase 2 Step 6.3.
    """
    query: Optional[str] = Field(None, description="Original user question")
    obligations: List[ExpandedObligationContext] = Field(
        default_factory=list,
        description="List of expanded obligation subgraphs",
    )
    total_obligations: int = Field(0, description="Total number of expanded obligations")
    total_evidence: int = Field(0, description="Total number of linked evidence artifacts")
    total_related: int = Field(0, description="Total number of related obligations (depends_on / supersedes)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution and provenance metadata")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire context to dictionary."""
        return self.model_dump(mode="json")

    def get_citation_sources(self) -> List[Dict[str, Any]]:
        """
        Extract all citation sources from the graph context (obligations, versions, evidence).
        Provides full provenance needed for auditor-grade citations in LLM generation.
        """
        citations: List[Dict[str, Any]] = []
        for ctx in self.obligations:
            ob = ctx.obligation
            fw_name = ctx.framework.name if ctx.framework else "Unknown Framework"
            ver_slug = ctx.version.version_slug if ctx.version else "Unknown Version"

            # Obligation-level citation
            citations.append({
                "type": "obligation",
                "node_id": ob.id,
                "clause": ob.clause or ob.code,
                "title": ob.title,
                "framework": fw_name,
                "version": ver_slug,
            })

            # Evidence-level citations
            for ev in ctx.evidence_artifacts:
                citations.append({
                    "type": "evidence",
                    "evidence_id": ev.id,
                    "name": ev.name,
                    "coverage": ev.coverage or ev.coverage_status,
                    "confidence": ev.confidence,
                    "file_path": ev.file_path,
                    "obligation_id": ob.id,
                    "clause": ob.clause or ob.code,
                })

        return citations

    def format_for_llm(self) -> str:
        """
        Render the structured graph context as a Markdown-formatted context block
        for the LLM prompt.
        """
        if not self.obligations:
            return "No relevant regulatory obligations found in graph."

        sections: List[str] = []
        if self.query:
            sections.append(f"### User Question:\n{self.query}\n")

        sections.append("### Regulatory Compliance Graph Context:")

        for idx, ctx in enumerate(self.obligations, start=1):
            ob = ctx.obligation
            fw = ctx.framework.name if ctx.framework else "N/A"
            ver = ctx.version.version_slug if ctx.version else "N/A"
            clause = ob.clause or ob.code or "N/A"
            score_str = f" (Similarity: {ctx.retrieval_score:.4f})" if ctx.retrieval_score is not None else ""

            block = [
                f"#### [{idx}] Obligation: {clause} - {ob.title or 'Requirement'}{score_str}",
                f"- **Node ID**: `{ob.id}`",
                f"- **Framework**: {fw} (Version: {ver})",
            ]

            if ctx.categories:
                cats_str = ", ".join(c.name for c in ctx.categories)
                block.append(f"- **Control Category**: {cats_str}")

            if ob.description:
                block.append(f"- **Requirement Statement**:\n  > {ob.description}")

            # Linked Evidence
            if ctx.evidence_artifacts:
                block.append("- **Satisfying Evidence Artifacts**:")
                for ev in ctx.evidence_artifacts:
                    cov = ev.coverage or ev.coverage_status or "UNSPECIFIED"
                    conf = f" (Confidence: {ev.confidence:.2f})" if ev.confidence is not None else ""
                    ev_line = f"  * **[{cov}]{conf}** `{ev.name}` (ID: `{ev.id}`)"
                    if ev.reasoning:
                        ev_line += f"\n    - *Auditor Reasoning*: {ev.reasoning}"
                    if ev.evidence_text:
                        ev_line += f"\n    - *Evidence Excerpt*: \"{ev.evidence_text}\""
                    block.append(ev_line)
            else:
                block.append("- **Satisfying Evidence**: None documented.")

            # Dependencies
            if ctx.dependencies:
                block.append("- **Related Dependencies**:")
                for dep in ctx.dependencies:
                    dep_clause = dep.code or dep.id
                    det = f" - {dep.details}" if dep.details else ""
                    block.append(f"  * {dep.direction} `[:{dep.rel_type}]` -> {dep_clause} ({dep.title or 'Obligation'}){det}")

            # Supersedes
            if ctx.supersedes:
                block.append("- **Superseded / Newer Versions**:")
                for sup in ctx.supersedes:
                    sup_clause = sup.code or sup.id
                    det = f" - {sup.details}" if sup.details else ""
                    block.append(f"  * {sup.direction} `[:{sup.rel_type}]` -> {sup_clause} ({sup.title or 'Obligation'}){det}")

            sections.append("\n".join(block))

        return "\n\n".join(sections)


# -----------------------------------------------------------------------------
# Graph RAG Engine Class
# -----------------------------------------------------------------------------


class GraphRAGEngine:
    """
    Graph RAG Query Engine.

    Phase 2 Step 6.1 & Step 6.2:
    - Step 6.1: Semantic retrieval of top-k obligations from Qdrant.
    - Step 6.2: Multi-hop graph context expansion in Neo4j (Version, Framework, Categories, Evidence, Dependencies, Supersedes).
    """

    def __init__(
        self,
        qdrant_client: Optional[QdrantClient] = None,
        graph_service: Optional[Any] = None,
        default_top_k: int = DEFAULT_TOP_K,
        collection_name: Optional[str] = None,
        max_traversal_depth: int = DEFAULT_TRAVERSAL_DEPTH,
    ):
        """
        Initialize GraphRAGEngine with vector and graph client integrations.

        :param qdrant_client: Reusable QdrantClient instance (defaults to global singleton)
        :param graph_service: Reusable GraphService instance (defaults to global singleton)
        :param default_top_k: Default number of top results to retrieve from Qdrant (default: 5)
        :param collection_name: Optional Qdrant collection name override
        :param max_traversal_depth: Default maximum graph traversal depth (default: 1)
        """
        self.qdrant_client = qdrant_client or global_qdrant_client
        self._graph_service = graph_service
        self.default_top_k = default_top_k
        self.collection_name = collection_name or DEFAULT_COLLECTION_NAME
        self.max_traversal_depth = max_traversal_depth

    @property
    def graph_service(self) -> Any:
        """
        Lazily resolved GraphService instance for Neo4j graph operations.
        """
        if self._graph_service is None:
            from app.services.graph_service import graph_service
            self._graph_service = graph_service
        return self._graph_service

    # -------------------------------------------------------------------------
    # Step 6.1: Vector Retrieval
    # -------------------------------------------------------------------------

    async def generate_query_embedding(self, question: str) -> List[float]:
        """
        Generate a vector embedding for a user question using the existing
        Qdrant client integration code (Google Gemini, FastEmbed, or deterministic local fallback).

        :param question: The user question or query string
        :return: Vector embedding list of floats
        """
        if not question or not question.strip():
            return []
        return await self.qdrant_client.generate_embedding(question.strip())

    async def retrieve_relevant_obligations(
        self,
        question: str,
        top_k: Optional[int] = None,
        framework: Optional[str] = None,
        version: Optional[str] = None,
        category: Optional[str] = None,
        score_threshold: Optional[float] = None,
        collection_name: Optional[str] = None,
        raise_on_error: bool = False,
    ) -> List[RetrievedObligation]:
        """
        Retrieve the top-k most relevant regulatory obligations for a user question from Qdrant.

        :param question: User question (e.g. 'What data retention requirements apply under GDPR?')
        :param top_k: Maximum number of relevant obligations to return (default: 5)
        :param framework: Optional filter by framework name (e.g. 'GDPR', 'SOC 2')
        :param version: Optional filter by version slug (e.g. '2016', '2017')
        :param category: Optional filter by control category
        :param score_threshold: Optional minimum similarity score threshold
        :param collection_name: Optional collection name override
        :param raise_on_error: If True, raises RAGRetrievalError on unhandled failure;
                               if False (default), logs and returns [] gracefully.
        :return: List of RetrievedObligation models sorted by similarity score
        """
        clean_question = question.strip() if question else ""
        if not clean_question:
            logger.debug("Empty or blank question provided to retrieve_relevant_obligations; returning empty list.")
            return []

        limit = top_k if (top_k is not None and top_k > 0) else self.default_top_k
        target_collection = collection_name or self.collection_name

        try:
            logger.info(
                f"Retrieving top {limit} relevant obligations for question: '{clean_question[:80]}' "
                f"(framework={framework or 'ALL'}, collection={target_collection})"
            )

            # 1. Generate query embedding using existing integration code
            query_vector = await self.generate_query_embedding(clean_question)
            if not query_vector:
                logger.warning("Query embedding generation returned empty vector; returning empty list.")
                return []

            # 2. Search Qdrant for top-k similar obligations
            raw_results = await self.qdrant_client.search_similar_obligations(
                query_text=clean_question,
                query_vector=query_vector,
                framework=framework,
                version=version,
                category=category,
                limit=limit,
                score_threshold=score_threshold,
                collection_name=target_collection,
            )

            # 3. Gracefully handle empty results from vector store
            if not raw_results:
                logger.info(f"Qdrant similarity search returned 0 results for question: '{clean_question[:80]}'")
                return []

            # 4. Map raw results into structured RetrievedObligation models
            retrieved: List[RetrievedObligation] = []
            for item in raw_results:
                try:
                    retrieved.append(RetrievedObligation.model_validate(item))
                except Exception as val_err:
                    logger.warning(f"Error parsing retrieved obligation item: {val_err}", exc_info=True)
                    ob_id = str(item.get("obligation_id") or item.get("id") or "")
                    retrieved.append(
                        RetrievedObligation(
                            obligation_id=ob_id,
                            node_id=ob_id,
                            score=float(item.get("score", 0.0)),
                            framework=item.get("framework"),
                            version=item.get("version"),
                            clause=item.get("clause"),
                            category=item.get("category"),
                            title=item.get("title"),
                            text=item.get("text"),
                            mandatory=item.get("mandatory", True),
                            keywords=item.get("keywords", []),
                            metadata=item.get("payload") or {},
                        )
                    )

            top_score_msg = f", top_score={retrieved[0].score:.4f}" if retrieved else ""
            logger.info(
                f"Successfully retrieved {len(retrieved)} obligation(s) for question '{clean_question[:80]}' "
                f"(limit={limit}{top_score_msg})"
            )
            return retrieved

        except Exception as err:
            logger.error(f"Error during Qdrant obligation retrieval: {err}", exc_info=True)
            if raise_on_error:
                raise RAGRetrievalError(f"RAG retrieval failed: {err}") from err
            return []

    # -------------------------------------------------------------------------
    # Step 6.2: Graph Context Expansion
    # -------------------------------------------------------------------------

    async def expand_graph_context(
        self,
        obligation_ids: Sequence[Union[str, UUID, RetrievedObligation, Dict[str, Any]]],
        max_depth: Optional[int] = None,
        query: Optional[str] = None,
        scores_map: Optional[Dict[str, float]] = None,
        raise_on_error: bool = False,
    ) -> GraphRAGContext:
        """
        Expand the regulatory knowledge graph in Neo4j around given obligation node IDs (Phase 2 Step 6.2).

        Traverses and retrieves connected information:
        - Parent RegulatoryVersion and RegulatoryFramework
        - Connected ControlCategories (via CATEGORIZED_AS)
        - Connected EvidenceArtifacts (via SATISFIES, including coverage, confidence, reasoning, evidence_text)
        - Related obligations via DEPENDS_ON (both outgoing and incoming)
        - Related obligations via SUPERSEDES (both outgoing and incoming)

        Traversals are bounded by max_depth to prevent graph explosion.
        All provenance (node IDs, relationship types, evidence IDs) is preserved.

        :param obligation_ids: Sequence of obligation node IDs or RetrievedObligation models
        :param max_depth: Traversal depth limit (default: 1)
        :param query: Optional original user question to include in context
        :param scores_map: Optional mapping of obligation ID -> similarity score
        :param raise_on_error: If True, raises RAGGraphExpansionError on failure;
                               if False (default), logs and returns empty GraphRAGContext gracefully.
        :return: Structured GraphRAGContext model
        """
        if not obligation_ids:
            logger.debug("No obligation IDs provided for graph context expansion.")
            return GraphRAGContext(query=query)

        depth = max_depth if (max_depth is not None and max_depth > 0) else self.max_traversal_depth

        # 1. Normalize obligation IDs and extract pre-computed scores
        clean_scores: Dict[str, float] = dict(scores_map or {})
        target_ids: List[str] = []
        seen = set()

        for item in obligation_ids:
            if isinstance(item, RetrievedObligation):
                nid = item.node_id or item.obligation_id
                if item.score is not None:
                    clean_scores[nid] = item.score
            elif isinstance(item, dict):
                nid = str(item.get("node_id") or item.get("obligation_id") or item.get("id") or "")
                if item.get("score") is not None:
                    try:
                        clean_scores[nid] = float(item["score"])
                    except (ValueError, TypeError):
                        pass
            elif isinstance(item, UUID):
                nid = str(item)
            else:
                nid = str(item).strip()

            if nid and nid not in seen:
                seen.add(nid)
                target_ids.append(nid)

        if not target_ids:
            logger.debug("Normalized obligation IDs list is empty.")
            return GraphRAGContext(query=query)

        # 2. Cypher query to retrieve 1-hop connected graph context per obligation
        cypher_query = """
        MATCH (o:RegulatoryObligation)
        WHERE o.id IN $obligation_ids

        // 1. Version and Framework
        OPTIONAL MATCH (v:RegulatoryVersion)-[:CONTAINS]->(o)
        OPTIONAL MATCH (f:RegulatoryFramework)-[:HAS_VERSION]->(v)

        // 2. Control Categories
        OPTIONAL MATCH (o)-[:CATEGORIZED_AS]->(c:ControlCategory)
        WITH o, v, f, collect(DISTINCT {
            id: c.id,
            name: c.name,
            code: c.code,
            description: c.description
        }) AS raw_categories

        // 3. Evidence Artifacts (SATISFIES)
        OPTIONAL MATCH (e:EvidenceArtifact)-[r_sat:SATISFIES]->(o)
        WITH o, v, f, raw_categories, collect(DISTINCT {
            id: e.id,
            name: e.name,
            file_path: e.file_path,
            status: coalesce(r_sat.status, e.status),
            coverage: r_sat.coverage,
            coverage_status: coalesce(r_sat.coverage_status, r_sat.coverage),
            confidence: r_sat.confidence,
            reasoning: r_sat.reasoning,
            evidence_text: r_sat.evidence_text,
            similarity_score: r_sat.similarity_score
        }) AS raw_evidence

        // 4. Dependencies (Outgoing and Incoming)
        OPTIONAL MATCH (o)-[r_dep:DEPENDS_ON]->(dep:RegulatoryObligation)
        OPTIONAL MATCH (dep_by:RegulatoryObligation)-[r_dep_by:DEPENDS_ON]->(o)
        WITH o, v, f, raw_categories, raw_evidence,
             collect(DISTINCT {
                 id: dep.id,
                 code: coalesce(dep.code, dep.clause),
                 title: dep.title,
                 description: dep.description,
                 direction: "OUTGOING",
                 rel_type: "DEPENDS_ON",
                 rel_description: r_dep.description
             }) AS raw_dependencies_out,
             collect(DISTINCT {
                 id: dep_by.id,
                 code: coalesce(dep_by.code, dep_by.clause),
                 title: dep_by.title,
                 description: dep_by.description,
                 direction: "INCOMING",
                 rel_type: "DEPENDED_ON_BY",
                 rel_description: r_dep_by.description
             }) AS raw_dependencies_in

        // 5. Supersedes (Outgoing and Incoming)
        OPTIONAL MATCH (o)-[r_sup:SUPERSEDES]->(sup:RegulatoryObligation)
        OPTIONAL MATCH (sup_by:RegulatoryObligation)-[r_sup_by:SUPERSEDES]->(o)
        WITH o, v, f, raw_categories, raw_evidence, raw_dependencies_out, raw_dependencies_in,
             collect(DISTINCT {
                 id: sup.id,
                 code: coalesce(sup.code, sup.clause),
                 title: sup.title,
                 description: sup.description,
                 direction: "OUTGOING",
                 rel_type: "SUPERSEDES",
                 reason: r_sup.reason
             }) AS raw_supersedes_out,
             collect(DISTINCT {
                 id: sup_by.id,
                 code: coalesce(sup_by.code, sup_by.clause),
                 title: sup_by.title,
                 description: sup_by.description,
                 direction: "INCOMING",
                 rel_type: "SUPERSEDED_BY",
                 reason: r_sup_by.reason
             }) AS raw_supersedes_in

        RETURN o.id AS obligation_id,
               properties(o) AS obligation,
               properties(v) AS version,
               properties(f) AS framework,
               [cat IN raw_categories WHERE cat.id IS NOT NULL] AS categories,
               [ev IN raw_evidence WHERE ev.id IS NOT NULL] AS evidence_artifacts,
               [d IN raw_dependencies_out WHERE d.id IS NOT NULL] + [d IN raw_dependencies_in WHERE d.id IS NOT NULL] AS dependencies,
               [s IN raw_supersedes_out WHERE s.id IS NOT NULL] + [s IN raw_supersedes_in WHERE s.id IS NOT NULL] AS supersedes
        """

        try:
            logger.info(
                f"Executing Neo4j graph context expansion for {len(target_ids)} obligation(s) "
                f"(max_depth={depth})..."
            )
            records = await self.graph_service.execute_query(
                cypher_query,
                parameters={"obligation_ids": target_ids},
            )

            # Map results by obligation ID for preserving original Qdrant ranking order
            expanded_by_id: Dict[str, ExpandedObligationContext] = {}

            for r in (records or []):
                ob_id = str(r.get("obligation_id") or "")
                raw_ob = r.get("obligation") or {}
                if not raw_ob.get("id"):
                    raw_ob["id"] = ob_id

                ob_node = ObligationGraphNode.model_validate(raw_ob)

                # Framework
                raw_fw = r.get("framework")
                fw_node = FrameworkGraphNode.model_validate(raw_fw) if raw_fw and raw_fw.get("id") else None

                # Version
                raw_ver = r.get("version")
                ver_node = VersionGraphNode.model_validate(raw_ver) if raw_ver and raw_ver.get("id") else None

                # Categories
                raw_cats = r.get("categories") or []
                cat_nodes = [ControlCategoryGraphNode.model_validate(c) for c in raw_cats if c and c.get("id")]

                # Evidence artifacts
                raw_ev = r.get("evidence_artifacts") or []
                ev_nodes = [EvidenceGraphNode.model_validate(e) for e in raw_ev if e and e.get("id")]

                # Dependencies
                raw_deps = r.get("dependencies") or []
                dep_nodes = [RelatedObligationNode.model_validate(d) for d in raw_deps if d and d.get("id")]

                # Supersedes
                raw_sups = r.get("supersedes") or []
                sup_nodes = [RelatedObligationNode.model_validate(s) for s in raw_sups if s and s.get("id")]

                # Similarity score
                score = clean_scores.get(ob_id)

                expanded_ctx = ExpandedObligationContext(
                    obligation=ob_node,
                    framework=fw_node,
                    version=ver_node,
                    categories=cat_nodes,
                    evidence_artifacts=ev_nodes,
                    dependencies=dep_nodes,
                    supersedes=sup_nodes,
                    retrieval_score=score,
                )
                expanded_by_id[ob_id] = expanded_ctx

            # Maintain the original Qdrant similarity rank order
            ordered_expanded: List[ExpandedObligationContext] = []
            for tid in target_ids:
                if tid in expanded_by_id:
                    ordered_expanded.append(expanded_by_id[tid])

            total_ev = sum(len(ctx.evidence_artifacts) for ctx in ordered_expanded)
            total_rel = sum(len(ctx.dependencies) + len(ctx.supersedes) for ctx in ordered_expanded)

            logger.info(
                f"Graph expansion complete: {len(ordered_expanded)}/{len(target_ids)} obligation(s) matched in Neo4j, "
                f"linked to {total_ev} evidence artifact(s) and {total_rel} related obligation(s)."
            )

            return GraphRAGContext(
                query=query,
                obligations=ordered_expanded,
                total_obligations=len(ordered_expanded),
                total_evidence=total_ev,
                total_related=total_rel,
                metadata={
                    "target_ids_count": len(target_ids),
                    "matched_ids_count": len(ordered_expanded),
                    "max_depth": depth,
                },
            )

        except Exception as err:
            logger.error(f"Error during graph context expansion in Neo4j: {err}", exc_info=True)
            if raise_on_error:
                raise RAGGraphExpansionError(f"Neo4j graph expansion failed: {err}") from err
            # Handle missing/error states safely by returning empty context
            return GraphRAGContext(query=query)

    # -------------------------------------------------------------------------
    # Combined Pipeline: Retrieval + Graph Expansion
    # -------------------------------------------------------------------------

    async def query_and_expand(
        self,
        question: str,
        top_k: Optional[int] = None,
        framework: Optional[str] = None,
        version: Optional[str] = None,
        category: Optional[str] = None,
        score_threshold: Optional[float] = None,
        collection_name: Optional[str] = None,
        max_depth: Optional[int] = None,
        raise_on_error: bool = False,
    ) -> GraphRAGContext:
        """
        Execute the end-to-end Graph RAG Context Pipeline (Steps 6.1 + 6.2):
        1. Accepts user question and retrieves top-k relevant obligations from Qdrant (Step 6.1).
        2. Takes the obligation node IDs and expands graph context in Neo4j (Step 6.2).
        3. Returns structured GraphRAGContext containing obligations, versions, categories,
           evidence artifacts, and dependencies.

        :param question: User question (e.g. 'What data retention requirements apply under GDPR?')
        :param top_k: Maximum number of obligations to retrieve from Qdrant
        :param framework: Optional framework filter (e.g. 'GDPR')
        :param version: Optional version filter
        :param category: Optional category filter
        :param score_threshold: Optional similarity score threshold
        :param collection_name: Optional Qdrant collection name
        :param max_depth: Graph traversal depth limit
        :param raise_on_error: Whether to raise on retrieval/expansion failures
        :return: Structured GraphRAGContext model
        """
        # Step 6.1: Qdrant Vector Retrieval
        retrieved = await self.retrieve_relevant_obligations(
            question=question,
            top_k=top_k,
            framework=framework,
            version=version,
            category=category,
            score_threshold=score_threshold,
            collection_name=collection_name,
            raise_on_error=raise_on_error,
        )

        if not retrieved:
            logger.info("No relevant obligations retrieved from Qdrant; returning empty GraphRAGContext.")
            return GraphRAGContext(query=question)

        # Step 6.2: Neo4j Graph Context Expansion
        return await self.expand_graph_context(
            obligation_ids=retrieved,
            max_depth=max_depth,
            query=question,
            raise_on_error=raise_on_error,
        )

    # -------------------------------------------------------------------------
    # Helper & Node Extraction Methods
    # -------------------------------------------------------------------------

    @staticmethod
    def extract_node_ids(obligations: Sequence[Union[RetrievedObligation, Dict[str, Any]]]) -> List[str]:
        """
        Extract list of unique Neo4j node IDs from retrieved obligations.
        Used to feed directly into graph traversal queries.

        :param obligations: Sequence of RetrievedObligation models or result dictionaries
        :return: List of distinct Neo4j node ID strings in rank order
        """
        seen = set()
        node_ids: List[str] = []
        for ob in obligations:
            if isinstance(ob, RetrievedObligation):
                nid = ob.node_id or ob.obligation_id
            elif isinstance(ob, dict):
                nid = str(ob.get("node_id") or ob.get("obligation_id") or ob.get("id") or "")
            else:
                nid = str(getattr(ob, "node_id", None) or getattr(ob, "obligation_id", "") or "")

            if nid and nid not in seen:
                seen.add(nid)
                node_ids.append(nid)
        return node_ids

    async def get_top_obligation_node_ids(
        self,
        question: str,
        top_k: Optional[int] = None,
        framework: Optional[str] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Convenience method to retrieve only the top Neo4j node IDs for a user question.

        :param question: User question text
        :param top_k: Maximum number of obligations to retrieve
        :param framework: Optional framework filter
        :return: List of Neo4j node ID strings
        """
        obligations = await self.retrieve_relevant_obligations(
            question=question,
            top_k=top_k,
            framework=framework,
            **kwargs,
        )
        return self.extract_node_ids(obligations)

    # -------------------------------------------------------------------------
    # Convenience Aliases
    # -------------------------------------------------------------------------

    retrieve = retrieve_relevant_obligations
    search_obligations = retrieve_relevant_obligations
    search = retrieve_relevant_obligations
    retrieve_and_expand = query_and_expand


# Convenience class alias
RAGEngine = GraphRAGEngine

# Global singleton instance for application usage
rag_engine = GraphRAGEngine()


async def get_rag_engine() -> GraphRAGEngine:
    """
    FastAPI dependency returning the global GraphRAGEngine instance.
    """
    return rag_engine
