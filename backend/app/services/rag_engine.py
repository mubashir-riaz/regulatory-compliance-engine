"""
Graph RAG Engine - Retrieval, Graph Context Expansion & Grounded Answer Synthesis (Phase 2, Step 6).

Implements the complete Graph RAG pipeline:
Step 6.1 — Qdrant Semantic Retrieval:
  - Vector search over regulatory obligations.
  - Returns top-k obligations with similarity scores and metadata.
Step 6.2 — Neo4j Graph Context Expansion:
  - Traverses from obligation node IDs to connected RegulatoryVersion, RegulatoryFramework,
    ControlCategory, EvidenceArtifacts, DEPENDS_ON, and SUPERSEDES relationships.
  - Limits traversal depth to prevent graph explosion.
  - Preserves full provenance (node IDs, clauses, evidence IDs).
Step 6.3 — LLM Answer Generation & Verifiable Citations:
  - Combines question + Qdrant results + Neo4j graph context.
  - Prompts configured LLM (Groq or Gemini) with strict grounding instructions.
  - Validates and returns structured answer, cited node IDs, and evidence IDs.
  - Provides a deterministic fallback for offline testing or LLM downtime.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Union
from uuid import UUID

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.config import settings
from app.integrations.qdrant_client import (
    QdrantClient,
    qdrant_client as global_qdrant_client,
)
from app.schemas.compliance import (
    CitationItem,
    ComplianceQueryRequest,
    ComplianceQueryResponse,
)

logger = logging.getLogger(__name__)

# Default configurations
DEFAULT_TOP_K = 5
DEFAULT_TRAVERSAL_DEPTH = 1
DEFAULT_COLLECTION_NAME = getattr(settings, "QDRANT_COLLECTION", "regulatory_obligations")

# Default prompt path
DEFAULT_ANSWER_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "rag_answer.txt"

# Default LLM models
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"

# Embedded prompt fallback if template file is not found
EMBEDDED_ANSWER_PROMPT_TEMPLATE = """
You are an expert regulatory compliance auditor and legal engineering AI.
Your task is to provide an accurate, strictly grounded answer to the user's compliance question based ONLY on the provided regulatory knowledge graph context.

### User Question:
{question}

### Regulatory Compliance Graph Context:
{graph_context}

### Strict Auditing Instructions:
1. Grounding: Answer ONLY using the provided regulatory knowledge graph context above. Do NOT assume, extrapolate, or invent requirements that are not explicitly stated in the context.
2. Insufficient Context: If the retrieved context does not contain enough information to answer the question, clearly state: "The retrieved regulatory context does not contain sufficient information to answer this question." Do not fabricate obligations.
3. Citations: Every factual regulatory statement must cite the corresponding obligation clause and Neo4j node_id.
4. Evidence: If relevant satisfying evidence artifacts exist in the context, reference their evidence IDs and document names in your answer.
5. JSON Output: Return strictly a valid JSON object matching the format below with no markdown fences, reasoning tags, or extra commentary.

### JSON Output Schema:
{{
  "answer": "Detailed, professional compliance answer based strictly on the retrieved context...",
  "citations": [
    {{
      "node_id": "exact_node_id_from_context",
      "clause": "exact_clause_from_context",
      "framework": "exact_framework_from_context",
      "title": "exact_title_from_context"
    }}
  ],
  "cited_node_ids": [
    "exact_node_id_from_context"
  ],
  "evidence_ids": [
    "exact_evidence_id_from_context"
  ]
}}
""".strip()


class RAGServiceError(Exception):
    """Base exception for Graph RAG service errors."""
    pass


class RAGRetrievalError(RAGServiceError):
    """Exception raised when retrieval in the RAG pipeline fails."""
    pass


class RAGGraphExpansionError(RAGServiceError):
    """Exception raised when graph context expansion in Neo4j fails."""
    pass


class RAGLLMError(RAGServiceError):
    """Raised when LLM provider request fails or authentication fails."""
    pass


class RAGParseError(RAGServiceError):
    """Raised when LLM response cannot be parsed or validated."""
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

    Phase 2 Steps 6.1, 6.2, and 6.3:
    - Step 6.1: Semantic retrieval of top-k obligations from Qdrant.
    - Step 6.2: Multi-hop graph context expansion in Neo4j.
    - Step 6.3: Grounded answer generation and verifiable citations via LLM.
    """

    def __init__(
        self,
        qdrant_client: Optional[QdrantClient] = None,
        graph_service: Optional[Any] = None,
        provider: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        prompt_path: Optional[Union[str, Path]] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        fallback_enabled: bool = True,
        default_top_k: int = DEFAULT_TOP_K,
        collection_name: Optional[str] = None,
        max_traversal_depth: int = DEFAULT_TRAVERSAL_DEPTH,
    ):
        """
        Initialize GraphRAGEngine with vector, graph, and LLM services.

        :param qdrant_client: Reusable QdrantClient instance (defaults to global singleton)
        :param graph_service: Reusable GraphService instance (defaults to global singleton)
        :param provider: LLM provider ("groq" or "gemini", defaults to settings.LLM_PROVIDER)
        :param groq_api_key: Groq API key
        :param gemini_api_key: Google Gemini API key
        :param prompt_path: Path to custom prompt template for answer synthesis
        :param http_client: Reusable httpx.AsyncClient
        :param fallback_enabled: If True, uses deterministic fallback when LLM is unavailable
        :param default_top_k: Default number of top results to retrieve from Qdrant (default: 5)
        :param collection_name: Optional Qdrant collection name override
        :param max_traversal_depth: Default maximum graph traversal depth (default: 1)
        """
        self.qdrant_client = qdrant_client or global_qdrant_client
        self._graph_service = graph_service
        self.provider = (provider or getattr(settings, "LLM_PROVIDER", "groq")).lower().strip()
        self.groq_api_key = (
            groq_api_key
            if groq_api_key is not None
            else getattr(settings, "GROQ_API_KEY", None)
        )
        self.gemini_api_key = (
            gemini_api_key
            if gemini_api_key is not None
            else getattr(settings, "GEMINI_API_KEY", None)
        )
        self.prompt_path = Path(prompt_path) if prompt_path else DEFAULT_ANSWER_PROMPT_PATH
        self._http_client = http_client
        self.fallback_enabled = fallback_enabled
        self.default_top_k = default_top_k
        self.collection_name = collection_name or DEFAULT_COLLECTION_NAME
        self.max_traversal_depth = max_traversal_depth
        self._prompt_template: Optional[str] = None

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
    # Prompt Template Management
    # -------------------------------------------------------------------------

    def load_prompt_template(self) -> str:
        """
        Load the RAG answer synthesis prompt template from file or fallback to embedded template.
        """
        if self._prompt_template is None:
            if self.prompt_path.exists():
                self._prompt_template = self.prompt_path.read_text(encoding="utf-8")
            else:
                logger.warning(
                    f"RAG answer prompt template file not found at {self.prompt_path}, using embedded template."
                )
                self._prompt_template = EMBEDDED_ANSWER_PROMPT_TEMPLATE
        return self._prompt_template

    def format_answer_prompt(self, question: str, context: GraphRAGContext) -> str:
        """
        Format the LLM answer synthesis prompt with user question and rendered graph context.
        """
        template = self.load_prompt_template()
        formatted_context = context.format_for_llm()
        prompt = template.replace("{question}", question.strip())
        prompt = prompt.replace("{graph_context}", formatted_context.strip())
        return prompt

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

        :param obligation_ids: Sequence of obligation node IDs or RetrievedObligation models
        :param max_depth: Traversal depth limit (default: 1)
        :param query: Optional original user question to include in context
        :param scores_map: Optional mapping of obligation ID -> similarity score
        :param raise_on_error: If True, raises RAGGraphExpansionError on failure
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
            return GraphRAGContext(query=query)

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
        Execute Steps 6.1 + 6.2 (Qdrant Vector Retrieval + Neo4j Graph Context Expansion).
        """
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

        return await self.expand_graph_context(
            obligation_ids=retrieved,
            max_depth=max_depth,
            query=question,
            raise_on_error=raise_on_error,
        )

    # -------------------------------------------------------------------------
    # Step 6.3: Grounded LLM Answer Synthesis & Citation Verification
    # -------------------------------------------------------------------------

    async def generate_grounded_answer(
        self,
        question: str,
        context: GraphRAGContext,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        use_fallback: bool = True,
    ) -> ComplianceQueryResponse:
        """
        Synthesize a grounded compliance answer from the expanded graph context using an LLM.

        Strict Auditing Rules:
        - Only answers using the retrieved regulatory context.
        - Cross-verifies citations against actual graph nodes.
        - Preserves evidence artifact IDs where evidence exists.
        - Clearly states when context is insufficient.

        :param question: User question
        :param context: Structured GraphRAGContext containing expanded obligations and evidence
        :param provider: LLM provider override ('groq' or 'gemini')
        :param model: LLM model override
        :param use_fallback: Whether to use deterministic fallback on LLM failure / missing API key
        :return: Validated ComplianceQueryResponse
        """
        # 1. Handle completely empty graph context safely
        if not context.obligations:
            logger.info(f"No regulatory obligations in context for question '{question[:80]}'. Returning insufficient context.")
            return ComplianceQueryResponse(
                answer=(
                    f"The retrieved regulatory context does not contain sufficient information to answer this question: "
                    f"'{question}'. No matching regulatory obligations or controls were found in the knowledge graph."
                ),
                citations=[],
                cited_node_ids=[],
                evidence_ids=[],
                metadata={
                    "status": "insufficient_context",
                    "obligations_retrieved": 0,
                    "evidence_count": 0,
                },
            )

        target_provider = (provider or self.provider).lower().strip()

        # 2. Check API key configuration
        has_key = False
        if target_provider == "groq" and self.groq_api_key and not self.groq_api_key.startswith("your-"):
            has_key = True
        elif target_provider in ("gemini", "google") and self.gemini_api_key and not self.gemini_api_key.startswith("your-"):
            has_key = True

        # Fallback if no LLM key is available
        if not has_key:
            if use_fallback and self.fallback_enabled:
                logger.info(
                    f"No API key configured for provider '{target_provider}'. "
                    "Synthesizing grounded compliance answer via deterministic context engine."
                )
                return self._generate_fallback_answer(question=question, context=context)
            raise RAGLLMError(f"API key not configured for LLM provider '{target_provider}'.")

        # 3. Format prompt for LLM
        prompt = self.format_answer_prompt(question=question, context=context)

        try:
            logger.info(f"Dispatching Graph RAG answer synthesis to LLM provider '{target_provider}'...")
            raw_response = await self._call_llm(
                prompt=prompt,
                provider=target_provider,
                model=model,
            )
            response = self.parse_and_validate_answer_response(
                response_text=raw_response,
                context=context,
            )
            logger.info(
                f"Generated grounded answer ({len(response.answer)} chars) with "
                f"{len(response.citations)} citation(s) and {len(response.evidence_ids)} evidence ID(s)."
            )
            return response

        except Exception as err:
            logger.warning(f"LLM answer synthesis failed ({err}).", exc_info=True)
            if use_fallback and self.fallback_enabled:
                logger.info("Falling back to deterministic answer synthesis.")
                return self._generate_fallback_answer(question=question, context=context)
            if isinstance(err, RAGServiceError):
                raise
            raise RAGServiceError(f"Answer synthesis failed: {err}") from err

    async def answer_question(
        self,
        question: str,
        top_k: Optional[int] = None,
        framework: Optional[str] = None,
        version: Optional[str] = None,
        category: Optional[str] = None,
        score_threshold: Optional[float] = None,
        collection_name: Optional[str] = None,
        max_depth: Optional[int] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        raise_on_error: bool = False,
    ) -> ComplianceQueryResponse:
        """
        Complete End-to-End Graph RAG Orchestration Flow:
        1. Accept user question.
        2. Retrieve top-k relevant obligations from Qdrant (Step 6.1).
        3. Expand graph context around retrieved obligation node IDs in Neo4j (Step 6.2).
        4. Synthesize grounded compliance answer with verifiable citations via LLM (Step 6.3).

        :param question: User question (e.g. 'What data retention requirements apply under GDPR?')
        :param top_k: Maximum number of obligations to retrieve
        :param framework: Optional framework filter
        :param version: Optional version filter
        :param category: Optional category filter
        :param score_threshold: Optional similarity threshold
        :param collection_name: Optional collection override
        :param max_depth: Graph traversal depth
        :param provider: LLM provider override
        :param model: LLM model override
        :param raise_on_error: Whether to raise exceptions
        :return: Structured ComplianceQueryResponse
        """
        clean_question = question.strip() if question else ""
        if not clean_question:
            return ComplianceQueryResponse(
                answer="No question was provided. Please submit a valid compliance question.",
                citations=[],
                cited_node_ids=[],
                evidence_ids=[],
                metadata={"status": "empty_question"},
            )

        # 1 & 2. Qdrant Retrieval + Neo4j Graph Context Expansion
        context = await self.query_and_expand(
            question=clean_question,
            top_k=top_k,
            framework=framework,
            version=version,
            category=category,
            score_threshold=score_threshold,
            collection_name=collection_name,
            max_depth=max_depth,
            raise_on_error=raise_on_error,
        )

        # 3. LLM Grounded Answer Synthesis
        return await self.generate_grounded_answer(
            question=clean_question,
            context=context,
            provider=provider,
            model=model,
            use_fallback=self.fallback_enabled,
        )

    # Convenience alias for full query execution
    query = answer_question

    # -------------------------------------------------------------------------
    # Response Parsing & Citation Verification
    # -------------------------------------------------------------------------

    def parse_and_validate_answer_response(
        self,
        response_text: str,
        context: GraphRAGContext,
    ) -> ComplianceQueryResponse:
        """
        Parse raw LLM response text into a validated ComplianceQueryResponse model,
        and cross-verify citations against actual graph nodes.
        """
        if not response_text or not response_text.strip():
            raise RAGParseError("Received empty response text from LLM.")

        cleaned = self._clean_json_text(response_text)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            data = self._extract_first_json_object(cleaned)
            if data is None:
                raise RAGParseError(f"No valid JSON object found in LLM response: {response_text[:300]}")

        if not isinstance(data, dict):
            raise RAGParseError(f"Expected JSON object, got {type(data).__name__}")

        raw_answer = str(data.get("answer") or "").strip()
        if not raw_answer:
            raise RAGParseError("LLM response did not contain an 'answer' field.")

        # Build index of valid graph nodes and evidence for verification
        valid_nodes: Dict[str, ExpandedObligationContext] = {
            ctx.node_id: ctx for ctx in context.obligations
        }
        valid_evidence_ids: Set[str] = {
            ev.id for ctx in context.obligations for ev in ctx.evidence_artifacts
        }

        # 1. Parse and verify citations against actual graph nodes
        raw_citations = data.get("citations") or []
        verified_citations: List[CitationItem] = []
        cited_node_ids: List[str] = []

        for cit in raw_citations:
            if isinstance(cit, dict):
                nid = str(cit.get("node_id") or cit.get("id") or "")
                if nid in valid_nodes:
                    matched_ctx = valid_nodes[nid]
                    verified_citations.append(
                        CitationItem(
                            node_id=nid,
                            clause=cit.get("clause") or matched_ctx.clause,
                            framework=cit.get("framework") or (matched_ctx.framework.name if matched_ctx.framework else None),
                            title=cit.get("title") or matched_ctx.obligation.title,
                        )
                    )
                    if nid not in cited_node_ids:
                        cited_node_ids.append(nid)

        # If LLM didn't produce structured citations or cited an alias, cross-reference clauses in answer text
        if not verified_citations and context.obligations:
            for ctx in context.obligations:
                clause = ctx.clause
                if clause and (clause.lower() in raw_answer.lower() or ctx.node_id in raw_answer):
                    verified_citations.append(
                        CitationItem(
                            node_id=ctx.node_id,
                            clause=clause,
                            framework=ctx.framework.name if ctx.framework else None,
                            title=ctx.obligation.title,
                        )
                    )
                    if ctx.node_id not in cited_node_ids:
                        cited_node_ids.append(ctx.node_id)

        # Fallback citations: If still empty, link top retrieved obligation from context
        if not verified_citations and context.obligations:
            top_ctx = context.obligations[0]
            verified_citations.append(
                CitationItem(
                    node_id=top_ctx.node_id,
                    clause=top_ctx.clause,
                    framework=top_ctx.framework.name if top_ctx.framework else None,
                    title=top_ctx.obligation.title,
                )
            )
            cited_node_ids.append(top_ctx.node_id)

        # 2. Parse and verify evidence IDs against actual graph evidence
        raw_ev_ids = data.get("evidence_ids") or []
        verified_evidence_ids: List[str] = []
        for eid in raw_ev_ids:
            eid_str = str(eid).strip()
            if eid_str in valid_evidence_ids and eid_str not in verified_evidence_ids:
                verified_evidence_ids.append(eid_str)

        # If LLM referenced evidence filenames in text, match their IDs
        if not verified_evidence_ids:
            for ctx in context.obligations:
                for ev in ctx.evidence_artifacts:
                    if ev.name.lower() in raw_answer.lower() and ev.id not in verified_evidence_ids:
                        verified_evidence_ids.append(ev.id)

        metadata = {
            "obligations_retrieved": context.total_obligations,
            "evidence_count": context.total_evidence,
            "provider": self.provider,
        }

        return ComplianceQueryResponse(
            answer=raw_answer,
            citations=verified_citations,
            cited_node_ids=cited_node_ids,
            evidence_ids=verified_evidence_ids,
            metadata=metadata,
        )

    # -------------------------------------------------------------------------
    # Deterministic Fallback Answer Synthesizer
    # -------------------------------------------------------------------------

    def _generate_fallback_answer(
        self,
        question: str,
        context: GraphRAGContext,
    ) -> ComplianceQueryResponse:
        """
        Generate a strictly grounded compliance answer directly from the retrieved
        graph context without requiring external LLM API calls.
        Guarantees verifiable citations and prevents arbitrary hallucination.
        """
        if not context.obligations:
            return ComplianceQueryResponse(
                answer=(
                    f"The retrieved regulatory context does not contain sufficient information to answer this question: "
                    f"'{question}'. No matching regulatory obligations or controls were found in the knowledge graph."
                ),
                citations=[],
                cited_node_ids=[],
                evidence_ids=[],
                metadata={"status": "insufficient_context"},
            )

        paragraphs: List[str] = []
        citations: List[CitationItem] = []
        cited_node_ids: List[str] = []
        evidence_ids: List[str] = []

        for ctx in context.obligations:
            ob = ctx.obligation
            fw_name = ctx.framework.name if ctx.framework else "Compliance Framework"
            ver_slug = f" ({ctx.version.version_slug})" if ctx.version else ""
            clause = ob.clause or ob.code or "Requirement"
            desc = ob.description or ob.title or "Obligation requirement statement"

            p = f"Under {fw_name}{ver_slug}, {clause} ({ob.title or 'Control'}) requires that: {desc}"

            # Evidence details
            if ctx.evidence_artifacts:
                ev_summaries = []
                for ev in ctx.evidence_artifacts:
                    cov = ev.coverage or ev.coverage_status or "DOCUMENTED"
                    ev_summaries.append(f"'{ev.name}' ({cov} coverage, ID: {ev.id})")
                    if ev.id not in evidence_ids:
                        evidence_ids.append(ev.id)
                p += f" This requirement is addressed by satisfying evidence: {', '.join(ev_summaries)}."

            # Dependencies
            if ctx.dependencies:
                dep_clauses = [d.code or d.id for d in ctx.dependencies]
                p += f" Note that compliance with this control depends on: {', '.join(dep_clauses)}."

            # Supersedes
            if ctx.supersedes:
                sup_clauses = [s.code or s.id for s in ctx.supersedes]
                p += f" This control supersedes legacy requirements: {', '.join(sup_clauses)}."

            paragraphs.append(p)

            citations.append(
                CitationItem(
                    node_id=ctx.node_id,
                    clause=clause,
                    framework=ctx.framework.name if ctx.framework else None,
                    title=ob.title,
                )
            )
            cited_node_ids.append(ctx.node_id)

        answer_text = "\n\n".join(paragraphs)

        return ComplianceQueryResponse(
            answer=answer_text,
            citations=citations,
            cited_node_ids=cited_node_ids,
            evidence_ids=evidence_ids,
            metadata={
                "mode": "deterministic_synthesis",
                "obligations_retrieved": context.total_obligations,
                "evidence_count": len(evidence_ids),
            },
        )

    # -------------------------------------------------------------------------
    # LLM Provider Dispatch
    # -------------------------------------------------------------------------

    async def _call_llm(
        self,
        prompt: str,
        provider: str,
        model: Optional[str] = None,
    ) -> str:
        """
        Dispatch prompt to the configured LLM provider.
        """
        if provider == "groq":
            return await self._call_groq(prompt=prompt, model=model)
        elif provider in ("gemini", "google"):
            return await self._call_gemini(prompt=prompt, model=model)
        else:
            raise RAGLLMError(f"Unsupported LLM provider '{provider}'. Must be 'groq' or 'gemini'.")

    async def _call_groq(self, prompt: str, model: Optional[str] = None) -> str:
        """
        Call Groq API with candidate model fallback.
        """
        api_key = self.groq_api_key
        if not api_key or not api_key.strip() or api_key.startswith("your-"):
            raise RAGLLMError("GROQ_API_KEY is not configured.")

        candidate_models = (
            [model] if model else ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3.6-27b", "openai/gpt-oss-20b"]
        )

        last_err: Optional[Exception] = None
        for target_model in candidate_models:
            try:
                try:
                    from groq import AsyncGroq
                    client = AsyncGroq(api_key=api_key)
                    completion = await client.chat.completions.create(
                        model=target_model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a regulatory compliance auditor. You output strictly valid JSON objects.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.1,
                        max_tokens=2048,
                    )
                    content = completion.choices[0].message.content
                    if not content:
                        raise RAGParseError("Groq returned an empty response.")
                    return content
                except ImportError:
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    }
                    payload = {
                        "model": target_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a regulatory compliance auditor. You output strictly valid JSON objects.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.1,
                        "max_tokens": 2048,
                    }

                    async with (self._http_client or httpx.AsyncClient(timeout=60.0)) as client:
                        response = await client.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            headers=headers,
                            json=payload,
                        )
                        if response.status_code != 200:
                            raise RAGLLMError(
                                f"Groq API returned HTTP {response.status_code}: {response.text}"
                            )
                        data = response.json()
                        content = data["choices"][0]["message"]["content"]
                        if not content:
                            raise RAGParseError("Groq returned an empty response.")
                        return content
            except Exception as e:
                err_msg = str(e).lower()
                last_err = e
                if "model_not_found" in err_msg or "decommissioned" in err_msg or "does not exist" in err_msg or "404" in err_msg or "400" in err_msg:
                    logger.debug(f"Groq model '{target_model}' unavailable ({e}), trying next candidate...")
                    continue
                if isinstance(e, (RAGLLMError, RAGParseError)):
                    raise
                raise RAGLLMError(f"Groq API call failed: {e}") from e

        if isinstance(last_err, (RAGLLMError, RAGParseError)):
            raise last_err
        raise RAGLLMError(f"All Groq candidate models failed: {last_err}") from last_err

    async def _call_gemini(self, prompt: str, model: Optional[str] = None) -> str:
        """
        Call Google Gemini API via REST endpoint.
        """
        api_key = self.gemini_api_key
        if not api_key or not api_key.strip() or api_key.startswith("your-"):
            raise RAGLLMError("GEMINI_API_KEY is not configured.")

        target_model = model or DEFAULT_GEMINI_MODEL
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"

        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }

        try:
            async with (self._http_client or httpx.AsyncClient(timeout=60.0)) as client:
                response = await client.post(url, json=payload)
                if response.status_code != 200:
                    raise RAGLLMError(
                        f"Gemini API returned HTTP {response.status_code}: {response.text}"
                    )
                data = response.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    raise RAGParseError(f"Gemini returned no candidates: {data}")
                content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if not content:
                    raise RAGParseError("Gemini returned empty text content.")
                return content
        except Exception as e:
            if isinstance(e, (RAGLLMError, RAGParseError)):
                raise
            raise RAGLLMError(f"Gemini API call failed: {e}") from e

    @staticmethod
    def _clean_json_text(text: str) -> str:
        """Strip reasoning <think>...</think> tags and markdown code block wrappers."""
        text = text.strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    @classmethod
    def _extract_first_json_object(cls, text: str) -> Optional[Dict[str, Any]]:
        """Extract first complete JSON object using JSONDecoder.raw_decode."""
        idx = text.find("{")
        while idx != -1:
            try:
                obj, _ = json.JSONDecoder().raw_decode(text[idx:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
            idx = text.find("{", idx + 1)
        return None

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
