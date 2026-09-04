"""
Pydantic Schemas for Compliance Query Engine (Phase 2, Step 6.3).

Defines request and response models for Graph RAG compliance queries,
verifiable citations, and evidence provenance.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class CitationItem(BaseModel):
    """
    Structured citation pointing to a verified regulatory obligation node in Neo4j.
    """
    node_id: str = Field(
        ...,
        description="The Neo4j node ID of the cited regulatory obligation",
    )
    clause: Optional[str] = Field(
        None,
        description="Clause, article, or control identifier (e.g. 'Article 5(1)(e)', 'CC6.1')",
    )
    framework: Optional[str] = Field(
        None,
        description="Regulatory framework name (e.g. 'GDPR', 'SOC 2', 'ISO 27001')",
    )
    title: Optional[str] = Field(
        None,
        description="Obligation title or summary header",
    )

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def normalize_citation(cls, data: Any) -> Any:
        if isinstance(data, dict):
            mapped = dict(data)
            nid = mapped.get("node_id") or mapped.get("obligation_id") or mapped.get("id")
            if nid is not None:
                mapped["node_id"] = str(nid)
            return mapped
        return data


class ComplianceQueryRequest(BaseModel):
    """
    Request model for querying the Graph RAG compliance engine.
    """
    question: str = Field(
        ...,
        min_length=1,
        description="Compliance or regulatory question to answer (e.g. 'What data retention requirements apply under GDPR?')",
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of relevant obligations to retrieve from Qdrant (default: 5)",
    )
    framework: Optional[str] = Field(
        default=None,
        description="Optional filter by regulatory framework name (e.g. 'GDPR', 'SOC 2')",
    )
    version: Optional[str] = Field(
        default=None,
        description="Optional filter by version slug (e.g. '2016', '2017')",
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional filter by control category (e.g. 'Storage Limitation', 'Access Control')",
    )
    score_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional minimum vector similarity score threshold (0.0 to 1.0)",
    )
    max_depth: Optional[int] = Field(
        default=1,
        ge=1,
        le=3,
        description="Maximum graph traversal depth in Neo4j (default: 1)",
    )

    model_config = ConfigDict(from_attributes=True)


class ComplianceQueryResponse(BaseModel):
    """
    Structured response from the Graph RAG compliance engine containing
    a grounded answer, verifiable citations, and supporting evidence IDs.
    """
    answer: str = Field(
        ...,
        description="Grounded compliance answer synthesized strictly from retrieved graph context",
    )
    citations: List[CitationItem] = Field(
        default_factory=list,
        description="List of cited regulatory obligations with node IDs and clauses",
    )
    cited_node_ids: List[str] = Field(
        default_factory=list,
        description="List of Neo4j node IDs cited in the answer",
    )
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="List of satisfying evidence artifact IDs referenced in the answer",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional query execution metadata (retrieval scores, counts, provider)",
    )

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    @property
    def node_ids(self) -> List[str]:
        """Convenience alias for cited_node_ids."""
        return self.cited_node_ids
