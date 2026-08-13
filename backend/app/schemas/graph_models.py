"""
Pydantic Schemas for Regulatory Compliance Graph Entities & Relationships.

Defines graph node models corresponding to domain entities and relationship
models representing connections in the Neo4j graph database.
"""

from enum import Enum
from datetime import datetime, date
from uuid import UUID, uuid4
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class GraphNodeLabel(str, Enum):
    FRAMEWORK = "RegulatoryFramework"
    VERSION = "RegulatoryVersion"
    OBLIGATION = "RegulatoryObligation"
    CONTROL_CATEGORY = "ControlCategory"
    EVIDENCE = "EvidenceArtifact"


class GraphRelationshipType(str, Enum):
    HAS_VERSION = "HAS_VERSION"
    CONTAINS = "CONTAINS"
    CATEGORIZED_AS = "CATEGORIZED_AS"
    SATISFIES = "SATISFIES"
    DEPENDS_ON = "DEPENDS_ON"
    SUPERSEDES = "SUPERSEDES"


# Convenience constants for direct imports
HAS_VERSION = GraphRelationshipType.HAS_VERSION.value
CONTAINS = GraphRelationshipType.CONTAINS.value
CATEGORIZED_AS = GraphRelationshipType.CATEGORIZED_AS.value
SATISFIES = GraphRelationshipType.SATISFIES.value
DEPENDS_ON = GraphRelationshipType.DEPENDS_ON.value
SUPERSEDES = GraphRelationshipType.SUPERSEDES.value


class BaseGraphNode(BaseModel):
    """Base Pydantic model for graph nodes."""
    model_config = ConfigDict(from_attributes=True)

    def to_cypher_properties(self) -> Dict[str, Any]:
        """Convert Pydantic model fields to JSON-serializable Cypher parameters."""
        data = self.model_dump(mode="json")
        return {k: v for k, v in data.items() if v is not None}


class RegulatoryFramework(BaseGraphNode):
    """Represents a regulatory framework entity (e.g. SOC2, GDPR, ISO 27001)."""
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class RegulatoryVersion(BaseGraphNode):
    """Represents a specific revision/version of a framework (e.g. 2017, v1)."""
    id: UUID = Field(default_factory=uuid4)
    framework_id: UUID
    version_slug: str
    description: Optional[str] = None
    publication_date: Optional[date] = None
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class RegulatoryObligation(BaseGraphNode):
    """Represents a specific requirement, control statement, or obligation."""
    id: UUID = Field(default_factory=uuid4)
    version_id: UUID
    code: str
    title: str
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ControlCategory(BaseGraphNode):
    """Represents a domain or control category (e.g. Access Control, Encryption)."""
    id: UUID = Field(default_factory=uuid4)
    name: str
    code: Optional[str] = None
    description: Optional[str] = None
    created_at: Optional[datetime] = None


class EvidenceArtifact(BaseGraphNode):
    """Represents an uploaded audit evidence file or policy document."""
    id: UUID = Field(default_factory=uuid4)
    organization_id: UUID
    name: str
    file_path: str
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    status: str = "PENDING"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class BaseGraphRelationship(BaseModel):
    """Base model representing a relationship edge between two graph nodes."""
    source_id: UUID
    target_id: UUID
    rel_type: GraphRelationshipType
    properties: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True)


class HasVersionRelationship(BaseGraphRelationship):
    """(RegulatoryFramework)-[:HAS_VERSION]->(RegulatoryVersion)"""
    rel_type: GraphRelationshipType = GraphRelationshipType.HAS_VERSION


class ContainsRelationship(BaseGraphRelationship):
    """(RegulatoryVersion)-[:CONTAINS]->(RegulatoryObligation)"""
    rel_type: GraphRelationshipType = GraphRelationshipType.CONTAINS


class CategorizedAsRelationship(BaseGraphRelationship):
    """(RegulatoryObligation)-[:CATEGORIZED_AS]->(ControlCategory)"""
    rel_type: GraphRelationshipType = GraphRelationshipType.CATEGORIZED_AS


class SatisfiesRelationship(BaseGraphRelationship):
    """(EvidenceArtifact)-[:SATISFIES]->(RegulatoryObligation)"""
    rel_type: GraphRelationshipType = GraphRelationshipType.SATISFIES
    similarity_score: Optional[float] = None
    status: Optional[str] = "pending"


class DependsOnRelationship(BaseGraphRelationship):
    """(RegulatoryObligation)-[:DEPENDS_ON]->(RegulatoryObligation)"""
    rel_type: GraphRelationshipType = GraphRelationshipType.DEPENDS_ON
    description: Optional[str] = None


class SupersedesRelationship(BaseGraphRelationship):
    """(RegulatoryVersion|RegulatoryObligation)-[:SUPERSEDES]->(RegulatoryVersion|RegulatoryObligation)"""
    rel_type: GraphRelationshipType = GraphRelationshipType.SUPERSEDES
    reason: Optional[str] = None
