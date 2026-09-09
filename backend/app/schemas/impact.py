"""
Pydantic Schemas for Change Impact Analysis (Phase 2, Step 7).

Defines data models for draft regulatory text input, extracted draft obligations,
and change impact analysis pipelines.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Union
from uuid import UUID, uuid4
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.extraction import ExtractedObligation


class DraftObligation(BaseModel):
    """
    Structured regulatory obligation extracted from a new or draft regulatory text,
    formatted and preserved for change impact comparison and graph analysis.
    """
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for the draft obligation",
    )
    clause: str = Field(
        ...,
        description="The specific article, section, clause, or control identifier (e.g. 'Article 5(1)(e)')",
    )
    text: str = Field(
        ...,
        description="A clear, complete statement of the requirement or obligation",
    )
    category: str = Field(
        ...,
        description="High-level regulatory or security domain (e.g. 'Data Retention', 'Access Control')",
    )
    mandatory: bool = Field(
        default=True,
        description="True if strictly mandatory (shall/must/required); False if recommendation/guidance (should/may)",
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Relevant keywords and domain concepts highlighting technologies, controls, or mechanisms",
    )
    framework: Optional[str] = Field(
        default=None,
        description="Regulatory framework identifier or name (e.g. 'GDPR', 'SOC 2')",
    )
    version: Optional[str] = Field(
        default=None,
        description="Draft or target version identifier (e.g. '2024-draft', 'v2.0')",
    )
    source_text: Optional[str] = Field(
        default=None,
        description="The source chunk or paragraph text from which this obligation was extracted",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional arbitrary metadata for downstream comparison or auditing",
    )

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    @classmethod
    def from_extracted(
        cls,
        extracted: ExtractedObligation,
        framework: Optional[str] = None,
        version: Optional[str] = None,
        source_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "DraftObligation":
        """
        Create a DraftObligation instance from an ExtractedObligation.
        """
        return cls(
            id=uuid4(),
            clause=extracted.clause,
            text=extracted.text,
            category=extracted.category,
            mandatory=extracted.mandatory,
            keywords=list(extracted.keywords or []),
            framework=framework,
            version=version,
            source_text=source_text,
            metadata=metadata or {},
        )

    @field_validator("clause", "text", "category", mode="before")
    @classmethod
    def strip_strings(cls, v: Any) -> str:
        if isinstance(v, str):
            return v.strip()
        return str(v) if v is not None else ""

    @field_validator("mandatory", mode="before")
    @classmethod
    def parse_mandatory_bool(cls, v: Any) -> bool:
        if isinstance(v, str):
            v_lower = v.strip().lower()
            if v_lower in ("true", "1", "yes", "mandatory", "required"):
                return True
            if v_lower in ("false", "0", "no", "optional", "recommended", "guidance"):
                return False
        return bool(v)

    @field_validator("keywords", mode="before")
    @classmethod
    def parse_keywords_list(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        if isinstance(v, (list, tuple, set)):
            return [str(k).strip() for k in v if str(k).strip()]
        return []


class DraftRegulationInput(BaseModel):
    """
    Input schema for providing draft regulatory text and framework/version identifiers.
    """
    draft_text: str = Field(
        ...,
        description="Raw or formatted text of the draft/new regulation, amendment, or standard",
    )
    framework: Optional[str] = Field(
        default=None,
        description="Framework identifier or name (e.g. 'GDPR', 'SOC 2')",
    )
    version: Optional[str] = Field(
        default=None,
        description="Draft version identifier (e.g. '2024-draft', 'draft-v2')",
    )
    existing_version_id: Optional[UUID] = Field(
        default=None,
        description="Optional UUID of existing baseline regulatory version to compare against in future steps",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata associated with the draft regulation input",
    )

    model_config = ConfigDict(from_attributes=True)


class DraftExtractionResult(BaseModel):
    """
    Output model representing the structured obligations extracted from a draft regulation.
    """
    framework: Optional[str] = Field(
        default=None,
        description="Framework identifier or name",
    )
    version: Optional[str] = Field(
        default=None,
        description="Version identifier",
    )
    obligations: List[DraftObligation] = Field(
        default_factory=list,
        description="List of structured, validated obligations extracted from the draft",
    )
    total_obligations: int = Field(
        default=0,
        description="Count of obligations extracted",
    )
    raw_text_length: int = Field(
        default=0,
        description="Total character length of the analyzed draft text",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metadata including chunk count, extraction timestamp, etc.",
    )

    model_config = ConfigDict(from_attributes=True)

    def __iter__(self) -> Iterator[DraftObligation]:
        return iter(self.obligations)

    def __len__(self) -> int:
        return len(self.obligations)

    def __getitem__(self, index: int) -> DraftObligation:
        return self.obligations[index]


class ObligationChangeType(str, Enum):
    """
    Classification of regulatory obligation change (Phase 2, Step 7.2).
    """
    ADDED = "ADDED"
    MODIFIED = "MODIFIED"
    REMOVED = "REMOVED"
    UNCHANGED = "UNCHANGED"


# Convenience constants
CHANGE_ADDED = ObligationChangeType.ADDED.value
CHANGE_MODIFIED = ObligationChangeType.MODIFIED.value
CHANGE_REMOVED = ObligationChangeType.REMOVED.value
CHANGE_UNCHANGED = ObligationChangeType.UNCHANGED.value


class ObligationComparisonItem(BaseModel):
    """
    Represents the comparison outcome between a baseline regulatory obligation
    and a draft obligation (Phase 2, Step 7.2).
    """
    change_type: ObligationChangeType = Field(
        ...,
        description="Classification of change: ADDED, MODIFIED, REMOVED, or UNCHANGED",
    )
    old_obligation_id: Optional[str] = Field(
        default=None,
        description="Identifier of the baseline obligation (e.g. 'GDPR_2024_ART5' or UUID)",
    )
    old_clause: Optional[str] = Field(
        default=None,
        description="Clause identifier in baseline regulation",
    )
    old_text: Optional[str] = Field(
        default=None,
        description="Requirement statement in baseline regulation",
    )
    new_obligation_id: Optional[str] = Field(
        default=None,
        description="Identifier of newly extracted draft obligation",
    )
    new_clause: Optional[str] = Field(
        default=None,
        description="Clause identifier in draft regulation",
    )
    new_text: Optional[str] = Field(
        default=None,
        description="Requirement statement in draft regulation",
    )
    category: Optional[str] = Field(
        default=None,
        description="Regulatory or control domain (e.g. 'Data Retention')",
    )
    similarity_score: Optional[float] = Field(
        default=None,
        description="Semantic or lexical similarity score (0.0 to 1.0) between old and new obligations",
    )
    reason: str = Field(
        ...,
        description="Auditor-grade rationale explaining the classification determination",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 to 1.0) for the change classification",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional evaluation metadata (e.g. matching method, diff anchors, model)",
    )

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    @field_validator("change_type", mode="before")
    @classmethod
    def normalize_change_type(cls, v: Any) -> ObligationChangeType:
        if isinstance(v, ObligationChangeType):
            return v
        if isinstance(v, str):
            v_upper = v.strip().upper()
            if v_upper in ObligationChangeType.__members__:
                return ObligationChangeType(v_upper)
        raise ValueError(
            f"Invalid change_type: '{v}'. Must be one of: "
            f"{[e.value for e in ObligationChangeType]}"
        )

    @field_validator("old_obligation_id", "new_obligation_id", mode="before")
    @classmethod
    def stringify_ids(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        return str(v)

    @model_validator(mode="before")
    @classmethod
    def populate_reason_alias(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "reason" not in data and "reasoning" in data:
                data["reason"] = data["reasoning"]
            elif "reasoning" not in data and "reason" in data:
                data["reasoning"] = data["reason"]
        return data

    @property
    def reasoning(self) -> str:
        return self.reason


class ObligationComparisonResult(BaseModel):
    """
    Complete summary result of comparing draft obligations against baseline obligations.
    """
    framework: Optional[str] = Field(
        default=None,
        description="Framework name or identifier (e.g. 'GDPR', 'SOC 2')",
    )
    baseline_version: Optional[str] = Field(
        default=None,
        description="Baseline version identifier (e.g. '2016', 'v1.0')",
    )
    draft_version: Optional[str] = Field(
        default=None,
        description="Draft version identifier (e.g. '2024-draft')",
    )
    changes: List[ObligationComparisonItem] = Field(
        default_factory=list,
        description="List of detected changes (ADDED, MODIFIED, REMOVED, UNCHANGED)",
    )
    total_changes: int = Field(
        default=0,
        description="Total number of evaluated changes",
    )
    summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Counts of each change type (e.g. {'ADDED': 1, 'MODIFIED': 1, 'REMOVED': 1, 'UNCHANGED': 1, 'TOTAL': 4})",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata including matching method counts, execution time, models used, etc.",
    )

    model_config = ConfigDict(from_attributes=True)

    def __iter__(self) -> Iterator[ObligationComparisonItem]:
        return iter(self.changes)

    def __len__(self) -> int:
        return len(self.changes)

    def __getitem__(self, index: int) -> ObligationComparisonItem:
        return self.changes[index]

    @property
    def added(self) -> List[ObligationComparisonItem]:
        return [c for c in self.changes if c.change_type == ObligationChangeType.ADDED]

    @property
    def modified(self) -> List[ObligationComparisonItem]:
        return [c for c in self.changes if c.change_type == ObligationChangeType.MODIFIED]

    @property
    def removed(self) -> List[ObligationComparisonItem]:
        return [c for c in self.changes if c.change_type == ObligationChangeType.REMOVED]

    @property
    def unchanged(self) -> List[ObligationComparisonItem]:
        return [c for c in self.changes if c.change_type == ObligationChangeType.UNCHANGED]


class CompareObligationsRequest(BaseModel):
    """
    Input request schema for comparing draft obligations against baseline obligations.
    """
    draft_input: Optional[Union[str, DraftRegulationInput, DraftExtractionResult, List[Dict[str, Any]], List[DraftObligation]]] = Field(
        default=None,
        description="Raw draft regulatory text, DraftRegulationInput, DraftExtractionResult, or obligation list",
    )
    existing_obligations: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Explicit list of baseline obligations. If omitted, loaded from graph/DB.",
    )
    framework: Optional[str] = Field(
        default=None,
        description="Framework identifier or name (e.g. 'GDPR', 'SOC 2')",
    )
    baseline_version: Optional[str] = Field(
        default=None,
        description="Baseline version identifier (e.g. '2016', 'v1')",
    )
    draft_version: Optional[str] = Field(
        default=None,
        description="Draft version identifier (e.g. '2024-draft')",
    )
    existing_version_id: Optional[UUID] = Field(
        default=None,
        description="Optional UUID of existing baseline regulatory version",
    )
    provider: Optional[str] = Field(
        default=None,
        description="LLM provider override ('groq' or 'gemini')",
    )
    model: Optional[str] = Field(
        default=None,
        description="LLM model override",
    )
    similarity_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Threshold for semantic matching when clause/ID matching is unavailable",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Phase 2 Step 7.3: Graph-Based Impact Traversal Schemas
# =============================================================================


class ImpactProvenance(BaseModel):
    """
    Provenance trail linking an impacted evidence artifact or control
    back to an actual graph node and the original changed obligation (Step 7.3).
    """
    root_obligation_id: str = Field(
        ...,
        description="ID of the changed baseline obligation triggering this traversal",
    )
    root_clause: Optional[str] = Field(
        default=None,
        description="Clause or article identifier of the root changed obligation",
    )
    target_node_id: str = Field(
        ...,
        description="ID of the reached target node (e.g. EvidenceArtifact or ControlCategory)",
    )
    target_node_label: str = Field(
        default="EvidenceArtifact",
        description="Neo4j label of the target node",
    )
    traversal_depth: int = Field(
        default=1,
        ge=1,
        description="Graph distance / hop count from the changed obligation to the target node",
    )
    path_nodes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered list of graph nodes encountered along the traversal path",
    )
    path_relationships: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered list of relationship edges traversed",
    )
    readable_path: str = Field(
        default="",
        description="Human-readable provenance path representation (e.g. 'CC6.1 -> SATISFIES -> okta.pdf')",
    )

    model_config = ConfigDict(from_attributes=True)


class AffectedEvidenceItem(BaseModel):
    """
    Represents an evidence artifact connected to an impacted obligation via SATISFIES,
    either directly (depth 1) or transitively through DEPENDS_ON / SUPERSEDES (depth 2+).
    """
    evidence_id: str = Field(
        ...,
        description="Unique identifier of the EvidenceArtifact graph node",
    )
    evidence_name: str = Field(
        ...,
        description="Title or file name of the evidence artifact",
    )
    evidence_title: Optional[str] = Field(
        default=None,
        description="Display title or formal document title",
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Storage or repository path for the evidence artifact",
    )
    evidence_status: Optional[str] = Field(
        default=None,
        description="Status property on the EvidenceArtifact node (e.g. 'COMPLETED', 'PENDING')",
    )
    obligation_id: str = Field(
        ...,
        description="ID of the obligation directly connected to this evidence artifact",
    )
    clause: Optional[str] = Field(
        default=None,
        description="Clause or code of the connected obligation (e.g. 'Article 5(1)(e)', 'CC6.1')",
    )
    obligation_title: Optional[str] = Field(
        default=None,
        description="Title of the connected obligation",
    )
    root_obligation_id: Optional[str] = Field(
        default=None,
        description="ID of the root modified/removed obligation that initiated graph traversal",
    )
    root_clause: Optional[str] = Field(
        default=None,
        description="Clause of the root modified/removed obligation",
    )
    change_type: Optional[str] = Field(
        default=None,
        description="Classification of the change: MODIFIED or REMOVED",
    )
    impact_type: str = Field(
        default="DIRECT",
        description="Impact category: 'DIRECT' (SATISFIES) or 'INDIRECT' (via DEPENDS_ON/SUPERSEDES)",
    )
    depth: int = Field(
        default=1,
        ge=1,
        description="Traversal depth from changed obligation to evidence artifact",
    )
    coverage_status: Optional[str] = Field(
        default=None,
        description="Existing coverage status on SATISFIES edge (e.g. 'FULL', 'PARTIAL', 'NONE', 'approved')",
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence score stored on the SATISFIES relationship",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Auditor reasoning stored on the SATISFIES relationship",
    )
    evidence_text: Optional[str] = Field(
        default=None,
        description="Relevant evidence text or snippet stored on the SATISFIES relationship",
    )
    similarity_score: Optional[float] = Field(
        default=None,
        description="Similarity score stored on the SATISFIES relationship",
    )
    relationship_type: str = Field(
        default="SATISFIES",
        description="Relationship edge type connecting evidence artifact",
    )
    relationship_direction: str = Field(
        default="INCOMING",
        description="Edge direction relative to obligation ('INCOMING' for (Evidence)-[:SATISFIES]->(Obligation))",
    )
    relationship_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Full preserved properties dictionary from the SATISFIES relationship edge",
    )
    provenance: Optional[ImpactProvenance] = Field(
        default=None,
        description="Graph provenance trace linking this finding back to the root changed obligation",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context metadata",
    )

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    @field_validator("evidence_id", "obligation_id", mode="before")
    @classmethod
    def stringify_ids(cls, v: Any) -> str:
        return str(v) if v is not None else ""


class AffectedControlItem(BaseModel):
    """
    Represents a control category connected to an impacted obligation via CATEGORIZED_AS (Step 7.3).
    """
    control_id: str = Field(
        ...,
        description="ID of the ControlCategory graph node",
    )
    control_name: str = Field(
        ...,
        description="Name of the control category (e.g. 'Access Control', 'Data Retention')",
    )
    control_code: Optional[str] = Field(
        default=None,
        description="Short code of the control category (e.g. 'AC')",
    )
    control_description: Optional[str] = Field(
        default=None,
        description="Description of the control category",
    )
    obligation_id: str = Field(
        ...,
        description="ID of the connected obligation",
    )
    clause: Optional[str] = Field(
        default=None,
        description="Clause or code of the connected obligation",
    )
    relationship_type: str = Field(
        default="CATEGORIZED_AS",
        description="Relationship type",
    )
    relationship_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Properties from the CATEGORIZED_AS relationship edge",
    )

    model_config = ConfigDict(from_attributes=True)


class DependentObligationItem(BaseModel):
    """
    Represents an obligation connected via DEPENDS_ON relationship (Step 7.3).
    """
    obligation_id: str = Field(..., description="ID of the dependent or dependency obligation")
    code: Optional[str] = Field(default=None, description="Obligation code (e.g. 'CC6.2')")
    clause: Optional[str] = Field(default=None, description="Obligation clause")
    title: Optional[str] = Field(default=None, description="Obligation title")
    description: Optional[str] = Field(default=None, description="Obligation description or text")
    direction: str = Field(
        default="OUTGOING",
        description="Edge direction: 'OUTGOING' ((o)-[:DEPENDS_ON]->(dep)) or 'INCOMING' ((dep)-[:DEPENDS_ON]->(o))",
    )
    dependency_description: Optional[str] = Field(
        default=None,
        description="Description property on the DEPENDS_ON relationship edge",
    )
    relationship_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Properties from the DEPENDS_ON edge",
    )

    model_config = ConfigDict(from_attributes=True)


class SupersededObligationItem(BaseModel):
    """
    Represents an obligation connected via SUPERSEDES relationship (Step 7.3).
    """
    obligation_id: str = Field(..., description="ID of the superseded or superseding obligation")
    code: Optional[str] = Field(default=None, description="Obligation code (e.g. 'CC6.1-2014')")
    clause: Optional[str] = Field(default=None, description="Obligation clause")
    title: Optional[str] = Field(default=None, description="Obligation title")
    description: Optional[str] = Field(default=None, description="Obligation description or text")
    direction: str = Field(
        default="OUTGOING",
        description="Edge direction: 'OUTGOING' ((o)-[:SUPERSEDES]->(sup)) or 'INCOMING' ((sup)-[:SUPERSEDES]->(o))",
    )
    supersedes_reason: Optional[str] = Field(
        default=None,
        description="Reason property on the SUPERSEDES relationship edge",
    )
    relationship_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Properties from the SUPERSEDES edge",
    )

    model_config = ConfigDict(from_attributes=True)


class ImpactedObligationTraversal(BaseModel):
    """
    Comprehensive graph impact traversal result for a single MODIFIED or REMOVED obligation (Step 7.3).
    """
    obligation_id: str = Field(
        ...,
        description="ID of the analyzed obligation in the regulatory graph",
    )
    clause: Optional[str] = Field(
        default=None,
        description="Clause or article identifier of the obligation",
    )
    title: Optional[str] = Field(
        default=None,
        description="Title of the obligation",
    )
    change_type: Optional[str] = Field(
        default=None,
        description="Classification of change (MODIFIED or REMOVED)",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Auditor reasoning explaining why this obligation changed",
    )
    direct_evidence: List[AffectedEvidenceItem] = Field(
        default_factory=list,
        description="Evidence artifacts directly connected through SATISFIES (depth 1)",
    )
    indirect_evidence: List[AffectedEvidenceItem] = Field(
        default_factory=list,
        description="Evidence artifacts transitively reached via DEPENDS_ON or SUPERSEDES (depth 2+)",
    )
    dependent_obligations: List[DependentObligationItem] = Field(
        default_factory=list,
        description="Obligations linked via DEPENDS_ON relationship edges",
    )
    superseded_obligations: List[SupersededObligationItem] = Field(
        default_factory=list,
        description="Obligations linked via SUPERSEDES relationship edges",
    )
    affected_controls: List[AffectedControlItem] = Field(
        default_factory=list,
        description="Control categories connected via CATEGORIZED_AS edges",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Execution and traversal metadata",
    )

    model_config = ConfigDict(from_attributes=True)

    @property
    def all_evidence(self) -> List[AffectedEvidenceItem]:
        """All affected evidence items (direct and indirect)."""
        return self.direct_evidence + self.indirect_evidence

    @property
    def affected_evidence_ids(self) -> List[str]:
        """Deduplicated list of affected evidence artifact IDs."""
        seen = set()
        ids = []
        for ev in self.all_evidence:
            if ev.evidence_id not in seen:
                seen.add(ev.evidence_id)
                ids.append(ev.evidence_id)
        return ids


class GraphImpactTraversalResult(BaseModel):
    """
    Aggregated result of graph-based impact traversal across all MODIFIED and REMOVED obligations (Step 7.3).
    """
    framework: Optional[str] = Field(
        default=None,
        description="Regulatory framework identifier (e.g. 'GDPR', 'SOC 2')",
    )
    baseline_version: Optional[str] = Field(
        default=None,
        description="Baseline version identifier (e.g. '2016', '2017')",
    )
    draft_version: Optional[str] = Field(
        default=None,
        description="Draft or target version identifier (e.g. '2024-draft')",
    )
    traversals: List[ImpactedObligationTraversal] = Field(
        default_factory=list,
        description="Per-obligation traversal results for each MODIFIED or REMOVED obligation",
    )
    affected_evidence_items: List[AffectedEvidenceItem] = Field(
        default_factory=list,
        description="Deduplicated list of all affected evidence items across all analyzed obligations",
    )
    affected_evidence_ids: List[str] = Field(
        default_factory=list,
        description="Deduplicated list of all affected evidence artifact IDs",
    )
    affected_control_ids: List[str] = Field(
        default_factory=list,
        description="Deduplicated list of all affected control category IDs",
    )
    total_impacted_obligations: int = Field(
        default=0,
        description="Total number of MODIFIED or REMOVED obligations traversed",
    )
    total_affected_evidence: int = Field(
        default=0,
        description="Total number of unique affected evidence artifacts found",
    )
    total_affected_controls: int = Field(
        default=0,
        description="Total number of unique affected control categories found",
    )
    max_depth: int = Field(
        default=2,
        description="Traversal depth limit used during graph traversal",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary metadata including execution timing, depth limits, and node counts",
    )

    model_config = ConfigDict(from_attributes=True)

    def __iter__(self) -> Iterator[ImpactedObligationTraversal]:
        return iter(self.traversals)

    def __len__(self) -> int:
        return len(self.traversals)

    def __getitem__(self, index: int) -> ImpactedObligationTraversal:
        return self.traversals[index]

    @property
    def direct_evidence(self) -> List[AffectedEvidenceItem]:
        """All directly affected evidence items (depth 1)."""
        return [e for e in self.affected_evidence_items if e.impact_type == "DIRECT"]

    @property
    def indirect_evidence(self) -> List[AffectedEvidenceItem]:
        """All indirectly affected evidence items (transitive, depth > 1)."""
        return [e for e in self.affected_evidence_items if e.impact_type != "DIRECT"]

    def get_evidence_for_obligation(self, obligation_id: str) -> List[AffectedEvidenceItem]:
        """Retrieve affected evidence items linked to a specific obligation ID."""
        target = str(obligation_id).lower()
        return [
            e for e in self.affected_evidence_items
            if (e.obligation_id and e.obligation_id.lower() == target)
            or (e.root_obligation_id and e.root_obligation_id.lower() == target)
        ]


class TraverseImpactRequest(BaseModel):
    """
    Input request schema for graph-based impact traversal (Step 7.3).
    """
    comparison_result: Optional[ObligationComparisonResult] = Field(
        default=None,
        description="Comparison result from Step 7.2 containing classified obligation changes",
    )
    changed_obligations: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Explicit list of changed obligations (MODIFIED or REMOVED) to traverse",
    )
    obligation_ids: Optional[List[Union[str, UUID]]] = Field(
        default=None,
        description="Explicit list of obligation IDs to query in the graph",
    )
    max_depth: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum graph traversal depth limit (default 2)",
    )
    include_dependencies: bool = Field(
        default=True,
        description="Whether to inspect DEPENDS_ON relationships",
    )
    include_supersedes: bool = Field(
        default=True,
        description="Whether to inspect SUPERSEDES relationships",
    )
    include_controls: bool = Field(
        default=True,
        description="Whether to inspect CATEGORIZED_AS control categories",
    )
    framework: Optional[str] = Field(
        default=None,
        description="Regulatory framework identifier override",
    )
    baseline_version: Optional[str] = Field(
        default=None,
        description="Baseline version identifier override",
    )
    draft_version: Optional[str] = Field(
        default=None,
        description="Draft version identifier override",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Phase 2 Step 7.4: Flag Potentially Invalid Evidence Schemas
# =============================================================================


class EvidenceReviewStatus(str, Enum):
    """
    Non-destructive review status for evidence affected by regulatory changes (Step 7.4).
    Prevents false conclusions by marking evidence as requiring review rather than
    prematurely declaring it definitively invalid.
    """
    NEEDS_REVIEW = "NEEDS_REVIEW"
    POTENTIALLY_INVALID = "POTENTIALLY_INVALID"
    VALID = "VALID"
    SUPERSEDED = "SUPERSEDED"


# Convenience constants
STATUS_NEEDS_REVIEW = EvidenceReviewStatus.NEEDS_REVIEW.value
STATUS_POTENTIALLY_INVALID = EvidenceReviewStatus.POTENTIALLY_INVALID.value
STATUS_VALID = EvidenceReviewStatus.VALID.value
STATUS_SUPERSEDED = EvidenceReviewStatus.SUPERSEDED.value


class FlaggedEvidenceRecord(BaseModel):
    """
    Structured impact review record for an evidence artifact connected to a
    MODIFIED or REMOVED obligation (Phase 2, Step 7.4).

    Flagged with a non-destructive status (NEEDS_REVIEW or POTENTIALLY_INVALID)
    while strictly preserving existing coverage assessments and graph relationships.
    """
    evidence_id: str = Field(
        ...,
        description="Unique identifier of the EvidenceArtifact graph node",
    )
    status: Union[EvidenceReviewStatus, str] = Field(
        default=EvidenceReviewStatus.NEEDS_REVIEW,
        description="Non-destructive review status: 'NEEDS_REVIEW' or 'POTENTIALLY_INVALID'",
    )
    reason: str = Field(
        ...,
        description="Auditor explanation why compliance review is required",
    )
    obligation_id: str = Field(
        ...,
        description="Identifier of the directly or indirectly affected regulatory obligation",
    )
    clause: Optional[str] = Field(
        default=None,
        description="Clause or control code of the affected obligation (e.g. 'Article 5(1)(e)')",
    )
    obligation_title: Optional[str] = Field(
        default=None,
        description="Title of the affected obligation",
    )
    change_type: str = Field(
        ...,
        description="Classification of regulatory change triggering this review: 'MODIFIED' or 'REMOVED'",
    )
    impact_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score of the change impact classification where available",
    )
    evidence_name: Optional[str] = Field(
        default=None,
        description="File name or title of the evidence artifact",
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Storage or repository path of the evidence artifact",
    )
    # Preservation of previous coverage information (must NOT overwrite or delete existing SATISFIES)
    previous_coverage_status: Optional[str] = Field(
        default=None,
        description="Preserved existing coverage status from SATISFIES edge (e.g. 'FULL', 'PARTIAL', 'approved')",
    )
    previous_confidence: Optional[float] = Field(
        default=None,
        description="Preserved previous assessment confidence score from SATISFIES edge",
    )
    previous_reasoning: Optional[str] = Field(
        default=None,
        description="Preserved previous auditor reasoning from SATISFIES edge",
    )
    previous_evidence_text: Optional[str] = Field(
        default=None,
        description="Preserved evidence snippet from SATISFIES edge",
    )
    root_obligation_id: Optional[str] = Field(
        default=None,
        description="Root modified/removed obligation ID that initiated graph traversal",
    )
    root_clause: Optional[str] = Field(
        default=None,
        description="Root modified/removed obligation clause",
    )
    impact_type: str = Field(
        default="DIRECT",
        description="Impact category: 'DIRECT' (depth 1) or 'INDIRECT' (depth >= 2)",
    )
    traversal_depth: int = Field(
        default=1,
        description="Graph hop distance from changed obligation to evidence artifact",
    )
    framework: Optional[str] = Field(
        default=None,
        description="Regulatory framework identifier (e.g. 'GDPR', 'SOC 2')",
    )
    draft_version: Optional[str] = Field(
        default=None,
        description="Draft or target regulation version triggering the review",
    )
    flagged_at: Optional[datetime] = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the evidence review flag was generated",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Supplementary audit and provenance metadata",
    )

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, v: Any) -> str:
        if isinstance(v, EvidenceReviewStatus):
            return v.value
        if isinstance(v, str):
            v_upper = v.strip().upper()
            if v_upper in EvidenceReviewStatus.__members__:
                return EvidenceReviewStatus(v_upper).value
            return v_upper
        return EvidenceReviewStatus.NEEDS_REVIEW.value

    @field_validator("evidence_id", "obligation_id", mode="before")
    @classmethod
    def stringify_ids(cls, v: Any) -> str:
        return str(v) if v is not None else ""

    def to_canonical_dict(self) -> Dict[str, Any]:
        """
        Export compact dictionary conforming to the canonical Step 7.4 specification example:
        {
          "evidence_id": "retention_policy_2026",
          "status": "NEEDS_REVIEW",
          "reason": "The obligation satisfied by this policy was modified in the draft regulation."
        }
        """
        status_val = self.status.value if isinstance(self.status, EvidenceReviewStatus) else str(self.status)
        return {
            "evidence_id": self.evidence_id,
            "status": status_val,
            "reason": self.reason,
        }

    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Export detailed dictionary including canonical fields plus affected obligation & change metadata.
        """
        status_val = self.status.value if isinstance(self.status, EvidenceReviewStatus) else str(self.status)
        return {
            "evidence_id": self.evidence_id,
            "status": status_val,
            "reason": self.reason,
            "obligation_id": self.obligation_id,
            "clause": self.clause,
            "change_type": self.change_type,
            "impact_confidence": self.impact_confidence,
            "previous_coverage_status": self.previous_coverage_status,
            "impact_type": self.impact_type,
            "draft_version": self.draft_version,
        }


class EvidenceImpactReviewResult(BaseModel):
    """
    Aggregated result of Phase 2 Step 7.4 affected evidence review flagging.
    Contains deduplicated list of flagged evidence records, summary metrics,
    and preserved coverage state.
    """
    framework: Optional[str] = Field(
        default=None,
        description="Regulatory framework identifier",
    )
    baseline_version: Optional[str] = Field(
        default=None,
        description="Baseline version identifier",
    )
    draft_version: Optional[str] = Field(
        default=None,
        description="Draft version identifier",
    )
    flagged_items: List[FlaggedEvidenceRecord] = Field(
        default_factory=list,
        description="List of all flagged evidence review records",
    )
    total_flagged: int = Field(
        default=0,
        description="Total number of evidence-obligation impact pairings flagged",
    )
    total_unique_evidence: int = Field(
        default=0,
        description="Total unique evidence artifacts requiring compliance review",
    )
    unique_evidence_ids: List[str] = Field(
        default_factory=list,
        description="Deduplicated list of affected evidence IDs",
    )
    status_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of records by status (e.g. {'NEEDS_REVIEW': 3})",
    )
    change_type_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of records by change type (e.g. {'MODIFIED': 2, 'REMOVED': 1})",
    )
    canonical_summary: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of compact evidence review summary dicts matching canonical Step 7.4 format",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metadata including graph update flags, timing, etc.",
    )

    model_config = ConfigDict(from_attributes=True)

    def __iter__(self) -> Iterator[FlaggedEvidenceRecord]:
        return iter(self.flagged_items)

    def __len__(self) -> int:
        return len(self.flagged_items)

    def __getitem__(self, index: int) -> FlaggedEvidenceRecord:
        return self.flagged_items[index]

    def get_for_evidence(self, evidence_id: str) -> List[FlaggedEvidenceRecord]:
        """Retrieve all review records for a specific evidence ID."""
        target = str(evidence_id).strip().lower()
        return [item for item in self.flagged_items if item.evidence_id.lower() == target]

    def get_for_obligation(self, obligation_id: str) -> List[FlaggedEvidenceRecord]:
        """Retrieve all review records for a specific obligation ID."""
        target = str(obligation_id).strip().lower()
        return [
            item for item in self.flagged_items
            if item.obligation_id.lower() == target
            or (item.root_obligation_id and item.root_obligation_id.lower() == target)
        ]


class FlagEvidenceRequest(BaseModel):
    """
    Input request schema for flagging affected evidence (Phase 2, Step 7.4).
    """
    traversal_result: Optional[GraphImpactTraversalResult] = Field(
        default=None,
        description="Traversal result from Step 7.3 containing impacted evidence",
    )
    comparison_result: Optional[ObligationComparisonResult] = Field(
        default=None,
        description="Comparison result from Step 7.2 containing classified obligation changes",
    )
    affected_evidence: Optional[List[Union[AffectedEvidenceItem, Dict[str, Any]]]] = Field(
        default=None,
        description="Explicit list of affected evidence items to flag",
    )
    default_status: Union[EvidenceReviewStatus, str] = Field(
        default=EvidenceReviewStatus.NEEDS_REVIEW,
        description="Non-destructive review status to assign ('NEEDS_REVIEW' or 'POTENTIALLY_INVALID')",
    )
    store_in_graph: bool = Field(
        default=True,
        description="Whether to record review flags and impact findings in the Neo4j graph",
    )
    update_relationship: bool = Field(
        default=True,
        description="Whether to attach non-destructive impact properties to existing SATISFIES edges",
    )
    create_impact_nodes: bool = Field(
        default=True,
        description="Whether to create separate DraftImpactFinding nodes to isolate draft findings from approved state",
    )
    custom_reason: Optional[str] = Field(
        default=None,
        description="Optional custom reason template (supports {clause}, {change_type}, {obligation_id})",
    )
    framework: Optional[str] = Field(
        default=None,
        description="Framework identifier override",
    )
    baseline_version: Optional[str] = Field(
        default=None,
        description="Baseline version identifier override",
    )
    draft_version: Optional[str] = Field(
        default=None,
        description="Draft version identifier override",
    )

    model_config = ConfigDict(from_attributes=True)



