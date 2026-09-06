"""
Pydantic Schemas for Change Impact Analysis (Phase 2, Step 7).

Defines data models for draft regulatory text input, extracted draft obligations,
and change impact analysis pipelines.
"""

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

