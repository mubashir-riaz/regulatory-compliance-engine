"""
Pydantic Schemas for Change Impact Analysis (Phase 2, Step 7).

Defines data models for draft regulatory text input, extracted draft obligations,
and change impact analysis pipelines.
"""

from typing import Any, Dict, Iterator, List, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, ConfigDict, Field, field_validator

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
