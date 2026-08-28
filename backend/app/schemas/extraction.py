"""
Pydantic Schemas for Regulatory Obligation Extraction.

Defines data models for structured obligations extracted by LLM from regulatory text chunks.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


class ExtractedObligation(BaseModel):
    """
    Structured regulatory obligation extracted from legal/compliance text chunk.
    """
    clause: str = Field(
        ...,
        description="The specific article, section, clause, or control identifier (e.g. 'Article 5(1)(a)', 'CC6.1')",
    )
    text: str = Field(
        ...,
        description="A clear, complete statement of the requirement or obligation",
    )
    category: str = Field(
        ...,
        description="High-level regulatory or security domain (e.g. 'Access Control', 'Data Protection & Privacy')",
    )
    mandatory: bool = Field(
        ...,
        description="True if strictly mandatory (shall/must/required); False if recommendation/guidance (should/may)",
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Relevant keywords and domain concepts highlighting technologies, controls, or mechanisms",
    )

    model_config = ConfigDict(from_attributes=True)

    @field_validator("clause", "text", "category", mode="before")
    @classmethod
    def strip_strings(cls, v):
        if isinstance(v, str):
            return v.strip()
        return str(v) if v is not None else ""

    @field_validator("mandatory", mode="before")
    @classmethod
    def parse_mandatory_bool(cls, v):
        if isinstance(v, str):
            v_lower = v.strip().lower()
            if v_lower in ("true", "1", "yes", "mandatory", "required"):
                return True
            if v_lower in ("false", "0", "no", "optional", "recommended", "guidance"):
                return False
        return bool(v)

    @field_validator("keywords", mode="before")
    @classmethod
    def parse_keywords_list(cls, v):
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        if isinstance(v, (list, tuple, set)):
            return [str(k).strip() for k in v if str(k).strip()]
        return []
