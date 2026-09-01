"""
Pydantic Schemas for Regulatory Coverage Assessment.

Defines data models for structured assessment of audit evidence against
regulatory obligations.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class CoverageStatus(str, Enum):
    """
    Coverage status categories indicating how completely evidence satisfies an obligation.
    """
    FULL = "FULL"
    PARTIAL = "PARTIAL"
    NONE = "NONE"


class CoverageAssessmentResult(BaseModel):
    """
    Structured result of comparing an audit evidence statement against a regulatory obligation.
    """
    status: CoverageStatus = Field(
        ...,
        description="Coverage determination: FULL, PARTIAL, or NONE",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0 reflecting clarity of assessment",
    )
    reasoning: str = Field(
        ...,
        description="Detailed auditor-grade rationale explaining why the evidence is FULL, PARTIAL, or NONE",
    )
    relevant_snippet: Optional[str] = Field(
        default=None,
        description="Exact quote or excerpt from the evidence text supporting the determination",
    )

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    @property
    def coverage_status(self) -> CoverageStatus:
        """Alias for status."""
        return self.status

    @property
    def confidence_score(self) -> float:
        """Alias for confidence."""
        return self.confidence

    @property
    def relevant_evidence(self) -> Optional[str]:
        """Alias for relevant_snippet."""
        return self.relevant_snippet

    @property
    def relevant_evidence_text(self) -> Optional[str]:
        """Alias for relevant_snippet."""
        return self.relevant_snippet

    @model_validator(mode="before")
    @classmethod
    def remap_input_keys(cls, data: Any) -> Any:
        """
        Remap alternative field names that LLMs or callers might use.
        """
        if isinstance(data, dict):
            mapped = dict(data)
            # Map alternative status keys
            if "status" not in mapped:
                for k in ("coverage_status", "coverage", "assessment"):
                    if k in mapped and mapped[k] is not None:
                        mapped["status"] = mapped[k]
                        break

            # Map alternative confidence keys
            if "confidence" not in mapped:
                for k in ("confidence_score", "score", "certainty"):
                    if k in mapped and mapped[k] is not None:
                        mapped["confidence"] = mapped[k]
                        break

            # Map alternative snippet keys
            if "relevant_snippet" not in mapped:
                for k in ("relevant_evidence", "relevant_evidence_text", "evidence_snippet", "snippet", "quote"):
                    if k in mapped and mapped[k] is not None:
                        mapped["relevant_snippet"] = mapped[k]
                        break

            # Map alternative reasoning keys
            if "reasoning" not in mapped:
                for k in ("explanation", "rationale", "justification", "analysis"):
                    if k in mapped and mapped[k] is not None:
                        mapped["reasoning"] = mapped[k]
                        break

            return mapped
        return data

    @field_validator("status", mode="before")
    @classmethod
    def parse_status(cls, v: Any) -> CoverageStatus:
        if isinstance(v, CoverageStatus):
            return v
        if isinstance(v, str):
            v_upper = v.strip().upper()
            if "FULL" in v_upper:
                return CoverageStatus.FULL
            if "PARTIAL" in v_upper:
                return CoverageStatus.PARTIAL
            if "NONE" in v_upper or "NO" in v_upper or "NOT" in v_upper:
                return CoverageStatus.NONE
        raise ValueError(f"Invalid coverage status '{v}'. Must be FULL, PARTIAL, or NONE.")

    @field_validator("confidence", mode="before")
    @classmethod
    def parse_confidence(cls, v: Any) -> float:
        if v is None:
            return 0.85
        if isinstance(v, str):
            v_upper = v.strip().upper()
            if "VERY HIGH" in v_upper:
                return 0.98
            if "HIGH" in v_upper:
                return 0.90
            if "MED" in v_upper:
                return 0.70
            if "LOW" in v_upper:
                return 0.40
        try:
            val = float(v)
            # Normalize percentage (e.g. 85.0 -> 0.85)
            if 1.0 < val <= 100.0:
                val = val / 100.0
            return max(0.0, min(1.0, round(val, 4)))
        except (ValueError, TypeError):
            return 0.85

    @field_validator("reasoning", mode="before")
    @classmethod
    def parse_reasoning(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v).strip()

    @field_validator("relevant_snippet", mode="before")
    @classmethod
    def parse_snippet(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        text = str(v).strip()
        return text if text else None


class CoverageAssessmentRequest(BaseModel):
    """
    Request model for assessing evidence coverage against an obligation.
    """
    evidence_text: str = Field(..., description="Audit evidence statement, policy excerpt, or configuration text")
    obligation_text: str = Field(..., description="Text of the regulatory requirement or control statement")
    clause: Optional[str] = Field(None, description="Clause or control ID (e.g. 'CC6.1', 'Article 5(1)(a)')")
    category: Optional[str] = Field(None, description="Domain or control category (e.g. 'Access Control')")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional extra metadata")

    model_config = ConfigDict(from_attributes=True)
