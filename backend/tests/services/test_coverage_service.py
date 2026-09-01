"""
Unit and Standalone Tests for Coverage Assessment Logic (Phase 2, Step 5.1).

Tests the coverage assessment service that compares audit evidence statements
against regulatory obligations and determines FULL, PARTIAL, or NONE coverage.
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch
import pytest

from app.schemas.coverage import CoverageAssessmentResult, CoverageStatus
from app.services.coverage_service import CoverageService, coverage_service


# -----------------------------------------------------------------------------
# Test Data: One Evidence Statement and One Obligation
# -----------------------------------------------------------------------------

SAMPLE_EVIDENCE_STATEMENT = (
    "All employees accessing company production environments must authenticate using "
    "Okta multi-factor authentication (MFA) with biometric or hardware security keys. "
    "Access permissions are reviewed on a quarterly basis and revoked immediately upon termination."
)

SAMPLE_OBLIGATION_TEXT = (
    "The entity implements logical access security software, infrastructure, and architectures "
    "over protected information assets to protect them from security events."
)

SAMPLE_CLAUSE = "CC6.1"
SAMPLE_CATEGORY = "Access Control"

PARTIAL_EVIDENCE_STATEMENT = (
    "Multi-factor authentication (MFA) and role-based access control are enforced for administrative "
    "and IT staff, but general employees currently only use single-factor passwords. Access reviews "
    "are only partially implemented across departments."
)

IRRELEVANT_EVIDENCE_STATEMENT = (
    "The office cafeteria offers subsidized healthy meal options and organic coffee for all "
    "on-site employees during standard working hours."
)


# -----------------------------------------------------------------------------
# Standalone Unit Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_standalone_coverage_assessment_full():
    """
    Test comparing one evidence statement with one obligation producing FULL coverage.
    """
    service = CoverageService(fallback_enabled=True)

    result = await service.assess_coverage(
        evidence_text=SAMPLE_EVIDENCE_STATEMENT,
        obligation_text=SAMPLE_OBLIGATION_TEXT,
        clause=SAMPLE_CLAUSE,
        category=SAMPLE_CATEGORY,
    )

    assert isinstance(result, CoverageAssessmentResult)
    assert result.status == CoverageStatus.FULL
    assert result.confidence >= 0.7
    assert len(result.reasoning) > 10
    assert result.relevant_snippet is not None
    assert "MFA" in result.relevant_snippet or "authenticate" in result.relevant_snippet

    # Verify property accessors
    assert result.coverage_status == CoverageStatus.FULL
    assert result.confidence_score == result.confidence
    assert result.relevant_evidence == result.relevant_snippet


@pytest.mark.asyncio
async def test_standalone_coverage_assessment_partial():
    """
    Test comparing evidence with partial implementation producing PARTIAL coverage.
    """
    service = CoverageService(fallback_enabled=True)

    result = await service.assess_coverage(
        evidence_text=PARTIAL_EVIDENCE_STATEMENT,
        obligation_text=SAMPLE_OBLIGATION_TEXT,
        clause=SAMPLE_CLAUSE,
        category=SAMPLE_CATEGORY,
    )

    assert isinstance(result, CoverageAssessmentResult)
    assert result.status == CoverageStatus.PARTIAL
    assert result.confidence > 0.0
    assert "PARTIAL" in result.reasoning.upper() or "roadmap" in result.reasoning.lower() or "partial" in result.reasoning.lower()


@pytest.mark.asyncio
async def test_standalone_coverage_assessment_none():
    """
    Test comparing irrelevant evidence statement producing NONE coverage.
    """
    service = CoverageService(fallback_enabled=True)

    result = await service.assess_coverage(
        evidence_text=IRRELEVANT_EVIDENCE_STATEMENT,
        obligation_text=SAMPLE_OBLIGATION_TEXT,
        clause=SAMPLE_CLAUSE,
        category=SAMPLE_CATEGORY,
    )

    assert isinstance(result, CoverageAssessmentResult)
    assert result.status == CoverageStatus.NONE
    assert result.confidence >= 0.7
    assert result.relevant_snippet is None or result.relevant_snippet == ""


@pytest.mark.asyncio
async def test_llm_mocked_coverage_assessment():
    """
    Test coverage assessment with mocked LLM provider returning structured JSON.
    """
    mock_llm_response = json.dumps({
        "status": "FULL",
        "confidence": 0.96,
        "reasoning": "The evidence demonstrates mandatory Okta MFA and quarterly reviews, fully meeting CC6.1.",
        "relevant_snippet": "All employees accessing company production environments must authenticate using Okta multi-factor authentication (MFA).",
    })

    service = CoverageService(provider="groq", groq_api_key="gsk_test_mock_key")

    with patch.object(service, "_call_llm", new=AsyncMock(return_value=mock_llm_response)):
        result = await service.assess_coverage(
            evidence_text=SAMPLE_EVIDENCE_STATEMENT,
            obligation_text=SAMPLE_OBLIGATION_TEXT,
            clause=SAMPLE_CLAUSE,
            category=SAMPLE_CATEGORY,
        )

        assert isinstance(result, CoverageAssessmentResult)
        assert result.status == CoverageStatus.FULL
        assert result.confidence == 0.96
        assert "meeting CC6.1" in result.reasoning
        assert "Okta multi-factor" in result.relevant_snippet


def test_coverage_result_pydantic_validation():
    """
    Test CoverageAssessmentResult Pydantic schema validation and field aliasing.
    """
    data = {
        "coverage_status": "full",
        "confidence_score": 95.0,  # percentage auto-normalized
        "explanation": "Controls are completely implemented.",
        "evidence_snippet": "Okta MFA is enforced for all logins.",
    }

    result = CoverageAssessmentResult.model_validate(data)
    assert result.status == CoverageStatus.FULL
    assert result.confidence == 0.95
    assert result.reasoning == "Controls are completely implemented."
    assert result.relevant_snippet == "Okta MFA is enforced for all logins."


def test_rule_based_coverage_assessment_direct():
    """
    Test direct rule-based fallback logic for FULL, PARTIAL, and NONE determinations.
    """
    service = CoverageService()

    # 1. Test FULL
    full_res = service.assess_coverage_rule_based(
        evidence_text=SAMPLE_EVIDENCE_STATEMENT,
        obligation_text=SAMPLE_OBLIGATION_TEXT,
        clause=SAMPLE_CLAUSE,
        category=SAMPLE_CATEGORY,
    )
    assert full_res.status == CoverageStatus.FULL
    assert full_res.confidence >= 0.70
    assert full_res.relevant_snippet is not None

    # 2. Test PARTIAL
    partial_res = service.assess_coverage_rule_based(
        evidence_text=PARTIAL_EVIDENCE_STATEMENT,
        obligation_text=SAMPLE_OBLIGATION_TEXT,
        clause=SAMPLE_CLAUSE,
        category=SAMPLE_CATEGORY,
    )
    assert partial_res.status == CoverageStatus.PARTIAL
    assert partial_res.confidence > 0.0

    # 3. Test NONE
    none_res = service.assess_coverage_rule_based(
        evidence_text=IRRELEVANT_EVIDENCE_STATEMENT,
        obligation_text=SAMPLE_OBLIGATION_TEXT,
        clause=SAMPLE_CLAUSE,
        category=SAMPLE_CATEGORY,
    )
    assert none_res.status == CoverageStatus.NONE
    assert none_res.relevant_snippet is None


# -----------------------------------------------------------------------------
# Standalone Runner Function
# -----------------------------------------------------------------------------


async def run_standalone_test() -> None:
    """Run a small standalone test comparing one evidence statement and one obligation."""
    print("=" * 70)
    print("  Phase 2 — Step 5.1: Standalone Coverage Assessment Test")
    print("=" * 70)

    service = CoverageService(fallback_enabled=True)

    print("\n[Input] Evidence Statement:")
    print(f"  \"{SAMPLE_EVIDENCE_STATEMENT}\"")

    print("\n[Input] Regulatory Obligation:")
    print(f"  Clause:    {SAMPLE_CLAUSE}")
    print(f"  Category:  {SAMPLE_CATEGORY}")
    print(f"  Statement: \"{SAMPLE_OBLIGATION_TEXT}\"")

    print("\n[Executing] Assessing coverage...")
    result = await service.assess_coverage(
        evidence_text=SAMPLE_EVIDENCE_STATEMENT,
        obligation_text=SAMPLE_OBLIGATION_TEXT,
        clause=SAMPLE_CLAUSE,
        category=SAMPLE_CATEGORY,
    )

    print("\n[Result] Coverage Assessment Output:")
    print(f"  Status:           {result.status.value}")
    print(f"  Confidence:       {result.confidence:.2%}")
    print(f"  Reasoning:        {result.reasoning}")
    print(f"  Relevant Snippet: \"{result.relevant_snippet}\"")

    assert result.status == CoverageStatus.FULL
    assert result.confidence >= 0.70
    assert result.relevant_snippet is not None

    print("\n" + "=" * 70)
    print("  STANDALONE TEST PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_standalone_test())
