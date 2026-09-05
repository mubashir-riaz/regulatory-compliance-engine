"""
Unit Tests for Draft Regulation Obligation Extraction (Phase 2, Step 7.1).

Tests the ImpactAnalysisService for accepting draft regulatory texts, extracting
structured obligations via ExtractionService, preserving required metadata,
handling empty inputs gracefully, and error handling.
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID
import pytest

from app.schemas.extraction import ExtractedObligation
from app.schemas.impact import (
    DraftExtractionResult,
    DraftObligation,
    DraftRegulationInput,
)
from app.services.extraction_service import ExtractionService
from app.services.impact_analysis import (
    DraftExtractionError,
    ImpactAnalysisService,
    SAMPLE_GDPR_DRAFT_ARTICLE_5_TEXT,
    extract_draft_obligations,
)

# Short modified regulatory text (Amended GDPR Article 5(1)(e) data retention)
SHORT_MODIFIED_GDPR_TEXT = """
Article 5(1)(e) - Storage Limitation Amendment
Personal data must not be retained for longer than a maximum of 12 months from the date of collection.
Controllers must implement automated purging procedures to ensure expired personal data is deleted.
""".strip()

MOCK_EXTRACTED_DRAFT_OBLIGATIONS = [
    ExtractedObligation(
        clause="Article 5(1)(e)",
        text="Personal data must not be retained for longer than a maximum of 12 months from the date of collection.",
        category="Data Retention",
        mandatory=True,
        keywords=["personal data", "retention", "12 months", "storage limitation"],
    ),
    ExtractedObligation(
        clause="Article 5(1)(e)-bis",
        text="Controllers must implement automated purging procedures to ensure expired personal data is deleted.",
        category="Data Deletion & Purging",
        mandatory=True,
        keywords=["purging", "deletion", "automated procedures", "controller"],
    ),
]


@pytest.mark.asyncio
async def test_extract_draft_obligations_short_modified_text():
    """
    Test Phase 2 Step 7.1: Extracting obligations from a short modified regulatory draft.
    Verifies that clause, text, category, mandatory, keywords, framework, version,
    and a generated unique ID are preserved for later comparison.
    """
    mock_extractor = MagicMock(spec=ExtractionService)
    mock_extractor.extract_obligations = AsyncMock(return_value=MOCK_EXTRACTED_DRAFT_OBLIGATIONS)

    service = ImpactAnalysisService(extraction_service=mock_extractor)

    result = await service.extract_draft_obligations(
        draft_input=SHORT_MODIFIED_GDPR_TEXT,
        framework="GDPR",
        version="2024-draft",
    )

    assert isinstance(result, DraftExtractionResult)
    assert result.framework == "GDPR"
    assert result.version == "2024-draft"
    assert result.total_obligations == 2
    assert len(result.obligations) == 2

    # Verify first obligation structure and metadata preservation
    ob1 = result.obligations[0]
    assert isinstance(ob1, DraftObligation)
    assert isinstance(ob1.id, UUID)
    assert ob1.clause == "Article 5(1)(e)"
    assert "12 months" in ob1.text
    assert ob1.category == "Data Retention"
    assert ob1.mandatory is True
    assert "retention" in ob1.keywords
    assert ob1.framework == "GDPR"
    assert ob1.version == "2024-draft"
    assert ob1.source_text is not None

    # Verify second obligation
    ob2 = result.obligations[1]
    assert ob2.clause == "Article 5(1)(e)-bis"
    assert "automated purging" in ob2.text
    assert ob2.category == "Data Deletion & Purging"
    assert ob2.mandatory is True

    # Verify mock was called with the short modified text
    mock_extractor.extract_obligations.assert_awaited_once_with(
        text=SHORT_MODIFIED_GDPR_TEXT,
        provider=None,
        model=None,
    )


@pytest.mark.asyncio
async def test_extract_draft_obligations_with_input_model():
    """
    Test extraction using DraftRegulationInput Pydantic model with framework and version.
    """
    mock_extractor = MagicMock(spec=ExtractionService)
    mock_extractor.extract_obligations = AsyncMock(return_value=MOCK_EXTRACTED_DRAFT_OBLIGATIONS[:1])

    service = ImpactAnalysisService(extraction_service=mock_extractor)

    draft_input = DraftRegulationInput(
        draft_text=SHORT_MODIFIED_GDPR_TEXT,
        framework="GDPR",
        version="2024-amendment",
        metadata={"author": "EU Commission", "stage": "first-reading"},
    )

    result = await service.extract_draft_obligations(draft_input)

    assert result.framework == "GDPR"
    assert result.version == "2024-amendment"
    assert result.total_obligations == 1
    assert result.obligations[0].clause == "Article 5(1)(e)"
    assert result.obligations[0].metadata.get("author") == "EU Commission"


@pytest.mark.asyncio
async def test_extract_draft_obligations_empty_or_whitespace():
    """
    Test graceful handling of empty or blank draft text without errors.
    """
    mock_extractor = MagicMock(spec=ExtractionService)
    mock_extractor.extract_obligations = AsyncMock()

    service = ImpactAnalysisService(extraction_service=mock_extractor)

    # Empty string
    res_empty = await service.extract_draft_obligations(
        draft_input="",
        framework="GDPR",
        version="draft-v1",
    )
    assert res_empty.total_obligations == 0
    assert res_empty.obligations == []
    assert res_empty.raw_text_length == 0
    assert res_empty.metadata.get("status") == "empty_or_invalid_text"

    # Whitespace only
    res_whitespace = await service.extract_draft_obligations(
        draft_input="   \n\t  ",
        framework="GDPR",
    )
    assert res_whitespace.total_obligations == 0
    assert res_whitespace.obligations == []

    # Extractor should not have been called for empty input
    mock_extractor.extract_obligations.assert_not_called()


@pytest.mark.asyncio
async def test_extract_draft_obligations_chunking():
    """
    Test that large draft texts exceeding chunk_size are properly chunked and extracted.
    """
    # Create text longer than 200 characters with small chunk_size
    large_text = (
        "Article 1. Scope and subject matter of this regulation.\n\n"
        "Article 2. Material scope of personal data processing activities.\n\n"
        "Article 3. Territorial scope of applicable privacy obligations.\n\n"
    )
    mock_extractor = MagicMock(spec=ExtractionService)
    mock_extractor.extract_obligations = AsyncMock(
        side_effect=[
            [MOCK_EXTRACTED_DRAFT_OBLIGATIONS[0]],
            [MOCK_EXTRACTED_DRAFT_OBLIGATIONS[1]],
        ]
    )

    service = ImpactAnalysisService(
        extraction_service=mock_extractor,
        chunk_size=100,
        chunk_overlap=20,
    )

    result = await service.extract_draft_obligations(
        draft_input=large_text,
        framework="GDPR",
        version="v2-draft",
    )

    assert mock_extractor.extract_obligations.await_count >= 2
    assert result.total_obligations == 2
    assert result.metadata["total_chunks"] >= 2


@pytest.mark.asyncio
async def test_extract_draft_obligations_error_handling():
    """
    Test error handling when underlying extraction service raises an exception.
    """
    mock_extractor = MagicMock(spec=ExtractionService)
    mock_extractor.extract_obligations = AsyncMock(side_effect=RuntimeError("LLM API Timeout"))

    service = ImpactAnalysisService(extraction_service=mock_extractor)

    # 1. Non-raising mode (default): logs and records failure in metadata
    result = await service.extract_draft_obligations(
        draft_input=SHORT_MODIFIED_GDPR_TEXT,
        raise_on_error=False,
    )
    assert result.total_obligations == 0
    assert result.metadata["failed_chunks"] == 1
    assert "LLM API Timeout" in str(result.metadata.get("extraction_errors", []))

    # 2. Raising mode: raises DraftExtractionError
    with pytest.raises(DraftExtractionError) as exc_info:
        await service.extract_draft_obligations(
            draft_input=SHORT_MODIFIED_GDPR_TEXT,
            raise_on_error=True,
        )
    assert "LLM API Timeout" in str(exc_info.value)


@pytest.mark.asyncio
async def test_convenience_function_extract_draft_obligations():
    """
    Test the top-level convenience function `extract_draft_obligations`.
    """
    # Empty input test with convenience function
    result = await extract_draft_obligations(
        draft_input="",
        framework="ISO 27001",
        version="2024-draft",
    )
    assert result.total_obligations == 0
    assert result.framework == "ISO 27001"
