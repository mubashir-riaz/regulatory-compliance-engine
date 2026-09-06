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
    ObligationChangeType,
    ObligationComparisonItem,
    ObligationComparisonResult,
)
from app.services.extraction_service import ExtractionService
from app.services.graph_service import GraphService
from app.services.impact_analysis import (
    DraftExtractionError,
    ImpactAnalysisService,
    SAMPLE_GDPR_BASELINE_ARTICLE_5_OBLIGATIONS,
    SAMPLE_GDPR_DRAFT_ARTICLE_5_TEXT,
    compare_obligations,
    compare_with_existing_obligations,
    extract_draft_obligations,
    load_existing_obligations,
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


# =============================================================================
# Phase 2 Step 7.2 Unit Tests: Regulatory Obligation Comparison
# =============================================================================


@pytest.mark.asyncio
async def test_compare_obligations_modified():
    """
    Test Step 7.2: Classification of MODIFIED obligations.
    Verifies that when an obligation requirement changes from general necessity
    to a 12-month limit, it is classified as MODIFIED with reasoning and high confidence.
    """
    baseline_obligations = [
        {
            "id": "GDPR_2024_ART5",
            "clause": "Article 5",
            "code": "Art. 5",
            "text": "Data should only be retained as long as necessary.",
            "category": "Data Retention",
            "mandatory": True,
        }
    ]

    draft_obligations = [
        {
            "id": "DRAFT_ART5",
            "clause": "Article 5",
            "text": "Data must be deleted within 12 months.",
            "category": "Data Retention",
            "mandatory": True,
        }
    ]

    service = ImpactAnalysisService()
    result = await service.compare_obligations(
        draft_obligations=draft_obligations,
        existing_obligations=baseline_obligations,
        framework="GDPR",
        baseline_version="2016",
        draft_version="2024-draft",
    )

    assert isinstance(result, ObligationComparisonResult)
    assert result.summary["MODIFIED"] == 1
    assert result.summary["ADDED"] == 0
    assert result.summary["REMOVED"] == 0
    assert result.summary["UNCHANGED"] == 0
    assert len(result.modified) == 1

    item = result.modified[0]
    assert item.change_type == ObligationChangeType.MODIFIED
    assert item.old_obligation_id == "GDPR_2024_ART5"
    assert item.new_clause == "Article 5"
    assert "12-month" in item.reason or "retention" in item.reason.lower()
    assert item.confidence >= 0.90


@pytest.mark.asyncio
async def test_compare_obligations_removed():
    """
    Test Step 7.2: Classification of REMOVED obligations.
    Verifies that when a baseline obligation has no matching draft obligation,
    it is classified as REMOVED.
    """
    baseline_obligations = [
        {
            "id": "GDPR_LOGS_01",
            "clause": "Article 5(3)",
            "text": "Organizations must maintain audit logs.",
            "category": "Audit Logging",
            "mandatory": True,
        }
    ]

    # No equivalent obligation in draft
    draft_obligations = []

    service = ImpactAnalysisService()
    result = await service.compare_obligations(
        draft_obligations=draft_obligations,
        existing_obligations=baseline_obligations,
        framework="GDPR",
        baseline_version="2016",
        draft_version="2024-draft",
    )

    assert result.summary["REMOVED"] == 1
    assert result.summary["MODIFIED"] == 0
    assert result.summary["ADDED"] == 0
    assert len(result.removed) == 1

    item = result.removed[0]
    assert item.change_type == ObligationChangeType.REMOVED
    assert item.old_obligation_id == "GDPR_LOGS_01"
    assert item.old_clause == "Article 5(3)"
    assert item.new_clause is None
    assert item.confidence >= 0.90
    assert "removed" in item.reason.lower() or "no matching" in item.reason.lower()


@pytest.mark.asyncio
async def test_compare_obligations_added():
    """
    Test Step 7.2: Classification of ADDED obligations.
    Verifies that when a draft obligation has no matching baseline obligation,
    it is classified as ADDED.
    """
    baseline_obligations = []

    # New draft obligation with no prior baseline match
    draft_obligations = [
        {
            "id": "DRAFT_NOTIF_01",
            "clause": "Article 33(1)",
            "text": "Organizations must notify users within 48 hours.",
            "category": "Incident Response",
            "mandatory": True,
        }
    ]

    service = ImpactAnalysisService()
    result = await service.compare_obligations(
        draft_obligations=draft_obligations,
        existing_obligations=baseline_obligations,
        framework="GDPR",
        baseline_version="2016",
        draft_version="2024-draft",
    )

    assert result.summary["ADDED"] == 1
    assert result.summary["MODIFIED"] == 0
    assert result.summary["REMOVED"] == 0
    assert len(result.added) == 1

    item = result.added[0]
    assert item.change_type == ObligationChangeType.ADDED
    assert item.new_clause == "Article 33(1)"
    assert item.old_obligation_id is None
    assert item.confidence >= 0.90
    assert "new" in item.reason.lower() or "introduced" in item.reason.lower()


@pytest.mark.asyncio
async def test_compare_obligations_unchanged_identical_and_minor_wording():
    """
    Test Step 7.2: Classification of UNCHANGED obligations.
    Verifies that:
    1. Identical text produces UNCHANGED with 1.0 confidence.
    2. Minor stylistic / wording differences (e.g. 'shall' -> 'must', minor phrasing)
       without changing regulatory meaning are classified as UNCHANGED.
    """
    baseline_obligations = [
        {
            "id": "OB_EXACT",
            "clause": "Article 5(1)(b)",
            "text": "Personal data shall be collected for specified, explicit and legitimate purposes.",
            "category": "Purpose Limitation",
            "mandatory": True,
        },
        {
            "id": "OB_MINOR",
            "clause": "Article 5(1)(a)",
            "text": "Personal data shall be processed lawfully, fairly and in a transparent manner in relation to the data subject.",
            "category": "Lawfulness & Transparency",
            "mandatory": True,
        },
    ]

    draft_obligations = [
        {
            "id": "DRAFT_EXACT",
            "clause": "Article 5(1)(b)",
            "text": "Personal data shall be collected for specified, explicit and legitimate purposes.",
            "category": "Purpose Limitation",
            "mandatory": True,
        },
        {
            "id": "DRAFT_MINOR",
            "clause": "Article 5(1)(a)",
            "text": "Personal data must be processed lawfully, fairly and in a transparent manner in relation to the data subject.",
            "category": "Lawfulness & Transparency",
            "mandatory": True,
        },
    ]

    service = ImpactAnalysisService()
    result = await service.compare_obligations(
        draft_obligations=draft_obligations,
        existing_obligations=baseline_obligations,
    )

    assert result.summary["UNCHANGED"] == 2
    assert result.summary["MODIFIED"] == 0
    assert len(result.unchanged) == 2

    # Check exact match
    exact_item = next(c for c in result.unchanged if c.new_clause == "Article 5(1)(b)")
    assert exact_item.change_type == ObligationChangeType.UNCHANGED
    assert exact_item.confidence == 1.0

    # Check minor wording match
    minor_item = next(c for c in result.unchanged if c.new_clause == "Article 5(1)(a)")
    assert minor_item.change_type == ObligationChangeType.UNCHANGED
    assert minor_item.confidence >= 0.90
    assert "minor wording" in minor_item.reason.lower() or "identical" in minor_item.reason.lower() or "stylistic" in minor_item.reason.lower()


@pytest.mark.asyncio
async def test_compare_obligations_semantic_matching_clause_changed():
    """
    Test Step 7.2: Semantic similarity matching when clause identifiers changed.
    Verifies that when exact clause matching is unavailable, obligations with
    equivalent domain concepts are paired via semantic similarity.
    """
    baseline_obligations = [
        {
            "id": "SOC2_OLD_ENC",
            "clause": "Section 4.1",
            "code": "Sec. 4.1",
            "text": "All sensitive customer data must be encrypted at rest and in transit using strong cryptography.",
            "category": "Encryption",
            "keywords": ["encryption", "cryptography", "sensitive data"],
            "mandatory": True,
        }
    ]

    draft_obligations = [
        {
            "id": "SOC2_NEW_ENC",
            "clause": "Control CC-7.4",  # Completely different clause code
            "text": "All sensitive customer data must be encrypted at rest and in transit using strong cryptography.",
            "category": "Encryption",
            "keywords": ["encryption", "cryptography", "customer data"],
            "mandatory": True,
        }
    ]

    service = ImpactAnalysisService()
    result = await service.compare_obligations(
        draft_obligations=draft_obligations,
        existing_obligations=baseline_obligations,
        similarity_threshold=0.60,
    )

    assert len(result.changes) == 1
    item = result.changes[0]
    # Matched via semantic similarity despite different clause IDs
    assert item.old_obligation_id == "SOC2_OLD_ENC"
    assert item.new_clause == "Control CC-7.4"
    assert item.change_type == ObligationChangeType.UNCHANGED
    assert item.metadata.get("matching_method") == "semantic_similarity"
    assert item.similarity_score >= 0.70


@pytest.mark.asyncio
async def test_load_existing_obligations_from_graph():
    """
    Test Step 7.2: Loading existing baseline obligations from Neo4j graph.
    """
    mock_graph = MagicMock(spec=GraphService)
    mock_records = [
        {
            "id": "REC-01",
            "code": "Art. 5(1)(a)",
            "clause": "Article 5(1)(a)",
            "title": "Lawfulness",
            "text": "Personal data shall be processed lawfully.",
            "category": "Lawfulness",
            "mandatory": True,
            "keywords": ["lawfulness"],
            "framework": "GDPR",
            "version": "2016",
        }
    ]
    mock_graph.execute_query = AsyncMock(return_value=mock_records)

    service = ImpactAnalysisService(graph_service=mock_graph)
    loaded = await service.load_existing_obligations(
        framework="GDPR",
        version="2016",
    )

    assert len(loaded) == 1
    assert loaded[0]["id"] == "REC-01"
    assert loaded[0]["clause"] == "Article 5(1)(a)"
    assert loaded[0]["text"] == "Personal data shall be processed lawfully."
    mock_graph.execute_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_compare_with_existing_obligations_comprehensive_matrix():
    """
    Test Step 7.2 end-to-end: Verifies simultaneous detection of
    ADDED, MODIFIED, REMOVED, and UNCHANGED in a single regulatory comparison.
    """
    baseline_obligations = [
        # Will be MODIFIED
        {
            "id": "GDPR_2024_ART5",
            "clause": "Article 5(1)(e)",
            "text": "Data should only be retained as long as necessary.",
            "category": "Data Retention",
            "mandatory": True,
        },
        # Will be UNCHANGED
        {
            "id": "GDPR_ART5_A",
            "clause": "Article 5(1)(a)",
            "text": "Personal data shall be processed lawfully, fairly and transparently.",
            "category": "Transparency",
            "mandatory": True,
        },
        # Will be REMOVED
        {
            "id": "GDPR_LOGS",
            "clause": "Article 5(3)",
            "text": "Organizations must maintain audit logs.",
            "category": "Audit Logs",
            "mandatory": True,
        },
    ]

    draft_obligations = [
        # MODIFIED match
        {
            "id": "DRAFT_5_E",
            "clause": "Article 5(1)(e)",
            "text": "Data must be deleted within 12 months.",
            "category": "Data Retention",
            "mandatory": True,
        },
        # UNCHANGED match
        {
            "id": "DRAFT_5_A",
            "clause": "Article 5(1)(a)",
            "text": "Personal data shall be processed lawfully, fairly and transparently.",
            "category": "Transparency",
            "mandatory": True,
        },
        # ADDED (no baseline equivalent)
        {
            "id": "DRAFT_NOTIF",
            "clause": "Article 33",
            "text": "Organizations must notify users within 48 hours.",
            "category": "Notification",
            "mandatory": True,
        },
    ]

    result = await compare_with_existing_obligations(
        draft_input=draft_obligations,
        existing_obligations=baseline_obligations,
        framework="GDPR",
        baseline_version="2016",
        draft_version="2024-draft",
    )

    assert isinstance(result, ObligationComparisonResult)
    assert result.summary["ADDED"] == 1
    assert result.summary["MODIFIED"] == 1
    assert result.summary["REMOVED"] == 1
    assert result.summary["UNCHANGED"] == 1
    assert result.summary["TOTAL"] == 4

    # Verify each category
    assert len(result.added) == 1
    assert result.added[0].new_clause == "Article 33"

    assert len(result.modified) == 1
    assert result.modified[0].old_obligation_id == "GDPR_2024_ART5"
    assert result.modified[0].new_clause == "Article 5(1)(e)"
    assert result.modified[0].confidence == 0.94 or result.modified[0].confidence >= 0.90

    assert len(result.removed) == 1
    assert result.removed[0].old_obligation_id == "GDPR_LOGS"

    assert len(result.unchanged) == 1
    assert result.unchanged[0].new_clause == "Article 5(1)(a)"
    assert result.unchanged[0].confidence == 1.0

