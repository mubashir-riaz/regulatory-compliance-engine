"""
Unit Tests for Celery Obligation Extraction Tasks (Phase 2, Step 3.3).
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4
import pytest

from app.schemas.extraction import ExtractedObligation
from app.workers.tasks import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    _extract_regulatory_obligations_pipeline,
    extract_regulatory_obligations_task,
    process_document_task,
)


@pytest.fixture
def sample_doc_uuid():
    return uuid4()


@pytest.fixture
def sample_ver_uuid():
    return uuid4()


@pytest.fixture
def mock_extracted_obligations():
    return [
        ExtractedObligation(
            clause="Art. 5(1)(a)",
            text="Personal data shall be processed lawfully, fairly and transparently.",
            category="Lawfulness & Transparency",
            mandatory=True,
            keywords=["personal data", "lawfulness", "fairness"],
        ),
        ExtractedObligation(
            clause="Art. 5(1)(b)",
            text="Personal data shall be collected for specified, explicit and legitimate purposes.",
            category="Purpose Limitation",
            mandatory=True,
            keywords=["purpose limitation", "specified purposes"],
        ),
    ]


@pytest.mark.asyncio
async def test_extract_obligations_pipeline_success(
    sample_doc_uuid, sample_ver_uuid, mock_extracted_obligations
):
    """Test full extraction pipeline with mocked DB, ExtractionService, and GraphService."""
    mock_artifact = MagicMock()
    mock_artifact.id = sample_doc_uuid
    mock_artifact.file_path = "evidence/gdpr.pdf"
    mock_artifact.extracted_text = "Article 5. Personal data shall be processed lawfully..."

    mock_version = MagicMock()
    mock_version.id = sample_ver_uuid
    mock_version.framework_id = uuid4()
    mock_version.version_slug = "2016"
    mock_version.description = "GDPR 2016"
    mock_version.is_active = True

    mock_req_created = MagicMock()
    mock_req_created.id = uuid4()
    mock_req_created.version_id = sample_ver_uuid
    mock_req_created.code = "Art. 5(1)(a)"

    mock_extractor = MagicMock()
    mock_extractor.extract_obligations = AsyncMock(return_value=mock_extracted_obligations)

    mock_graph = MagicMock()
    mock_graph.upsert_version = AsyncMock()
    mock_graph.upsert_obligation = AsyncMock()
    mock_graph.link_version_obligation = AsyncMock()
    mock_graph.upsert_control_category = AsyncMock()
    mock_graph.link_obligation_category = AsyncMock()
    mock_graph.link_evidence_obligation = AsyncMock()

    with patch("app.workers.tasks.AsyncSessionLocal") as mock_session_cls, \
         patch("app.workers.tasks.EvidenceArtifactRepository") as mock_evidence_repo_cls, \
         patch("app.workers.tasks.RegulatoryVersionRepository") as mock_version_repo_cls, \
         patch("app.workers.tasks.RegulatoryRequirementRepository") as mock_req_repo_cls:

        mock_session = AsyncMock()
        mock_session_cls.return_value.__aenter__.return_value = mock_session

        mock_evidence_repo = MagicMock()
        mock_evidence_repo.get_by_id = AsyncMock(return_value=mock_artifact)
        mock_evidence_repo_cls.return_value = mock_evidence_repo

        mock_version_repo = MagicMock()
        mock_version_repo.get_by_id = AsyncMock(return_value=mock_version)
        mock_version_repo_cls.return_value = mock_version_repo

        mock_req_repo = MagicMock()
        mock_req_repo.get_by_code = AsyncMock(return_value=None)
        mock_req_repo.create = AsyncMock(return_value=mock_req_created)
        mock_req_repo_cls.return_value = mock_req_repo

        result = await _extract_regulatory_obligations_pipeline(
            document_id=str(sample_doc_uuid),
            version_id=str(sample_ver_uuid),
            custom_extraction_service=mock_extractor,
            custom_graph_service=mock_graph,
        )

        assert result["status"] == STATUS_COMPLETED
        assert result["obligations_extracted"] == 2
        assert result["created_in_db"] == 2
        assert result["failed_chunks"] == 0

        # Verify Neo4j calls
        mock_graph.upsert_version.assert_called_once()
        assert mock_graph.upsert_obligation.call_count == 2
        assert mock_graph.link_version_obligation.call_count == 2
        assert mock_graph.upsert_control_category.call_count == 2
        assert mock_graph.link_obligation_category.call_count == 2


@pytest.mark.asyncio
async def test_extract_obligations_pipeline_idempotent_update(
    sample_doc_uuid, sample_ver_uuid, mock_extracted_obligations
):
    """Test that existing requirements are updated instead of creating duplicates."""
    mock_artifact = MagicMock()
    mock_artifact.id = sample_doc_uuid
    mock_artifact.extracted_text = "Article 5. Personal data..."

    mock_version = MagicMock()
    mock_version.id = sample_ver_uuid
    mock_version.framework_id = uuid4()
    mock_version.version_slug = "2016"
    mock_version.description = "GDPR"
    mock_version.is_active = True

    mock_existing_req = MagicMock()
    mock_existing_req.id = uuid4()
    mock_existing_req.code = "Art. 5(1)(a)"

    mock_extractor = MagicMock()
    mock_extractor.extract_obligations = AsyncMock(return_value=[mock_extracted_obligations[0]])

    mock_graph = MagicMock()
    mock_graph.upsert_version = AsyncMock()
    mock_graph.upsert_obligation = AsyncMock()
    mock_graph.link_version_obligation = AsyncMock()
    mock_graph.upsert_control_category = AsyncMock()
    mock_graph.link_obligation_category = AsyncMock()
    mock_graph.link_evidence_obligation = AsyncMock()

    with patch("app.workers.tasks.AsyncSessionLocal") as mock_session_cls, \
         patch("app.workers.tasks.EvidenceArtifactRepository") as mock_evidence_repo_cls, \
         patch("app.workers.tasks.RegulatoryVersionRepository") as mock_version_repo_cls, \
         patch("app.workers.tasks.RegulatoryRequirementRepository") as mock_req_repo_cls:

        mock_session = AsyncMock()
        mock_session_cls.return_value.__aenter__.return_value = mock_session

        mock_evidence_repo = MagicMock()
        mock_evidence_repo.get_by_id = AsyncMock(return_value=mock_artifact)
        mock_evidence_repo_cls.return_value = mock_evidence_repo

        mock_version_repo = MagicMock()
        mock_version_repo.get_by_id = AsyncMock(return_value=mock_version)
        mock_version_repo_cls.return_value = mock_version_repo

        mock_req_repo = MagicMock()
        mock_req_repo.get_by_code = AsyncMock(return_value=mock_existing_req)
        mock_req_repo.update = AsyncMock(return_value=mock_existing_req)
        mock_req_repo_cls.return_value = mock_req_repo

        result = await _extract_regulatory_obligations_pipeline(
            document_id=str(sample_doc_uuid),
            version_id=str(sample_ver_uuid),
            custom_extraction_service=mock_extractor,
            custom_graph_service=mock_graph,
        )

        assert result["status"] == STATUS_COMPLETED
        assert result["obligations_extracted"] == 1
        assert result["updated_in_db"] == 1
        assert result["created_in_db"] == 0
        mock_req_repo.update.assert_called_once()
        mock_req_repo.create.assert_not_called()


@pytest.mark.asyncio
async def test_extract_obligations_pipeline_doc_not_found(sample_doc_uuid, sample_ver_uuid):
    """Test pipeline handling when EvidenceArtifact is not found."""
    with patch("app.workers.tasks.AsyncSessionLocal") as mock_session_cls, \
         patch("app.workers.tasks.EvidenceArtifactRepository") as mock_evidence_repo_cls:

        mock_session = AsyncMock()
        mock_session_cls.return_value.__aenter__.return_value = mock_session

        mock_evidence_repo = MagicMock()
        mock_evidence_repo.get_by_id = AsyncMock(return_value=None)
        mock_evidence_repo_cls.return_value = mock_evidence_repo

        result = await _extract_regulatory_obligations_pipeline(
            document_id=str(sample_doc_uuid),
            version_id=str(sample_ver_uuid),
        )

        assert result["status"] == STATUS_FAILED
        assert "not found" in result["error"]


def test_extract_regulatory_obligations_task_sync_wrapper(sample_doc_uuid, sample_ver_uuid):
    """Test the synchronous Celery task wrapper."""
    with patch("app.workers.tasks._extract_regulatory_obligations_pipeline", new_callable=AsyncMock) as mock_pipe:
        mock_pipe.return_value = {"status": STATUS_COMPLETED, "obligations_extracted": 5}

        res = extract_regulatory_obligations_task(
            document_id=str(sample_doc_uuid),
            version_id=str(sample_ver_uuid),
        )

        assert res["status"] == STATUS_COMPLETED
        assert res["obligations_extracted"] == 5
        mock_pipe.assert_called_once()
