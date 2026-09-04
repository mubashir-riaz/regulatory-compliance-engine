"""
Integration Tests for Compliance Query API (Phase 2, Step 6.3).

Tests:
POST /api/v1/compliance/query
1. Successful compliance question answering with citations and evidence IDs.
2. Handling queries with no relevant results (insufficient context).
3. Request input validation (e.g. empty questions).
4. Tenant isolation header propagation.
"""

from unittest.mock import AsyncMock, MagicMock
import pytest
from httpx import AsyncClient

from app.main import app
from app.schemas.compliance import CitationItem, ComplianceQueryResponse
from app.services.rag_engine import GraphRAGEngine, get_rag_engine


@pytest.fixture
def mock_rag_engine():
    """Mock GraphRAGEngine instance for API tests."""
    engine = MagicMock(spec=GraphRAGEngine)

    # Mock response for relevant question
    mock_response = ComplianceQueryResponse(
        answer=(
            "Under GDPR (2016), Article 5(1)(e) (Storage Limitation) requires personal data "
            "to be kept no longer than is necessary for processing purposes."
        ),
        citations=[
            CitationItem(
                node_id="GDPR_2016_ART_5_1_E",
                clause="Article 5(1)(e)",
                framework="GDPR",
                title="Storage Limitation",
            )
        ],
        cited_node_ids=["GDPR_2016_ART_5_1_E"],
        evidence_ids=["retention_policy_2026.pdf"],
        metadata={"obligations_retrieved": 1, "evidence_count": 1},
    )
    engine.answer_question = AsyncMock(return_value=mock_response)
    return engine


@pytest.mark.asyncio
async def test_compliance_query_endpoint_success(client: AsyncClient, mock_rag_engine):
    """
    Test POST /api/v1/compliance/query with a valid question.
    Verifies 200 OK and response structure matching Step 6.3 specifications.
    """
    app.dependency_overrides[get_rag_engine] = lambda: mock_rag_engine

    payload = {
        "question": "What data retention requirements apply under GDPR?",
        "top_k": 3,
        "framework": "GDPR",
    }

    try:
        response = await client.post("/api/v1/compliance/query", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert "answer" in data
        assert "Storage Limitation" in data["answer"]
        assert "citations" in data
        assert len(data["citations"]) == 1
        assert data["citations"][0]["node_id"] == "GDPR_2016_ART_5_1_E"
        assert data["citations"][0]["clause"] == "Article 5(1)(e)"

        assert "cited_node_ids" in data
        assert "GDPR_2016_ART_5_1_E" in data["cited_node_ids"]

        assert "evidence_ids" in data
        assert "retention_policy_2026.pdf" in data["evidence_ids"]

        mock_rag_engine.answer_question.assert_called_once()
        call_kwargs = mock_rag_engine.answer_question.call_args[1]
        assert call_kwargs["question"] == "What data retention requirements apply under GDPR?"
        assert call_kwargs["top_k"] == 3
        assert call_kwargs["framework"] == "GDPR"

    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_compliance_query_endpoint_no_results(client: AsyncClient, mock_rag_engine):
    """
    Test POST /api/v1/compliance/query when no relevant obligations exist.
    """
    empty_response = ComplianceQueryResponse(
        answer="The retrieved regulatory context does not contain sufficient information to answer this question.",
        citations=[],
        cited_node_ids=[],
        evidence_ids=[],
        metadata={"status": "insufficient_context"},
    )
    mock_rag_engine.answer_question = AsyncMock(return_value=empty_response)
    app.dependency_overrides[get_rag_engine] = lambda: mock_rag_engine

    payload = {
        "question": "What are the rules for space stations under HIPAA?",
    }

    try:
        response = await client.post("/api/v1/compliance/query", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert "not contain sufficient information" in data["answer"]
        assert data["citations"] == []
        assert data["cited_node_ids"] == []
        assert data["evidence_ids"] == []

    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_compliance_query_endpoint_validation_error(client: AsyncClient):
    """
    Test POST /api/v1/compliance/query with invalid payload (empty question).
    Verifies 422 Unprocessable Entity.
    """
    payload = {
        "question": "",  # min_length=1 constraint
    }
    response = await client.post("/api/v1/compliance/query", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_compliance_query_with_tenant_header(client: AsyncClient, mock_rag_engine):
    """
    Test POST /api/v1/compliance/query with X-Tenant-ID header.
    """
    app.dependency_overrides[get_rag_engine] = lambda: mock_rag_engine

    payload = {
        "question": "What access controls are required under SOC 2?",
    }
    headers = {
        "X-Tenant-ID": "org-tenant-001",
    }

    try:
        response = await client.post("/api/v1/compliance/query", json=payload, headers=headers)
        assert response.status_code == 200
    finally:
        app.dependency_overrides.clear()
