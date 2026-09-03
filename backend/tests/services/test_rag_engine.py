"""
Unit and Integration Tests for Graph RAG Retrieval Service (Phase 2, Step 6.1).

Tests Qdrant vector retrieval for regulatory obligations:
1. Retrieval with user questions and top-k ranking.
2. Neo4j Node ID and metadata extraction.
3. Configurable top_k and filter options.
4. Graceful handling of empty queries and empty vector store results.
5. Error handling and exception resilience.
6. Local deterministic embedding generation integration.
"""

from unittest.mock import AsyncMock, MagicMock
import pytest

from app.services.rag_engine import (
    DEFAULT_TOP_K,
    GraphRAGEngine,
    RAGRetrievalError,
    RetrievedObligation,
    rag_engine,
)


# Sample mock data for GDPR obligations
MOCK_QDRANT_OBLIGATIONS = [
    {
        "obligation_id": "GDPR_2026_ART_5_1_E",
        "node_id": "GDPR_2026_ART_5_1_E",
        "score": 0.9142,
        "framework": "GDPR",
        "version": "2016",
        "clause": "Article 5(1)(e)",
        "category": "Storage Limitation",
        "title": "Storage Limitation & Data Retention",
        "text": "Personal data shall be kept in a form which permits identification of data subjects for no longer than is necessary.",
        "mandatory": True,
        "keywords": ["storage limitation", "retention", "data subjects"],
        "payload": {
            "obligation_id": "GDPR_2026_ART_5_1_E",
            "framework": "GDPR",
            "version": "2016",
            "clause": "Article 5(1)(e)",
            "category": "Storage Limitation",
            "title": "Storage Limitation & Data Retention",
            "text": "Personal data shall be kept in a form which permits identification of data subjects for no longer than is necessary.",
            "mandatory": True,
            "keywords": ["storage limitation", "retention", "data subjects"],
        },
    },
    {
        "obligation_id": "GDPR_2026_ART_17",
        "node_id": "GDPR_2026_ART_17",
        "score": 0.8715,
        "framework": "GDPR",
        "version": "2016",
        "clause": "Article 17",
        "category": "Data Subject Rights",
        "title": "Right to Erasure ('Right to be Forgotten')",
        "text": "The data subject shall have the right to obtain from the controller the erasure of personal data concerning him or her without undue delay.",
        "mandatory": True,
        "keywords": ["erasure", "right to be forgotten", "retention period"],
        "payload": {
            "obligation_id": "GDPR_2026_ART_17",
            "framework": "GDPR",
            "version": "2016",
            "clause": "Article 17",
            "category": "Data Subject Rights",
            "title": "Right to Erasure ('Right to be Forgotten')",
            "text": "The data subject shall have the right to obtain from the controller the erasure of personal data concerning him or her without undue delay.",
            "mandatory": True,
            "keywords": ["erasure", "right to be forgotten", "retention period"],
        },
    },
]


@pytest.fixture
def mock_qdrant_client():
    """Mock QdrantClient instance."""
    client = MagicMock()
    client.generate_embedding = AsyncMock(return_value=[0.1] * 768)
    client.search_similar_obligations = AsyncMock(return_value=MOCK_QDRANT_OBLIGATIONS)
    return client


# -----------------------------------------------------------------------------
# Unit Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_relevant_obligations_success(mock_qdrant_client):
    """
    Test successful retrieval of obligations for a user question.
    Verifies obligation IDs, Neo4j node IDs, similarity scores, and metadata.
    """
    engine = GraphRAGEngine(qdrant_client=mock_qdrant_client)

    question = "What are the GDPR requirements for data retention?"
    results = await engine.retrieve_relevant_obligations(question, top_k=2)

    assert len(results) == 2
    assert all(isinstance(r, RetrievedObligation) for r in results)

    # Verify top result
    top = results[0]
    assert top.obligation_id == "GDPR_2026_ART_5_1_E"
    assert top.node_id == "GDPR_2026_ART_5_1_E"
    assert top.clause == "Article 5(1)(e)"
    assert top.category == "Storage Limitation"
    assert top.framework == "GDPR"
    assert top.score == pytest.approx(0.9142, rel=1e-3)
    assert top.mandatory is True
    assert "retention" in top.keywords

    # Verify second result
    second = results[1]
    assert second.obligation_id == "GDPR_2026_ART_17"
    assert second.node_id == "GDPR_2026_ART_17"
    assert second.clause == "Article 17"
    assert second.score == pytest.approx(0.8715, rel=1e-3)

    # Verify existing integration code was reused
    mock_qdrant_client.generate_embedding.assert_called_once_with(question)
    mock_qdrant_client.search_similar_obligations.assert_called_once()
    call_kwargs = mock_qdrant_client.search_similar_obligations.call_args[1]
    assert call_kwargs["limit"] == 2
    assert call_kwargs["query_text"] == question


@pytest.mark.asyncio
async def test_retrieve_configurable_top_k(mock_qdrant_client):
    """
    Test that top_k is configurable and defaults to 5.
    """
    engine = GraphRAGEngine(qdrant_client=mock_qdrant_client)

    # Default top_k
    await engine.retrieve_relevant_obligations("What are the access controls?")
    call_kwargs_default = mock_qdrant_client.search_similar_obligations.call_args[1]
    assert call_kwargs_default["limit"] == DEFAULT_TOP_K

    # Custom top_k = 10
    await engine.retrieve_relevant_obligations("What are the access controls?", top_k=10)
    call_kwargs_custom = mock_qdrant_client.search_similar_obligations.call_args[1]
    assert call_kwargs_custom["limit"] == 10


@pytest.mark.asyncio
async def test_retrieve_empty_question_returns_empty_list(mock_qdrant_client):
    """
    Test that empty or whitespace-only questions gracefully return empty list.
    """
    engine = GraphRAGEngine(qdrant_client=mock_qdrant_client)

    assert await engine.retrieve_relevant_obligations("") == []
    assert await engine.retrieve_relevant_obligations("   ") == []
    assert await engine.retrieve_relevant_obligations(None) == []

    # Verify Qdrant search was never called for blank queries
    mock_qdrant_client.search_similar_obligations.assert_not_called()


@pytest.mark.asyncio
async def test_retrieve_empty_qdrant_results_returns_empty_list(mock_qdrant_client):
    """
    Test that empty search results from Qdrant are handled gracefully.
    """
    mock_qdrant_client.search_similar_obligations.return_value = []
    engine = GraphRAGEngine(qdrant_client=mock_qdrant_client)

    results = await engine.retrieve_relevant_obligations("Non-existent obligation query")
    assert results == []


@pytest.mark.asyncio
async def test_retrieve_with_filters(mock_qdrant_client):
    """
    Test metadata filters (framework, version, category, score_threshold).
    """
    engine = GraphRAGEngine(qdrant_client=mock_qdrant_client)

    await engine.retrieve_relevant_obligations(
        question="What are the access control requirements?",
        top_k=3,
        framework="SOC 2",
        version="2017",
        category="Access Control",
        score_threshold=0.80,
    )

    call_kwargs = mock_qdrant_client.search_similar_obligations.call_args[1]
    assert call_kwargs["framework"] == "SOC 2"
    assert call_kwargs["version"] == "2017"
    assert call_kwargs["category"] == "Access Control"
    assert call_kwargs["score_threshold"] == 0.80
    assert call_kwargs["limit"] == 3


@pytest.mark.asyncio
async def test_extract_node_ids_and_get_top_node_ids(mock_qdrant_client):
    """
    Test extraction of Neo4j node IDs from retrieved obligations.
    """
    engine = GraphRAGEngine(qdrant_client=mock_qdrant_client)

    # 1. Test extract_node_ids directly
    retrieved = await engine.retrieve_relevant_obligations("Test retention question")
    node_ids = engine.extract_node_ids(retrieved)
    assert node_ids == ["GDPR_2026_ART_5_1_E", "GDPR_2026_ART_17"]

    # 2. Test get_top_obligation_node_ids convenience method
    direct_node_ids = await engine.get_top_obligation_node_ids("Test retention question", top_k=2)
    assert direct_node_ids == ["GDPR_2026_ART_5_1_E", "GDPR_2026_ART_17"]


@pytest.mark.asyncio
async def test_retrieved_obligation_accessors():
    """
    Test dict conversion and subscript access on RetrievedObligation model.
    """
    ob = RetrievedObligation(
        obligation_id="OBL-101",
        node_id="OBL-101",
        score=0.95,
        framework="ISO 27001",
        clause="A.9.1.1",
        category="Access Control",
        title="Access Control Policy",
        text="An access control policy shall be established.",
    )

    # Subscript access
    assert ob["obligation_id"] == "OBL-101"
    assert ob["node_id"] == "OBL-101"
    assert ob["clause"] == "A.9.1.1"
    assert ob["score"] == 0.95

    # to_dict conversion
    d = ob.to_dict()
    assert isinstance(d, dict)
    assert d["node_id"] == "OBL-101"
    assert d["framework"] == "ISO 27001"


@pytest.mark.asyncio
async def test_exception_handling(mock_qdrant_client):
    """
    Test exception handling: graceful default vs raise_on_error.
    """
    mock_qdrant_client.search_similar_obligations.side_effect = ConnectionError("Qdrant unreachable")
    engine = GraphRAGEngine(qdrant_client=mock_qdrant_client)

    # Default: returns [] gracefully without raising
    results = await engine.retrieve_relevant_obligations("Query when Qdrant is down")
    assert results == []

    # With raise_on_error=True: raises RAGRetrievalError
    with pytest.raises(RAGRetrievalError) as exc_info:
        await engine.retrieve_relevant_obligations("Query when Qdrant is down", raise_on_error=True)
    assert "Qdrant unreachable" in str(exc_info.value)
