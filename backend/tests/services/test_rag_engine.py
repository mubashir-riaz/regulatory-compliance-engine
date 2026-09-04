"""
Unit and Integration Tests for Graph RAG Retrieval & Graph Context Expansion (Phase 2, Steps 6.1 & 6.2).

Tests:
1. Step 6.1: Qdrant semantic vector retrieval, top-k ranking, metadata extraction.
2. Step 6.2: Neo4j graph context expansion around relevant obligation IDs:
   - Traversal to RegulatoryFramework and RegulatoryVersion.
   - Traversal to ControlCategory (CATEGORIZED_AS).
   - Traversal to EvidenceArtifact (SATISFIES, coverage, confidence, reasoning).
   - Traversal to Related Obligations (DEPENDS_ON and SUPERSEDES).
3. Verification using existing sample graph from Phase 2 Step 2 (SOC 2 CC6.1).
4. Safe handling of missing nodes and empty graph results.
5. End-to-end query_and_expand pipeline.
6. Structured LLM prompt context formatting and citation source generation.
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID
import pytest

from app.services.graph_service import (
    SAMPLE_CATEGORY_ID,
    SAMPLE_DEP_OBLIGATION_ID,
    SAMPLE_EVIDENCE_ID,
    SAMPLE_FRAMEWORK_ID,
    SAMPLE_OBLIGATION_ID,
    SAMPLE_SUPERSEDED_OBLIGATION_ID,
    SAMPLE_VERSION_ID,
)
from app.services.rag_engine import (
    DEFAULT_TOP_K,
    GraphRAGContext,
    GraphRAGEngine,
    RAGGraphExpansionError,
    RAGRetrievalError,
    RetrievedObligation,
    rag_engine,
)


# -----------------------------------------------------------------------------
# Test Data Fixtures
# -----------------------------------------------------------------------------

MOCK_QDRANT_OBLIGATIONS = [
    {
        "obligation_id": str(SAMPLE_OBLIGATION_ID),
        "node_id": str(SAMPLE_OBLIGATION_ID),
        "score": 0.9450,
        "framework": "SOC 2",
        "version": "2017",
        "clause": "CC6.1",
        "category": "Access Control",
        "title": "Logical and Physical Access Controls",
        "text": "The entity implements logical access security software, infrastructure, and architectures over protected information assets.",
        "mandatory": True,
        "keywords": ["logical access", "authentication", "security"],
        "payload": {
            "obligation_id": str(SAMPLE_OBLIGATION_ID),
            "framework": "SOC 2",
            "version": "2017",
            "clause": "CC6.1",
            "category": "Access Control",
            "title": "Logical and Physical Access Controls",
            "text": "The entity implements logical access security software, infrastructure, and architectures over protected information assets.",
            "mandatory": True,
            "keywords": ["logical access", "authentication", "security"],
        },
    },
    {
        "obligation_id": str(SAMPLE_DEP_OBLIGATION_ID),
        "node_id": str(SAMPLE_DEP_OBLIGATION_ID),
        "score": 0.8820,
        "framework": "SOC 2",
        "version": "2017",
        "clause": "CC6.2",
        "category": "Access Control",
        "title": "User Registration and Access Authorization",
        "text": "Prior to issuing system credentials and granting access, the entity registers and authorizes new users.",
        "mandatory": True,
        "keywords": ["registration", "credentials", "authorization"],
        "payload": {
            "obligation_id": str(SAMPLE_DEP_OBLIGATION_ID),
            "framework": "SOC 2",
            "version": "2017",
            "clause": "CC6.2",
            "category": "Access Control",
            "title": "User Registration and Access Authorization",
            "text": "Prior to issuing system credentials and granting access, the entity registers and authorizes new users.",
            "mandatory": True,
            "keywords": ["registration", "credentials", "authorization"],
        },
    },
]

# Mock Neo4j graph expansion record for SAMPLE_OBLIGATION_ID (Phase 2 Step 2 sample graph)
MOCK_SAMPLE_GRAPH_EXPANSION_RECORD = {
    "obligation_id": str(SAMPLE_OBLIGATION_ID),
    "obligation": {
        "id": str(SAMPLE_OBLIGATION_ID),
        "code": "CC6.1",
        "title": "Logical and Physical Access Controls",
        "description": "The entity implements logical access security software, infrastructure, and architectures over protected information assets to protect them from security events.",
        "category": "Access Control",
        "mandatory": True,
        "keywords": ["access control", "security events"],
    },
    "version": {
        "id": str(SAMPLE_VERSION_ID),
        "version_slug": "2017",
        "framework_id": str(SAMPLE_FRAMEWORK_ID),
        "description": "SOC 2 Trust Services Criteria (2017 Revision)",
        "is_active": True,
        "publication_date": "2017-01-01",
    },
    "framework": {
        "id": str(SAMPLE_FRAMEWORK_ID),
        "name": "SOC 2",
        "description": "Service Organization Control 2 Trust Services Criteria",
    },
    "categories": [
        {
            "id": str(SAMPLE_CATEGORY_ID),
            "name": "Access Control",
            "code": "AC",
            "description": "Controls governing user authentication, access authorizations, and perimeter security.",
        }
    ],
    "evidence_artifacts": [
        {
            "id": str(SAMPLE_EVIDENCE_ID),
            "name": "okta_mfa_policy_2026.pdf",
            "file_path": "evidence/org-001/okta_mfa_policy_2026.pdf",
            "status": "approved",
            "coverage": "FULL",
            "coverage_status": "FULL",
            "confidence": 0.95,
            "reasoning": "Evidence verifies mandatory Okta MFA across all production access points.",
            "evidence_text": "All employees accessing company production environments must authenticate using Okta MFA.",
            "similarity_score": 0.95,
        }
    ],
    "dependencies": [
        {
            "id": str(SAMPLE_DEP_OBLIGATION_ID),
            "code": "CC6.2",
            "title": "User Registration and Access Authorization",
            "description": "Prior to issuing system credentials and granting access, the entity registers and authorizes new users.",
            "direction": "OUTGOING",
            "rel_type": "DEPENDS_ON",
            "rel_description": "Logical access control enforcement depends on verified user access authorization.",
        }
    ],
    "supersedes": [
        {
            "id": str(SAMPLE_SUPERSEDED_OBLIGATION_ID),
            "code": "CC6.1-2014",
            "title": "Logical Access Controls (2014 Criteria)",
            "description": "Legacy logical access control requirement superseded by the 2017 TSC revision.",
            "direction": "OUTGOING",
            "rel_type": "SUPERSEDES",
            "reason": "2017 Trust Services Criteria revision supersedes 2014 criteria requirement.",
        }
    ],
}


@pytest.fixture
def mock_qdrant_client():
    """Mock QdrantClient instance."""
    client = MagicMock()
    client.generate_embedding = AsyncMock(return_value=[0.1] * 768)
    client.search_similar_obligations = AsyncMock(return_value=MOCK_QDRANT_OBLIGATIONS)
    return client


@pytest.fixture
def mock_graph_service():
    """Mock GraphService instance."""
    service = MagicMock()
    service.execute_query = AsyncMock(return_value=[MOCK_SAMPLE_GRAPH_EXPANSION_RECORD])
    return service


# -----------------------------------------------------------------------------
# Step 6.1 Tests: Vector Retrieval
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_relevant_obligations_success(mock_qdrant_client):
    """
    Test Step 6.1: Successful retrieval of obligations for a user question.
    Verifies obligation IDs, Neo4j node IDs, similarity scores, and metadata.
    """
    engine = GraphRAGEngine(qdrant_client=mock_qdrant_client)

    question = "What logical access controls apply under SOC 2?"
    results = await engine.retrieve_relevant_obligations(question, top_k=2)

    assert len(results) == 2
    assert all(isinstance(r, RetrievedObligation) for r in results)

    top = results[0]
    assert top.obligation_id == str(SAMPLE_OBLIGATION_ID)
    assert top.node_id == str(SAMPLE_OBLIGATION_ID)
    assert top.clause == "CC6.1"
    assert top.framework == "SOC 2"
    assert top.score == pytest.approx(0.9450, rel=1e-3)


@pytest.mark.asyncio
async def test_retrieve_empty_question_returns_empty_list(mock_qdrant_client):
    """Test that empty questions gracefully return empty list."""
    engine = GraphRAGEngine(qdrant_client=mock_qdrant_client)
    assert await engine.retrieve_relevant_obligations("") == []
    assert await engine.retrieve_relevant_obligations("   ") == []
    assert await engine.retrieve_relevant_obligations(None) == []


# -----------------------------------------------------------------------------
# Step 6.2 Tests: Graph Context Expansion
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_graph_context_sample_graph(mock_graph_service):
    """
    Verify Phase 2 Step 6.2: Graph context expansion using Phase 2 Step 2 sample graph.

    Verifies that the obligation (CC6.1) can be expanded into its connected:
    - version (2017)
    - framework (SOC 2)
    - category (Access Control)
    - evidence (okta_mfa_policy_2026.pdf) with SATISFIES status and score
    - related dependencies (CC6.2 via DEPENDS_ON)
    - related supersedes (CC6.1-2014 via SUPERSEDES)
    """
    engine = GraphRAGEngine(graph_service=mock_graph_service)

    # Expand graph context around the sample obligation ID
    context = await engine.expand_graph_context(
        obligation_ids=[SAMPLE_OBLIGATION_ID],
        max_depth=1,
        query="What are the access control requirements under SOC 2?",
    )

    assert isinstance(context, GraphRAGContext)
    assert context.total_obligations == 1
    assert context.total_evidence == 1
    assert context.total_related == 2
    assert len(context.obligations) == 1

    expanded_ob = context.obligations[0]

    # 1. Verify Obligation Node
    assert expanded_ob.obligation.id == str(SAMPLE_OBLIGATION_ID)
    assert expanded_ob.obligation.code == "CC6.1"
    assert expanded_ob.clause == "CC6.1"
    assert expanded_ob.node_id == str(SAMPLE_OBLIGATION_ID)
    assert "logical access" in expanded_ob.obligation.description.lower()

    # 2. Verify Framework Node
    assert expanded_ob.framework is not None
    assert expanded_ob.framework.id == str(SAMPLE_FRAMEWORK_ID)
    assert expanded_ob.framework.name == "SOC 2"

    # 3. Verify Version Node
    assert expanded_ob.version is not None
    assert expanded_ob.version.id == str(SAMPLE_VERSION_ID)
    assert expanded_ob.version.version_slug == "2017"

    # 4. Verify Control Category
    assert len(expanded_ob.categories) == 1
    assert expanded_ob.categories[0].id == str(SAMPLE_CATEGORY_ID)
    assert expanded_ob.categories[0].name == "Access Control"
    assert expanded_ob.categories[0].code == "AC"

    # 5. Verify Evidence Artifact (SATISFIES)
    assert len(expanded_ob.evidence_artifacts) == 1
    ev = expanded_ob.evidence_artifacts[0]
    assert ev.id == str(SAMPLE_EVIDENCE_ID)
    assert ev.name == "okta_mfa_policy_2026.pdf"
    assert ev.coverage == "FULL"
    assert ev.confidence == 0.95
    assert ev.status == "approved"
    assert "Okta MFA" in ev.evidence_text

    # 6. Verify Related Dependencies (DEPENDS_ON)
    assert len(expanded_ob.dependencies) == 1
    dep = expanded_ob.dependencies[0]
    assert dep.id == str(SAMPLE_DEP_OBLIGATION_ID)
    assert dep.code == "CC6.2"
    assert dep.rel_type == "DEPENDS_ON"
    assert dep.direction == "OUTGOING"
    assert "registration" in dep.title.lower()

    # 7. Verify Related Supersedes (SUPERSEDES)
    assert len(expanded_ob.supersedes) == 1
    sup = expanded_ob.supersedes[0]
    assert sup.id == str(SAMPLE_SUPERSEDED_OBLIGATION_ID)
    assert sup.code == "CC6.1-2014"
    assert sup.rel_type == "SUPERSEDES"
    assert sup.direction == "OUTGOING"
    assert "2017 Trust Services Criteria revision" in sup.details


@pytest.mark.asyncio
async def test_expand_graph_context_empty_and_missing_nodes(mock_graph_service):
    """
    Test safe handling of empty inputs and missing nodes in Neo4j.
    """
    engine = GraphRAGEngine(graph_service=mock_graph_service)

    # 1. Empty obligation IDs list
    empty_context = await engine.expand_graph_context([])
    assert isinstance(empty_context, GraphRAGContext)
    assert empty_context.total_obligations == 0
    assert empty_context.obligations == []

    # 2. Obligation IDs not found in Neo4j (returns empty list)
    mock_graph_service.execute_query.return_value = []
    missing_context = await engine.expand_graph_context(["NON_EXISTENT_ID_999"])
    assert missing_context.total_obligations == 0
    assert missing_context.obligations == []


@pytest.mark.asyncio
async def test_query_and_expand_pipeline(mock_qdrant_client, mock_graph_service):
    """
    Test the complete end-to-end Step 6.1 + Step 6.2 pipeline:
    Question -> Qdrant retrieval -> Neo4j graph context expansion -> GraphRAGContext.
    """
    engine = GraphRAGEngine(
        qdrant_client=mock_qdrant_client,
        graph_service=mock_graph_service,
    )

    question = "What are the logical access controls required under SOC 2?"
    context = await engine.query_and_expand(question=question, top_k=1)

    assert isinstance(context, GraphRAGContext)
    assert context.query == question
    assert context.total_obligations == 1
    assert len(context.obligations) == 1

    # Verify retrieval score from Qdrant was preserved on the expanded context
    expanded_ob = context.obligations[0]
    assert expanded_ob.retrieval_score == pytest.approx(0.9450, rel=1e-3)
    assert expanded_ob.clause == "CC6.1"
    assert expanded_ob.framework.name == "SOC 2"


@pytest.mark.asyncio
async def test_graph_rag_context_llm_formatting_and_citations(mock_graph_service):
    """
    Test prompt context rendering and citation provenance extraction.
    """
    engine = GraphRAGEngine(graph_service=mock_graph_service)
    context = await engine.expand_graph_context(
        obligation_ids=[SAMPLE_OBLIGATION_ID],
        query="What access controls are needed?",
    )

    # 1. Test format_for_llm
    prompt_text = context.format_for_llm()
    assert "CC6.1" in prompt_text
    assert "SOC 2" in prompt_text
    assert "Access Control" in prompt_text
    assert "okta_mfa_policy_2026.pdf" in prompt_text
    assert "CC6.2" in prompt_text
    assert "CC6.1-2014" in prompt_text

    # 2. Test get_citation_sources
    citations = context.get_citation_sources()
    assert len(citations) >= 2

    # Obligation citation
    ob_cit = next(c for c in citations if c["type"] == "obligation")
    assert ob_cit["clause"] == "CC6.1"
    assert ob_cit["framework"] == "SOC 2"
    assert ob_cit["version"] == "2017"
    assert ob_cit["node_id"] == str(SAMPLE_OBLIGATION_ID)

    # Evidence citation
    ev_cit = next(c for c in citations if c["type"] == "evidence")
    assert ev_cit["name"] == "okta_mfa_policy_2026.pdf"
    assert ev_cit["coverage"] == "FULL"
    assert ev_cit["evidence_id"] == str(SAMPLE_EVIDENCE_ID)


@pytest.mark.asyncio
async def test_expand_graph_error_handling(mock_graph_service):
    """
    Test error resilience in graph context expansion.
    """
    mock_graph_service.execute_query.side_effect = ConnectionError("Neo4j database unavailable")
    engine = GraphRAGEngine(graph_service=mock_graph_service)

    # Default: returns empty GraphRAGContext gracefully
    safe_context = await engine.expand_graph_context([SAMPLE_OBLIGATION_ID], raise_on_error=False)
    assert isinstance(safe_context, GraphRAGContext)
    assert safe_context.total_obligations == 0

    # With raise_on_error=True: raises RAGGraphExpansionError
    with pytest.raises(RAGGraphExpansionError) as exc_info:
        await engine.expand_graph_context([SAMPLE_OBLIGATION_ID], raise_on_error=True)
    assert "Neo4j database unavailable" in str(exc_info.value)
