"""
Unit and Integration Tests for Graph RAG Retrieval, Context Expansion & Answer Synthesis (Phase 2, Step 6).

Tests:
1. Step 6.1: Qdrant semantic vector retrieval, top-k ranking, metadata extraction.
2. Step 6.2: Neo4j graph context expansion around relevant obligation IDs:
   - Traversal to RegulatoryFramework and RegulatoryVersion.
   - Traversal to ControlCategory (CATEGORIZED_AS).
   - Traversal to EvidenceArtifact (SATISFIES, coverage, confidence, reasoning).
   - Traversal to Related Obligations (DEPENDS_ON and SUPERSEDES).
   - Sample graph verification using Phase 2 Step 2 sample graph (SOC 2 CC6.1).
3. Step 6.3: LLM answer synthesis, citations, and grounding:
   - A question with relevant results.
   - A question with no relevant results (insufficient context).
   - Verification that returned citations correspond to actual graph nodes.
   - Verification that the answer is generated from retrieved context rather than arbitrary information.
   - Fallback answer generation when LLM is unavailable.
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID
import pytest

from app.schemas.compliance import (
    CitationItem,
    ComplianceQueryRequest,
    ComplianceQueryResponse,
)
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

    # 5. Verify Evidence Artifact (SATISFIES)
    assert len(expanded_ob.evidence_artifacts) == 1
    ev = expanded_ob.evidence_artifacts[0]
    assert ev.id == str(SAMPLE_EVIDENCE_ID)
    assert ev.name == "okta_mfa_policy_2026.pdf"
    assert ev.coverage == "FULL"
    assert ev.confidence == 0.95
    assert "Okta MFA" in ev.evidence_text

    # 6. Verify Related Dependencies (DEPENDS_ON)
    assert len(expanded_ob.dependencies) == 1
    dep = expanded_ob.dependencies[0]
    assert dep.id == str(SAMPLE_DEP_OBLIGATION_ID)
    assert dep.code == "CC6.2"
    assert dep.rel_type == "DEPENDS_ON"

    # 7. Verify Related Supersedes (SUPERSEDES)
    assert len(expanded_ob.supersedes) == 1
    sup = expanded_ob.supersedes[0]
    assert sup.id == str(SAMPLE_SUPERSEDED_OBLIGATION_ID)
    assert sup.code == "CC6.1-2014"
    assert sup.rel_type == "SUPERSEDES"


@pytest.mark.asyncio
async def test_expand_graph_context_empty_and_missing_nodes(mock_graph_service):
    """Test safe handling of empty inputs and missing nodes in Neo4j."""
    engine = GraphRAGEngine(graph_service=mock_graph_service)

    # Empty obligation IDs list
    empty_context = await engine.expand_graph_context([])
    assert isinstance(empty_context, GraphRAGContext)
    assert empty_context.total_obligations == 0
    assert empty_context.obligations == []

    # Obligation IDs not found in Neo4j
    mock_graph_service.execute_query.return_value = []
    missing_context = await engine.expand_graph_context(["NON_EXISTENT_ID_999"])
    assert missing_context.total_obligations == 0
    assert missing_context.obligations == []


@pytest.mark.asyncio
async def test_expand_graph_error_handling(mock_graph_service):
    """Test error resilience in graph context expansion."""
    mock_graph_service.execute_query.side_effect = ConnectionError("Neo4j database unavailable")
    engine = GraphRAGEngine(graph_service=mock_graph_service)

    # Default: returns empty GraphRAGContext gracefully
    safe_context = await engine.expand_graph_context([SAMPLE_OBLIGATION_ID], raise_on_error=False)
    assert isinstance(safe_context, GraphRAGContext)
    assert safe_context.total_obligations == 0

    # With raise_on_error=True: raises RAGGraphExpansionError
    with pytest.raises(RAGGraphExpansionError):
        await engine.expand_graph_context([SAMPLE_OBLIGATION_ID], raise_on_error=True)


# -----------------------------------------------------------------------------
# Step 6.3 Tests: LLM Answer Synthesis, Citations, and Grounding
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_answer_question_with_relevant_results(mock_qdrant_client, mock_graph_service):
    """
    Test Step 6.3: A question with relevant results.
    Verifies that a grounded compliance answer is generated containing
    citations, cited node IDs, and evidence IDs.
    """
    engine = GraphRAGEngine(
        qdrant_client=mock_qdrant_client,
        graph_service=mock_graph_service,
        fallback_enabled=True,
    )

    question = "What logical access controls are required under SOC 2?"
    response = await engine.answer_question(question=question, top_k=2)

    assert isinstance(response, ComplianceQueryResponse)
    assert len(response.answer) > 20
    assert "CC6.1" in response.answer or "SOC 2" in response.answer

    # Verify citations
    assert len(response.citations) >= 1
    assert any(c.clause == "CC6.1" for c in response.citations)
    assert str(SAMPLE_OBLIGATION_ID) in response.cited_node_ids

    # Verify evidence IDs
    assert str(SAMPLE_EVIDENCE_ID) in response.evidence_ids


@pytest.mark.asyncio
async def test_answer_question_with_no_relevant_results(mock_qdrant_client, mock_graph_service):
    """
    Test Step 6.3: A question with no relevant results.
    Verifies that the engine clearly states that the retrieved context is insufficient
    and returns empty citations and evidence IDs without hallucinating.
    """
    mock_qdrant_client.search_similar_obligations.return_value = []
    mock_graph_service.execute_query.return_value = []

    engine = GraphRAGEngine(
        qdrant_client=mock_qdrant_client,
        graph_service=mock_graph_service,
        fallback_enabled=True,
    )

    question = "What are the quantum computing data transfer rules under GDPR?"
    response = await engine.answer_question(question=question)

    assert isinstance(response, ComplianceQueryResponse)
    assert "not contain sufficient information" in response.answer.lower() or "no matching" in response.answer.lower()
    assert response.citations == []
    assert response.cited_node_ids == []
    assert response.evidence_ids == []


@pytest.mark.asyncio
async def test_citations_correspond_to_actual_graph_nodes(mock_graph_service):
    """
    Test Step 6.3: Verification that returned citations correspond to actual graph nodes.
    Ensures that any hallucinated or non-existent node IDs are stripped out.
    """
    engine = GraphRAGEngine(graph_service=mock_graph_service)

    # Expand graph to obtain valid context
    context = await engine.expand_graph_context(
        obligation_ids=[SAMPLE_OBLIGATION_ID],
        query="What are the access controls?",
    )

    # Simulate LLM output containing one real node ID and one hallucinated node ID
    mock_llm_json = json_payload = f"""{{
        "answer": "Under SOC 2 CC6.1, logical access security software is required.",
        "citations": [
            {{
                "node_id": "{str(SAMPLE_OBLIGATION_ID)}",
                "clause": "CC6.1",
                "framework": "SOC 2"
            }},
            {{
                "node_id": "HALLUCINATED_NODE_ID_DOES_NOT_EXIST",
                "clause": "CC9.9",
                "framework": "SOC 2"
            }}
        ],
        "cited_node_ids": ["{str(SAMPLE_OBLIGATION_ID)}", "HALLUCINATED_NODE_ID_DOES_NOT_EXIST"],
        "evidence_ids": ["{str(SAMPLE_EVIDENCE_ID)}", "HALLUCINATED_EVIDENCE_999"]
    }}"""

    response = engine.parse_and_validate_answer_response(mock_llm_json, context=context)

    # Verify only actual graph nodes are preserved
    assert len(response.citations) == 1
    assert response.citations[0].node_id == str(SAMPLE_OBLIGATION_ID)
    assert response.citations[0].clause == "CC6.1"
    assert "HALLUCINATED_NODE_ID_DOES_NOT_EXIST" not in response.cited_node_ids
    assert response.cited_node_ids == [str(SAMPLE_OBLIGATION_ID)]

    # Verify only actual evidence IDs are preserved
    assert str(SAMPLE_EVIDENCE_ID) in response.evidence_ids
    assert "HALLUCINATED_EVIDENCE_999" not in response.evidence_ids


@pytest.mark.asyncio
async def test_answer_generated_from_retrieved_context(mock_graph_service):
    """
    Test Step 6.3: Verification that the answer is generated from retrieved context
    rather than arbitrary information.
    """
    engine = GraphRAGEngine(graph_service=mock_graph_service, fallback_enabled=True)

    context = await engine.expand_graph_context(
        obligation_ids=[SAMPLE_OBLIGATION_ID],
        query="Explain logical access requirements.",
    )

    response = engine._generate_fallback_answer(
        question="Explain logical access requirements.",
        context=context,
    )

    # Check that text is directly derived from the graph context properties
    assert "SOC 2" in response.answer
    assert "CC6.1" in response.answer
    assert "Logical and Physical Access Controls" in response.answer
    assert "okta_mfa_policy_2026.pdf" in response.answer
    assert "CC6.2" in response.answer  # Dependency
    assert "CC6.1-2014" in response.answer  # Supersedes
