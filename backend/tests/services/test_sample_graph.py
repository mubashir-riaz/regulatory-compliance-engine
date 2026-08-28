"""
Unit and Integration Tests for GraphService Sample Graph Creation & Verification (Phase 2, Step 2.3).
"""

from datetime import date
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from app.schemas.graph_models import (
    ControlCategory,
    EvidenceArtifact,
    GraphNodeLabel,
    GraphRelationshipType,
    RegulatoryFramework,
    RegulatoryObligation,
    RegulatoryVersion,
)
from app.services.graph_service import (
    SAMPLE_CATEGORY_ID,
    SAMPLE_DEP_OBLIGATION_ID,
    SAMPLE_EVIDENCE_ID,
    SAMPLE_FRAMEWORK_ID,
    SAMPLE_OBLIGATION_ID,
    SAMPLE_SUPERSEDED_OBLIGATION_ID,
    SAMPLE_VERSION_ID,
    GraphService,
)


@pytest.fixture
def mock_neo4j_client():
    """Fixture providing a mocked Neo4jClient."""
    mock_client = MagicMock()
    mock_client.execute_query = AsyncMock()
    mock_client.test_connection = AsyncMock(return_value=True)
    return mock_client


@pytest.mark.asyncio
async def test_upsert_framework(mock_neo4j_client):
    """Test upserting a RegulatoryFramework node."""
    mock_neo4j_client.execute_query.return_value = [
        {"node": {"id": str(SAMPLE_FRAMEWORK_ID), "name": "SOC 2"}}
    ]
    service = GraphService(client=mock_neo4j_client)
    framework = RegulatoryFramework(id=SAMPLE_FRAMEWORK_ID, name="SOC 2")

    result = await service.upsert_framework(framework)

    assert result["name"] == "SOC 2"
    assert result["id"] == str(SAMPLE_FRAMEWORK_ID)
    mock_neo4j_client.execute_query.assert_called_once()
    call_args = mock_neo4j_client.execute_query.call_args
    assert "MERGE (n:RegulatoryFramework {id: $id})" in call_args[0][0]


@pytest.mark.asyncio
async def test_upsert_version(mock_neo4j_client):
    """Test upserting a RegulatoryVersion node."""
    mock_neo4j_client.execute_query.return_value = [
        {
            "node": {
                "id": str(SAMPLE_VERSION_ID),
                "version_slug": "2017",
                "framework_id": str(SAMPLE_FRAMEWORK_ID),
            }
        }
    ]
    service = GraphService(client=mock_neo4j_client)
    version = RegulatoryVersion(
        id=SAMPLE_VERSION_ID,
        framework_id=SAMPLE_FRAMEWORK_ID,
        version_slug="2017",
        publication_date=date(2017, 1, 1),
    )

    result = await service.upsert_version(version)

    assert result["version_slug"] == "2017"
    mock_neo4j_client.execute_query.assert_called_once()


@pytest.mark.asyncio
async def test_upsert_obligation(mock_neo4j_client):
    """Test upserting a RegulatoryObligation node."""
    mock_neo4j_client.execute_query.return_value = [
        {
            "node": {
                "id": str(SAMPLE_OBLIGATION_ID),
                "code": "CC6.1",
                "title": "Logical and Physical Access Controls",
            }
        }
    ]
    service = GraphService(client=mock_neo4j_client)
    obligation = RegulatoryObligation(
        id=SAMPLE_OBLIGATION_ID,
        version_id=SAMPLE_VERSION_ID,
        code="CC6.1",
        title="Logical and Physical Access Controls",
    )

    result = await service.upsert_obligation(obligation)

    assert result["code"] == "CC6.1"
    mock_neo4j_client.execute_query.assert_called_once()


@pytest.mark.asyncio
async def test_upsert_control_category(mock_neo4j_client):
    """Test upserting a ControlCategory node."""
    mock_neo4j_client.execute_query.return_value = [
        {"node": {"id": str(SAMPLE_CATEGORY_ID), "name": "Access Control", "code": "AC"}}
    ]
    service = GraphService(client=mock_neo4j_client)
    category = ControlCategory(id=SAMPLE_CATEGORY_ID, name="Access Control", code="AC")

    result = await service.upsert_control_category(category)

    assert result["name"] == "Access Control"
    mock_neo4j_client.execute_query.assert_called_once()


@pytest.mark.asyncio
async def test_upsert_evidence_artifact(mock_neo4j_client):
    """Test upserting an EvidenceArtifact node."""
    mock_neo4j_client.execute_query.return_value = [
        {
            "node": {
                "id": str(SAMPLE_EVIDENCE_ID),
                "name": "okta_mfa_policy_2026.pdf",
                "status": "COMPLETED",
            }
        }
    ]
    service = GraphService(client=mock_neo4j_client)
    artifact = EvidenceArtifact(
        id=SAMPLE_EVIDENCE_ID,
        organization_id=UUID("00000000-0000-0000-0000-000000000001"),
        name="okta_mfa_policy_2026.pdf",
        file_path="evidence/okta.pdf",
    )

    result = await service.upsert_evidence_artifact(artifact)

    assert result["name"] == "okta_mfa_policy_2026.pdf"
    mock_neo4j_client.execute_query.assert_called_once()


@pytest.mark.asyncio
async def test_create_relationship(mock_neo4j_client):
    """Test creating/merging a relationship edge."""
    mock_neo4j_client.execute_query.return_value = [
        {
            "rel_type": "HAS_VERSION",
            "properties": {},
            "source_id": str(SAMPLE_FRAMEWORK_ID),
            "target_id": str(SAMPLE_VERSION_ID),
        }
    ]
    service = GraphService(client=mock_neo4j_client)
    result = await service.link_framework_version(
        framework_id=SAMPLE_FRAMEWORK_ID,
        version_id=SAMPLE_VERSION_ID,
    )

    assert result["rel_type"] == "HAS_VERSION"
    assert result["source_id"] == str(SAMPLE_FRAMEWORK_ID)
    assert result["target_id"] == str(SAMPLE_VERSION_ID)


@pytest.mark.asyncio
async def test_create_sample_graph_mocked(mock_neo4j_client):
    """Test full create_sample_graph pipeline with mocked Neo4j client."""
    mock_neo4j_client.execute_query.return_value = [
        {
            "node": {"id": "test-id"},
            "rel_type": "TEST_REL",
            "properties": {},
            "source_id": "s-id",
            "target_id": "t-id",
        }
    ]
    service = GraphService(client=mock_neo4j_client)
    result = await service.create_sample_graph()

    assert result["success"] is True
    assert "framework" in result["nodes"]
    assert "version" in result["nodes"]
    assert "obligation" in result["nodes"]
    assert "control_category" in result["nodes"]
    assert "evidence_artifact" in result["nodes"]

    assert "HAS_VERSION" in result["relationships"]
    assert "CONTAINS_MAIN" in result["relationships"]
    assert "CATEGORIZED_AS" in result["relationships"]
    assert "SATISFIES" in result["relationships"]
    assert "DEPENDS_ON" in result["relationships"]
    assert "SUPERSEDES" in result["relationships"]


@pytest.mark.asyncio
async def test_verify_sample_graph_mocked(mock_neo4j_client):
    """Test verify_sample_graph pipeline with valid node and relationship responses."""
    def mock_query_handler(query, parameters=None, db=None):
        if "RETURN f.name AS framework" in query:
            return [{
                "framework": "SOC 2",
                "version": "2017",
                "obligation": "CC6.1",
                "category": "Access Control",
                "evidence": "okta_mfa_policy_2026.pdf",
                "depends_on": "CC6.2",
                "supersedes": "CC6.1-2014",
            }]
        elif ":RegulatoryFramework" in query:
            return [{"node": {"id": str(SAMPLE_FRAMEWORK_ID), "name": "SOC 2"}}]
        elif ":RegulatoryVersion" in query:
            return [
                {
                    "node": {
                        "id": str(SAMPLE_VERSION_ID),
                        "version_slug": "2017",
                        "framework_id": str(SAMPLE_FRAMEWORK_ID),
                    }
                }
            ]
        elif parameters and parameters.get("id") == str(SAMPLE_SUPERSEDED_OBLIGATION_ID):
            return [{"node": {"id": str(SAMPLE_SUPERSEDED_OBLIGATION_ID), "code": "CC6.1-2014"}}]
        elif parameters and parameters.get("id") == str(SAMPLE_DEP_OBLIGATION_ID):
            return [{"node": {"id": str(SAMPLE_DEP_OBLIGATION_ID), "code": "CC6.2"}}]
        elif ":RegulatoryObligation" in query:
            return [
                {
                    "node": {
                        "id": str(SAMPLE_OBLIGATION_ID),
                        "code": "CC6.1",
                        "title": "Logical and Physical Access Controls",
                    }
                }
            ]
        elif ":ControlCategory" in query:
            return [{"node": {"id": str(SAMPLE_CATEGORY_ID), "name": "Access Control", "code": "AC"}}]
        elif ":EvidenceArtifact" in query:
            return [
                {
                    "node": {
                        "id": str(SAMPLE_EVIDENCE_ID),
                        "name": "okta_mfa_policy_2026.pdf",
                        "status": "COMPLETED",
                    }
                }
            ]
        elif "HAS_VERSION" in query:
            return [{"rel_type": "HAS_VERSION", "properties": {}, "source_id": str(SAMPLE_FRAMEWORK_ID), "target_id": str(SAMPLE_VERSION_ID)}]
        elif "CONTAINS" in query:
            return [{"rel_type": "CONTAINS", "properties": {}, "source_id": str(SAMPLE_VERSION_ID), "target_id": str(SAMPLE_OBLIGATION_ID)}]
        elif "CATEGORIZED_AS" in query:
            return [{"rel_type": "CATEGORIZED_AS", "properties": {}, "source_id": str(SAMPLE_OBLIGATION_ID), "target_id": str(SAMPLE_CATEGORY_ID)}]
        elif "SATISFIES" in query:
            return [{"rel_type": "SATISFIES", "properties": {"similarity_score": 0.95, "status": "approved"}, "source_id": str(SAMPLE_EVIDENCE_ID), "target_id": str(SAMPLE_OBLIGATION_ID)}]
        elif "DEPENDS_ON" in query:
            return [{"rel_type": "DEPENDS_ON", "properties": {"description": "Access control depends on auth"}, "source_id": str(SAMPLE_OBLIGATION_ID), "target_id": str(SAMPLE_DEP_OBLIGATION_ID)}]
        elif "SUPERSEDES" in query:
            return [{"rel_type": "SUPERSEDES", "properties": {"reason": "2017 supersedes 2014"}, "source_id": str(SAMPLE_OBLIGATION_ID), "target_id": str(SAMPLE_SUPERSEDED_OBLIGATION_ID)}]
        return []


    mock_neo4j_client.execute_query.side_effect = mock_query_handler
    service = GraphService(client=mock_neo4j_client)

    result = await service.verify_sample_graph()

    assert result["success"] is True
    assert "RegulatoryFramework" in result["verified_nodes"]
    assert "HAS_VERSION" in result["verified_relationships"]
    assert "SATISFIES" in result["verified_relationships"]
    assert "DEPENDS_ON" in result["verified_relationships"]
    assert "SUPERSEDES" in result["verified_relationships"]
    assert result["traversal"]["framework"] == "SOC 2"
