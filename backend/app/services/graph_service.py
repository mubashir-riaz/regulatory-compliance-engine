"""
Graph Service.

High-level domain service for interacting with the Neo4j graph database.
Provides query execution, schema-aligned entity upserts, relationship creation,
and sample graph creation & verification capabilities.
"""

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from app.integrations.neo4j_client import Neo4jClient, neo4j_client
from app.schemas.graph_models import (
    GraphNodeLabel,
    GraphRelationshipType,
    RegulatoryFramework,
    RegulatoryVersion,
    RegulatoryObligation,
    ControlCategory,
    EvidenceArtifact,
)

logger = logging.getLogger(__name__)

# Deterministic sample UUIDs for reproducible, idempotent graph creation & verification
SAMPLE_FRAMEWORK_ID = UUID("11111111-1111-1111-1111-111111111111")
SAMPLE_VERSION_ID = UUID("22222222-2222-2222-2222-222222222222")
SAMPLE_OBLIGATION_ID = UUID("33333333-3333-3333-3333-333333333331")
SAMPLE_DEP_OBLIGATION_ID = UUID("33333333-3333-3333-3333-333333333332")
SAMPLE_SUPERSEDED_OBLIGATION_ID = UUID("33333333-3333-3333-3333-333333333330")
SAMPLE_CATEGORY_ID = UUID("44444444-4444-4444-4444-444444444444")
SAMPLE_EVIDENCE_ID = UUID("55555555-5555-5555-5555-555555555555")
SAMPLE_ORG_ID = UUID("00000000-0000-0000-0000-000000000001")


class GraphService:
    """
    Service managing Neo4j graph operations, entity persistence,
    relationship creation, and graph integrity checks.
    """

    def __init__(self, client: Optional[Neo4jClient] = None):
        self.client = client or neo4j_client

    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        db: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a raw Cypher query through the Neo4j client.

        :param query: Cypher query string
        :param parameters: Optional dictionary of query parameters
        :param db: Optional target database name
        :return: List of record dictionaries returned by query
        """
        return await self.client.execute_query(query, parameters=parameters, db=db)

    # -------------------------------------------------------------------------
    # Generic Node Operations
    # -------------------------------------------------------------------------

    async def upsert_node(
        self,
        label: str,
        id_value: Union[str, UUID],
        properties: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Idempotently create or update a node identified by its 'id' property.

        :param label: Node label (e.g. 'RegulatoryFramework')
        :param id_value: Unique identifier for the node
        :param properties: Node properties dictionary
        :return: Properties dictionary of the merged node
        """
        clean_props = {k: v for k, v in properties.items() if v is not None}
        clean_props["id"] = str(id_value)

        query = f"""
        MERGE (n:{label} {{id: $id}})
        SET n += $properties
        RETURN properties(n) AS node
        """
        results = await self.execute_query(
            query,
            parameters={"id": str(id_value), "properties": clean_props},
        )
        if results and "node" in results[0]:
            return results[0]["node"]
        return clean_props

    async def get_node_by_id(
        self,
        label: str,
        node_id: Union[str, UUID],
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve node properties by label and ID.

        :param label: Node label
        :param node_id: Node unique ID
        :return: Dictionary of properties if found, else None
        """
        query = f"""
        MATCH (n:{label} {{id: $id}})
        RETURN properties(n) AS node
        """
        results = await self.execute_query(query, parameters={"id": str(node_id)})
        if results and results[0].get("node") is not None:
            return results[0]["node"]
        return None

    # -------------------------------------------------------------------------
    # Domain Entity Upserts
    # -------------------------------------------------------------------------

    async def upsert_framework(self, framework: RegulatoryFramework) -> Dict[str, Any]:
        """Upsert a RegulatoryFramework node."""
        return await self.upsert_node(
            label=GraphNodeLabel.FRAMEWORK.value,
            id_value=framework.id,
            properties=framework.to_cypher_properties(),
        )

    async def upsert_version(self, version: RegulatoryVersion) -> Dict[str, Any]:
        """Upsert a RegulatoryVersion node."""
        return await self.upsert_node(
            label=GraphNodeLabel.VERSION.value,
            id_value=version.id,
            properties=version.to_cypher_properties(),
        )

    async def upsert_obligation(self, obligation: RegulatoryObligation) -> Dict[str, Any]:
        """Upsert a RegulatoryObligation node."""
        return await self.upsert_node(
            label=GraphNodeLabel.OBLIGATION.value,
            id_value=obligation.id,
            properties=obligation.to_cypher_properties(),
        )

    async def upsert_control_category(self, category: ControlCategory) -> Dict[str, Any]:
        """Upsert a ControlCategory node."""
        return await self.upsert_node(
            label=GraphNodeLabel.CONTROL_CATEGORY.value,
            id_value=category.id,
            properties=category.to_cypher_properties(),
        )

    async def upsert_evidence_artifact(self, artifact: EvidenceArtifact) -> Dict[str, Any]:
        """Upsert an EvidenceArtifact node."""
        return await self.upsert_node(
            label=GraphNodeLabel.EVIDENCE.value,
            id_value=artifact.id,
            properties=artifact.to_cypher_properties(),
        )

    # -------------------------------------------------------------------------
    # Relationship Operations
    # -------------------------------------------------------------------------

    async def create_relationship(
        self,
        source_label: str,
        source_id: Union[str, UUID],
        target_label: str,
        target_id: Union[str, UUID],
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Idempotently create or update a relationship edge between two nodes.

        :param source_label: Label of the source node
        :param source_id: ID of the source node
        :param target_label: Label of the target node
        :param target_id: ID of the target node
        :param rel_type: Relationship type (e.g. 'HAS_VERSION', 'CONTAINS', etc.)
        :param properties: Optional properties to attach to the relationship
        :return: Dict containing rel_type, properties, source_id, and target_id
        """
        clean_props = {k: v for k, v in (properties or {}).items() if v is not None}
        query = f"""
        MATCH (s:{source_label} {{id: $source_id}})
        MATCH (t:{target_label} {{id: $target_id}})
        MERGE (s)-[r:{rel_type}]->(t)
        SET r += $properties
        RETURN type(r) AS rel_type, properties(r) AS properties, s.id AS source_id, t.id AS target_id
        """
        params = {
            "source_id": str(source_id),
            "target_id": str(target_id),
            "properties": clean_props,
        }
        results = await self.execute_query(query, parameters=params)
        if not results:
            raise ValueError(
                f"Failed to create relationship ':{rel_type}' from {source_label}({source_id}) "
                f"to {target_label}({target_id}). Ensure both nodes exist in the graph."
            )
        return results[0]

    async def link_framework_version(
        self,
        framework_id: Union[str, UUID],
        version_id: Union[str, UUID],
        properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """(RegulatoryFramework)-[:HAS_VERSION]->(RegulatoryVersion)"""
        return await self.create_relationship(
            source_label=GraphNodeLabel.FRAMEWORK.value,
            source_id=framework_id,
            target_label=GraphNodeLabel.VERSION.value,
            target_id=version_id,
            rel_type=GraphRelationshipType.HAS_VERSION.value,
            properties=properties,
        )

    async def link_version_obligation(
        self,
        version_id: Union[str, UUID],
        obligation_id: Union[str, UUID],
        properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """(RegulatoryVersion)-[:CONTAINS]->(RegulatoryObligation)"""
        return await self.create_relationship(
            source_label=GraphNodeLabel.VERSION.value,
            source_id=version_id,
            target_label=GraphNodeLabel.OBLIGATION.value,
            target_id=obligation_id,
            rel_type=GraphRelationshipType.CONTAINS.value,
            properties=properties,
        )

    async def link_obligation_category(
        self,
        obligation_id: Union[str, UUID],
        category_id: Union[str, UUID],
        properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """(RegulatoryObligation)-[:CATEGORIZED_AS]->(ControlCategory)"""
        return await self.create_relationship(
            source_label=GraphNodeLabel.OBLIGATION.value,
            source_id=obligation_id,
            target_label=GraphNodeLabel.CONTROL_CATEGORY.value,
            target_id=category_id,
            rel_type=GraphRelationshipType.CATEGORIZED_AS.value,
            properties=properties,
        )

    async def link_evidence_obligation(
        self,
        evidence_id: Union[str, UUID],
        obligation_id: Union[str, UUID],
        similarity_score: Optional[float] = None,
        status: str = "approved",
        properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """(EvidenceArtifact)-[:SATISFIES]->(RegulatoryObligation)"""
        props = dict(properties or {})
        if similarity_score is not None:
            props["similarity_score"] = float(similarity_score)
        if status is not None:
            props["status"] = str(status)

        return await self.create_relationship(
            source_label=GraphNodeLabel.EVIDENCE.value,
            source_id=evidence_id,
            target_label=GraphNodeLabel.OBLIGATION.value,
            target_id=obligation_id,
            rel_type=GraphRelationshipType.SATISFIES.value,
            properties=props,
        )

    async def link_obligation_depends_on(
        self,
        source_obligation_id: Union[str, UUID],
        target_obligation_id: Union[str, UUID],
        description: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """(RegulatoryObligation)-[:DEPENDS_ON]->(RegulatoryObligation)"""
        props = dict(properties or {})
        if description is not None:
            props["description"] = str(description)

        return await self.create_relationship(
            source_label=GraphNodeLabel.OBLIGATION.value,
            source_id=source_obligation_id,
            target_label=GraphNodeLabel.OBLIGATION.value,
            target_id=target_obligation_id,
            rel_type=GraphRelationshipType.DEPENDS_ON.value,
            properties=props,
        )

    async def link_obligation_supersedes(
        self,
        source_obligation_id: Union[str, UUID],
        target_obligation_id: Union[str, UUID],
        reason: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """(RegulatoryObligation)-[:SUPERSEDES]->(RegulatoryObligation)"""
        props = dict(properties or {})
        if reason is not None:
            props["reason"] = str(reason)

        return await self.create_relationship(
            source_label=GraphNodeLabel.OBLIGATION.value,
            source_id=source_obligation_id,
            target_label=GraphNodeLabel.OBLIGATION.value,
            target_id=target_obligation_id,
            rel_type=GraphRelationshipType.SUPERSEDES.value,
            properties=props,
        )

    async def get_relationship(
        self,
        source_id: Union[str, UUID],
        target_id: Union[str, UUID],
        rel_type: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Find a specific relationship between two nodes.

        :param source_id: ID of the source node
        :param target_id: ID of the target node
        :param rel_type: Relationship type string
        :return: Relationship data dict if found, else None
        """
        query = f"""
        MATCH (s {{id: $source_id}})-[r:{rel_type}]->(t {{id: $target_id}})
        RETURN type(r) AS rel_type, properties(r) AS properties, s.id AS source_id, t.id AS target_id
        """
        results = await self.execute_query(
            query,
            parameters={"source_id": str(source_id), "target_id": str(target_id)},
        )
        return results[0] if results else None

    # -------------------------------------------------------------------------
    # Sample Graph Creation & Verification (Phase 2, Step 2.3)
    # -------------------------------------------------------------------------

    async def create_sample_graph(self) -> Dict[str, Any]:
        """
        Create all sample nodes and relationships in Neo4j idempotently.

        Nodes created:
        1. RegulatoryFramework (SOC 2)
        2. RegulatoryVersion (2017)
        3. RegulatoryObligation (CC6.1 - Logical Access)
        4. RegulatoryObligation (CC6.2 - Dependency Target)
        5. RegulatoryObligation (CC6.1-2014 - Superseded Target)
        6. ControlCategory (Access Control)
        7. EvidenceArtifact (okta_mfa_policy_2026.pdf)

        Relationships created:
        1. Framework -> Version (HAS_VERSION)
        2. Version -> Obligation (CONTAINS)
        3. Obligation -> ControlCategory (CATEGORIZED_AS)
        4. EvidenceArtifact -> Obligation (SATISFIES)
        5. Obligation -> Obligation (DEPENDS_ON)
        6. Obligation -> Obligation (SUPERSEDES)

        :return: Dict with created node and relationship details
        """
        logger.info("Creating / Merging sample graph in Neo4j...")

        # 1. Instantiate Sample Models
        framework = RegulatoryFramework(
            id=SAMPLE_FRAMEWORK_ID,
            name="SOC 2",
            description="Service Organization Control 2 Trust Services Criteria",
            created_at=datetime(2026, 1, 1, 0, 0, 0),
            updated_at=datetime(2026, 1, 1, 0, 0, 0),
        )

        version = RegulatoryVersion(
            id=SAMPLE_VERSION_ID,
            framework_id=SAMPLE_FRAMEWORK_ID,
            version_slug="2017",
            description="SOC 2 Trust Services Criteria (2017 Revision)",
            publication_date=date(2017, 1, 1),
            is_active=True,
            created_at=datetime(2026, 1, 1, 0, 0, 0),
            updated_at=datetime(2026, 1, 1, 0, 0, 0),
        )

        obligation = RegulatoryObligation(
            id=SAMPLE_OBLIGATION_ID,
            version_id=SAMPLE_VERSION_ID,
            code="CC6.1",
            title="Logical and Physical Access Controls",
            description=(
                "The entity implements logical access security software, infrastructure, "
                "and architectures over protected information assets to protect them from security events."
            ),
            created_at=datetime(2026, 1, 1, 0, 0, 0),
            updated_at=datetime(2026, 1, 1, 0, 0, 0),
        )

        dep_obligation = RegulatoryObligation(
            id=SAMPLE_DEP_OBLIGATION_ID,
            version_id=SAMPLE_VERSION_ID,
            code="CC6.2",
            title="User Registration and Access Authorization",
            description=(
                "Prior to issuing system credentials and granting system access, "
                "the entity registers and authorizes new internal and external users."
            ),
            created_at=datetime(2026, 1, 1, 0, 0, 0),
            updated_at=datetime(2026, 1, 1, 0, 0, 0),
        )

        superseded_obligation = RegulatoryObligation(
            id=SAMPLE_SUPERSEDED_OBLIGATION_ID,
            version_id=SAMPLE_VERSION_ID,
            code="CC6.1-2014",
            title="Logical Access Controls (2014 Criteria)",
            description="Legacy logical access control requirement superseded by the 2017 TSC revision.",
            created_at=datetime(2026, 1, 1, 0, 0, 0),
            updated_at=datetime(2026, 1, 1, 0, 0, 0),
        )

        category = ControlCategory(
            id=SAMPLE_CATEGORY_ID,
            name="Access Control",
            code="AC",
            description="Controls governing user authentication, access authorizations, and perimeter security.",
            created_at=datetime(2026, 1, 1, 0, 0, 0),
        )

        evidence = EvidenceArtifact(
            id=SAMPLE_EVIDENCE_ID,
            organization_id=SAMPLE_ORG_ID,
            name="okta_mfa_policy_2026.pdf",
            file_path="evidence/org-001/okta_mfa_policy_2026.pdf",
            file_size=1048576,
            mime_type="application/pdf",
            status="COMPLETED",
            created_at=datetime(2026, 1, 1, 0, 0, 0),
            updated_at=datetime(2026, 1, 1, 0, 0, 0),
        )

        # 2. Upsert Nodes
        created_nodes = {
            "framework": await self.upsert_framework(framework),
            "version": await self.upsert_version(version),
            "obligation": await self.upsert_obligation(obligation),
            "dependent_obligation": await self.upsert_obligation(dep_obligation),
            "superseded_obligation": await self.upsert_obligation(superseded_obligation),
            "control_category": await self.upsert_control_category(category),
            "evidence_artifact": await self.upsert_evidence_artifact(evidence),
        }

        # 3. Upsert Relationships
        # Framework -> Version (HAS_VERSION)
        rel_has_version = await self.link_framework_version(
            framework_id=SAMPLE_FRAMEWORK_ID,
            version_id=SAMPLE_VERSION_ID,
        )

        # Version -> Obligation (CONTAINS)
        rel_contains_main = await self.link_version_obligation(
            version_id=SAMPLE_VERSION_ID,
            obligation_id=SAMPLE_OBLIGATION_ID,
        )
        rel_contains_dep = await self.link_version_obligation(
            version_id=SAMPLE_VERSION_ID,
            obligation_id=SAMPLE_DEP_OBLIGATION_ID,
        )
        rel_contains_sup = await self.link_version_obligation(
            version_id=SAMPLE_VERSION_ID,
            obligation_id=SAMPLE_SUPERSEDED_OBLIGATION_ID,
        )

        # Obligation -> ControlCategory (CATEGORIZED_AS)
        rel_categorized_as = await self.link_obligation_category(
            obligation_id=SAMPLE_OBLIGATION_ID,
            category_id=SAMPLE_CATEGORY_ID,
        )

        # EvidenceArtifact -> Obligation (SATISFIES)
        rel_satisfies = await self.link_evidence_obligation(
            evidence_id=SAMPLE_EVIDENCE_ID,
            obligation_id=SAMPLE_OBLIGATION_ID,
            similarity_score=0.95,
            status="approved",
        )

        # Obligation -> Obligation (DEPENDS_ON)
        rel_depends_on = await self.link_obligation_depends_on(
            source_obligation_id=SAMPLE_OBLIGATION_ID,
            target_obligation_id=SAMPLE_DEP_OBLIGATION_ID,
            description="Logical access control enforcement depends on verified user access authorization.",
        )

        # Obligation -> Obligation (SUPERSEDES)
        rel_supersedes = await self.link_obligation_supersedes(
            source_obligation_id=SAMPLE_OBLIGATION_ID,
            target_obligation_id=SAMPLE_SUPERSEDED_OBLIGATION_ID,
            reason="2017 Trust Services Criteria revision supersedes 2014 criteria requirement.",
        )

        created_relationships = {
            "HAS_VERSION": rel_has_version,
            "CONTAINS_MAIN": rel_contains_main,
            "CONTAINS_DEP": rel_contains_dep,
            "CONTAINS_SUP": rel_contains_sup,
            "CATEGORIZED_AS": rel_categorized_as,
            "SATISFIES": rel_satisfies,
            "DEPENDS_ON": rel_depends_on,
            "SUPERSEDES": rel_supersedes,
        }

        logger.info("Sample graph creation successfully completed.")
        return {
            "success": True,
            "nodes": created_nodes,
            "relationships": created_relationships,
        }

    async def verify_sample_graph(self) -> Dict[str, Any]:
        """
        Query Neo4j to verify the existence, properties, and relationships
        of the sample regulatory compliance graph.

        :return: Verification report dict with verified details
        :raises AssertionError: If any verification condition fails
        """
        logger.info("Verifying sample graph in Neo4j...")
        verified_nodes = {}
        verified_relationships = {}

        # 1. Verify Nodes & Properties
        # RegulatoryFramework
        fw = await self.get_node_by_id(GraphNodeLabel.FRAMEWORK.value, SAMPLE_FRAMEWORK_ID)
        assert fw is not None, f"Node RegulatoryFramework({SAMPLE_FRAMEWORK_ID}) not found in Neo4j."
        assert fw.get("name") == "SOC 2", f"Framework name mismatch: expected 'SOC 2', got '{fw.get('name')}'"
        verified_nodes["RegulatoryFramework"] = fw

        # RegulatoryVersion
        ver = await self.get_node_by_id(GraphNodeLabel.VERSION.value, SAMPLE_VERSION_ID)
        assert ver is not None, f"Node RegulatoryVersion({SAMPLE_VERSION_ID}) not found in Neo4j."
        assert ver.get("version_slug") == "2017", f"Version slug mismatch: expected '2017', got '{ver.get('version_slug')}'"
        assert ver.get("framework_id") == str(SAMPLE_FRAMEWORK_ID), "Version framework_id mismatch"
        verified_nodes["RegulatoryVersion"] = ver

        # RegulatoryObligation (Primary)
        ob = await self.get_node_by_id(GraphNodeLabel.OBLIGATION.value, SAMPLE_OBLIGATION_ID)
        assert ob is not None, f"Node RegulatoryObligation({SAMPLE_OBLIGATION_ID}) not found in Neo4j."
        assert ob.get("code") == "CC6.1", f"Obligation code mismatch: expected 'CC6.1', got '{ob.get('code')}'"
        assert ob.get("title") == "Logical and Physical Access Controls", "Obligation title mismatch"
        verified_nodes["RegulatoryObligation_CC6.1"] = ob

        # RegulatoryObligation (Dependent)
        ob_dep = await self.get_node_by_id(GraphNodeLabel.OBLIGATION.value, SAMPLE_DEP_OBLIGATION_ID)
        assert ob_dep is not None, f"Dependent Obligation({SAMPLE_DEP_OBLIGATION_ID}) not found in Neo4j."
        assert ob_dep.get("code") == "CC6.2", f"Dependent code mismatch: expected 'CC6.2', got '{ob_dep.get('code')}'"
        verified_nodes["RegulatoryObligation_CC6.2"] = ob_dep

        # RegulatoryObligation (Superseded)
        ob_sup = await self.get_node_by_id(GraphNodeLabel.OBLIGATION.value, SAMPLE_SUPERSEDED_OBLIGATION_ID)
        assert ob_sup is not None, f"Superseded Obligation({SAMPLE_SUPERSEDED_OBLIGATION_ID}) not found in Neo4j."
        assert ob_sup.get("code") == "CC6.1-2014", f"Superseded code mismatch: expected 'CC6.1-2014', got '{ob_sup.get('code')}'"
        verified_nodes["RegulatoryObligation_CC6.1-2014"] = ob_sup

        # ControlCategory
        cat = await self.get_node_by_id(GraphNodeLabel.CONTROL_CATEGORY.value, SAMPLE_CATEGORY_ID)
        assert cat is not None, f"Node ControlCategory({SAMPLE_CATEGORY_ID}) not found in Neo4j."
        assert cat.get("name") == "Access Control", f"Category name mismatch: expected 'Access Control', got '{cat.get('name')}'"
        assert cat.get("code") == "AC", f"Category code mismatch: expected 'AC', got '{cat.get('code')}'"
        verified_nodes["ControlCategory"] = cat

        # EvidenceArtifact
        ev = await self.get_node_by_id(GraphNodeLabel.EVIDENCE.value, SAMPLE_EVIDENCE_ID)
        assert ev is not None, f"Node EvidenceArtifact({SAMPLE_EVIDENCE_ID}) not found in Neo4j."
        assert ev.get("name") == "okta_mfa_policy_2026.pdf", f"Evidence name mismatch: '{ev.get('name')}'"
        assert ev.get("status") == "COMPLETED", f"Evidence status mismatch: expected 'COMPLETED', got '{ev.get('status')}'"
        verified_nodes["EvidenceArtifact"] = ev

        # 2. Verify Relationships & Properties
        # HAS_VERSION
        rel_hv = await self.get_relationship(SAMPLE_FRAMEWORK_ID, SAMPLE_VERSION_ID, GraphRelationshipType.HAS_VERSION.value)
        assert rel_hv is not None, "Relationship (Framework)-[:HAS_VERSION]->(Version) not found."
        verified_relationships["HAS_VERSION"] = rel_hv

        # CONTAINS
        rel_cnt = await self.get_relationship(SAMPLE_VERSION_ID, SAMPLE_OBLIGATION_ID, GraphRelationshipType.CONTAINS.value)
        assert rel_cnt is not None, "Relationship (Version)-[:CONTAINS]->(Obligation CC6.1) not found."
        verified_relationships["CONTAINS"] = rel_cnt

        # CATEGORIZED_AS
        rel_cat = await self.get_relationship(SAMPLE_OBLIGATION_ID, SAMPLE_CATEGORY_ID, GraphRelationshipType.CATEGORIZED_AS.value)
        assert rel_cat is not None, "Relationship (Obligation CC6.1)-[:CATEGORIZED_AS]->(ControlCategory) not found."
        verified_relationships["CATEGORIZED_AS"] = rel_cat

        # SATISFIES
        rel_sat = await self.get_relationship(SAMPLE_EVIDENCE_ID, SAMPLE_OBLIGATION_ID, GraphRelationshipType.SATISFIES.value)
        assert rel_sat is not None, "Relationship (EvidenceArtifact)-[:SATISFIES]->(Obligation CC6.1) not found."
        assert rel_sat.get("properties", {}).get("similarity_score") == 0.95, "SATISFIES similarity_score mismatch"
        assert rel_sat.get("properties", {}).get("status") == "approved", "SATISFIES status mismatch"
        verified_relationships["SATISFIES"] = rel_sat

        # DEPENDS_ON
        rel_dep = await self.get_relationship(SAMPLE_OBLIGATION_ID, SAMPLE_DEP_OBLIGATION_ID, GraphRelationshipType.DEPENDS_ON.value)
        assert rel_dep is not None, "Relationship (Obligation CC6.1)-[:DEPENDS_ON]->(Obligation CC6.2) not found."
        assert "description" in rel_dep.get("properties", {}), "DEPENDS_ON description property missing."
        verified_relationships["DEPENDS_ON"] = rel_dep

        # SUPERSEDES
        rel_sup = await self.get_relationship(SAMPLE_OBLIGATION_ID, SAMPLE_SUPERSEDED_OBLIGATION_ID, GraphRelationshipType.SUPERSEDES.value)
        assert rel_sup is not None, "Relationship (Obligation CC6.1)-[:SUPERSEDES]->(Obligation CC6.1-2014) not found."
        assert "reason" in rel_sup.get("properties", {}), "SUPERSEDES reason property missing."
        verified_relationships["SUPERSEDES"] = rel_sup

        # 3. Verify End-to-End Traversal Query
        traversal_query = """
        MATCH (f:RegulatoryFramework {id: $framework_id})-[:HAS_VERSION]->(v:RegulatoryVersion {id: $version_id})
        MATCH (v)-[:CONTAINS]->(o:RegulatoryObligation {id: $obligation_id})
        MATCH (o)-[:CATEGORIZED_AS]->(c:ControlCategory {id: $category_id})
        MATCH (e:EvidenceArtifact {id: $evidence_id})-[:SATISFIES]->(o)
        MATCH (o)-[:DEPENDS_ON]->(dep:RegulatoryObligation {id: $dep_obligation_id})
        MATCH (o)-[:SUPERSEDES]->(sup:RegulatoryObligation {id: $sup_obligation_id})
        RETURN f.name AS framework,
               v.version_slug AS version,
               o.code AS obligation,
               c.name AS category,
               e.name AS evidence,
               dep.code AS depends_on,
               sup.code AS supersedes
        """
        traversal_results = await self.execute_query(
            traversal_query,
            parameters={
                "framework_id": str(SAMPLE_FRAMEWORK_ID),
                "version_id": str(SAMPLE_VERSION_ID),
                "obligation_id": str(SAMPLE_OBLIGATION_ID),
                "category_id": str(SAMPLE_CATEGORY_ID),
                "evidence_id": str(SAMPLE_EVIDENCE_ID),
                "dep_obligation_id": str(SAMPLE_DEP_OBLIGATION_ID),
                "sup_obligation_id": str(SAMPLE_SUPERSEDED_OBLIGATION_ID),
            },
        )
        assert traversal_results, "End-to-end traversal query returned no results."
        traversal_data = traversal_results[0]
        logger.info(f"Graph traversal verified successfully: {traversal_data}")

        return {
            "success": True,
            "verified_nodes": verified_nodes,
            "verified_relationships": verified_relationships,
            "traversal": traversal_data,
        }

    # -------------------------------------------------------------------------
    # Legacy / Simple Test Node Methods (Retained for backwards compatibility)
    # -------------------------------------------------------------------------

    async def create_test_node(self, name: str = "graph-test") -> Dict[str, Any]:
        """
        Create or merge a simple test node (:Test {name: $name}).
        """
        query = """
        MERGE (n:Test {name: $name})
        RETURN n.name AS name
        """
        results = await self.execute_query(query, parameters={"name": name})
        if results:
            logger.info(f"Test node created/verified with name: {name}")
            return results[0]
        return {"name": name}

    async def get_test_node(self, name: str = "graph-test") -> Optional[Dict[str, Any]]:
        """
        Retrieve a test node by name (:Test {name: $name}).
        """
        query = """
        MATCH (n:Test {name: $name})
        RETURN n.name AS name
        """
        results = await self.execute_query(query, parameters={"name": name})
        if results:
            return results[0]
        return None
