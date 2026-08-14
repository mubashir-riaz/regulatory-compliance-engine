"""
Neo4j Graph Schema & Constraints Initialization Service.

Defines and executes Cypher DDL statements to set up uniqueness constraints and
indexes for regulatory compliance graph nodes in Neo4j.
"""

import logging
from typing import List, Optional
from app.services.graph_service import GraphService

logger = logging.getLogger(__name__)

# List of Cypher DDL statements for Neo4j uniqueness constraints and indexes
SCHEMA_CYPHER_STATEMENTS: List[str] = [
    # --- Constraints ---
    # RegulatoryFramework
    "CREATE CONSTRAINT framework_id_unique IF NOT EXISTS FOR (f:RegulatoryFramework) REQUIRE f.id IS UNIQUE",
    "CREATE CONSTRAINT framework_name_unique IF NOT EXISTS FOR (f:RegulatoryFramework) REQUIRE f.name IS UNIQUE",

    # RegulatoryVersion
    "CREATE CONSTRAINT version_id_unique IF NOT EXISTS FOR (v:RegulatoryVersion) REQUIRE v.id IS UNIQUE",
    "CREATE CONSTRAINT version_framework_slug_unique IF NOT EXISTS FOR (v:RegulatoryVersion) REQUIRE (v.framework_id, v.version_slug) IS UNIQUE",

    # RegulatoryObligation
    "CREATE CONSTRAINT obligation_id_unique IF NOT EXISTS FOR (o:RegulatoryObligation) REQUIRE o.id IS UNIQUE",
    "CREATE CONSTRAINT obligation_version_code_unique IF NOT EXISTS FOR (o:RegulatoryObligation) REQUIRE (o.version_id, o.code) IS UNIQUE",

    # ControlCategory
    "CREATE CONSTRAINT control_category_id_unique IF NOT EXISTS FOR (c:ControlCategory) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT control_category_name_unique IF NOT EXISTS FOR (c:ControlCategory) REQUIRE c.name IS UNIQUE",

    # EvidenceArtifact
    "CREATE CONSTRAINT evidence_id_unique IF NOT EXISTS FOR (e:EvidenceArtifact) REQUIRE e.id IS UNIQUE",

    # --- Indexes ---
    "CREATE INDEX evidence_org_id_idx IF NOT EXISTS FOR (e:EvidenceArtifact) ON (e.organization_id)",
    "CREATE INDEX obligation_code_idx IF NOT EXISTS FOR (o:RegulatoryObligation) ON (o.code)",
    "CREATE INDEX version_slug_idx IF NOT EXISTS FOR (v:RegulatoryVersion) ON (v.version_slug)",
]


class GraphSchemaService:
    """
    Service responsible for initializing and managing Neo4j constraints and indexes.
    """

    def __init__(self, graph_service: Optional[GraphService] = None):
        self.graph_service = graph_service or GraphService()

    async def init_schema(self) -> List[str]:
        """
        Execute all schema initialization statements (constraints & indexes).
        Uses 'IF NOT EXISTS' syntax to ensure safe, idempotent execution.

        :return: List of successfully executed Cypher statements
        """
        logger.info("Initializing Neo4j graph schema (constraints and indexes)...")
        executed_statements = []
        for statement in SCHEMA_CYPHER_STATEMENTS:
            try:
                await self.graph_service.execute_query(statement)
                executed_statements.append(statement)
                logger.debug(f"Successfully executed schema statement: {statement}")
            except Exception as e:
                logger.error(f"Error executing schema statement '{statement}': {e}")
                raise e

        logger.info(f"Neo4j graph schema initialization complete ({len(executed_statements)} statements executed).")
        return executed_statements


async def init_graph_schema() -> List[str]:
    """
    Convenience function to initialize the Neo4j graph schema.
    """
    service = GraphSchemaService()
    return await service.init_schema()
