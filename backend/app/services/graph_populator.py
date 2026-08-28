"""
Regulatory Compliance Graph Population Service (Phase 2, Step 4.1).

Responsible for taking extracted regulatory obligations and populating them into Neo4j:
1. Creates / merges the RegulatoryFramework node if it doesn't exist.
2. Creates / merges the RegulatoryVersion node if it doesn't exist.
3. Connects Framework -> Version using (RegulatoryFramework)-[:HAS_VERSION]->(RegulatoryVersion).
4. Creates / merges a RegulatoryObligation node for each extracted obligation.
5. Generates deterministic unique IDs (UUIDv5) for each obligation using framework + version + clause.
6. Connects Version -> Obligation using (RegulatoryVersion)-[:CONTAINS]->(RegulatoryObligation).
7. If an obligation has a known category, creates/finds the ControlCategory node and connects
   it using (RegulatoryObligation)-[:CATEGORIZED_AS]->(ControlCategory).
8. Uses MERGE and existing constraints to avoid duplicates and ensure full idempotency.
"""

import logging
import uuid
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from uuid import UUID, uuid4

from app.integrations.neo4j_client import Neo4jClient
from app.schemas.extraction import ExtractedObligation
from app.schemas.graph_models import (
    ControlCategory,
    GraphNodeLabel,
    GraphRelationshipType,
    RegulatoryFramework,
    RegulatoryObligation,
    RegulatoryVersion,
)
from app.services.graph_service import GraphService, graph_service as global_graph_service

logger = logging.getLogger(__name__)

# Deterministic namespace UUIDs for UUIDv5 generation
NAMESPACE_FRAMEWORK = UUID("a0000000-0000-0000-0000-000000000001")
NAMESPACE_VERSION = UUID("b0000000-0000-0000-0000-000000000002")
NAMESPACE_OBLIGATION = UUID("c0000000-0000-0000-0000-000000000003")
NAMESPACE_CATEGORY = UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


def generate_framework_id(
    framework_name: str,
    namespace: UUID = NAMESPACE_FRAMEWORK,
) -> UUID:
    """
    Generate a deterministic UUIDv5 for a regulatory framework from its name.

    :param framework_name: Unique name of the regulatory framework (e.g. 'SOC 2', 'GDPR')
    :param namespace: UUID namespace
    :return: Deterministic UUID
    """
    return uuid.uuid5(namespace, framework_name.strip().lower())


def generate_version_id(
    framework_id_or_name: Union[str, UUID],
    version_slug: str,
    namespace: UUID = NAMESPACE_VERSION,
) -> UUID:
    """
    Generate a deterministic UUIDv5 for a regulatory version from its framework and version slug.

    :param framework_id_or_name: Framework identifier or name
    :param version_slug: Version revision slug (e.g. '2017', 'v1')
    :param namespace: UUID namespace
    :return: Deterministic UUID
    """
    fw_str = str(framework_id_or_name).strip().lower()
    ver_str = str(version_slug).strip().lower()
    return uuid.uuid5(namespace, f"{fw_str}:{ver_str}")


def generate_obligation_id(
    framework: Union[str, UUID, Any],
    version: Union[str, UUID, Any],
    clause: str,
    namespace: UUID = NAMESPACE_OBLIGATION,
) -> UUID:
    """
    Generate a deterministic unique ID (UUIDv5) for each obligation using framework + version + clause.

    :param framework: Framework name (str), UUID, or RegulatoryFramework instance/dict
    :param version: Version slug (str), UUID, or RegulatoryVersion instance/dict
    :param clause: Obligation clause or requirement identifier (e.g. 'Article 5(1)(a)', 'CC6.1')
    :param namespace: UUID namespace for UUIDv5 generation
    :return: Deterministic UUID
    """
    # Extract framework key (prefer name if present, otherwise id/str)
    if isinstance(framework, str):
        fw_key = framework.strip().lower()
    elif isinstance(framework, UUID):
        fw_key = str(framework).lower()
    elif isinstance(framework, dict):
        fw_key = str(framework.get("name") or framework.get("id") or "").strip().lower()
    elif hasattr(framework, "name") and framework.name:
        fw_key = str(framework.name).strip().lower()
    elif hasattr(framework, "id") and framework.id:
        fw_key = str(framework.id).lower()
    else:
        fw_key = str(framework).strip().lower()

    # Extract version key (prefer version_slug if present, otherwise id/str)
    if isinstance(version, str):
        ver_key = version.strip().lower()
    elif isinstance(version, UUID):
        ver_key = str(version).lower()
    elif isinstance(version, dict):
        ver_key = str(version.get("version_slug") or version.get("id") or "").strip().lower()
    elif hasattr(version, "version_slug") and version.version_slug:
        ver_key = str(version.version_slug).strip().lower()
    elif hasattr(version, "id") and version.id:
        ver_key = str(version.id).lower()
    else:
        ver_key = str(version).strip().lower()

    clause_key = str(clause).strip().lower()
    seed = f"{fw_key}:{ver_key}:{clause_key}"
    return uuid.uuid5(namespace, seed)


def generate_category_id(
    category_name: str,
    namespace: UUID = NAMESPACE_CATEGORY,
) -> UUID:
    """
    Generate a deterministic UUIDv5 for a control category by name.

    :param category_name: Name of the control category (e.g. 'Access Control')
    :param namespace: UUID namespace
    :return: Deterministic UUID
    """
    return uuid.uuid5(namespace, category_name.strip().lower())


class GraphPopulator:
    """
    Graph population service for putting extracted regulatory obligations into Neo4j.

    Provides end-to-end idempotent population:
    - Creates/merges RegulatoryFramework
    - Creates/merges RegulatoryVersion
    - Creates (Framework)-[:HAS_VERSION]->(Version)
    - Creates/merges RegulatoryObligation with deterministic UUIDs
    - Creates (Version)-[:CONTAINS]->(Obligation)
    - Creates/merges ControlCategory (if category is present)
    - Creates (Obligation)-[:CATEGORIZED_AS]->(ControlCategory)
    """

    def __init__(
        self,
        graph_service: Optional[GraphService] = None,
        client: Optional[Neo4jClient] = None,
    ):
        """
        Initialize GraphPopulator.

        :param graph_service: Optional custom GraphService instance
        :param client: Optional custom Neo4jClient instance
        """
        if graph_service is not None:
            self.graph_service = graph_service
        elif client is not None:
            self.graph_service = GraphService(client=client)
        else:
            self.graph_service = global_graph_service

    # -------------------------------------------------------------------------
    # Input Normalization Helpers
    # -------------------------------------------------------------------------

    def _normalize_framework(
        self,
        framework: Union[RegulatoryFramework, Any, Dict[str, Any], str],
    ) -> RegulatoryFramework:
        """
        Normalize framework input into a Pydantic RegulatoryFramework graph node.
        """
        if isinstance(framework, RegulatoryFramework):
            if not framework.id:
                framework.id = generate_framework_id(framework.name)
            return framework

        if isinstance(framework, str):
            name = framework.strip()
            return RegulatoryFramework(
                id=generate_framework_id(name),
                name=name,
            )

        if isinstance(framework, dict):
            name = str(framework.get("name", "")).strip()
            raw_id = framework.get("id")
            node_id = UUID(str(raw_id)) if raw_id else generate_framework_id(name)
            return RegulatoryFramework(
                id=node_id,
                name=name,
                description=framework.get("description"),
                created_at=framework.get("created_at"),
                updated_at=framework.get("updated_at"),
            )

        # Handle SQLAlchemy model or generic object with attributes
        name = getattr(framework, "name", str(framework)).strip()
        raw_id = getattr(framework, "id", None)
        node_id = UUID(str(raw_id)) if raw_id else generate_framework_id(name)
        return RegulatoryFramework(
            id=node_id,
            name=name,
            description=getattr(framework, "description", None),
            created_at=getattr(framework, "created_at", None),
            updated_at=getattr(framework, "updated_at", None),
        )

    def _normalize_version(
        self,
        version: Union[RegulatoryVersion, Any, Dict[str, Any], str],
        framework_id: UUID,
    ) -> RegulatoryVersion:
        """
        Normalize version input into a Pydantic RegulatoryVersion graph node.
        """
        if isinstance(version, RegulatoryVersion):
            version.framework_id = framework_id
            if not version.id:
                version.id = generate_version_id(framework_id, version.version_slug)
            return version

        if isinstance(version, str):
            slug = version.strip()
            return RegulatoryVersion(
                id=generate_version_id(framework_id, slug),
                framework_id=framework_id,
                version_slug=slug,
            )

        if isinstance(version, dict):
            slug = str(version.get("version_slug", "")).strip()
            raw_id = version.get("id")
            node_id = UUID(str(raw_id)) if raw_id else generate_version_id(framework_id, slug)
            pub_date = version.get("publication_date")
            return RegulatoryVersion(
                id=node_id,
                framework_id=framework_id,
                version_slug=slug,
                description=version.get("description"),
                publication_date=pub_date,
                is_active=version.get("is_active", True),
                created_at=version.get("created_at"),
                updated_at=version.get("updated_at"),
            )

        # Handle SQLAlchemy model or generic object with attributes
        slug = getattr(version, "version_slug", str(version)).strip()
        raw_id = getattr(version, "id", None)
        node_id = UUID(str(raw_id)) if raw_id else generate_version_id(framework_id, slug)
        return RegulatoryVersion(
            id=node_id,
            framework_id=framework_id,
            version_slug=slug,
            description=getattr(version, "description", None),
            publication_date=getattr(version, "publication_date", None),
            is_active=getattr(version, "is_active", True),
            created_at=getattr(version, "created_at", None),
            updated_at=getattr(version, "updated_at", None),
        )

    def _normalize_obligation(
        self,
        item: Union[ExtractedObligation, RegulatoryObligation, Dict[str, Any], Any],
        framework_node: RegulatoryFramework,
        version_node: RegulatoryVersion,
    ) -> Tuple[RegulatoryObligation, Optional[ControlCategory]]:
        """
        Normalize an extracted obligation into a RegulatoryObligation node with a
        deterministic unique ID, and extract an optional ControlCategory node.
        """
        # Extract clause / identifier
        clause: Optional[str] = None
        if hasattr(item, "clause") and item.clause:
            clause = str(item.clause).strip()
        elif isinstance(item, dict) and item.get("clause"):
            clause = str(item["clause"]).strip()
        elif hasattr(item, "code") and item.code:
            clause = str(item.code).strip()
        elif isinstance(item, dict) and item.get("code"):
            clause = str(item["code"]).strip()

        # Extract text / description
        text: Optional[str] = None
        if hasattr(item, "text") and item.text:
            text = str(item.text).strip()
        elif hasattr(item, "description") and item.description:
            text = str(item.description).strip()
        elif isinstance(item, dict):
            text = str(item.get("text") or item.get("description") or "").strip()

        # Extract category
        category: Optional[str] = None
        if hasattr(item, "category") and item.category:
            category = str(item.category).strip()
        elif isinstance(item, dict) and item.get("category"):
            category = str(item["category"]).strip()

        # Extract mandatory
        mandatory: bool = True
        if hasattr(item, "mandatory") and item.mandatory is not None:
            mandatory = bool(item.mandatory)
        elif isinstance(item, dict) and "mandatory" in item:
            mandatory = bool(item["mandatory"])

        # Extract keywords
        keywords: List[str] = []
        if hasattr(item, "keywords") and item.keywords:
            keywords = list(item.keywords)
        elif isinstance(item, dict) and item.get("keywords"):
            keywords = list(item["keywords"])

        # Extract code and title
        code: str = clause or (item.get("code") if isinstance(item, dict) else getattr(item, "code", None)) or f"REQ-{uuid4().hex[:8].upper()}"
        code = str(code).strip()

        title: Optional[str] = getattr(item, "title", None) if hasattr(item, "title") else (item.get("title") if isinstance(item, dict) else None)
        if not title:
            title = f"[{category}] {code}" if category else code
        title = str(title).strip()[:255]

        source_text: Optional[str] = getattr(item, "source_text", None) if hasattr(item, "source_text") else (item.get("source_text") if isinstance(item, dict) else None)

        # Generate deterministic unique ID for the obligation
        obligation_id = generate_obligation_id(
            framework=framework_node,
            version=version_node,
            clause=clause or code,
        )

        obligation_node = RegulatoryObligation(
            id=obligation_id,
            version_id=version_node.id,
            code=code,
            title=title,
            description=text if text else None,
            clause=clause,
            category=category if category else None,
            mandatory=mandatory,
            keywords=keywords,
            source_text=source_text,
        )

        # Build ControlCategory if a known category is provided
        category_node: Optional[ControlCategory] = None
        if category:
            cat_id = generate_category_id(category)
            category_node = ControlCategory(
                id=cat_id,
                name=category,
                code=category[:50],
                description=f"Control category for {category}",
            )

        return obligation_node, category_node

    # -------------------------------------------------------------------------
    # Main Population Pipeline
    # -------------------------------------------------------------------------

    async def populate(
        self,
        framework: Union[RegulatoryFramework, Any, Dict[str, Any], str],
        version: Union[RegulatoryVersion, Any, Dict[str, Any], str],
        obligations: Sequence[Union[ExtractedObligation, RegulatoryObligation, Dict[str, Any], Any]],
    ) -> Dict[str, Any]:
        """
        Populate framework, version, obligations, categories, and relationships into Neo4j.

        :param framework: Regulatory framework (model, dict, or name string)
        :param version: Regulatory version (model, dict, or version slug string)
        :param obligations: Sequence of extracted obligations
        :return: Summary dictionary containing created/merged nodes and relationships
        """
        logger.info("Starting graph population into Neo4j...")

        # 1. Create the RegulatoryFramework node if it doesn't exist
        framework_node = self._normalize_framework(framework)
        fw_result = await self.graph_service.upsert_framework(framework_node)
        logger.info(f"Upserted RegulatoryFramework node: {framework_node.name} (id={framework_node.id})")

        # 2. Create the RegulatoryVersion node if it doesn't exist
        version_node = self._normalize_version(version, framework_id=framework_node.id)
        ver_result = await self.graph_service.upsert_version(version_node)
        logger.info(f"Upserted RegulatoryVersion node: {version_node.version_slug} (id={version_node.id})")

        # 3. Connect Framework -> Version using HAS_VERSION
        has_version_rel = await self.graph_service.link_framework_version(
            framework_id=framework_node.id,
            version_id=version_node.id,
        )
        logger.info(f"Connected (Framework:{framework_node.name})-[:HAS_VERSION]->(Version:{version_node.version_slug})")

        # 4. Create RegulatoryObligation nodes & relationships
        populated_obligations: List[Dict[str, Any]] = []
        populated_categories: Dict[str, Dict[str, Any]] = {}
        contains_relationships: List[Dict[str, Any]] = []
        categorized_relationships: List[Dict[str, Any]] = []

        for item in obligations:
            obligation_node, category_node = self._normalize_obligation(
                item=item,
                framework_node=framework_node,
                version_node=version_node,
            )

            # Upsert RegulatoryObligation node
            ob_result = await self.graph_service.upsert_obligation(obligation_node)
            populated_obligations.append(ob_result)

            # 6. Connect Version -> Obligation using CONTAINS
            contains_rel = await self.graph_service.link_version_obligation(
                version_id=version_node.id,
                obligation_id=obligation_node.id,
            )
            contains_relationships.append(contains_rel)

            # 7. If obligation has a known category, create/find ControlCategory & connect via CATEGORIZED_AS
            if category_node:
                cat_id_str = str(category_node.id)
                if cat_id_str not in populated_categories:
                    cat_result = await self.graph_service.upsert_control_category(category_node)
                    populated_categories[cat_id_str] = cat_result
                    logger.debug(f"Upserted ControlCategory node: {category_node.name} (id={category_node.id})")

                cat_rel = await self.graph_service.link_obligation_category(
                    obligation_id=obligation_node.id,
                    category_id=category_node.id,
                )
                categorized_relationships.append(cat_rel)

        logger.info(
            f"Successfully populated graph: framework='{framework_node.name}', "
            f"version='{version_node.version_slug}', obligations={len(populated_obligations)}, "
            f"categories={len(populated_categories)}"
        )

        return {
            "success": True,
            "framework": fw_result,
            "version": ver_result,
            "obligations": populated_obligations,
            "categories": list(populated_categories.values()),
            "relationships": {
                "HAS_VERSION": has_version_rel,
                "CONTAINS": contains_relationships,
                "CATEGORIZED_AS": categorized_relationships,
            },
            "counts": {
                "frameworks": 1,
                "versions": 1,
                "obligations": len(populated_obligations),
                "categories": len(populated_categories),
                "relationships_total": 1 + len(contains_relationships) + len(categorized_relationships),
            },
        }

    # -------------------------------------------------------------------------
    # Convenience Aliases
    # -------------------------------------------------------------------------

    async def populate_obligations(
        self,
        framework: Union[RegulatoryFramework, Any, Dict[str, Any], str],
        version: Union[RegulatoryVersion, Any, Dict[str, Any], str],
        obligations: Sequence[Union[ExtractedObligation, RegulatoryObligation, Dict[str, Any], Any]],
    ) -> Dict[str, Any]:
        """Convenience alias for populate."""
        return await self.populate(framework=framework, version=version, obligations=obligations)

    async def populate_framework_obligations(
        self,
        framework: Union[RegulatoryFramework, Any, Dict[str, Any], str],
        version: Union[RegulatoryVersion, Any, Dict[str, Any], str],
        obligations: Sequence[Union[ExtractedObligation, RegulatoryObligation, Dict[str, Any], Any]],
    ) -> Dict[str, Any]:
        """Convenience alias for populate."""
        return await self.populate(framework=framework, version=version, obligations=obligations)

    async def populate_graph(
        self,
        framework: Union[RegulatoryFramework, Any, Dict[str, Any], str],
        version: Union[RegulatoryVersion, Any, Dict[str, Any], str],
        obligations: Sequence[Union[ExtractedObligation, RegulatoryObligation, Dict[str, Any], Any]],
    ) -> Dict[str, Any]:
        """Convenience alias for populate."""
        return await self.populate(framework=framework, version=version, obligations=obligations)


# Global singleton instance for application usage
graph_populator = GraphPopulator()
