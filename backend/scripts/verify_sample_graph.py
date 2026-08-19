"""
Phase 2 — Step 2.3: Create & Verify Sample Graph Script.

Demonstration and verification script that:
1. Initializes Neo4j uniqueness constraints and indexes.
2. Creates/merges sample regulatory compliance nodes:
   - RegulatoryFramework (SOC 2)
   - RegulatoryVersion (2017)
   - RegulatoryObligation (CC6.1 - Logical Access)
   - RegulatoryObligation (CC6.2 - Dependency Target)
   - RegulatoryObligation (CC6.1-2014 - Superseded Target)
   - ControlCategory (Access Control)
   - EvidenceArtifact (okta_mfa_policy_2026.pdf)
3. Creates/merges all required relationships:
   - Framework -> Version (HAS_VERSION)
   - Version -> Obligation (CONTAINS)
   - Obligation -> ControlCategory (CATEGORIZED_AS)
   - EvidenceArtifact -> Obligation (SATISFIES)
   - Obligation -> Obligation (DEPENDS_ON)
   - Obligation -> Obligation (SUPERSEDES)
4. Queries the graph back and verifies that all nodes, properties,
   and relationships exist and are correctly connected.
5. Verifies idempotency (safe to run repeatedly with no duplicates).
"""

import sys
import asyncio
import logging
from pathlib import Path

# Configure sys.path so script can be run directly from repo root or backend folder
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent if current_dir.name == "scripts" else current_dir
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.integrations.neo4j_client import neo4j_client
from app.services.graph_schema import init_graph_schema
from app.services.graph_service import (
    GraphService,
    SAMPLE_FRAMEWORK_ID,
    SAMPLE_VERSION_ID,
    SAMPLE_OBLIGATION_ID,
    SAMPLE_DEP_OBLIGATION_ID,
    SAMPLE_SUPERSEDED_OBLIGATION_ID,
    SAMPLE_CATEGORY_ID,
    SAMPLE_EVIDENCE_ID,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("verify_sample_graph")


async def main() -> None:
    """Run full graph creation and verification workflow."""
    print("=" * 70)
    print("  Phase 2 — Step 2.3: Create & Verify Sample Graph")
    print("=" * 70)

    graph_service = GraphService(client=neo4j_client)

    try:
        # 1. Verify Neo4j connectivity
        print("\n[Step 1/5] Testing Neo4j connection...")
        connected = await neo4j_client.test_connection()
        if not connected:
            print("[-] Error: Unable to connect to Neo4j database.")
            sys.exit(1)
        print("[+] Neo4j connection established successfully.")

        # 2. Initialize Constraints and Indexes
        print("\n[Step 2/5] Initializing graph schema (constraints & indexes)...")
        statements = await init_graph_schema()
        print(f"[+] Schema initialized with {len(statements)} constraints/indexes.")

        # 3. Create Sample Graph
        print("\n[Step 3/5] Creating sample graph nodes and relationships...")
        create_res = await graph_service.create_sample_graph()
        print(f"[+] Nodes created/merged: {len(create_res['nodes'])}")
        for key, node_props in create_res["nodes"].items():
            name = node_props.get("name") or node_props.get("title") or node_props.get("version_slug")
            print(f"    - {key}: {name} (id: {node_props.get('id')})")

        print(f"[+] Relationships created/merged: {len(create_res['relationships'])}")
        for key, rel_info in create_res["relationships"].items():
            print(f"    - {rel_info.get('rel_type')}: {rel_info.get('source_id')} -> {rel_info.get('target_id')}")

        # 4. Query & Verify Graph
        print("\n[Step 4/5] Querying graph back and verifying nodes, properties & relationships...")
        verify_res = await graph_service.verify_sample_graph()
        assert verify_res["success"] is True, "Graph verification reported failure."

        print("[+] Verified Nodes:")
        for label_key, node_data in verify_res["verified_nodes"].items():
            display_val = node_data.get("name") or node_data.get("code") or node_data.get("version_slug")
            print(f"    [OK] {label_key}: {display_val}")

        print("\n[+] Verified Relationships:")
        for rel_key, rel_data in verify_res["verified_relationships"].items():
            props = rel_data.get("properties", {})
            props_str = f" with properties {props}" if props else ""
            print(f"    [OK] {rel_key}: ({rel_data.get('source_id')[:8]}...) -[:{rel_data.get('rel_type')}]-> ({rel_data.get('target_id')[:8]}...){props_str}")

        print("\n[+] End-to-End Subgraph Traversal:")
        traversal = verify_res["traversal"]
        print(f"    Framework:     {traversal.get('framework')}")
        print(f"    Version:       {traversal.get('version')}")
        print(f"    Obligation:    {traversal.get('obligation')}")
        print(f"    Category:      {traversal.get('category')}")
        print(f"    Evidence:      {traversal.get('evidence')}")
        print(f"    Depends On:    {traversal.get('depends_on')}")
        print(f"    Supersedes:    {traversal.get('supersedes')}")

        # 5. Verify Idempotency
        print("\n[Step 5/5] Verifying idempotency (re-running creation without duplicates)...")
        # Count nodes before 2nd run
        count_query = """
        MATCH (n)
        WHERE n.id IN [
            $framework_id, $version_id, $obligation_id,
            $dep_obligation_id, $sup_obligation_id,
            $category_id, $evidence_id
        ]
        RETURN count(n) AS node_count
        """
        params = {
            "framework_id": str(SAMPLE_FRAMEWORK_ID),
            "version_id": str(SAMPLE_VERSION_ID),
            "obligation_id": str(SAMPLE_OBLIGATION_ID),
            "dep_obligation_id": str(SAMPLE_DEP_OBLIGATION_ID),
            "sup_obligation_id": str(SAMPLE_SUPERSEDED_OBLIGATION_ID),
            "category_id": str(SAMPLE_CATEGORY_ID),
            "evidence_id": str(SAMPLE_EVIDENCE_ID),
        }
        before_counts = await graph_service.execute_query(count_query, parameters=params)
        node_count_before = before_counts[0]["node_count"]

        # Run creation again
        await graph_service.create_sample_graph()

        after_counts = await graph_service.execute_query(count_query, parameters=params)
        node_count_after = after_counts[0]["node_count"]

        assert node_count_before == node_count_after, (
            f"Idempotency violation! Node count changed from {node_count_before} to {node_count_after}"
        )
        print(f"[+] Idempotency confirmed: Node count remained constant at {node_count_after}.")

        print("\n" + "=" * 70)
        print("  ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
        print("=" * 70)

    finally:
        await neo4j_client.close()


if __name__ == "__main__":
    asyncio.run(main())
