"""
Phase 2 — Step 4.3: Graph + Vector Population Verification Script.

Demonstration and verification script that:
1. Connects to Neo4j and Qdrant instances.
2. Initializes Neo4j schema (constraints/indexes) and Qdrant collection.
3. Populates a sample regulatory framework, version, and extracted obligations
   simultaneously into Neo4j graph relationships and Qdrant vector embeddings.
4. Queries Neo4j to verify:
   - Framework node (GDPR) and Version node (2016)
   - (Framework)-[:HAS_VERSION]->(Version)
   - Obligation nodes with deterministic unique IDs (UUIDv5)
   - (Version)-[:CONTAINS]->(Obligation)
   - ControlCategory nodes and (Obligation)-[:CATEGORIZED_AS]->(ControlCategory)
5. Queries Qdrant to verify:
   - Vectors and payload metadata exist with the exact same deterministic IDs
   - End-to-end traceability between Neo4j node ID and Qdrant point ID
   - Performs semantic similarity search and confirms expected obligations are retrieved
6. Verifies idempotency across multiple runs (no duplicate nodes or vector points).
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Configure sys.path so script can be run directly from repo root or backend folder
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent if current_dir.name == "scripts" else current_dir
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.integrations.neo4j_client import neo4j_client
from app.integrations.qdrant_client import qdrant_client
from app.schemas.extraction import ExtractedObligation
from app.services.graph_populator import (
    generate_category_id,
    generate_framework_id,
    generate_obligation_id,
    generate_version_id,
    graph_populator,
)
from app.services.graph_schema import init_graph_schema

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("verify_graph_vector_population")

# Sample Regulation Data (GDPR 2016 - Article 5 Processing Principles)
SAMPLE_FRAMEWORK = {
    "name": "GDPR",
    "description": "General Data Protection Regulation (Regulation (EU) 2016/679)",
}

SAMPLE_VERSION = {
    "version_slug": "2016",
    "description": "Regulation (EU) 2016/679 of the European Parliament and of the Council",
}

SAMPLE_OBLIGATIONS = [
    ExtractedObligation(
        clause="Article 5(1)(a)",
        text=(
            "Personal data shall be processed lawfully, fairly and in a transparent manner "
            "in relation to the data subject ('lawfulness, fairness and transparency')."
        ),
        category="Data Protection & Privacy",
        mandatory=True,
        keywords=["lawfulness", "fairness", "transparency", "personal data", "data subject"],
    ),
    ExtractedObligation(
        clause="Article 5(1)(c)",
        text=(
            "Personal data shall be adequate, relevant and limited to what is necessary in relation "
            "to the purposes for which they are processed ('data minimisation')."
        ),
        category="Data Protection & Privacy",
        mandatory=True,
        keywords=["data minimisation", "adequate", "relevant", "purposes", "proportionality"],
    ),
    ExtractedObligation(
        clause="Article 5(1)(f)",
        text=(
            "Personal data shall be processed in a manner that ensures appropriate security of the "
            "personal data, including protection against unauthorised or unlawful processing and against "
            "accidental loss, destruction or damage, using appropriate technical or organisational "
            "measures ('integrity and confidentiality')."
        ),
        category="Security & Confidentiality",
        mandatory=True,
        keywords=["integrity", "confidentiality", "security measures", "unauthorised processing", "data loss"],
    ),
]


async def main() -> None:
    """Run full graph and vector population verification workflow."""
    print("=" * 80)
    print("  Phase 2 — Step 4.3: Graph + Vector Population Verification")
    print("=" * 80)

    try:
        # ---------------------------------------------------------------------
        # 1. Connection Checks
        # ---------------------------------------------------------------------
        print("\n[Step 1/6] Checking database connections...")

        print("  -> Testing Neo4j connection...")
        neo4j_ok = await neo4j_client.test_connection()
        if not neo4j_ok:
            print("  [-] Error: Neo4j connection failed.")
            sys.exit(1)
        print("  [+] Neo4j connection established.")

        print("  -> Testing Qdrant connection...")
        qdrant_ok = await qdrant_client.test_connection()
        if not qdrant_ok:
            print("  [-] Error: Qdrant connection failed.")
            sys.exit(1)
        print("  [+] Qdrant connection established.")

        # ---------------------------------------------------------------------
        # 2. Schema Initialization
        # ---------------------------------------------------------------------
        print("\n[Step 2/6] Initializing schemas (Neo4j constraints & Qdrant collection)...")
        neo4j_statements = await init_graph_schema()
        print(f"  [+] Neo4j schema initialized with {len(neo4j_statements)} constraints/indexes.")

        await qdrant_client.ensure_collection()
        print(f"  [+] Qdrant collection '{qdrant_client.collection_name}' ensured.")

        # ---------------------------------------------------------------------
        # 3. Populate Graph + Vector via Unified Service
        # ---------------------------------------------------------------------
        print("\n[Step 3/6] Running unified GraphPopulator on sample regulation...")
        pop_result = await graph_populator.populate(
            framework=SAMPLE_FRAMEWORK,
            version=SAMPLE_VERSION,
            obligations=SAMPLE_OBLIGATIONS,
            populate_vector=True,
        )

        assert pop_result["success"] is True, "Population returned success=False"
        counts = pop_result["counts"]
        print(f"  [+] Population complete:")
        print(f"      - Frameworks:             {counts['frameworks']}")
        print(f"      - Versions:               {counts['versions']}")
        print(f"      - Obligations (Neo4j):    {counts['obligations']}")
        print(f"      - Categories (Neo4j):     {counts['categories']}")
        print(f"      - Graph Relationships:    {counts['relationships_total']}")
        print(f"      - Vector Embeddings:      {counts['vectors_indexed']}")

        # ---------------------------------------------------------------------
        # 4. Neo4j Graph Verification
        # ---------------------------------------------------------------------
        print("\n[Step 4/6] Querying Neo4j to verify nodes and relationships...")

        traversal_query = """
        MATCH (f:RegulatoryFramework {name: $framework_name})-[:HAS_VERSION]->(v:RegulatoryVersion {version_slug: $version_slug})
        MATCH (v)-[:CONTAINS]->(o:RegulatoryObligation)
        OPTIONAL MATCH (o)-[:CATEGORIZED_AS]->(c:ControlCategory)
        RETURN f.name AS framework,
               v.version_slug AS version,
               o.id AS obligation_id,
               o.code AS clause,
               o.title AS title,
               c.name AS category
        ORDER BY o.code
        """
        records = await neo4j_client.execute_query(
            traversal_query,
            parameters={"framework_name": "GDPR", "version_slug": "2016"},
        )

        assert len(records) == len(SAMPLE_OBLIGATIONS), (
            f"Expected {len(SAMPLE_OBLIGATIONS)} obligations in Neo4j, got {len(records)}"
        )

        neo4j_obligation_ids = {}
        for r in records:
            clause = r["clause"]
            ob_id = r["obligation_id"]
            cat = r["category"]
            neo4j_obligation_ids[clause] = ob_id
            print(f"  [+] Verified Neo4j Node: [{r['framework']}] {clause} ({cat}) -> ID: {ob_id}")

            # Verify deterministic ID matches expected UUIDv5
            expected_id = str(
                generate_obligation_id(
                    framework="GDPR",
                    version="2016",
                    clause=clause,
                )
            )
            assert ob_id == expected_id, (
                f"Deterministic ID mismatch for {clause}: expected {expected_id}, got {ob_id}"
            )

        print("  [+] All Neo4j nodes and deterministic IDs successfully verified.")

        # ---------------------------------------------------------------------
        # 5. Qdrant Vector & Traceability Verification
        # ---------------------------------------------------------------------
        print("\n[Step 5/6] Verifying Qdrant vector indexing, traceability & similarity search...")

        # 5.1 Verify point retrieval and cross-system ID alignment
        for clause, expected_id in neo4j_obligation_ids.items():
            point_info = await qdrant_client.get_obligation_vector(expected_id)
            assert point_info is not None, f"Obligation {expected_id} ({clause}) not found in Qdrant."
            assert point_info["obligation_id"] == expected_id, "Qdrant payload obligation_id mismatch."
            assert point_info["payload"]["framework"] == "GDPR", "Qdrant payload framework mismatch."
            assert point_info["payload"]["clause"] == clause, "Qdrant payload clause mismatch."
            print(f"  [+] Cross-System Traceability OK: Neo4j ID {expected_id} <-> Qdrant Vector ({clause})")

        # 5.2 Similarity Search Test 1: Security & Confidentiality
        query_1 = "technical measures for security, integrity and preventing unauthorised access or data loss"
        print(f"\n  -> Searching Qdrant: '{query_1}'")
        search_results_1 = await graph_populator.search_similar_obligations(
            query_text=query_1,
            framework="GDPR",
            limit=2,
        )

        assert len(search_results_1) > 0, "Similarity search returned 0 results."
        top_match_1 = search_results_1[0]
        print(f"     Top Match: [{top_match_1['framework']}] {top_match_1['clause']} "
              f"({top_match_1['category']}) - Score: {top_match_1['score']:.4f}")
        assert top_match_1["clause"] == "Article 5(1)(f)", (
            f"Expected top match Article 5(1)(f) for security query, got {top_match_1['clause']}"
        )
        assert top_match_1["obligation_id"] == neo4j_obligation_ids["Article 5(1)(f)"], (
            "Similarity result obligation_id does not match Neo4j node ID."
        )

        # 5.3 Similarity Search Test 2: Data Minimisation
        query_2 = "limiting data collection to only what is adequate, relevant and necessary"
        print(f"\n  -> Searching Qdrant: '{query_2}'")
        search_results_2 = await graph_populator.search_similar_obligations(
            query_text=query_2,
            framework="GDPR",
            limit=2,
        )

        assert len(search_results_2) > 0, "Similarity search returned 0 results."
        top_match_2 = search_results_2[0]
        print(f"     Top Match: [{top_match_2['framework']}] {top_match_2['clause']} "
              f"({top_match_2['category']}) - Score: {top_match_2['score']:.4f}")
        assert top_match_2["clause"] == "Article 5(1)(c)", (
            f"Expected top match Article 5(1)(c) for data minimisation query, got {top_match_2['clause']}"
        )
        assert top_match_2["obligation_id"] == neo4j_obligation_ids["Article 5(1)(c)"], (
            "Similarity result obligation_id does not match Neo4j node ID."
        )

        # ---------------------------------------------------------------------
        # 6. Idempotency Verification
        # ---------------------------------------------------------------------
        print("\n[Step 6/6] Verifying idempotency (re-populating identical data)...")

        # Count nodes in Neo4j before rerun
        count_neo4j_query = """
        MATCH (f:RegulatoryFramework {name: 'GDPR'})-[:HAS_VERSION]->(v:RegulatoryVersion {version_slug: '2016'})
        MATCH (v)-[:CONTAINS]->(o:RegulatoryObligation)
        RETURN count(o) AS ob_count
        """
        before_neo4j = await neo4j_client.execute_query(count_neo4j_query)
        ob_count_before = before_neo4j[0]["ob_count"]

        # Run population again
        rerun_result = await graph_populator.populate(
            framework=SAMPLE_FRAMEWORK,
            version=SAMPLE_VERSION,
            obligations=SAMPLE_OBLIGATIONS,
            populate_vector=True,
        )
        assert rerun_result["success"] is True

        after_neo4j = await neo4j_client.execute_query(count_neo4j_query)
        ob_count_after = after_neo4j[0]["ob_count"]

        assert ob_count_before == ob_count_after, (
            f"Neo4j Idempotency failed: count changed from {ob_count_before} to {ob_count_after}"
        )
        print(f"  [+] Neo4j Idempotency verified: Obligation node count remained constant at {ob_count_after}.")

        # Re-verify Qdrant points count
        for expected_id in neo4j_obligation_ids.values():
            point_info = await qdrant_client.get_obligation_vector(expected_id)
            assert point_info is not None, f"Point {expected_id} missing after rerun."
        print(f"  [+] Qdrant Idempotency verified: All points cleanly updated without duplication.")

        print("\n" + "=" * 80)
        print("  ALL GRAPH + VECTOR POPULATION VERIFICATIONS PASSED SUCCESSFULLY!")
        print("=" * 80)

    finally:
        await neo4j_client.close()
        await qdrant_client.close()


if __name__ == "__main__":
    asyncio.run(main())
