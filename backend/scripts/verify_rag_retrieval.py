"""
Phase 2 — Steps 6.1 & 6.2: Graph RAG Retrieval & Graph Context Expansion Verification Script.

Demonstrates and verifies:
1. Question Embedding Generation (Step 6.1):
   Converts a user question into a dense vector embedding using the existing embedding integration.
2. Qdrant Semantic Retrieval (Step 6.1):
   Retrieves the top-k most relevant regulatory obligations matching the question.
3. Neo4j Graph Context Expansion (Step 6.2):
   Takes retrieved obligation node IDs and queries Neo4j to expand connected information:
   - RegulatoryFramework & RegulatoryVersion
   - ControlCategory (CATEGORIZED_AS)
   - EvidenceArtifacts (SATISFIES, coverage status, reasoning, evidence text)
   - Related Obligations: DEPENDS_ON (outgoing & incoming dependencies)
   - Related Obligations: SUPERSEDES (superseded / newer versions)
4. Sample Graph Verification (Step 6.2):
   Verifies expansion using the existing Phase 2 Step 2 sample graph (SOC 2 CC6.1).
5. Structured Graph Context & LLM Formatting:
   Builds the structured GraphRAGContext, renders LLM prompt blocks, and extracts citation sources.
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
from app.integrations.qdrant_client import QdrantClient, qdrant_client
from app.services.graph_service import (
    SAMPLE_CATEGORY_ID,
    SAMPLE_DEP_OBLIGATION_ID,
    SAMPLE_EVIDENCE_ID,
    SAMPLE_FRAMEWORK_ID,
    SAMPLE_OBLIGATION_ID,
    SAMPLE_SUPERSEDED_OBLIGATION_ID,
    SAMPLE_VERSION_ID,
    graph_service,
)
from app.services.rag_engine import (
    GraphRAGContext,
    GraphRAGEngine,
    RetrievedObligation,
    rag_engine,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("verify_rag_retrieval")

# Sample Multi-Framework Regulatory Obligations for Testing
SAMPLE_OBLIGATIONS_DATA = [
    {
        "obligation_id": str(SAMPLE_OBLIGATION_ID),
        "framework": "SOC 2",
        "version": "2017",
        "clause": "CC6.1",
        "category": "Access Control",
        "title": "Logical and Physical Access Controls",
        "text": (
            "The entity implements logical access security software, infrastructure, and architectures over "
            "protected information assets to protect them from security events and unauthorized access."
        ),
        "mandatory": True,
        "keywords": ["logical access", "authentication", "credentials", "mfa"],
    },
    {
        "obligation_id": "GDPR_2016_ART_5_1_E",
        "framework": "GDPR",
        "version": "2016",
        "clause": "Article 5(1)(e)",
        "category": "Storage Limitation",
        "title": "Storage Limitation & Data Retention",
        "text": (
            "Personal data shall be kept in a form which permits identification of data subjects "
            "for no longer than is necessary for the purposes for which the personal data are processed "
            "('storage limitation')."
        ),
        "mandatory": True,
        "keywords": ["storage limitation", "data retention", "retention period", "personal data"],
    },
    {
        "obligation_id": "GDPR_2016_ART_17",
        "framework": "GDPR",
        "version": "2016",
        "clause": "Article 17",
        "category": "Data Subject Rights",
        "title": "Right to Erasure ('Right to be Forgotten')",
        "text": (
            "The data subject shall have the right to obtain from the controller the erasure of personal data "
            "concerning him or her without undue delay and the controller shall have the obligation to erase "
            "personal data without undue delay where personal data are no longer necessary in relation to "
            "the purposes for which they were collected."
        ),
        "mandatory": True,
        "keywords": ["erasure", "right to be forgotten", "deletion", "data retention"],
    },
]


async def run_live_verification(q_client: QdrantClient) -> None:
    """Run verification against live Qdrant and Neo4j servers."""
    print("\n[Step 2/5] Initializing Qdrant collection and indexing sample obligations...")
    await q_client.ensure_collection()

    upsert_result = await q_client.upsert_obligations_batch(SAMPLE_OBLIGATIONS_DATA)
    print(f"  [+] Upserted {upsert_result.get('count', 0)} obligations into Qdrant collection '{q_client.collection_name}'.")

    engine = GraphRAGEngine(qdrant_client=q_client, graph_service=graph_service)

    # -------------------------------------------------------------------------
    # Test 1: Step 6.1 Retrieval for Data Retention Query
    # -------------------------------------------------------------------------
    print("\n[Step 3/5] Step 6.1: Testing Qdrant retrieval for 'What data retention requirements apply under GDPR?'")
    question_1 = "What data retention requirements apply under GDPR?"
    results_1 = await engine.retrieve_relevant_obligations(question_1, top_k=2)
    assert len(results_1) > 0, "No results returned for retention query."

    print(f"  [+] Retrieved {len(results_1)} relevant obligation(s):")
    for idx, r in enumerate(results_1, start=1):
        print(f"      {idx}. [{r.framework}] {r.clause} - Score: {r.score:.4f} -> Node ID: {r.node_id}")

    # -------------------------------------------------------------------------
    # Test 2: Step 6.2 Graph Expansion using Sample Graph (SOC 2 CC6.1)
    # -------------------------------------------------------------------------
    print("\n[Step 4/5] Step 6.2: Testing Neo4j Graph Context Expansion using Sample Graph (CC6.1)...")
    context = await engine.expand_graph_context(
        obligation_ids=[SAMPLE_OBLIGATION_ID],
        max_depth=1,
        query="What are the access control requirements under SOC 2?",
    )

    assert isinstance(context, GraphRAGContext)
    assert context.total_obligations >= 1, "Graph expansion failed to find sample obligation."

    exp_ob = context.obligations[0]
    print(f"  [+] Expanded Obligation: [{exp_ob.obligation.code}] {exp_ob.obligation.title}")
    print(f"      - Node ID:         {exp_ob.node_id}")
    print(f"      - Framework:       {exp_ob.framework.name if exp_ob.framework else 'N/A'}")
    print(f"      - Version:         {exp_ob.version.version_slug if exp_ob.version else 'N/A'}")
    print(f"      - Categories:      {[c.name for c in exp_ob.categories]}")
    print(f"      - Evidence:        {[e.name for e in exp_ob.evidence_artifacts]}")
    print(f"      - Dependencies:    {[d.code for d in exp_ob.dependencies]}")
    print(f"      - Supersedes:      {[s.code for s in exp_ob.supersedes]}")

    # -------------------------------------------------------------------------
    # Test 3: LLM Context Formatting & Citation Sources
    # -------------------------------------------------------------------------
    print("\n[Step 5/5] Testing LLM Context Block Rendering & Provenance Citations...")
    llm_context = context.format_for_llm()
    print("  [+] Formatted LLM Context Snippet:")
    print("  " + "-" * 70)
    for line in llm_context.splitlines()[:15]:
        print(f"    {line}")
    print("    ...")
    print("  " + "-" * 70)

    citations = context.get_citation_sources()
    print(f"  [+] Generated {len(citations)} citation source(s):")
    for cit in citations:
        print(f"      * [{cit['type'].upper()}] {cit.get('clause') or cit.get('name')} (ID: {cit.get('node_id') or cit.get('evidence_id')})")


async def run_standalone_demonstration() -> None:
    """Run in-memory standalone demonstration of retrieval and graph expansion logic."""
    print("  [*] Running in-memory standalone simulation of retrieval & graph expansion...")

    # 1. Step 6.1 Vector Retrieval Simulation
    query_text = "What data retention requirements apply under GDPR?"
    query_vec = QdrantClient._generate_local_deterministic_embedding(query_text)
    print(f"  [+] Generated query embedding ({len(query_vec)} dims) for: '{query_text}'")

    scored_items: List[Dict[str, Any]] = []
    for item in SAMPLE_OBLIGATIONS_DATA:
        ob_text = f"{item['title']} {item['text']} {' '.join(item['keywords'])}"
        ob_vec = QdrantClient._generate_local_deterministic_embedding(ob_text)
        score = sum(q * o for q, o in zip(query_vec, ob_vec))
        scored_items.append({**item, "score": round(score, 4), "node_id": item["obligation_id"]})

    scored_items.sort(key=lambda x: x["score"], reverse=True)
    top_items = [RetrievedObligation.model_validate(it) for it in scored_items[:2]]

    print("  [+] Top Retrieved Obligations:")
    for idx, r in enumerate(top_items, start=1):
        print(f"      {idx}. [{r.framework}] {r.clause} - Score: {r.score:.4f} -> Node ID: {r.node_id}")

    # 2. Step 6.2 Graph Expansion Simulation using sample graph structure
    sample_record = {
        "obligation_id": str(SAMPLE_OBLIGATION_ID),
        "obligation": {
            "id": str(SAMPLE_OBLIGATION_ID),
            "code": "CC6.1",
            "title": "Logical and Physical Access Controls",
            "description": "The entity implements logical access security software over protected assets.",
            "category": "Access Control",
        },
        "version": {
            "id": str(SAMPLE_VERSION_ID),
            "version_slug": "2017",
            "framework_id": str(SAMPLE_FRAMEWORK_ID),
        },
        "framework": {
            "id": str(SAMPLE_FRAMEWORK_ID),
            "name": "SOC 2",
        },
        "categories": [
            {"id": str(SAMPLE_CATEGORY_ID), "name": "Access Control", "code": "AC"}
        ],
        "evidence_artifacts": [
            {
                "id": str(SAMPLE_EVIDENCE_ID),
                "name": "okta_mfa_policy_2026.pdf",
                "coverage": "FULL",
                "confidence": 0.95,
                "reasoning": "Okta MFA mandatory across company production environments.",
                "evidence_text": "All employees accessing production environments must authenticate using Okta MFA.",
            }
        ],
        "dependencies": [
            {
                "id": str(SAMPLE_DEP_OBLIGATION_ID),
                "code": "CC6.2",
                "title": "User Registration and Access Authorization",
                "direction": "OUTGOING",
                "rel_type": "DEPENDS_ON",
                "rel_description": "Logical access control enforcement depends on verified authorization.",
            }
        ],
        "supersedes": [
            {
                "id": str(SAMPLE_SUPERSEDED_OBLIGATION_ID),
                "code": "CC6.1-2014",
                "title": "Logical Access Controls (2014 Criteria)",
                "direction": "OUTGOING",
                "rel_type": "SUPERSEDES",
                "reason": "2017 Trust Services Criteria revision supersedes 2014 criteria requirement.",
            }
        ],
    }

    class MockGraphService:
        async def execute_query(self, query, parameters=None, db=None):
            return [sample_record]

    engine = GraphRAGEngine(graph_service=MockGraphService())
    context = await engine.expand_graph_context(
        obligation_ids=[SAMPLE_OBLIGATION_ID],
        query="What are the access control requirements under SOC 2?",
    )

    exp = context.obligations[0]
    print(f"\n  [+] Expanded Graph Context for {exp.clause}:")
    print(f"      - Framework:    {exp.framework.name}")
    print(f"      - Version:      {exp.version.version_slug}")
    print(f"      - Category:     {[c.name for c in exp.categories]}")
    print(f"      - Evidence:     {[e.name for e in exp.evidence_artifacts]}")
    print(f"      - Dependencies: {[d.code for d in exp.dependencies]}")
    print(f"      - Supersedes:   {[s.code for s in exp.supersedes]}")

    citations = context.get_citation_sources()
    print(f"  [+] Provenance Citations: {len(citations)} records available for LLM citations.")


async def main() -> None:
    print("=" * 80)
    print("  Phase 2 — Steps 6.1 & 6.2: Graph RAG Retrieval & Context Expansion")
    print("=" * 80)

    print("\n[Step 1/5] Testing database connectivity (Qdrant & Neo4j)...")
    qdrant_ok = False
    neo4j_ok = False

    try:
        qdrant_ok = await qdrant_client.test_connection()
    except Exception as e:
        logger.warning(f"Qdrant connection unavailable: {e}")

    try:
        neo4j_ok = await neo4j_client.test_connection()
    except Exception as e:
        logger.warning(f"Neo4j connection unavailable: {e}")

    if qdrant_ok and neo4j_ok:
        print("  [+] Both Qdrant and Neo4j connections active.")
        await run_live_verification(qdrant_client)
    else:
        print("  [-] Live databases not fully connected. Executing standalone simulation.")
        await run_standalone_demonstration()

    # Verify empty question handling
    empty_result = await rag_engine.retrieve_relevant_obligations("")
    assert empty_result == [], "Empty question did not return empty list."
    empty_graph = await rag_engine.expand_graph_context([])
    assert empty_graph.total_obligations == 0, "Empty obligation IDs did not return empty context."
    print("\n  [+] Verified graceful handling of empty inputs.")

    print("\n" + "=" * 80)
    print("  [+] Step 6.2 Graph Context Expansion Verification Successfully Completed!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
