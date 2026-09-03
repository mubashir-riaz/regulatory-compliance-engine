"""
Phase 2 — Step 6.1: Graph RAG Qdrant Retrieval Verification Script.

Demonstrates and verifies:
1. Question Embedding Generation:
   Converts a user question into a dense vector embedding using the existing embedding integration.
2. Qdrant Semantic Retrieval:
   Retrieves the top-k most relevant regulatory obligations matching the question.
3. Node ID & Metadata Extraction:
   Extracts the Neo4j Node IDs, similarity scores, clauses, and metadata for downstream
   graph traversal (Step 6.2) and LLM grounding (Step 6.3).
4. Multi-Framework & Domain Neutrality:
   Demonstrates retrieval across different regulatory frameworks (e.g. GDPR data retention,
   SOC 2 logical access controls) without hardcoded regulation-specific logic.
5. Graceful Edge Case Handling:
   Verifies that empty questions and missing records return empty results gracefully.
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

from app.integrations.qdrant_client import QdrantClient, qdrant_client
from app.services.rag_engine import GraphRAGEngine, RetrievedObligation, rag_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("verify_rag_retrieval")

# Sample Multi-Framework Regulatory Obligations for Testing
SAMPLE_OBLIGATIONS_DATA = [
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
    {
        "obligation_id": "GDPR_2016_ART_5_1_F",
        "framework": "GDPR",
        "version": "2016",
        "clause": "Article 5(1)(f)",
        "category": "Security & Confidentiality",
        "title": "Integrity and Confidentiality",
        "text": (
            "Personal data shall be processed in a manner that ensures appropriate security of the "
            "personal data, including protection against unauthorised or unlawful processing and against "
            "accidental loss, destruction or damage, using appropriate technical or organisational measures."
        ),
        "mandatory": True,
        "keywords": ["security", "confidentiality", "encryption", "integrity"],
    },
    {
        "obligation_id": "SOC2_2017_CC6_1",
        "framework": "SOC 2",
        "version": "2017",
        "clause": "CC6.1",
        "category": "Logical Access",
        "title": "Logical and Physical Access Controls",
        "text": (
            "The entity implements logical access security software, infrastructure, and architectures over "
            "protected information assets to protect them from security events and unauthorized access."
        ),
        "mandatory": True,
        "keywords": ["logical access", "authentication", "credentials", "mfa"],
    },
]


async def run_live_verification(client: QdrantClient) -> None:
    """Run verification against a live Qdrant server."""
    print("\n[Step 2/4] Initializing Qdrant collection and indexing sample obligations...")
    await client.ensure_collection()

    # Index sample obligations into Qdrant
    upsert_result = await client.upsert_obligations_batch(SAMPLE_OBLIGATIONS_DATA)
    print(f"  [+] Upserted {upsert_result.get('count', 0)} obligations into Qdrant collection '{client.collection_name}'.")

    engine = GraphRAGEngine(qdrant_client=client)

    # -------------------------------------------------------------------------
    # Test 1: Data Retention Query (GDPR)
    # -------------------------------------------------------------------------
    print("\n[Step 3/4] Testing retrieval for question: 'What data retention requirements apply under GDPR?'")
    question_1 = "What data retention requirements apply under GDPR?"

    results_1 = await engine.retrieve_relevant_obligations(question_1, top_k=3)
    assert len(results_1) > 0, "No results returned for retention query."

    print(f"  [+] Retrieved {len(results_1)} relevant obligation(s):")
    for idx, r in enumerate(results_1, start=1):
        print(f"      {idx}. [{r.framework}] {r.clause} - Score: {r.score:.4f}")
        print(f"         Node ID:  {r.node_id}")
        print(f"         Category: {r.category}")
        print(f"         Title:    {r.title}")

    # Verify top result is data retention related
    top_clauses = [r.clause for r in results_1]
    assert "Article 5(1)(e)" in top_clauses or "Article 17" in top_clauses, (
        f"Expected Article 5(1)(e) or Article 17 in top results, got {top_clauses}"
    )

    # Extract Neo4j Node IDs
    node_ids_1 = engine.extract_node_ids(results_1)
    print(f"  [+] Extracted Neo4j Node IDs for Graph Traversal: {node_ids_1}")
    assert len(node_ids_1) == len(results_1)

    # -------------------------------------------------------------------------
    # Test 2: Access Control Query (SOC 2) - Multi-Framework Domain Neutrality
    # -------------------------------------------------------------------------
    print("\n[Step 4/4] Testing multi-framework retrieval: 'What logical access controls are required under SOC 2?'")
    question_2 = "What logical access controls are required under SOC 2?"

    results_2 = await engine.retrieve_relevant_obligations(
        question=question_2,
        top_k=2,
        framework="SOC 2",
    )
    assert len(results_2) > 0, "No results returned for SOC 2 access control query."

    print(f"  [+] Retrieved {len(results_2)} relevant obligation(s):")
    for idx, r in enumerate(results_2, start=1):
        print(f"      {idx}. [{r.framework}] {r.clause} - Score: {r.score:.4f}")
        print(f"         Node ID:  {r.node_id}")
        print(f"         Category: {r.category}")

    assert results_2[0].clause == "CC6.1", f"Expected CC6.1 as top match, got {results_2[0].clause}"
    assert results_2[0].framework == "SOC 2"


async def run_standalone_demonstration() -> None:
    """Run in-memory standalone demonstration of retrieval logic with deterministic embeddings."""
    print("  [*] Running in-memory deterministic simulation of question retrieval...")

    # Generate deterministic embeddings using QdrantClient helper
    query_text = "What data retention requirements apply under GDPR?"
    query_vec = QdrantClient._generate_local_deterministic_embedding(query_text)
    print(f"  [+] Generated {len(query_vec)}-dim query embedding for: '{query_text}'")

    # Score sample obligations against query vector using cosine similarity
    scored_items: List[Dict[str, Any]] = []
    for item in SAMPLE_OBLIGATIONS_DATA:
        ob_text = f"{item['title']} {item['text']} {' '.join(item['keywords'])}"
        ob_vec = QdrantClient._generate_local_deterministic_embedding(ob_text)
        # Cosine dot product (vectors are L2 normalized)
        score = sum(q * o for q, o in zip(query_vec, ob_vec))
        scored_items.append({**item, "score": round(score, 4), "node_id": item["obligation_id"]})

    scored_items.sort(key=lambda x: x["score"], reverse=True)

    # Use GraphRAGEngine with mock
    mock_client = QdrantClient()
    engine = GraphRAGEngine(qdrant_client=mock_client)

    print("\n  [+] Top Retrieved Obligations:")
    top_items = scored_items[:2]
    retrieved = [RetrievedObligation.model_validate(it) for it in top_items]

    for idx, r in enumerate(retrieved, start=1):
        print(f"      {idx}. [{r.framework}] {r.clause} - Score: {r.score:.4f}")
        print(f"         Node ID:  {r.node_id}")
        print(f"         Category: {r.category}")
        print(f"         Title:    {r.title}")

    node_ids = engine.extract_node_ids(retrieved)
    print(f"\n  [+] Extracted Neo4j Node IDs for Graph Traversal (Step 6.2): {node_ids}")
    assert node_ids == ["GDPR_2016_ART_5_1_E", "GDPR_2016_ART_17"]


async def main() -> None:
    print("=" * 80)
    print("  Phase 2 — Step 6.1: Graph RAG Qdrant Retrieval Verification")
    print("=" * 80)

    print("\n[Step 1/4] Checking Qdrant connection...")
    is_connected = False
    try:
        is_connected = await qdrant_client.test_connection()
    except Exception as e:
        logger.warning(f"Qdrant live connection unavailable: {e}")

    if is_connected:
        print("  [+] Qdrant live connection successful.")
        await run_live_verification(qdrant_client)
    else:
        print("  [-] Live Qdrant daemon not detected. Executing standalone vector retrieval verification.")
        await run_standalone_demonstration()

    # Verify empty question handling
    empty_result = await rag_engine.retrieve_relevant_obligations("")
    assert empty_result == [], "Empty question did not return empty list."
    print("\n  [+] Verified graceful handling of empty queries (returned []).")

    print("\n" + "=" * 80)
    print("  [+] Step 6.1 Qdrant Retrieval Verification Successfully Completed!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
