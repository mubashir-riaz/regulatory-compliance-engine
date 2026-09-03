"""
Graph RAG Engine - Qdrant Retrieval Service (Phase 2, Step 6.1).

Responsible for the semantic vector retrieval stage of the Graph RAG query pipeline:
1. Accepts a user query/question about compliance or regulatory obligations.
2. Generates an embedding vector using the existing embedding/integration code
   (Google Gemini text-embedding-004, FastEmbed, or deterministic local fallback).
3. Searches Qdrant for the top-k most relevant regulatory obligations.
4. Returns structured results containing:
   - Obligation IDs / Neo4j Node IDs (for downstream graph traversal in Step 6.2)
   - Similarity scores
   - Regulatory metadata (framework, version, clause, category, title, text, keywords, etc.)
5. Reuses the existing Qdrant integration without duplicate connection logic.
6. Makes top_k configurable with a sensible default (5).
7. Handles empty results and missing queries gracefully without crashing.
8. Asynchronous architecture aligned with the existing FastAPI / async service stack.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.config import settings
from app.integrations.qdrant_client import (
    QdrantClient,
    qdrant_client as global_qdrant_client,
)

logger = logging.getLogger(__name__)

# Default retrieval configuration
DEFAULT_TOP_K = 5
DEFAULT_COLLECTION_NAME = getattr(settings, "QDRANT_COLLECTION", "regulatory_obligations")


class RAGRetrievalError(Exception):
    """Exception raised when retrieval in the RAG pipeline fails."""
    pass


class RetrievedObligation(BaseModel):
    """
    Structured model representing a regulatory obligation retrieved from vector search.

    Preserves the exact Neo4j node ID for downstream graph traversal and provides
    access to similarity scores, clause identifiers, and metadata.
    """
    obligation_id: str = Field(
        ...,
        description="Unique obligation ID (corresponds directly to the Neo4j node ID)",
    )
    node_id: str = Field(
        ...,
        description="Neo4j node ID (alias for obligation_id)",
    )
    score: float = Field(
        0.0,
        description="Cosine similarity score (0.0 to 1.0)",
    )
    framework: Optional[str] = Field(
        None,
        description="Regulatory framework name (e.g. 'GDPR', 'SOC 2', 'ISO 27001')",
    )
    version: Optional[str] = Field(
        None,
        description="Regulatory version slug (e.g. '2016', '2017')",
    )
    clause: Optional[str] = Field(
        None,
        description="Clause or article identifier (e.g. 'Article 5(1)(e)', 'CC6.1')",
    )
    category: Optional[str] = Field(
        None,
        description="Control or requirement category (e.g. 'Storage Limitation', 'Access Control')",
    )
    title: Optional[str] = Field(
        None,
        description="Obligation title or summary header",
    )
    text: Optional[str] = Field(
        None,
        description="Obligation requirement text or description",
    )
    mandatory: Optional[bool] = Field(
        True,
        description="Whether the requirement is mandatory",
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Associated keywords and domain concepts",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional payload metadata from the vector store",
    )

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def normalize_retrieved_data(cls, data: Any) -> Any:
        """
        Normalize input dictionary from Qdrant search results or raw payloads.
        Ensures both obligation_id and node_id are set and aliases are resolved.
        """
        if isinstance(data, dict):
            mapped = dict(data)
            payload = mapped.get("payload") or {}

            # 1. Resolve obligation_id and node_id
            ob_id = (
                mapped.get("obligation_id")
                or mapped.get("node_id")
                or mapped.get("id")
                or payload.get("obligation_id")
                or payload.get("id")
            )
            if ob_id is not None:
                id_str = str(ob_id)
                mapped["obligation_id"] = id_str
                if not mapped.get("node_id"):
                    mapped["node_id"] = id_str

            # 2. Extract score
            if "score" in mapped and mapped["score"] is not None:
                try:
                    mapped["score"] = float(mapped["score"])
                except (ValueError, TypeError):
                    mapped["score"] = 0.0

            # 3. Fallback fields from payload if top-level fields are missing
            for field_name in (
                "framework",
                "version",
                "clause",
                "category",
                "title",
                "text",
                "mandatory",
                "keywords",
            ):
                if mapped.get(field_name) is None and field_name in payload:
                    mapped[field_name] = payload[field_name]

            # 4. Populate metadata payload
            if not mapped.get("metadata") and payload:
                mapped["metadata"] = payload
            elif not mapped.get("metadata"):
                mapped["metadata"] = {
                    k: v for k, v in mapped.items()
                    if k not in ("obligation_id", "node_id", "score", "metadata")
                }

            return mapped
        return data

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "obligation_id": self.obligation_id,
            "node_id": self.node_id,
            "score": self.score,
            "framework": self.framework,
            "version": self.version,
            "clause": self.clause,
            "category": self.category,
            "title": self.title,
            "text": self.text,
            "mandatory": self.mandatory,
            "keywords": self.keywords,
            "metadata": self.metadata,
        }

    def __getitem__(self, item: str) -> Any:
        """Allow subscripting like a dict for backward compatibility."""
        return getattr(self, item)


class GraphRAGEngine:
    """
    Graph RAG Query Engine.

    Phase 2 Step 6.1 implements the vector retrieval component:
    - User Question -> Question Embedding -> Qdrant Semantic Search -> Top relevant Obligations
    - Extracts Neo4j Node IDs, similarity scores, and metadata for downstream graph expansion.
    """

    def __init__(
        self,
        qdrant_client: Optional[QdrantClient] = None,
        default_top_k: int = DEFAULT_TOP_K,
        collection_name: Optional[str] = None,
    ):
        """
        Initialize GraphRAGEngine with vector client integration.

        :param qdrant_client: Reusable QdrantClient instance (defaults to global singleton)
        :param default_top_k: Default number of top results to retrieve (default: 5)
        :param collection_name: Optional Qdrant collection name override
        """
        self.qdrant_client = qdrant_client or global_qdrant_client
        self.default_top_k = default_top_k
        self.collection_name = collection_name or DEFAULT_COLLECTION_NAME

    # -------------------------------------------------------------------------
    # Embedding Generation
    # -------------------------------------------------------------------------

    async def generate_query_embedding(self, question: str) -> List[float]:
        """
        Generate a vector embedding for a user question using the existing
        Qdrant client integration code (Google Gemini, FastEmbed, or deterministic local fallback).

        :param question: The user question or query string
        :return: Vector embedding list of floats
        """
        if not question or not question.strip():
            return []
        return await self.qdrant_client.generate_embedding(question.strip())

    # -------------------------------------------------------------------------
    # Vector Retrieval
    # -------------------------------------------------------------------------

    async def retrieve_relevant_obligations(
        self,
        question: str,
        top_k: Optional[int] = None,
        framework: Optional[str] = None,
        version: Optional[str] = None,
        category: Optional[str] = None,
        score_threshold: Optional[float] = None,
        collection_name: Optional[str] = None,
        raise_on_error: bool = False,
    ) -> List[RetrievedObligation]:
        """
        Retrieve the top-k most relevant regulatory obligations for a user question from Qdrant.

        Pipeline:
        1. Validates user question; returns an empty list gracefully if blank.
        2. Generates an embedding vector for the question using existing integration logic.
        3. Performs semantic similarity search against the configured Qdrant collection.
        4. Maps search hits into structured RetrievedObligation instances containing obligation/node IDs,
           similarity scores, and metadata.
        5. Handles empty search results gracefully.

        :param question: User question (e.g. 'What data retention requirements apply under GDPR?')
        :param top_k: Maximum number of relevant obligations to return (default: 5)
        :param framework: Optional filter by framework name (e.g. 'GDPR', 'SOC 2')
        :param version: Optional filter by version slug (e.g. '2016', '2017')
        :param category: Optional filter by control category
        :param score_threshold: Optional minimum similarity score threshold
        :param collection_name: Optional collection name override
        :param raise_on_error: If True, raises RAGRetrievalError on unhandled failure;
                               if False (default), logs and returns [] gracefully.
        :return: List of RetrievedObligation models sorted by similarity score
        """
        clean_question = question.strip() if question else ""
        if not clean_question:
            logger.debug("Empty or blank question provided to retrieve_relevant_obligations; returning empty list.")
            return []

        limit = top_k if (top_k is not None and top_k > 0) else self.default_top_k
        target_collection = collection_name or self.collection_name

        try:
            logger.info(
                f"Retrieving top {limit} relevant obligations for question: '{clean_question[:80]}' "
                f"(framework={framework or 'ALL'}, collection={target_collection})"
            )

            # 1. Generate query embedding using existing integration code
            query_vector = await self.generate_query_embedding(clean_question)
            if not query_vector:
                logger.warning("Query embedding generation returned empty vector; returning empty list.")
                return []

            # 2. Search Qdrant for top-k similar obligations
            raw_results = await self.qdrant_client.search_similar_obligations(
                query_text=clean_question,
                query_vector=query_vector,
                framework=framework,
                version=version,
                category=category,
                limit=limit,
                score_threshold=score_threshold,
                collection_name=target_collection,
            )

            # 3. Gracefully handle empty results from vector store
            if not raw_results:
                logger.info(f"Qdrant similarity search returned 0 results for question: '{clean_question[:80]}'")
                return []

            # 4. Map raw results into structured RetrievedObligation models
            retrieved: List[RetrievedObligation] = []
            for item in raw_results:
                try:
                    retrieved.append(RetrievedObligation.model_validate(item))
                except Exception as val_err:
                    logger.warning(f"Error parsing retrieved obligation item: {val_err}", exc_info=True)
                    # Fallback mapping
                    ob_id = str(item.get("obligation_id") or item.get("id") or "")
                    retrieved.append(
                        RetrievedObligation(
                            obligation_id=ob_id,
                            node_id=ob_id,
                            score=float(item.get("score", 0.0)),
                            framework=item.get("framework"),
                            version=item.get("version"),
                            clause=item.get("clause"),
                            category=item.get("category"),
                            title=item.get("title"),
                            text=item.get("text"),
                            mandatory=item.get("mandatory", True),
                            keywords=item.get("keywords", []),
                            metadata=item.get("payload") or {},
                        )
                    )

            top_score_msg = f", top_score={retrieved[0].score:.4f}" if retrieved else ""
            logger.info(
                f"Successfully retrieved {len(retrieved)} obligation(s) for question '{clean_question[:80]}' "
                f"(limit={limit}{top_score_msg})"
            )
            return retrieved

        except Exception as err:
            logger.error(f"Error during Qdrant obligation retrieval: {err}", exc_info=True)
            if raise_on_error:
                raise RAGRetrievalError(f"RAG retrieval failed: {err}") from err
            return []

    # -------------------------------------------------------------------------
    # Node ID Extraction & Graph Preparation (For Step 6.2)
    # -------------------------------------------------------------------------

    @staticmethod
    def extract_node_ids(obligations: Sequence[Union[RetrievedObligation, Dict[str, Any]]]) -> List[str]:
        """
        Extract list of unique Neo4j node IDs from retrieved obligations.
        Used to feed directly into graph traversal queries in Step 6.2.

        :param obligations: Sequence of RetrievedObligation models or result dictionaries
        :return: List of distinct Neo4j node ID strings in rank order
        """
        seen = set()
        node_ids: List[str] = []
        for ob in obligations:
            if isinstance(ob, RetrievedObligation):
                nid = ob.node_id or ob.obligation_id
            elif isinstance(ob, dict):
                nid = str(ob.get("node_id") or ob.get("obligation_id") or ob.get("id") or "")
            else:
                nid = str(getattr(ob, "node_id", None) or getattr(ob, "obligation_id", "") or "")

            if nid and nid not in seen:
                seen.add(nid)
                node_ids.append(nid)
        return node_ids

    async def get_top_obligation_node_ids(
        self,
        question: str,
        top_k: Optional[int] = None,
        framework: Optional[str] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Convenience method to retrieve only the top Neo4j node IDs for a user question.

        :param question: User question text
        :param top_k: Maximum number of obligations to retrieve
        :param framework: Optional framework filter
        :return: List of Neo4j node ID strings
        """
        obligations = await self.retrieve_relevant_obligations(
            question=question,
            top_k=top_k,
            framework=framework,
            **kwargs,
        )
        return self.extract_node_ids(obligations)

    # -------------------------------------------------------------------------
    # Convenience Aliases
    # -------------------------------------------------------------------------

    retrieve = retrieve_relevant_obligations
    search_obligations = retrieve_relevant_obligations
    search = retrieve_relevant_obligations


# Convenience class alias
RAGEngine = GraphRAGEngine

# Global singleton instance for application usage
rag_engine = GraphRAGEngine()


async def get_rag_engine() -> GraphRAGEngine:
    """
    FastAPI dependency returning the global GraphRAGEngine instance.
    """
    return rag_engine
