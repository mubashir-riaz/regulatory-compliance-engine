"""
Compliance Query API Endpoints (Phase 2, Step 6.3).

Provides the Graph RAG query endpoint:
POST /api/v1/compliance/query
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.schemas.compliance import (
    ComplianceQueryRequest,
    ComplianceQueryResponse,
)
from app.services.rag_engine import (
    GraphRAGEngine,
    get_rag_engine,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/query",
    response_model=ComplianceQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query Compliance Graph RAG Engine",
    description=(
        "Accepts a natural language compliance question, executes vector retrieval against Qdrant, "
        "expands the regulatory knowledge graph in Neo4j (Version, Framework, Categories, Evidence, Dependencies), "
        "and synthesizes a grounded compliance answer with verifiable citations and evidence IDs."
    ),
)
async def query_compliance(
    query_request: ComplianceQueryRequest,
    raw_request: Request,
    rag_engine: GraphRAGEngine = Depends(get_rag_engine),
) -> ComplianceQueryResponse:
    """
    Execute a grounded compliance Graph RAG query.

    :param query_request: Validated ComplianceQueryRequest containing question and optional filters
    :param raw_request: Underlying HTTP request (for tenant tracking)
    :param rag_engine: Injected GraphRAGEngine dependency
    :return: ComplianceQueryResponse with grounded answer, citations, and evidence IDs
    """
    tenant_id: Optional[str] = getattr(raw_request.state, "tenant_id", None)
    logger.info(
        f"Processing compliance query from tenant '{tenant_id or 'default'}': "
        f"'{query_request.question[:80]}' (top_k={query_request.top_k}, framework={query_request.framework or 'ALL'})"
    )

    try:
        response = await rag_engine.answer_question(
            question=query_request.question,
            top_k=query_request.top_k,
            framework=query_request.framework,
            version=query_request.version,
            category=query_request.category,
            score_threshold=query_request.score_threshold,
            max_depth=query_request.max_depth,
        )
        return response

    except Exception as err:
        logger.error(f"Compliance query execution failed: {err}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the compliance query. Please try again later.",
        )
