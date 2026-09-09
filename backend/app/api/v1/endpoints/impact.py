"""
Regulatory Impact Analysis API Endpoints (Phase 2, Step 7.5).

Provides the regulatory change impact analysis endpoint:
POST /api/v1/impact/analyze

Accepts draft regulatory text, framework, and baseline/current version information,
runs the complete change impact analysis pipeline (extraction, comparison, graph traversal,
and evidence review flagging), and returns a comprehensive, audit-ready impact report.
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.impact import (
    ImpactAnalysisReport,
    ImpactAnalysisRequest,
)
from app.services.impact_analysis import (
    ImpactAnalysisService,
    impact_analysis_service as default_impact_analysis_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def get_impact_analysis_service() -> ImpactAnalysisService:
    """Dependency provider for ImpactAnalysisService."""
    return default_impact_analysis_service


@router.post(
    "/analyze",
    response_model=ImpactAnalysisReport,
    status_code=status.HTTP_200_OK,
    summary="Analyze Regulatory Change Impact",
    description=(
        "Executes the full Change Impact Analysis pipeline for a draft regulatory text: "
        "extracts discrete draft obligations, compares against baseline obligations, "
        "traverses the Neo4j graph for impacted evidence artifacts and controls, "
        "flags affected evidence with non-destructive review status (NEEDS_REVIEW), "
        "and synthesizes a structured impact report with audit-grade traceability."
    ),
)
async def analyze_regulatory_impact(
    request_body: ImpactAnalysisRequest,
    raw_request: Request,
    service: ImpactAnalysisService = Depends(get_impact_analysis_service),
    db: AsyncSession = Depends(get_db),
) -> ImpactAnalysisReport:
    """
    Execute end-to-end Change Impact Analysis.

    :param request_body: Validated ImpactAnalysisRequest containing draft text and framework/version info
    :param raw_request: Underlying HTTP request for tenant tracking
    :param service: Injected ImpactAnalysisService dependency
    :param db: Injected SQLAlchemy session for relational repository fallback
    :return: ImpactAnalysisReport containing summary, changes, affected evidence, and controls
    """
    tenant_id: Optional[str] = getattr(raw_request.state, "tenant_id", None)
    logger.info(
        f"Processing regulatory impact analysis from tenant '{tenant_id or 'default'}': "
        f"framework='{request_body.framework}', current_version='{request_body.current_version}', "
        f"draft_version='{request_body.draft_version}' (text_length={len(request_body.draft_text or '')})"
    )

    if not request_body.draft_text or not request_body.draft_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Draft regulatory text must not be empty or blank.",
        )

    try:
        report = await service.generate_impact_report(
            draft_input=request_body.draft_text,
            framework=request_body.framework,
            current_version=request_body.current_version,
            draft_version=request_body.draft_version,
            existing_version_id=request_body.existing_version_id,
            similarity_threshold=request_body.similarity_threshold,
            max_depth=request_body.max_depth,
            store_flags=request_body.store_flags,
            tenant_id=tenant_id,
            provider=request_body.provider,
            model=request_body.model,
            db_session=db,
        )
        return report

    except HTTPException:
        raise
    except Exception as err:
        logger.error(f"Regulatory impact analysis execution failed: {err}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while analyzing regulatory change impact. Please try again later.",
        )
