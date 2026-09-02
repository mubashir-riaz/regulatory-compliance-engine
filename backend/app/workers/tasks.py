"""
Celery Background Tasks for Document Processing and Regulatory Obligation Extraction.

Implements asynchronous workflows:
1. Document text extraction (PDF, DOCX, TXT via FileProcessor).
2. Regulatory obligation extraction (LLM chunking, Groq/Gemini extraction, PostgreSQL persistence, Neo4j graph linking).
"""

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional
from uuid import UUID

from app.core.database import AsyncSessionLocal
from app.models.framework import RegulatoryRequirement, RegulatoryVersion
from app.repositories.evidence_repo import EvidenceArtifactRepository
from app.repositories.framework_repo import (
    RegulatoryRequirementRepository,
    RegulatoryVersionRepository,
)
from app.integrations.qdrant_client import QdrantClient, qdrant_client
from app.schemas.coverage import CoverageAssessmentResult, CoverageStatus
from app.schemas.extraction import ExtractedObligation
from app.schemas.graph_models import (
    ControlCategory as Neo4jControlCategory,
    EvidenceArtifact as Neo4jEvidenceArtifact,
    RegulatoryObligation as Neo4jRegulatoryObligation,
    RegulatoryVersion as Neo4jRegulatoryVersion,
)
from app.services.coverage_service import CoverageService, coverage_service
from app.services.extraction_service import ExtractionService, extraction_service
from app.services.file_processor import FileProcessor
from app.services.graph_service import GraphService, graph_service
from app.services.text_chunker import chunk_regulatory_text
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

STATUS_PENDING = "PENDING"
STATUS_PROCESSING = "PROCESSING"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"

CATEGORY_NAMESPACE = UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


async def _update_evidence_artifact(
    document_id: str,
    status: str,
    extracted_text: Optional[str] = None,
    page_count: Optional[int] = None,
    word_count: Optional[int] = None,
) -> bool:
    """
    Async helper to update status and extracted results of EvidenceArtifact in DB.
    """
    try:
        doc_uuid = UUID(document_id)
    except (ValueError, TypeError):
        logger.warning(f"Invalid UUID string format for document_id={document_id}")
        return False

    try:
        async with AsyncSessionLocal() as session:
            repo = EvidenceArtifactRepository(session)
            artifact = await repo.get_by_id(doc_uuid)
            if artifact:
                update_data: Dict[str, Any] = {"status": status}
                if extracted_text is not None:
                    update_data["extracted_text"] = extracted_text
                if page_count is not None:
                    update_data["page_count"] = page_count
                if word_count is not None:
                    update_data["word_count"] = word_count
                await repo.update(artifact, update_data)
                logger.info(f"Updated EvidenceArtifact document_id={document_id} status to {status}")
                return True
            else:
                logger.warning(f"EvidenceArtifact with id {document_id} not found in database.")
                return False
    except Exception as db_exc:
        logger.error(
            f"Failed to update EvidenceArtifact in database for document_id={document_id}: {db_exc}",
            exc_info=True,
        )
        return False


async def _process_document_pipeline(
    document_id: str,
    file_path: str,
    version_id: Optional[str] = None,
    provider: Optional[str] = None,
    trigger_coverage: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Async implementation of the document processing pipeline.
    Extracts text/metadata from file, saves to DB, and automatically triggers:
    - regulatory obligation extraction if version_id is provided, OR
    - evidence coverage mapping (Phase 2, Step 5.3) for audit evidence artifacts.
    """
    logger.info(f"Starting background processing pipeline for document_id={document_id}, file_path={file_path}")

    # 1. Update processing status to PROCESSING
    await _update_evidence_artifact(document_id=document_id, status=STATUS_PROCESSING)

    try:
        processor = FileProcessor()
        result = processor.process_file(file_path)

        extracted_text = result.get("text", "")
        page_count = result.get("page_count", 0)
        word_count = result.get("word_count", 0)

        # 2. Save extracted text, page count, word count, and set status to COMPLETED
        await _update_evidence_artifact(
            document_id=document_id,
            status=STATUS_COMPLETED,
            extracted_text=extracted_text,
            page_count=page_count,
            word_count=word_count,
        )

        logger.info(
            f"Successfully finished file parsing for document_id={document_id}: "
            f"page_count={page_count}, word_count={word_count}, text_length={len(extracted_text)}"
        )

        pipeline_result: Dict[str, Any] = {
            "document_id": document_id,
            "status": STATUS_COMPLETED,
            "page_count": page_count,
            "word_count": word_count,
            "text_length": len(extracted_text),
        }

        # 3. If version_id is provided, automatically trigger regulatory obligation extraction
        if version_id:
            logger.info(f"Triggering obligation extraction for document_id={document_id}, version_id={version_id}")
            extraction_result = await _extract_regulatory_obligations_pipeline(
                document_id=document_id,
                version_id=version_id,
                provider=provider,
            )
            pipeline_result["extraction"] = extraction_result

        # 4. If trigger_coverage is requested or this is an evidence document (no version_id), trigger coverage mapping
        should_map_coverage = trigger_coverage is True or (trigger_coverage is None and not version_id)
        if should_map_coverage:
            logger.info(f"Triggering automatic coverage mapping for document_id={document_id}")
            coverage_result = await _map_evidence_coverage_pipeline(
                document_id=document_id,
                provider=provider,
            )
            pipeline_result["coverage"] = coverage_result

        return pipeline_result

    except Exception as e:
        logger.error(f"Error during document processing pipeline for document_id={document_id}: {e}", exc_info=True)
        await _update_evidence_artifact(document_id=document_id, status=STATUS_FAILED)
        return {
            "document_id": document_id,
            "status": STATUS_FAILED,
            "error": str(e),
        }


async def _extract_regulatory_obligations_pipeline(
    document_id: str,
    version_id: str,
    provider: Optional[str] = None,
    custom_extraction_service: Optional[ExtractionService] = None,
    custom_graph_service: Optional[GraphService] = None,
) -> Dict[str, Any]:
    """
    Async pipeline for chunking regulatory text, extracting obligations via LLM,
    storing them in PostgreSQL (RegulatoryRequirement), and creating/linking nodes in Neo4j.
    """
    logger.info(f"Starting regulatory obligation extraction pipeline for doc={document_id}, version={version_id}")

    try:
        doc_uuid = UUID(document_id)
        ver_uuid = UUID(version_id)
    except (ValueError, TypeError) as err:
        logger.error(f"Invalid UUID in extraction pipeline: {err}")
        return {"status": STATUS_FAILED, "error": f"Invalid UUID: {err}"}

    srv_extractor = custom_extraction_service or extraction_service
    srv_graph = custom_graph_service or graph_service

    # 1. Fetch document text and version from PostgreSQL
    async with AsyncSessionLocal() as session:
        evidence_repo = EvidenceArtifactRepository(session)
        version_repo = RegulatoryVersionRepository(session)
        req_repo = RegulatoryRequirementRepository(session)

        artifact = await evidence_repo.get_by_id(doc_uuid)
        if not artifact:
            logger.error(f"EvidenceArtifact {document_id} not found in database.")
            return {"status": STATUS_FAILED, "error": f"EvidenceArtifact {document_id} not found"}

        version_obj = await version_repo.get_by_id(ver_uuid)
        if not version_obj:
            logger.error(f"RegulatoryVersion {version_id} not found in database.")
            return {"status": STATUS_FAILED, "error": f"RegulatoryVersion {version_id} not found"}

        # Extract text from file if not already populated
        raw_text = artifact.extracted_text
        if not raw_text or not raw_text.strip():
            logger.info(f"Artifact {document_id} has no extracted text. Parsing from file_path={artifact.file_path}...")
            proc = FileProcessor()
            proc_res = proc.process_file(artifact.file_path)
            raw_text = proc_res.get("text", "")
            if raw_text:
                await evidence_repo.update(artifact, {"extracted_text": raw_text})

        if not raw_text or not raw_text.strip():
            logger.warning(f"No text content available in document {document_id} for obligation extraction.")
            return {
                "status": STATUS_COMPLETED,
                "document_id": document_id,
                "version_id": version_id,
                "total_chunks": 0,
                "obligations_extracted": 0,
                "message": "No text content available in document.",
            }

        # 2. Ensure RegulatoryVersion node exists in Neo4j
        try:
            neo4j_version = Neo4jRegulatoryVersion(
                id=version_obj.id,
                framework_id=version_obj.framework_id,
                version_slug=version_obj.version_slug,
                description=version_obj.description,
                is_active=version_obj.is_active,
            )
            await srv_graph.upsert_version(neo4j_version)
        except Exception as graph_err:
            logger.warning(f"Could not upsert RegulatoryVersion in Neo4j (non-fatal): {graph_err}")

        # 3. Chunk text
        chunks = chunk_regulatory_text(raw_text, chunk_size=3000, chunk_overlap=200)
        logger.info(f"Split document {document_id} into {len(chunks)} chunk(s) for LLM extraction.")

        total_extracted = 0
        created_count = 0
        updated_count = 0
        graph_synced_count = 0
        failed_chunks = 0
        processed_clauses = set()

        # 4. Process each chunk through ExtractionService
        for idx, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {idx + 1}/{len(chunks)} ({len(chunk)} chars)...")
            try:
                extracted_list: List[ExtractedObligation] = await srv_extractor.extract_obligations(
                    text=chunk,
                    provider=provider,
                )
            except Exception as chunk_exc:
                logger.error(f"Extraction failed for chunk {idx + 1}: {chunk_exc}", exc_info=True)
                failed_chunks += 1
                continue

            for item in extracted_list:
                clause_code = (item.clause or "").strip()
                if not clause_code:
                    clause_code = f"REQ-{uuid.uuid4().hex[:8].upper()}"

                clause_key = clause_code.lower()
                title_str = f"[{item.category}] {clause_code}" if item.category else clause_code
                title_str = title_str[:255]

                # 5. Persist to PostgreSQL (Idempotent upsert via RegulatoryRequirementRepository)
                try:
                    existing_req = await req_repo.get_by_code(ver_uuid, clause_code)
                    if existing_req:
                        req_obj = await req_repo.update(
                            existing_req,
                            {
                                "title": title_str,
                                "description": item.text,
                            },
                        )
                        updated_count += 1
                    else:
                        req_obj = await req_repo.create(
                            {
                                "id": uuid.uuid4(),
                                "version_id": ver_uuid,
                                "code": clause_code,
                                "title": title_str,
                                "description": item.text,
                            }
                        )
                        created_count += 1

                    total_extracted += 1
                    processed_clauses.add(clause_key)

                except Exception as db_save_err:
                    logger.error(f"Error saving requirement '{clause_code}' to PostgreSQL: {db_save_err}", exc_info=True)
                    continue

                # 6. Create corresponding node & relationships in Neo4j
                try:
                    # Upsert RegulatoryObligation node
                    neo4j_obligation = Neo4jRegulatoryObligation(
                        id=req_obj.id,
                        version_id=ver_uuid,
                        code=clause_code,
                        title=title_str,
                        description=item.text,
                        clause=item.clause,
                        category=item.category,
                        mandatory=item.mandatory,
                        keywords=item.keywords,
                        source_text=chunk[:500],
                    )
                    await srv_graph.upsert_obligation(neo4j_obligation)

                    # Link Version -> Obligation (CONTAINS)
                    await srv_graph.link_version_obligation(
                        version_id=ver_uuid,
                        obligation_id=req_obj.id,
                    )

                    # If category is specified, upsert ControlCategory & link Obligation -> ControlCategory (CATEGORIZED_AS)
                    if item.category and item.category.strip():
                        cat_name = item.category.strip()
                        cat_id = uuid.uuid5(CATEGORY_NAMESPACE, cat_name.lower())
                        neo4j_category = Neo4jControlCategory(
                            id=cat_id,
                            name=cat_name,
                            code=cat_name[:50],
                        )
                        await srv_graph.upsert_control_category(neo4j_category)
                        await srv_graph.link_obligation_category(
                            obligation_id=req_obj.id,
                            category_id=cat_id,
                        )

                    # Link EvidenceArtifact -> Obligation (SATISFIES)
                    await srv_graph.link_evidence_obligation(
                        evidence_id=doc_uuid,
                        obligation_id=req_obj.id,
                        similarity_score=1.0,
                        status="extracted",
                    )

                    graph_synced_count += 1

                except Exception as graph_err:
                    logger.warning(f"Error syncing obligation '{clause_code}' to Neo4j (non-fatal): {graph_err}")

        logger.info(
            f"Extraction complete for document {document_id}: "
            f"total_chunks={len(chunks)}, failed_chunks={failed_chunks}, "
            f"created_in_db={created_count}, updated_in_db={updated_count}, graph_synced={graph_synced_count}"
        )

        return {
            "status": STATUS_COMPLETED,
            "document_id": document_id,
            "version_id": version_id,
            "total_chunks": len(chunks),
            "failed_chunks": failed_chunks,
            "obligations_extracted": total_extracted,
            "created_in_db": created_count,
            "updated_in_db": updated_count,
            "graph_synced": graph_synced_count,
        }


async def _map_evidence_coverage_pipeline(
    document_id: str,
    framework: Optional[str] = None,
    version: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 10,
    score_threshold: Optional[float] = None,
    provider: Optional[str] = None,
    custom_coverage_service: Optional[CoverageService] = None,
    custom_graph_service: Optional[GraphService] = None,
    custom_qdrant_client: Optional[QdrantClient] = None,
) -> Dict[str, Any]:
    """
    Async pipeline for evidence coverage mapping (Phase 2, Step 5.3).

    Workflow:
    1. Fetch EvidenceArtifact text from PostgreSQL / local file.
    2. Ensure EvidenceArtifact node exists in Neo4j.
    3. Retrieve candidate regulatory obligations:
       - Uses Qdrant vector similarity search to narrow candidates.
       - Falls back gracefully to PostgreSQL / Neo4j if vector search is empty or offline.
    4. Call CoverageService for each candidate obligation to determine coverage status (FULL, PARTIAL, NONE).
    5. Store the resulting SATISFIES relationship in Neo4j with confidence, reasoning,
       coverage status, and supporting evidence text.
    6. Uses Cypher MERGE to avoid duplicate relationships on retries and isolates
       individual assessment failures so the entire mapping job succeeds.

    :param document_id: ID of the EvidenceArtifact
    :param framework: Optional framework filter (e.g. 'SOC 2')
    :param version: Optional version filter (e.g. '2017')
    :param category: Optional category filter (e.g. 'Access Control')
    :param limit: Maximum candidate obligations to evaluate (default: 10)
    :param score_threshold: Minimum vector similarity threshold
    :param provider: LLM provider override ('groq' or 'gemini')
    :param custom_coverage_service: Optional CoverageService instance (for testing)
    :param custom_graph_service: Optional GraphService instance (for testing)
    :param custom_qdrant_client: Optional QdrantClient instance (for testing)
    :return: Pipeline execution summary dictionary
    """
    logger.info(f"Starting evidence coverage mapping pipeline for document_id={document_id}")

    try:
        doc_uuid = UUID(document_id)
    except (ValueError, TypeError) as err:
        logger.error(f"Invalid UUID in coverage mapping pipeline: {err}")
        return {"status": STATUS_FAILED, "error": f"Invalid UUID: {err}"}

    srv_coverage = custom_coverage_service or coverage_service
    srv_graph = custom_graph_service or graph_service
    srv_qdrant = custom_qdrant_client if custom_qdrant_client is not None else qdrant_client

    # 1. Fetch EvidenceArtifact from PostgreSQL
    async with AsyncSessionLocal() as session:
        evidence_repo = EvidenceArtifactRepository(session)
        artifact = await evidence_repo.get_by_id(doc_uuid)
        if not artifact:
            logger.error(f"EvidenceArtifact {document_id} not found in database.")
            return {"status": STATUS_FAILED, "error": f"EvidenceArtifact {document_id} not found"}

        evidence_text = artifact.extracted_text
        if not evidence_text or not evidence_text.strip():
            # Parse from file if text not yet populated in DB
            if artifact.file_path:
                logger.info(f"Artifact {document_id} has no extracted text. Parsing from file_path={artifact.file_path}...")
                proc = FileProcessor()
                proc_res = proc.process_file(artifact.file_path)
                evidence_text = proc_res.get("text", "")
                if evidence_text:
                    await evidence_repo.update(artifact, {"extracted_text": evidence_text})

        if not evidence_text or not evidence_text.strip():
            logger.warning(f"No text content available in document {document_id} for coverage assessment.")
            return {
                "status": STATUS_COMPLETED,
                "document_id": document_id,
                "candidates_found": 0,
                "assessments_completed": 0,
                "assessments_failed": 0,
                "successful_mappings": [],
                "failed_assessments": [],
                "message": "No text content available in document.",
            }

        # 2. Ensure EvidenceArtifact node exists in Neo4j
        try:
            neo4j_evidence = Neo4jEvidenceArtifact(
                id=artifact.id,
                organization_id=artifact.organization_id,
                name=artifact.name,
                file_path=artifact.file_path,
                file_size=artifact.file_size,
                mime_type=artifact.mime_type,
                status=artifact.status or "COMPLETED",
            )
            await srv_graph.upsert_evidence_artifact(neo4j_evidence)
        except Exception as graph_err:
            logger.warning(f"Could not upsert EvidenceArtifact in Neo4j (non-fatal): {graph_err}")

        # 3. Retrieve Candidate Regulatory Obligations
        candidates: List[Dict[str, Any]] = []
        query_excerpt = evidence_text[:4000].strip()

        # Strategy A: Use Qdrant similarity search to narrow candidates
        try:
            if srv_qdrant is not None:
                qdrant_results = await srv_qdrant.search_similar_obligations(
                    query_text=query_excerpt,
                    framework=framework,
                    version=version,
                    category=category,
                    limit=limit,
                    score_threshold=score_threshold,
                )
                if qdrant_results:
                    candidates = qdrant_results
                    logger.info(
                        f"Retrieved {len(candidates)} candidate obligations from Qdrant vector search "
                        f"for document {document_id}."
                    )
        except Exception as qdrant_err:
            logger.warning(
                f"Qdrant similarity search failed ({qdrant_err}). Falling back to database/graph lookup.",
                exc_info=True,
            )

        # Strategy B: Fallback to PostgreSQL or Neo4j if Qdrant returned no candidates
        if not candidates:
            logger.info("No candidates returned from Qdrant. Checking PostgreSQL / Neo4j obligations fallback...")
            req_repo = RegulatoryRequirementRepository(session)
            requirements = await req_repo.list(limit=limit)
            if requirements:
                for r in requirements:
                    candidates.append({
                        "obligation_id": str(r.id),
                        "clause": r.code,
                        "title": r.title,
                        "text": r.description or r.title,
                        "category": None,
                        "framework": framework,
                        "version": version,
                        "score": 1.0,
                    })
                logger.info(f"Retrieved {len(candidates)} candidate obligations from PostgreSQL.")
            else:
                try:
                    query = """
                    MATCH (o:RegulatoryObligation)
                    RETURN o.id AS obligation_id, o.code AS clause, o.title AS title,
                           o.description AS text, o.category AS category
                    LIMIT $limit
                    """
                    neo4j_obs = await srv_graph.execute_query(query, parameters={"limit": limit})
                    for no in neo4j_obs:
                        candidates.append({
                            "obligation_id": str(no.get("obligation_id")),
                            "clause": no.get("clause"),
                            "title": no.get("title"),
                            "text": no.get("text") or no.get("title"),
                            "category": no.get("category"),
                            "framework": framework,
                            "version": version,
                            "score": 1.0,
                        })
                    logger.info(f"Retrieved {len(candidates)} candidate obligations from Neo4j.")
                except Exception as neo_err:
                    logger.warning(f"Neo4j fallback obligation query failed: {neo_err}")

        if not candidates:
            logger.warning(f"No candidate obligations found in system for document {document_id}.")
            return {
                "status": STATUS_COMPLETED,
                "document_id": document_id,
                "candidates_found": 0,
                "assessments_completed": 0,
                "assessments_failed": 0,
                "successful_mappings": [],
                "failed_assessments": [],
                "message": "No candidate obligations found in system.",
            }

        # 4. Assess Coverage and Persist SATISFIES Relationships
        successful_mappings: List[Dict[str, Any]] = []
        failed_assessments: List[Dict[str, Any]] = []

        for idx, cand in enumerate(candidates):
            cand_id = cand.get("obligation_id") or cand.get("id")
            if not cand_id:
                continue

            clause = cand.get("clause") or cand.get("code") or ""
            cand_text = cand.get("text") or cand.get("description") or cand.get("title") or ""
            cat = cand.get("category")
            meta = cand.get("payload") if isinstance(cand.get("payload"), dict) else cand

            logger.info(
                f"Assessing coverage ({idx + 1}/{len(candidates)}): "
                f"document_id={document_id} -> obligation_id={cand_id} ({clause})..."
            )

            try:
                # Call CoverageService
                assessment_res: CoverageAssessmentResult = await srv_coverage.assess_coverage(
                    evidence_text=evidence_text,
                    obligation_text=cand_text,
                    clause=clause,
                    category=cat,
                    metadata=meta,
                    provider=provider,
                    use_fallback=True,
                )

                # Store SATISFIES relationship in Neo4j idempotently via MERGE
                rel = await srv_graph.store_coverage_assessment(
                    evidence_id=doc_uuid,
                    obligation_id=cand_id,
                    assessment=assessment_res,
                    evidence_text=assessment_res.relevant_snippet or evidence_text[:1000],
                    create_nodes_if_missing=True,
                )

                successful_mappings.append({
                    "obligation_id": str(cand_id),
                    "clause": clause,
                    "status": assessment_res.status.value,
                    "confidence": assessment_res.confidence,
                    "reasoning": assessment_res.reasoning,
                    "relevant_snippet": assessment_res.relevant_snippet,
                    "rel_type": rel.get("rel_type", "SATISFIES"),
                })
                logger.info(
                    f"Successfully stored SATISFIES edge: doc={document_id} -> ob={cand_id}, "
                    f"status={assessment_res.status.value}, confidence={assessment_res.confidence:.2f}"
                )

            except Exception as cand_err:
                logger.error(
                    f"Coverage assessment failed for obligation {cand_id} ({clause}): {cand_err}",
                    exc_info=True,
                )
                failed_assessments.append({
                    "obligation_id": str(cand_id),
                    "clause": clause,
                    "error": str(cand_err),
                })
                # Isolate failure so other candidates continue
                continue

        logger.info(
            f"Coverage mapping completed for document {document_id}: "
            f"candidates={len(candidates)}, successful={len(successful_mappings)}, "
            f"failed={len(failed_assessments)}"
        )

        return {
            "status": STATUS_COMPLETED,
            "document_id": document_id,
            "candidates_found": len(candidates),
            "assessments_completed": len(successful_mappings),
            "assessments_failed": len(failed_assessments),
            "successful_mappings": successful_mappings,
            "failed_assessments": failed_assessments,
        }


# -----------------------------------------------------------------------------
# Celery Tasks
# -----------------------------------------------------------------------------


@celery_app.task(name="app.workers.tasks.process_document_task")
def process_document_task(
    document_id: str,
    file_path: str,
    version_id: Optional[str] = None,
    provider: Optional[str] = None,
    trigger_coverage: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Background Celery task to parse uploaded file, extract text, save to EvidenceArtifact,
    and automatically trigger obligation extraction (if version_id provided) or
    coverage mapping against candidate obligations.
    """
    return asyncio.run(
        _process_document_pipeline(
            document_id=document_id,
            file_path=file_path,
            version_id=version_id,
            provider=provider,
            trigger_coverage=trigger_coverage,
        )
    )


@celery_app.task(name="app.workers.tasks.extract_regulatory_obligations_task")
def extract_regulatory_obligations_task(
    document_id: str,
    version_id: str,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Background Celery task dedicated to regulatory obligation extraction from an EvidenceArtifact.
    Chunks the document text, extracts structured obligations via LLM, persists them to PostgreSQL,
    and creates/links nodes in Neo4j.
    """
    return asyncio.run(
        _extract_regulatory_obligations_pipeline(
            document_id=document_id,
            version_id=version_id,
            provider=provider,
        )
    )


@celery_app.task(name="app.workers.tasks.map_evidence_coverage_task")
def map_evidence_coverage_task(
    document_id: str,
    framework: Optional[str] = None,
    version: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 10,
    score_threshold: Optional[float] = None,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Background Celery task for evidence coverage mapping (Phase 2, Step 5.3).
    Narrows candidate regulatory obligations using Qdrant similarity search,
    evaluates coverage for each candidate via CoverageService, and idempotently
    stores the resulting SATISFIES relationships in Neo4j.
    """
    return asyncio.run(
        _map_evidence_coverage_pipeline(
            document_id=document_id,
            framework=framework,
            version=version,
            category=category,
            limit=limit,
            score_threshold=score_threshold,
            provider=provider,
        )
    )


@celery_app.task(name="app.workers.tasks.coverage_mapping_task")
def coverage_mapping_task(
    document_id: str,
    framework: Optional[str] = None,
    version: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 10,
    score_threshold: Optional[float] = None,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Alias for map_evidence_coverage_task."""
    return map_evidence_coverage_task(
        document_id=document_id,
        framework=framework,
        version=version,
        category=category,
        limit=limit,
        score_threshold=score_threshold,
        provider=provider,
    )
