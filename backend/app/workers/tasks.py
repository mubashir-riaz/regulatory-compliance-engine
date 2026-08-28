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
from app.schemas.extraction import ExtractedObligation
from app.schemas.graph_models import (
    ControlCategory as Neo4jControlCategory,
    RegulatoryObligation as Neo4jRegulatoryObligation,
    RegulatoryVersion as Neo4jRegulatoryVersion,
)
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
) -> Dict[str, Any]:
    """
    Async implementation of the document processing pipeline.
    Extracts text/metadata from file, saves to DB, and optionally triggers obligation extraction.
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


# -----------------------------------------------------------------------------
# Celery Tasks
# -----------------------------------------------------------------------------


@celery_app.task(name="app.workers.tasks.process_document_task")
def process_document_task(
    document_id: str,
    file_path: str,
    version_id: Optional[str] = None,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Background Celery task to parse uploaded file, extract text, save to EvidenceArtifact,
    and optionally trigger regulatory obligation extraction if version_id is provided.
    """
    return asyncio.run(
        _process_document_pipeline(
            document_id=document_id,
            file_path=file_path,
            version_id=version_id,
            provider=provider,
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
