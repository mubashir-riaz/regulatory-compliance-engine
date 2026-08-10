import asyncio
import logging
from uuid import UUID
from typing import Optional, Dict, Any

from app.workers.celery_app import celery_app
from app.services.file_processor import FileProcessor
from app.core.database import AsyncSessionLocal
from app.repositories.evidence_repo import EvidenceArtifactRepository

logger = logging.getLogger(__name__)

STATUS_PROCESSING = "PROCESSING"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"

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
        logger.error(f"Failed to update EvidenceArtifact in database for document_id={document_id}: {db_exc}", exc_info=True)
        return False

async def _process_document_pipeline(document_id: str, file_path: str) -> Dict[str, Any]:
    """
    Async implementation of the document processing pipeline.
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
            f"Successfully finished processing document_id={document_id}: "
            f"page_count={page_count}, word_count={word_count}, text_length={len(extracted_text)}"
        )
        
        return {
            "document_id": document_id,
            "status": STATUS_COMPLETED,
            "page_count": page_count,
            "word_count": word_count,
            "text_length": len(extracted_text),
        }
        
    except Exception as e:
        logger.error(f"Error during document processing pipeline for document_id={document_id}: {e}", exc_info=True)
        
        # 3. Update processing status to FAILED on errors
        await _update_evidence_artifact(document_id=document_id, status=STATUS_FAILED)
        
        return {
            "document_id": document_id,
            "status": STATUS_FAILED,
            "error": str(e),
        }

@celery_app.task(name="app.workers.tasks.process_document_task")
def process_document_task(document_id: str, file_path: str):
    """
    Background Celery task to download an uploaded file, extract text and page/word counts,
    save results to EvidenceArtifact, and manage task status (PROCESSING -> COMPLETED / FAILED).
    """
    return asyncio.run(_process_document_pipeline(document_id=document_id, file_path=file_path))

