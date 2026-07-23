import logging
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

@celery_app.task(name="app.workers.tasks.process_document_task")
def process_document_task(document_id: str, file_path: str):
    """
    Background Celery task skeleton to process an uploaded document.
    """
    logger.info(f"Starting background processing for document_id={document_id}, file_path={file_path}")
    # Logic for parsing and extracting content will be implemented in later steps
    logger.info(f"Finished background processing for document_id={document_id}")
    return {"document_id": document_id, "status": "completed"}
