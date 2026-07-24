import logging
from app.workers.celery_app import celery_app
from app.services.file_processor import FileProcessor

logger = logging.getLogger(__name__)

@celery_app.task(name="app.workers.tasks.process_document_task")
def process_document_task(document_id: str, file_path: str):
    """
    Background Celery task to download an uploaded file and extract text and counts.
    """
    logger.info(f"Starting background processing for document_id={document_id}, file_path={file_path}")
    try:
        processor = FileProcessor()
        result = processor.process_file(file_path)
        logger.info(f"Successfully processed document_id={document_id}: page_count={result['page_count']}, word_count={result['word_count']}")
        return {
            "document_id": document_id,
            "status": "completed",
            "page_count": result["page_count"],
            "word_count": result["word_count"],
            "text_length": len(result["text"]),
        }
    except Exception as e:
        logger.error(f"Error processing document_id={document_id}: {e}")
        return {
            "document_id": document_id,
            "status": "failed",
            "error": str(e)
        }
