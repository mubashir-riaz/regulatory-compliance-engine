import os
import uuid
from uuid import UUID
from typing import Optional
from fastapi import UploadFile, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.workers.tasks import process_document_task

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
}

class DocumentService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def upload_document(
        self,
        file: UploadFile,
        framework_id: UUID,
        version_id: UUID,
        title: Optional[str] = None
    ):
        filename = file.filename or ""
        ext = os.path.splitext(filename.lower())[1]

        # Validate file type using extension or MIME type
        if ext not in ALLOWED_EXTENSIONS and file.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type. Allowed types are: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

        # Generate a temporary document ID for tracking before database integration
        document_id = uuid.uuid4()

        # Enqueue background Celery task
        task = process_document_task.delay(document_id=str(document_id), file_path=filename)

        return {
            "task_id": task.id,
            "status": "queued"
        }
