import os
import uuid
from uuid import UUID
from typing import Optional
from fastapi import UploadFile, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.workers.tasks import process_document_task
from app.repositories.evidence_repo import EvidenceArtifactRepository
from app.models.evidence import EvidenceArtifact
from app.services.file_processor import FileProcessor

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

        # 1. Generate document ID
        document_id = uuid.uuid4()
        
        # 2. Create database record
        repo = EvidenceArtifactRepository(self.db)
        new_artifact = EvidenceArtifact(
            id=document_id,
            organization_id=UUID("11111111-1111-1111-1111-111111111111"),
            name=title or filename,
            file_path=filename,
            status="PENDING"
        )
        await repo.create(new_artifact)
        await self.db.commit()

        # 3. Upload file to MinIO
        processor = FileProcessor()
        content = await file.read()
        processor.upload_file(filename, content)

        # 4. Enqueue background Celery task
        task = process_document_task.delay(document_id=str(document_id), file_path=filename)

        return {
            "task_id": task.id,
            "status": "queued",
            "document_id": str(document_id)
        }

    async def get_document_status(self, document_id: UUID):
        """
        Retrieve EvidenceArtifact from database and return processing status and text preview.
        Raises 404 HTTP exception if document does not exist.
        """
        repo = EvidenceArtifactRepository(self.db)
        artifact = await repo.get_by_id(document_id)

        if not artifact:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID '{document_id}' not found."
            )

        current_status = artifact.status.upper() if artifact.status else "PENDING"
        text_preview = None

        # Return text_preview (first 300 chars) only if processing is completed
        if current_status == "COMPLETED" and artifact.extracted_text:
            text_preview = artifact.extracted_text[:300]

        return {
            "document_id": str(artifact.id),
            "status": current_status,
            "text_preview": text_preview,
            "page_count": artifact.page_count,
            "word_count": artifact.word_count,
        }