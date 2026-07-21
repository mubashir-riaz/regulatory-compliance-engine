import os
from uuid import UUID
from typing import Optional
from fastapi import UploadFile, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

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

        # Placeholder response indicating the document is ready for processing
        return {
            "message": "Document uploaded successfully and is ready for processing",
            "filename": filename,
            "framework_id": str(framework_id),
            "version_id": str(version_id),
            "title": title,
            "status": "pending"
        }
