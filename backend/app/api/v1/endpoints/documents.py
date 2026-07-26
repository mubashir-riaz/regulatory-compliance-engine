from uuid import UUID
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.services.document_service import DocumentService
from app.schemas.document import DocumentStatusResponse

router = APIRouter()

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    framework_id: UUID = Form(...),
    version_id: UUID = Form(...),
    title: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a compliance document.
    """
    document_service = DocumentService(db)
    return await document_service.upload_document(
        file=file,
        framework_id=framework_id,
        version_id=version_id,
        title=title
    )

@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    document_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get document processing status and extracted content.
    """
    document_service = DocumentService(db)
    return await document_service.get_document_status(document_id)
