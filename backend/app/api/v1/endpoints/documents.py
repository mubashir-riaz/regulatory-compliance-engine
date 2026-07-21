from uuid import UUID
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form

router = APIRouter()

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    framework_id: UUID = Form(...),
    version_id: UUID = Form(...),
    title: Optional[str] = Form(None)
):
    """
    Upload a compliance document.
    """
    return {
        "message": "Upload endpoint created"
    }
