from pydantic import BaseModel, ConfigDict
from uuid import UUID
from datetime import datetime
from typing import Optional

class DocumentCreate(BaseModel):
    framework_id: UUID
    version_id: UUID
    title: Optional[str] = None

class DocumentResponse(BaseModel):
    id: UUID
    organization_id: UUID
    name: str
    file_path: str
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
