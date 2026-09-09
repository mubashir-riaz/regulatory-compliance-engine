from fastapi import APIRouter
from app.api.v1.endpoints import compliance, documents, impact

v1_router = APIRouter()
v1_router.include_router(documents.router, prefix="/documents", tags=["documents"])
v1_router.include_router(compliance.router, prefix="/compliance", tags=["compliance"])
v1_router.include_router(impact.router, prefix="/impact", tags=["impact"])

