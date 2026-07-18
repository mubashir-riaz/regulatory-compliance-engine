from fastapi import FastAPI
from app.api.router import api_router
from app.core.config import settings
from app.core.database import engine
from app.db.base import Base

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Optional: create tables on startup (for dev only)
@app.on_event("startup")
async def startup():
    # In production, use Alembic migrations instead
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)