import pytest
from fastapi import FastAPI, Request
from httpx import AsyncClient, ASGITransport
from app.middleware.tenant import TenantMiddleware


@pytest.mark.asyncio
async def test_tenant_middleware_with_header():
    test_app = FastAPI()
    test_app.add_middleware(TenantMiddleware)

    @test_app.get("/test-tenant")
    async def sample_endpoint(request: Request):
        return {"tenant_id": getattr(request.state, "tenant_id", None)}

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/test-tenant", headers={"X-Tenant-ID": "tenant-abc-123"})
        assert response.status_code == 200
        assert response.json() == {"tenant_id": "tenant-abc-123"}


@pytest.mark.asyncio
async def test_tenant_middleware_without_header():
    test_app = FastAPI()
    test_app.add_middleware(TenantMiddleware)

    @test_app.get("/test-tenant")
    async def sample_endpoint(request: Request):
        return {"tenant_id": getattr(request.state, "tenant_id", None)}

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/test-tenant")
        assert response.status_code == 200
        assert response.json() == {"tenant_id": None}
