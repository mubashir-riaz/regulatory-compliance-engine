import pytest
from fastapi import FastAPI, Request
from httpx import AsyncClient, ASGITransport
from app.middleware.tenant import TenantMiddleware


@pytest.mark.asyncio
async def test_tenant_middleware_simple(caplog):
    test_app = FastAPI()
    test_app.add_middleware(TenantMiddleware)

    @test_app.get("/test-tenant")
    async def sample_endpoint(request: Request):
        return {"tenant_id": getattr(request.state, "tenant_id", None)}

    transport = ASGITransport(app=test_app)
    with caplog.at_level("INFO"):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/test-tenant", headers={"X-Tenant-ID": "test-org"})
            assert response.status_code == 200
            assert response.json() == {"tenant_id": "test-org"}
            assert "Captured tenant ID: test-org" in caplog.text


@pytest.mark.asyncio
async def test_tenant_middleware_with_header(caplog):
    test_app = FastAPI()
    test_app.add_middleware(TenantMiddleware)

    @test_app.get("/test-tenant")
    async def sample_endpoint(request: Request):
        return {"tenant_id": getattr(request.state, "tenant_id", None)}

    transport = ASGITransport(app=test_app)
    with caplog.at_level("INFO"):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/test-tenant", headers={"X-Tenant-ID": "tenant-abc-123"})
            assert response.status_code == 200
            assert response.json() == {"tenant_id": "tenant-abc-123"}
            assert "Captured tenant ID: tenant-abc-123" in caplog.text


@pytest.mark.asyncio
async def test_tenant_middleware_without_header(caplog):
    test_app = FastAPI()
    test_app.add_middleware(TenantMiddleware)

    @test_app.get("/test-tenant")
    async def sample_endpoint(request: Request):
        return {"tenant_id": getattr(request.state, "tenant_id", None)}

    transport = ASGITransport(app=test_app)
    with caplog.at_level("INFO"):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/test-tenant")
            assert response.status_code == 200
            assert response.json() == {"tenant_id": None}
            assert "No tenant ID was provided" in caplog.text
