from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class TenantMiddleware(BaseHTTPMiddleware):
    """
    Middleware to extract the tenant ID from the 'X-Tenant-ID' request header
    and store it in request.state.tenant_id for tenant isolation.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        tenant_id = request.headers.get("X-Tenant-ID")
        request.state.tenant_id = tenant_id
        response = await call_next(request)
        return response
