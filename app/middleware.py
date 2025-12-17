import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("medrag")


class PrivacyAwareLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs only metadata (path, status, latency). Does NOT log request body or retrieved text.
    This is a basic privacy-aware posture suitable for clinical tooling.
    """

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - start) * 1000

        logger.info(
            "request path=%s status=%s latency_ms=%.2f",
            request.url.path,
            response.status_code,
            ms,
        )
        return response
