"""Security middleware — rate limiting, HSTS, CSP, and security headers.

Implements SECURITY_CHECKLIST requirements:
    § 3 – API: Headers de segurança, Rate limiting, CORS configurado, Debug off.

The ``_rate_limit_exceeded_handler`` is re-exported here so the app factory
can register it without importing from slowapi directly.
"""

from __future__ import annotations

from typing import Callable

from fastapi import Request, Response
from slowapi import Limiter, _rate_limit_exceeded_handler  # noqa: F401 (re-exported)
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

# ── Rate limiter (shared across all routes) ───────────────────────────────────

limiter = Limiter(key_func=get_remote_address)
"""Global SlowAPI limiter; apply per-route with ``@limiter.limit(...)``."""


# ── Security headers middleware ────────────────────────────────────────────────


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Inject security-related HTTP response headers on every request.

    Headers added (per SECURITY_CHECKLIST § 3):
        - ``X-Content-Type-Options: nosniff`` — prevent MIME type sniffing.
        - ``X-Frame-Options: DENY`` — prevent clickjacking.
        - ``Strict-Transport-Security`` — enforce HTTPS for 1 year.
        - ``Content-Security-Policy: default-src 'self'`` — restrict resource
          origins to the same domain.
        - ``X-XSS-Protection: 1; mode=block`` — legacy XSS filter hint.
        - ``Referrer-Policy: strict-origin-when-cross-origin`` — limit referer
          header exposure.
        - ``Permissions-Policy`` — disable unused browser features.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> StarletteResponse:
        """Process the request and attach security headers to the response.

        Args:
            request (Request): Incoming FastAPI request.
            call_next (Callable): Next middleware or route handler.

        Returns:
            StarletteResponse: Response with security headers attached.
        """
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains; preload"
        )
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        return response
