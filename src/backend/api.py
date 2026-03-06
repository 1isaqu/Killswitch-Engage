"""FastAPI application factory — entry point for the Killswitch Engage API.

Run locally:
    uvicorn src.backend.api:app --reload

Production (Docker):
    uvicorn src.backend.api:app --host 0.0.0.0 --port 8000 --proxy-headers

Security notes (SECURITY_CHECKLIST § 3):
    - CORS origins are loaded from ``settings.ALLOWED_ORIGINS`` (env-driven).
    - DEBUG mode is off in all non-development environments.
    - Rate limiting is attached via SlowAPI.
    - Security headers are added by ``SecurityHeadersMiddleware``.
    - Logs never expose PII (user data or credentials).
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded

from src.backend.database import db
from src.backend.middleware.security import (
    SecurityHeadersMiddleware,
    _rate_limit_exceeded_handler,
    limiter,
)
from src.backend.routes import analiticos, jogos, recomendacoes, usuarios
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Manage database pool lifecycle.

    Opens the asyncpg connection pool on startup and closes it on shutdown.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None: Control is yielded to the application during its lifetime.
    """
    await db.connect()
    yield
    await db.disconnect()


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application.

    Returns:
        FastAPI: Fully configured application instance.
    """
    application = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="API para exploração e recomendação de jogos da Steam",
        debug=settings.DEBUG,
        lifespan=lifespan,
        # Disable OpenAPI docs in production (SECURITY_CHECKLIST § 3)
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
    )

    # ── Rate limiting ──────────────────────────────────────────────────────
    application.state.limiter = limiter
    application.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── Security headers (must come before CORS) ───────────────────────────
    application.add_middleware(SecurityHeadersMiddleware)

    # ── CORS (specific origins only — never '*') ───────────────────────────
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ── Request logging middleware ─────────────────────────────────────────
    @application.middleware("http")
    async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1_000
        # Security: log path + status only — never body/params (SECURITY_CHECKLIST § 3)
        logger.info(
            "%s %s → %d (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    # ── Routes ────────────────────────────────────────────────────────────
    application.include_router(jogos.router, prefix="/api/v1/jogos", tags=["Jogos"])
    application.include_router(usuarios.router, prefix="/api/v1/usuarios", tags=["Usuários"])
    application.include_router(analiticos.router, prefix="/api/v1/analiticos", tags=["Analíticos"])
    application.include_router(
        recomendacoes.router,
        prefix="/api/v1/recomendacoes",
        tags=["Recomendações"],
    )

    # ── Root & health ──────────────────────────────────────────────────────
    @application.get("/", include_in_schema=False)
    async def root() -> dict:
        return {"app": settings.APP_NAME, "docs": "/docs", "status": "online"}

    @application.get("/health", tags=["Infra"])
    async def health() -> dict:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "version": settings.APP_VERSION,
        }

    return application


# ── Module-level app instance (used by uvicorn) ───────────────────────────────

app = create_app()
