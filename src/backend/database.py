"""Async PostgreSQL connection pool — database layer for the FastAPI backend.

Wraps ``asyncpg`` with SSL-aware connection management. SSL mode is driven
entirely by the ``DB_SSL_MODE`` setting — never hardcoded.

Security notes (SECURITY_CHECKLIST § 2):
    - All queries must use parameterised statements (``$1``, ``$2``, …).
      Never interpolate user input into SQL strings.
    - SSL is ``require`` by default; only ``disable`` for local Docker compose
      (set ``DB_SSL_MODE=disable`` in the container env).
    - Connection credentials come exclusively from ``src.config.settings``.
"""

from __future__ import annotations

import ssl
from typing import Any, Optional

import asyncpg

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Database:
    """Async connection pool wrapping ``asyncpg``.

    Usage:
        Instantiate once at module level; call :meth:`connect` in the
        FastAPI ``startup`` event and :meth:`disconnect`` in ``shutdown``.

    Attributes:
        pool (asyncpg.Pool | None): Active connection pool, or ``None``.
    """

    def __init__(self) -> None:
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """Create the asyncpg connection pool.

        Respects ``DB_SSL_MODE``:
            - ``disable``  — no SSL (local Docker only).
            - ``insecure`` — SSL without certificate verification (debug only).
            - ``require``  — SSL with full certificate verification (default).

        Does nothing if a pool is already open.

        Raises:
            Logs a warning and sets ``pool = None`` on failure (graceful
            degradation; endpoints that need the DB will return 503).
        """
        if self.pool:
            return

        ssl_param = self._build_ssl_context()

        try:
            self.pool = await asyncpg.create_pool(
                settings.DATABASE_URL,
                min_size=1,
                max_size=10,
                statement_cache_size=0,  # required for PgBouncer / Supabase pooler
                ssl=ssl_param,
            )
            logger.info("Database pool established (ssl_mode=%s).", settings.DB_SSL_MODE)
        except Exception:
            logger.exception("Failed to connect to the database. Running in DB-less mode.")
            self.pool = None

    async def disconnect(self) -> None:
        """Close the connection pool gracefully."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Database pool closed.")

    # ── Query helpers ──────────────────────────────────────────────────────

    async def fetch(self, query: str, *args: Any) -> list:
        """Execute a SELECT query and return all rows.

        Args:
            query (str): Parameterised SQL (use ``$1``, ``$2``, …).
            *args: Positional parameters bound to the placeholders.

        Returns:
            list: List of asyncpg ``Record`` objects.
        """
        await self._ensure_connected()
        async with self.pool.acquire() as conn:  # type: ignore[union-attr]
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> Optional[asyncpg.Record]:
        """Execute a SELECT query and return the first row.

        Args:
            query (str): Parameterised SQL.
            *args: Positional parameters.

        Returns:
            asyncpg.Record | None: First result row, or ``None`` if empty.
        """
        await self._ensure_connected()
        async with self.pool.acquire() as conn:  # type: ignore[union-attr]
            return await conn.fetchrow(query, *args)

    async def execute(self, query: str, *args: Any) -> str:
        """Execute a DML statement (INSERT / UPDATE / DELETE).

        Args:
            query (str): Parameterised SQL.
            *args: Positional parameters.

        Returns:
            str: Command completion tag returned by PostgreSQL.
        """
        await self._ensure_connected()
        async with self.pool.acquire() as conn:  # type: ignore[union-attr]
            return await conn.execute(query, *args)

    # ── Internal helpers ───────────────────────────────────────────────────

    async def _ensure_connected(self) -> None:
        """Re-open the pool if it was closed or never initialised."""
        if not self.pool:
            await self.connect()

    @staticmethod
    def _build_ssl_context() -> Optional[ssl.SSLContext]:
        """Build an SSL context based on ``DB_SSL_MODE``.

        Returns:
            ssl.SSLContext | None:
                - ``None``                   for ``disable``
                - Unverified ``SSLContext``  for ``insecure`` (debug only)
                - Verified ``SSLContext``    for ``require`` (default)
        """
        mode = settings.DB_SSL_MODE

        if mode == "disable":
            logger.info("DB_SSL_MODE=disable — connecting without SSL (local/Docker only).")
            return None

        if mode == "insecure":
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            # Security: this is logged as WARNING, not INFO, per SECURITY_CHECKLIST § 2
            logger.warning(
                "DB_SSL_MODE=insecure — SSL certificate verification disabled. "
                "Use only for local debugging."
            )
            return ctx

        # Default: require — full certificate verification
        logger.info("DB_SSL_MODE=require — SSL with certificate verification enabled.")
        return ssl.create_default_context()


# ── Module-level singleton (one pool per process) ─────────────────────────────

db = Database()


async def get_db() -> Database:
    """FastAPI dependency that returns the shared database instance.

    Returns:
        Database: The module-level :data:`db` singleton.
    """
    return db
