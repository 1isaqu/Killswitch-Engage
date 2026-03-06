"""Analytics (analiticos) router — aggregated market and trend data.

All routes are cached in Redis with a 1-hour TTL to avoid repeated
expensive DB queries (SECURITY_CHECKLIST § 4 — Redis com autenticação,
configurado via settings.REDIS_URL).

Security notes (SECURITY_CHECKLIST § 2 & 3):
    - All SQL uses parameterised $N placeholders.
    - Redis key names are static strings — never built from user input.
    - Cache errors are caught and the request falls through to the DB.
"""

from __future__ import annotations

import json
from typing import List

import redis
from fastapi import APIRouter, Depends, Query, Request

from src.backend.database import Database, get_db
from src.backend.middleware.security import limiter
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# ── Redis cache ───────────────────────────────────────────────────────────────

_cache: redis.Redis = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
CACHE_TTL_SECONDS: int = 60 * 60  # 1 hour


def _get_cached(key: str):
    """Fetch a value from Redis cache; return None on miss or error.

    Args:
        key (str): Cache key (must be a static string, not user-supplied).

    Returns:
        Parsed JSON value or None.
    """
    try:
        raw = _cache.get(key)
        return json.loads(raw) if raw else None
    except Exception:  # noqa: BLE001
        logger.warning("Redis cache miss or error for key '%s' — falling back to DB.", key)
        return None


def _set_cached(key: str, value: object) -> None:
    """Store a value in Redis cache with the default TTL.

    Args:
        key (str): Cache key.
        value: JSON-serialisable value.
    """
    try:
        _cache.setex(key, CACHE_TTL_SECONDS, json.dumps(value))
    except Exception:  # noqa: BLE001
        logger.warning("Failed to write Redis cache for key '%s'.", key)


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/preco_por_categoria", summary="Preço médio por categoria")
@limiter.limit("100/minute")
async def get_preco_por_categoria(
    request: Request,
    db: Database = Depends(get_db),
) -> List[dict]:
    """Return average game price grouped by category.

    Results are cached for 1 hour.

    Args:
        request: FastAPI request.
        db: Injected database dependency.

    Returns:
        List[dict]: ``[{categoria, preco_medio}]``
    """
    key = "analiticos:preco_por_categoria"
    if (cached := _get_cached(key)) is not None:
        return cached

    rows = await db.fetch("""
        SELECT c.nome AS categoria, ROUND(AVG(j.preco_base)::numeric, 2) AS preco_medio
        FROM jogos j
        JOIN jogos_categorias jc ON j.id = jc.jogo_id
        JOIN categorias c ON jc.categoria_id = c.id
        GROUP BY c.nome
        ORDER BY preco_medio DESC
        LIMIT 20
        """)
    payload = [dict(row) for row in rows]
    _set_cached(key, payload)
    return payload


@router.get("/tempo_por_genero", summary="Tempo médio de jogo por gênero")
@limiter.limit("100/minute")
async def get_tempo_por_genero(
    request: Request,
    db: Database = Depends(get_db),
) -> List[dict]:
    """Return average session time grouped by game genre.

    Args:
        request: FastAPI request.
        db: Injected database dependency.

    Returns:
        List[dict]: ``[{genero, media_horas}]``
    """
    key = "analiticos:tempo_por_genero"
    if (cached := _get_cached(key)) is not None:
        return cached

    rows = await db.fetch("""
        SELECT c.nome AS genero,
               ROUND(AVG(COALESCE(s.horas_jogadas, 0))::numeric, 2) AS media_horas
        FROM sessoes_jogo s
        JOIN jogos_categorias jc ON s.jogo_id = jc.jogo_id
        JOIN categorias c ON jc.categoria_id = c.id
        GROUP BY c.nome
        ORDER BY media_horas DESC
        LIMIT 20
        """)
    payload = [dict(row) for row in rows]
    _set_cached(key, payload)
    return payload


@router.get("/desenvolvedores/top", summary="Top desenvolvedores por avaliação")
@limiter.limit("100/minute")
async def get_top_devs(
    request: Request,
    db: Database = Depends(get_db),
) -> List[dict]:
    """Return top developers ranked by average game rating.

    Args:
        request: FastAPI request.
        db: Injected database dependency.

    Returns:
        List[dict]: ``[{desenvolvedor, media_avaliacao, total_jogos}]``
    """
    key = "analiticos:top_devs"
    if (cached := _get_cached(key)) is not None:
        return cached

    rows = await db.fetch("""
        SELECT d.nome AS desenvolvedor,
               ROUND(AVG(j.avaliacao_media)::numeric, 2) AS media_avaliacao,
               COUNT(j.id) AS total_jogos
        FROM jogos j
        JOIN desenvolvedores d ON j.dev_id = d.id
        GROUP BY d.nome
        HAVING COUNT(j.id) >= 3
        ORDER BY media_avaliacao DESC
        LIMIT 10
        """)
    payload = [dict(row) for row in rows]
    _set_cached(key, payload)
    return payload


@router.get("/tendencias", summary="Jogos em alta nos últimos N dias")
@limiter.limit("100/minute")
async def get_tendencias(
    request: Request,
    dias: int = Query(default=30, ge=1, le=365, description="Janela de dias"),
    db: Database = Depends(get_db),
) -> List[dict]:
    """Return the most-played games in the last ``dias`` days.

    Cache key includes the ``dias`` parameter so different windows are
    cached independently.

    Args:
        request: FastAPI request.
        dias: Lookback window in days (1–365).
        db: Injected database dependency.

    Returns:
        List[dict]: ``[{id, titulo, acessos}]`` ordered by session count.
    """
    key = f"analiticos:tendencias:{dias}"
    if (cached := _get_cached(key)) is not None:
        return cached

    rows = await db.fetch(
        """
        SELECT j.id, j.titulo, COUNT(s.id) AS acessos
        FROM sessoes_jogo s
        JOIN jogos j ON s.jogo_id = j.id
        WHERE s.fim > NOW() - ($1 || ' days')::interval
        GROUP BY j.id, j.titulo
        ORDER BY acessos DESC
        LIMIT 10
        """,
        str(dias),
    )
    payload = [dict(row) for row in rows]
    _set_cached(key, payload)
    return payload
