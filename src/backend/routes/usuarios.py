"""Users (usuarios) router — user profile, library, and session endpoints.

Security notes (SECURITY_CHECKLIST § 2 & 3):
    - All SQL uses parameterised $N placeholders.
    - Session inserts accept typed parameters only (UUID, float, str) — no raw
      user-supplied SQL fragments.
    - Rate-limited to 100 req/min per IP.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request

from src.backend.database import Database, get_db
from src.backend.middleware.security import limiter
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/{usuario_id}/biblioteca", summary="Biblioteca de jogos do usuário")
@limiter.limit("100/minute")
async def get_biblioteca(
    request: Request,
    usuario_id: UUID,
    db: Database = Depends(get_db),
) -> list:
    """Return all games in a user's library.

    Args:
        request: FastAPI request.
        usuario_id: User UUID — validated by Pydantic.
        db: Injected database dependency.

    Returns:
        list: Game records from the user's session history.
    """
    rows = await db.fetch(
        """
        SELECT DISTINCT j.id, j.titulo, j.preco_base, j.avaliacao_media,
               SUM(COALESCE(s.horas_jogadas, 0)) AS total_horas
        FROM sessoes_jogo s
        JOIN jogos j ON s.jogo_id = j.id
        WHERE s.usuario_id = $1
        GROUP BY j.id, j.titulo, j.preco_base, j.avaliacao_media
        ORDER BY total_horas DESC
        """,
        usuario_id,
    )
    return [dict(row) for row in rows]


@router.get("/{usuario_id}/estatisticas", summary="Estatísticas de jogo do usuário")
@limiter.limit("100/minute")
async def get_estatisticas(
    request: Request,
    usuario_id: UUID,
    db: Database = Depends(get_db),
) -> dict:
    """Return aggregated play statistics for a user.

    Args:
        request: FastAPI request.
        usuario_id: User UUID.
        db: Injected database dependency.

    Returns:
        dict: ``{total_jogos, total_horas, categoria_preferida}``.
    """
    row = await db.fetchrow(
        """
        SELECT
            COUNT(DISTINCT s.jogo_id)          AS total_jogos,
            COALESCE(SUM(s.horas_jogadas), 0)  AS total_horas,
            NULL                               AS categoria_preferida
        FROM sessoes_jogo s
        WHERE s.usuario_id = $1
        """,
        usuario_id,
    )
    if not row:
        return {"total_jogos": 0, "total_horas": 0.0, "categoria_preferida": None}
    return dict(row)


@router.post("/{usuario_id}/sessao", summary="Registrar sessão de jogo")
@limiter.limit("60/minute")
async def post_sessao(
    request: Request,
    usuario_id: UUID,
    jogo_id: UUID,
    minutos: float = Query(..., gt=0, description="Duração da sessão em minutos"),
    plataforma: str = Query(default="PC", max_length=50),
    db: Database = Depends(get_db),
) -> dict:
    """Record a new game session for a user.

    Args:
        request: FastAPI request.
        usuario_id: User UUID.
        jogo_id: Game UUID.
        minutos: Session duration in minutes (must be > 0).
        plataforma: Platform name (max 50 chars).
        db: Injected database dependency.

    Returns:
        dict: Confirmation with session details.
    """
    fim = datetime.now()
    inicio = fim - timedelta(minutes=minutos)
    await db.execute(
        """
        INSERT INTO sessoes_jogo (usuario_id, jogo_id, inicio, fim, plataforma)
        VALUES ($1, $2, $3, $4, $5)
        """,
        usuario_id,
        jogo_id,
        inicio,
        fim,
        plataforma,
    )
    # Security: log action without PII details (SECURITY_CHECKLIST § 3)
    logger.info("Session recorded for user [...] game [...] %.1f min", minutos)
    return {
        "status": "sessão registrada",
        "usuario_id": str(usuario_id),
        "jogo_id": str(jogo_id),
        "duracao_min": minutos,
    }
