"""Games (jogos) router — browse and search the Steam game catalogue.

Security notes (SECURITY_CHECKLIST § 2 & 3):
    - All SQL uses parameterised $N placeholders — no string interpolation.
    - Rate-limited to 100 req/min per IP.
    - Input validation enforced by Pydantic (UUID type, Query bounds).
"""

from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request

from src.backend.database import Database, get_db
from src.backend.middleware.security import limiter
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/", summary="Listar jogos do catálogo")
@limiter.limit("100/minute")
async def list_jogos(
    request: Request,
    categoria: Optional[str] = Query(default=None, description="Filtrar por categoria"),
    limite: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Database = Depends(get_db),
) -> List[dict]:
    """Return a paginated list of games from the catalogue.

    Args:
        request: FastAPI request (required by SlowAPI).
        categoria: Optional category filter.
        limite: Page size (1–100).
        offset: Pagination offset.
        db: Injected database dependency.

    Returns:
        List[dict]: Game records with id, titulo, preco_base, avaliacao_media.
    """
    if categoria:
        rows = await db.fetch(
            """
            SELECT j.id, j.titulo, j.preco_base, j.avaliacao_media, j.total_avaliacoes
            FROM jogos j
            JOIN jogos_categorias jc ON j.id = jc.jogo_id
            JOIN categorias c ON jc.categoria_id = c.id
            WHERE c.nome = $1
            LIMIT $2 OFFSET $3
            """,
            categoria,
            limite,
            offset,
        )
    else:
        rows = await db.fetch(
            """
            SELECT id, titulo, preco_base, avaliacao_media, total_avaliacoes, idade_requerida
            FROM jogos
            LIMIT $1 OFFSET $2
            """,
            limite,
            offset,
        )
    return [dict(row) for row in rows]


@router.get("/{jogo_id}", summary="Detalhes de um jogo")
@limiter.limit("100/minute")
async def get_jogo(
    request: Request,
    jogo_id: UUID,
    db: Database = Depends(get_db),
) -> dict:
    """Return detailed metadata for a single game.

    Args:
        request: FastAPI request.
        jogo_id: Game UUID — validated by Pydantic.
        db: Injected database dependency.

    Returns:
        dict: Full game record.

    Raises:
        HTTPException 404: If the game does not exist.
    """
    from fastapi import HTTPException

    row = await db.fetchrow("SELECT * FROM jogos WHERE id = $1", jogo_id)
    if not row:
        raise HTTPException(status_code=404, detail="Jogo não encontrado.")
    return dict(row)


@router.get("/{jogo_id}/nota_ajustada", summary="Nota de avaliação ajustada por preço")
@limiter.limit("100/minute")
async def get_nota_ajustada(
    request: Request,
    jogo_id: UUID,
    db: Database = Depends(get_db),
) -> dict:
    """Return a price-adjusted rating score for a game.

    Games priced above R$50 receive a small negative adjustment,
    reflecting that expensive games have higher quality expectations.

    Args:
        request: FastAPI request.
        jogo_id: Game UUID.
        db: Injected database dependency.

    Returns:
        dict: ``{id, nota_original, nota_ajustada, fator_preco}``.
    """
    from fastapi import HTTPException

    row = await db.fetchrow("SELECT preco_base, avaliacao_media FROM jogos WHERE id = $1", jogo_id)
    if not row:
        raise HTTPException(status_code=404, detail="Jogo não encontrado.")

    preco: float = row["preco_base"] or 0.0
    nota: float = row["avaliacao_media"] or 0.0
    ajuste: float = -0.5 if preco > 50 else 0.0
    nota_ajustada = max(0.0, min(10.0, nota + ajuste))

    return {
        "id": str(jogo_id),
        "nota_original": nota,
        "nota_ajustada": nota_ajustada,
        "fator_preco": ajuste,
    }
