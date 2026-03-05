from fastapi import APIRouter, Depends, Query, Request
from typing import List
from app.models.database import get_db, Database
from app.models import queries, schemas
from app.utils.exceptions import NotFoundError
from app.utils.security import limiter
from uuid import UUID

router = APIRouter()

@router.get("/", response_model=List[schemas.JogoBase])
@limiter.limit("100/minute")
async def list_jogos(
    request: Request,
    categoria: str = None,
    limite: int = 10,
    offset: int = 0,
    db: Database = Depends(get_db)
):
    if categoria:
        rows = await db.fetch(queries.JOGOS_BY_CATEGORIA, categoria, limite, offset)
    else:
        # Consulta base com nomes de colunas oficiais
        rows = await db.fetch("SELECT id, titulo, preco_base, avaliacao_media, total_avaliacoes, idade_requerida FROM jogos LIMIT $1 OFFSET $2", limite, offset)
    return [dict(row) for row in rows]

@router.get("/{id}", response_model=schemas.JogoDetalhes)
@limiter.limit("100/minute")
async def get_jogo(request: Request, id: UUID, db: Database = Depends(get_db)):
    row = await db.fetchrow(queries.JOGO_DETALHES, id)
    if not row:
        raise NotFoundError("Jogo não encontrado")
    return dict(row)

@router.get("/{id}/similares", response_model=List[schemas.SimilarJogo])
@limiter.limit("100/minute")
async def get_similares(request: Request, id: UUID, limite: int = 5, db: Database = Depends(get_db)):
    rows = await db.fetch(queries.JOGOS_SIMILARES, id, limite)
    return [dict(row) for row in rows]

@router.get("/{id}/achievement_tier", response_model=schemas.AchievementTier)
@limiter.limit("100/minute")
async def get_achievement_tier(request: Request, id: UUID, db: Database = Depends(get_db)):
    row = await db.fetchrow(queries.ACHIEVEMENT_TIER, id)
    if not row:
        raise NotFoundError("Jogo não encontrado")
    return dict(row)

@router.get("/{id}/nota_ajustada")
@limiter.limit("100/minute")
async def get_nota_ajustada(request: Request, id: UUID, db: Database = Depends(get_db)):
    row = await db.fetchrow("SELECT preco_base, avaliacao_media FROM jogos WHERE id = $1", id)
    if not row:
        raise NotFoundError()
    
    preco = row['preco_base']
    nota = row['avaliacao_media']
    
    # Exemplo de ajuste baseado no insight que jogos caros tendem a ter notas maiores
    ajuste = -0.5 if preco > 50 else 0.0
    nota_ajustada = max(0, min(10, nota + ajuste))
    
    return {"id": id, "nota_original": nota, "nota_ajustada": nota_ajustada, "fator_preco": ajuste}
