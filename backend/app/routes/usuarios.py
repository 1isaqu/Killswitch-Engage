from fastapi import APIRouter, Depends, Request
from typing import List
from app.models.database import get_db, Database
from app.models import queries, schemas
from app.utils.exceptions import NotFoundError
from app.utils.security import limiter
from uuid import UUID

router = APIRouter()

@router.get("/{id}/biblioteca", response_model=List[schemas.UserLibraryItem])
@limiter.limit("100/minute")
async def get_biblioteca(request: Request, id: UUID, db: Database = Depends(get_db)):
    rows = await db.fetch(queries.USER_LIBRARY, id)
    return [dict(row) for row in rows]

@router.get("/{id}/estatisticas", response_model=schemas.UserStats)
@limiter.limit("100/minute")
async def get_estatisticas(request: Request, id: UUID, db: Database = Depends(get_db)):
    row = await db.fetchrow(queries.USER_STATS, id)
    if not row:
        return {"total_jogos": 0, "total_horas": 0, "categoria_preferida": None}
    return dict(row)

@router.post("/{id}/sessao")
@limiter.limit("100/minute")
async def post_sessao(request: Request, id: UUID, jogo_id: UUID, minutos: float, plataforma: str = 'PC', db: Database = Depends(get_db)):
    from datetime import datetime, timedelta
    fim = datetime.now()
    inicio = fim - timedelta(minutes=minutos)
    await db.execute(
        "INSERT INTO sessoes_jogo (usuario_id, jogo_id, inicio, fim, plataforma) VALUES ($1, $2, $3, $4, $5)",
        id, jogo_id, inicio, fim, plataforma
    )
    return {"status": "sessão registrada", "usuario_id": str(id), "jogo_id": str(jogo_id), "duracao_min": minutos}
