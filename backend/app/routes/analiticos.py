from fastapi import APIRouter, Depends, Request
from typing import List
import json
import redis
from app.models.database import get_db, Database
from app.models import queries, schemas
from app.config import settings
from app.utils.security import limiter

router = APIRouter()

# FIXED: adicionar cache Redis para rotas analíticas que não mudam com alta frequência
_cache = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
CACHE_TTL_SECONDS = 60 * 60  # 1h


@router.get("/preco_por_categoria", response_model=List[schemas.AnaliticoPreco])
@limiter.limit("100/minute")
async def get_preco_por_categoria(request: Request, db: Database = Depends(get_db)):
    cache_key = "analiticos:preco_por_categoria"
    cached = _cache.get(cache_key)
    if cached:
        return json.loads(cached)

    rows = await db.fetch(queries.PRECO_MEDIO_POR_CATEGORIA)
    payload = [dict(row) for row in rows]
    _cache.setex(cache_key, CACHE_TTL_SECONDS, json.dumps(payload))
    return payload


@router.get("/tempo_por_genero", response_model=List[schemas.AnaliticoTempo])
@limiter.limit("100/minute")
async def get_tempo_por_genero(request: Request, db: Database = Depends(get_db)):
    cache_key = "analiticos:tempo_por_genero"
    cached = _cache.get(cache_key)
    if cached:
        return json.loads(cached)

    rows = await db.fetch(queries.TEMPO_MEDIO_POR_GENERO)
    payload = [dict(row) for row in rows]
    _cache.setex(cache_key, CACHE_TTL_SECONDS, json.dumps(payload))
    return payload


@router.get("/desenvolvedores/top", response_model=List[schemas.TopDev])
@limiter.limit("100/minute")
async def get_top_devs(request: Request, db: Database = Depends(get_db)):
    cache_key = "analiticos:top_devs"
    cached = _cache.get(cache_key)
    if cached:
        return json.loads(cached)

    rows = await db.fetch(queries.TOP_DESENVOLVEDORES)
    payload = [dict(row) for row in rows]
    _cache.setex(cache_key, CACHE_TTL_SECONDS, json.dumps(payload))
    return payload


@router.get("/tendencias")
@limiter.limit("100/minute")
async def get_tendencias(request: Request, dias: int = 30, db: Database = Depends(get_db)):
    cache_key = f"analiticos:tendencias:{dias}"
    cached = _cache.get(cache_key)
    if cached:
        return json.loads(cached)

    query = """
        SELECT j.id, j.titulo, COUNT(s.id) as acessos
        FROM sessoes_jogo s
        JOIN jogos j ON s.jogo_id = j.id
        WHERE s.fim > NOW() - INTERVAL '1 day' * $1
        GROUP BY j.id, j.titulo
        ORDER BY acessos DESC
        LIMIT 10
    """
    rows = await db.fetch(query, dias)
    payload = [dict(row) for row in rows]
    _cache.setex(cache_key, CACHE_TTL_SECONDS, json.dumps(payload))
    return payload
