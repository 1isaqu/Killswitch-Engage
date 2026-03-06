"""Recommendations router — Layer-4 serving endpoint.

Security notes (SECURITY_CHECKLIST § 3):
    - ``usuario_id`` is validated as a ``UUID`` by Pydantic at the route level.
    - ``modo`` is validated against an explicit allowlist before reaching the service.
    - Rate-limited to 100 requests / minute per IP.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request

from src.backend.middleware.security import limiter
from src.backend.services.recomendador import recomendador
from src.utils.exceptions import InvalidModeError, ModelNotLoadedError
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

MODOS_VALIDOS = frozenset({"conservador", "equilibrado", "aventureiro"})


@router.get("/{usuario_id}", summary="Obter recomendações personalizadas")
@limiter.limit("100/minute")
async def get_recomendacoes(
    request: Request,
    usuario_id: UUID,
    k: int = Query(default=10, ge=1, le=50, description="Quantidade de jogos a recomendar"),
    modo: Optional[str] = Query(
        default=None,
        description=(
            "Modo: 'conservador' (precisão) "
            "| 'equilibrado' (padrão) "
            "| 'aventureiro' (diversidade)"
        ),
    ),
) -> dict:
    """Return personalised game recommendations for a user.

    Args:
        request: FastAPI request (required by SlowAPI).
        usuario_id: Target user UUID — validated by Pydantic.
        k: Number of items to return (1–50).
        modo: Recommendation mode. Defaults to 'equilibrado'.

    Returns:
        dict: ``{usuario_id, modo, total, recomendacoes}`` on success,
        or ``{usuario_id, modo, recomendacoes: [], status}`` when empty.

    Raises:
        HTTPException 400: If ``modo`` is not one of the valid modes.
        HTTPException 503: If the recommender models are not loaded.
    """
    if modo is not None and modo not in MODOS_VALIDOS:
        raise HTTPException(
            status_code=400,
            detail=f"Modo '{modo}' inválido. Use: {sorted(MODOS_VALIDOS)}",
        )

    try:
        recs = await recomendador.get_recomendacoes(usuario_id, k=k, modo=modo)
    except ModelNotLoadedError:
        logger.error("Recommender not loaded — returning 503.")
        raise HTTPException(
            status_code=503,
            detail="Sistema de recomendação em manutenção ou modelos não carregados.",
        )
    except InvalidModeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    resolved_modo = modo or "equilibrado"

    if not recs:
        return {
            "usuario_id": str(usuario_id),
            "modo": resolved_modo,
            "recomendacoes": [],
            "status": "no_recs_found",
        }

    return {
        "usuario_id": str(usuario_id),
        "modo": resolved_modo,
        "total": len(recs),
        "recomendacoes": recs,
    }
