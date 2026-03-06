from fastapi import APIRouter, HTTPException, Request, Query
from typing import Optional
from uuid import UUID
from app.services.recomendador import recomendador
from app.utils.security import limiter

router = APIRouter()

MODOS_VALIDOS = {"conservador", "equilibrado", "aventureiro"}

@router.get("/{usuario_id}")
@limiter.limit("100/minute")
async def get_recomendacoes(
    request: Request,
    usuario_id: UUID,
    k: int = Query(default=10, ge=1, le=50, description="Quantidade de jogos a recomendar"),
    modo: Optional[str] = Query(
        default=None,
        description="Arquétipo de recomendação: 'conservador' (precisão), 'equilibrado' (padrão) ou 'aventureiro' (diversidade)"
    ),
):
    """
    Retorna recomendações de jogos personalizadas usando o sistema multi-camada
    (Random Forest + KMeans + SVD).

    **Modos disponíveis:**
    - `conservador` — threshold 0.7, máxima precisão
    - `equilibrado` — threshold 0.5, balanço entre precisão e cobertura (padrão)
    - `aventureiro` — threshold 0.3, maior diversidade e descoberta
    """
    if modo is not None and modo not in MODOS_VALIDOS:
        raise HTTPException(
            status_code=400,
            detail=f"Modo '{modo}' inválido. Use: {sorted(MODOS_VALIDOS)}"
        )

    if not recomendador.is_loaded:
        raise HTTPException(status_code=503, detail="Sistema de recomendação em manutenção ou modelos não carregados.")

    recs = await recomendador.get_recomendacoes(usuario_id, k=k, modo=modo)

    if not recs:
        return {"usuario_id": usuario_id, "modo": modo or "equilibrado", "recomendacoes": [], "status": "no_recs_found"}

    return {
        "usuario_id": usuario_id,
        "modo": modo or "equilibrado",
        "total": len(recs),
        "recomendacoes": recs,
    }
