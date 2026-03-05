from fastapi import APIRouter, HTTPException, Request
from uuid import UUID
from app.services.recomendador import recomendador
from app.utils.security import limiter

router = APIRouter()

@router.get("/{usuario_id}")
@limiter.limit("100/minute")
async def get_recomendacoes(request: Request, usuario_id: UUID, k: int = 10):
    """
    Retorna recomendações de jogos personalizadas usando o sistema multi-camada
    (Random Forest + KMeans + SVD).
    """
    recs = await recomendador.get_recomendacoes(usuario_id, k)
    
    if not recs and recomendador.is_loaded:
        # Se não houver recs mas o modelo está carregado, pode ser um erro inesperado
        return {"usuario_id": usuario_id, "recomendacoes": [], "status": "no_recs_found"}
    elif not recomendador.is_loaded:
        raise HTTPException(status_code=503, detail="Sistema de recomendação em manutenção ou modelos não carregados.")
        
    return {
        "usuario_id": usuario_id,
        "total": len(recs),
        "recomendacoes": recs
    }
