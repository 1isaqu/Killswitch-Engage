import asyncio
import sys
import os
from uuid import UUID

# Adicionar o caminho do backend ao sys.path para importar o serviço
sys.path.append(os.path.join(os.getcwd(), "backend"))

from app.services.recomendador import recomendador

async def test_recommender():
    print("Iniciando Verificação do Sistema de Recomendação...")
    
    # 1. Testar Usuário Existente (Warm Start)
    user_warm = UUID("1594a29b-5481-45a6-b66e-ff54b09ec9e5")
    print(f"\nTestando Recomendação para Usuário WARM: {user_warm}")
    recs_warm = await recomendador.get_recomendacoes(user_warm, k=5)
    
    print(f"Total de recomendações geradas: {len(recs_warm)}")
    for i, r in enumerate(recs_warm):
        print(f"{i+1}. Jogo ID: {r['id']} | Score: {r['score']:.4f} | Razão: {r['explicacao']}")

    # 2. Testar Usuário Novo (Cold Start)
    user_cold = UUID("00000000-0000-0000-0000-000000000000") # UUID inexistente
    print(f"\nTestando Recomendação para Usuário COLD: {user_cold}")
    recs_cold = await recomendador.get_recomendacoes(user_cold, k=5)
    
    print(f"Total de recomendações geradas: {len(recs_cold)}")
    for i, r in enumerate(recs_cold):
        print(f"{i+1}. Jogo ID: {r['id']} | Score: {r['score']:.4f} | Razão: {r['explicacao']}")

if __name__ == "__main__":
    asyncio.run(test_recommender())
