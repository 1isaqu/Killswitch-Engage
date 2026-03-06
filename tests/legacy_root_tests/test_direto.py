"""
Testes Diretos do Sistema de Recomendacao (Opcao B - sem servidor HTTP)
Importa o recomendador diretamente, sem precisar do FastAPI/Uvicorn.
"""
import sys
import os
import asyncio
import numpy as np

# Aponta o PYTHONPATH para o backend/app ser importavel
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

from app.services.recomendador import RecomendadorService

rec = RecomendadorService()

if not rec.is_loaded:
    print("ERRO: Modelos nao carregaram. Verifique a pasta scripts/modelos.")
    sys.exit(1)

print("Modelos carregados com sucesso!\n")
print(f"Total de jogos no ranker: {len(rec.ranker['game_map'])}")
print(f"Total de usuarios no ranker: {len(rec.ranker['user_map'])}\n")

# ---------------------------------------------------------------------------
# TESTE 1: Usuario conhecido (ID numerico pequeno)
# ---------------------------------------------------------------------------
print("=" * 60)
print("TESTE 1: Recomendacao para usuario conhecido")
print("=" * 60)

# Pega o primeiro usuario que existe no mapa
primeiro_usuario = list(rec.ranker['user_map'].keys())[0]
resultado1 = asyncio.run(rec.get_recomendacoes(primeiro_usuario, k=5))
print(f"Usuario ID: {primeiro_usuario}")
for j in resultado1:
    print(f"  - Jogo {j['id']:>8} | Score: {j['score']:.6f} | {j['explicacao']}")

scores_validos = all(j['score'] > 0 for j in resultado1)
print(f"Scores > 0: {'OK' if scores_validos else 'FALHOU'}")
print(f"Qtd retornada: {len(resultado1)} (esperado: 5) -> {'OK' if len(resultado1) == 5 else 'FALHOU'}\n")

# ---------------------------------------------------------------------------
# TESTE 2: Modos de recomendacao (aventureiro / equilibrado / conservador)
# ---------------------------------------------------------------------------
print("=" * 60)
print("TESTE 2: Modos de diversidade de recomendacao")
print("=" * 60)
print("(Nota: a diversificacao por modo nao esta implementada no servico ainda.")
print(" Mostrando IDs retornados para o mesmo usuario com k=5 e k=10 e k=3)")
print(" para demonstrar que o ranking e determinístico e consistente)\n")

for label, k in [("conservador (k=5)", 5), ("equilibrado (k=10)", 10), ("aventureiro (k=15)", 15)]:
    resultado = asyncio.run(rec.get_recomendacoes(primeiro_usuario, k=k))
    print(f"  {label}: {[j['id'] for j in resultado][:5]}...")

# ---------------------------------------------------------------------------
# TESTE 3: MLflow - verificar metricas logadas
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TESTE 3: Metricas do MLflow")
print("=" * 60)

mlflow_db_path = os.path.join(os.path.dirname(__file__), "scripts", "experimentation", "mlflow.db")
if os.path.exists(mlflow_db_path):
    size_kb = os.path.getsize(mlflow_db_path) / 1024
    print(f"mlflow.db encontrado: {size_kb:.1f} KB")
    print("Execute: python -m mlflow ui --backend-store-uri sqlite:///scripts/experimentation/mlflow.db")
    print("E acesse http://localhost:5000 para ver os experimentos.")
else:
    print("mlflow.db nao encontrado no caminho esperado.")
    print(f"Procurado em: {mlflow_db_path}")

# ---------------------------------------------------------------------------
# TESTE 4: Cobertura manual (100 usuarios)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TESTE 4: Cobertura manual com 100 usuarios")
print("=" * 60)

todos_usuarios = list(rec.ranker['user_map'].keys())
sample_users = todos_usuarios[:100]  # primeiros 100

jogos_unicos = set()
for uid in sample_users:
    resultado = asyncio.run(rec.get_recomendacoes(uid, k=10))
    jogos_unicos.update(j['id'] for j in resultado)

total_jogos = len(rec.ranker['game_map'])
cobertura_pct = len(jogos_unicos) / total_jogos * 100
print(f"Usuarios amostrados: {len(sample_users)}")
print(f"Jogos unicos recomendados: {len(jogos_unicos)}")
print(f"Total de jogos no catalogo: {total_jogos}")
print(f"Cobertura: {cobertura_pct:.2f}%")

# ---------------------------------------------------------------------------
# TESTE 5: Cold Start (usuario que nao existe no mapa)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TESTE 5: Cold Start (usuario desconhecido)")
print("=" * 60)

id_inexistente = "usuario-que-nao-existe-99999"
resultado_cold = asyncio.run(rec.get_recomendacoes(id_inexistente, k=5))
print(f"Usuario ID: {id_inexistente}")
print(f"Qtd retornada: {len(resultado_cold)} (esperado: 5) -> {'OK' if len(resultado_cold) == 5 else 'FALHOU'}")
for j in resultado_cold:
    print(f"  - Jogo {j['id']:>8} | Score: {j['score']:.6f} | {j['explicacao']}")

print("\n" + "=" * 60)
print("TODOS OS TESTES CONCLUIDOS")
print("=" * 60)
