"""
Testes Comparativos dos 3 Modos de Recomendacao
Roda os 5 cenarios para cada modo e gera tabela comparativa.
"""
import sys
import os
import asyncio
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

from app.services.recomendador import RecomendadorService

rec = RecomendadorService()

if not rec.is_loaded:
    print("ERRO: Modelos nao carregaram.")
    sys.exit(1)

print(f"Modelos carregados. Jogos no catalogo: {len(rec.ranker['game_map'])}")
print(f"Usuarios no ranker: {len(rec.ranker['user_map'])}\n")

MODOS = ['conservador', 'equilibrado', 'aventureiro']
todos_usuarios = list(rec.ranker['user_map'].keys())
primeiro_usuario = todos_usuarios[0]
K = 10

# ============================================================
# TESTE 1 e 2: Usuario conhecido + verificacao de diversidade
# ============================================================
print("=" * 70)
print("TESTE 1 + 2: Recomendacao para usuario conhecido (variacoes por modo)")
print("=" * 70)

resultados_modo = {}
for modo in MODOS:
    resultado = asyncio.run(rec.get_recomendacoes(primeiro_usuario, k=K, modo=modo))
    ids = [j['id'] for j in resultado]
    scores = [j['score'] for j in resultado]
    resultados_modo[modo] = {'ids': ids, 'scores': scores, 'resultado': resultado}
    print(f"\n  [{modo.upper()}] threshold={rec.MODOS[modo]['threshold']}")
    print(f"    IDs:   {ids[:3]}...{ids[-1]}")
    print(f"    Score min/max: {min(scores):.4f} / {max(scores):.4f}")
    print(f"    Score medio:   {np.mean(scores):.4f}")

# Verifica diversidade entre modos
ids_conservador = set(resultados_modo['conservador']['ids'])
ids_equilibrado = set(resultados_modo['equilibrado']['ids'])
ids_aventureiro = set(resultados_modo['aventureiro']['ids'])
print(f"\n  Sobreposicao conservador vs aventureiro: {len(ids_conservador & ids_aventureiro)}/{K} jogos em comum")

# ============================================================
# TESTE 3: MLflow
# ============================================================
print("\n" + "=" * 70)
print("TESTE 3: MLflow DB")
print("=" * 70)
mlflow_db = os.path.join(os.path.dirname(__file__), "scripts", "experimentation", "mlflow.db")
if os.path.exists(mlflow_db):
    print(f"  mlflow.db: {os.path.getsize(mlflow_db)/1024:.1f} KB -- OK")
    print("  Comando: python -m mlflow ui --backend-store-uri sqlite:///scripts/experimentation/mlflow.db")
else:
    print("  mlflow.db nao encontrado.")

# ============================================================
# TESTE 4: Cobertura comparativa (100 usuarios, cada modo)
# ============================================================
print("\n" + "=" * 70)
print("TESTE 4: Cobertura com 100 usuarios por modo")
print("=" * 70)

sample_users = todos_usuarios[:100]
total_jogos = len(rec.ranker['game_map'])

cobertura = {}
for modo in MODOS:
    jogos_unicos = set()
    scores_totais = []
    for uid in sample_users:
        resultado = asyncio.run(rec.get_recomendacoes(uid, k=K, modo=modo))
        jogos_unicos.update(j['id'] for j in resultado)
        scores_totais.extend(j['score'] for j in resultado)
    pct = len(jogos_unicos) / total_jogos * 100
    cobertura[modo] = {'unicos': len(jogos_unicos), 'pct': pct, 'score_medio': np.mean(scores_totais)}
    print(f"  [{modo.upper()}]: {len(jogos_unicos)} jogos unicos | {pct:.2f}% de cobertura | score medio: {np.mean(scores_totais):.4f}")

# ============================================================
# TESTE 5: Cold Start por modo
# ============================================================
print("\n" + "=" * 70)
print("TESTE 5: Cold Start por modo")
print("=" * 70)

id_inexistente = "usuario-que-nao-existe-99999"
for modo in MODOS:
    resultado = asyncio.run(rec.get_recomendacoes(id_inexistente, k=5, modo=modo))
    scores = [j['score'] for j in resultado]
    print(f"  [{modo.upper()}]: {len(resultado)} jogos | score min/max: {min(scores):.4f}/{max(scores):.4f}")
    print(f"    {[j['id'] for j in resultado]}")

# ============================================================
# TABELA COMPARATIVA FINAL
# ============================================================
print("\n" + "=" * 70)
print("TABELA COMPARATIVA FINAL")
print("=" * 70)
print(f"{'Modo':<14} {'Threshold':<12} {'Exploracao':<12} {'Cobertura':<12} {'Score Medio':<14}")
print("-" * 70)
for modo in MODOS:
    cfg = rec.MODOS[modo]
    cob = cobertura[modo]
    print(f"{modo:<14} {cfg['threshold']:<12.1f} {cfg['exploracao']:<12.0%} {cob['pct']:<12.2f}% {cob['score_medio']:<14.4f}")

print("\nRecomendacao de modo padrao: equilibrado (threshold=0.5)")
print("  -> melhor democracia entre precisao e descoberta para maioria dos usuarios")
