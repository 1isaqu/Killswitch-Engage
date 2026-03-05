import asyncio
import os
import pandas as pd
import asyncpg
from dotenv import load_dotenv
from uuid import UUID
import json

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

async def extract_data():
    conn = await asyncpg.connect(DATABASE_URL, statement_cache_size=0, command_timeout=300)
    await conn.execute("SET statement_timeout = 300000") # 5 min em ms
    print("Conectado ao Supabase para extração...")

    # 1. Extrair Jogos e Features Básicas
    print("Extraindo Jogos...")
    jogos_query = """
        SELECT 
            j.id, j.titulo, j.preco_base, j.avaliacao_media, j.total_avaliacoes, 
            j.metacritic_score, j.idade_requerida, j.data_lancamento,
            d.nome as dev_nome, d.id as dev_id,
            (SELECT array_agg(c.nome) FROM jogos_categorias jc JOIN categorias c ON jc.categoria_id = c.id WHERE jc.jogo_id = j.id) as categorias
        FROM jogos j
        LEFT JOIN desenvolvedores d ON j.desenvolvedor_id = d.id
    """
    jogos_rows = await conn.fetch(jogos_query)
    df_jogos = pd.DataFrame([dict(r) for r in jogos_rows])

    # 2. Extrair Usuários (removido 'idade' que não existe no schema real)
    print("Extraindo Usuários...")
    usuarios_query = "SELECT id, pais, tipo_assinatura FROM usuarios"
    usuarios_rows = await conn.fetch(usuarios_query)
    df_usuarios = pd.DataFrame([dict(r) for r in usuarios_rows])

    # 3. Extrair Interações (Sessões)
    print("Extraindo Interações...")
    sessoes_query = "SELECT usuario_id, jogo_id, horas_jogadas, inicio FROM sessoes_jogo"
    sessoes_rows = await conn.fetch(sessoes_query)
    df_sessoes = pd.DataFrame([dict(r) for r in sessoes_rows])

    # 4. Salvar para CSVs temporários de treino
    os.makedirs("data/ml_ready", exist_ok=True)
    df_jogos.to_csv("data/ml_ready/jogos_features.csv", index=False)
    df_usuarios.to_csv("data/ml_ready/usuarios_features.csv", index=False)
    df_sessoes.to_csv("data/ml_ready/interacoes_sessoes.csv", index=False)

    print(f"Extração concluída: {len(df_jogos)} jogos, {len(df_usuarios)} usuários, {len(df_sessoes)} sessões.")
    await conn.close()

if __name__ == "__main__":
    asyncio.run(extract_data())
