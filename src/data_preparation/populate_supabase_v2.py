# populate_supabase_v2.py
# Revisão de Sanidade v2 — Distribuição realista de sessões por perfil comportamental
import asyncio
import asyncpg
import pandas as pd
import numpy as np
import uuid
import random
import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

NAMESPACE_BASE = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')

def generate_uuid(name):
    return uuid.uuid5(NAMESPACE_BASE, str(name))

def random_past_datetime(days_back=730):
    """Gera um timestamp aleatório nos últimos 2 anos com distribuição realista."""
    # Distribuição com mais atividade nos últimos 6 meses
    weights = np.exp(np.linspace(-2, 0, 720))  # Mais recente = mais peso
    weights = weights / weights.sum()
    chosen_day = int(np.random.choice(range(720), p=weights))
    dt = datetime.now(tz=timezone.utc) - timedelta(days=chosen_day)
    # Hora com pesos: picos às 19-23h e fins de semana
    hour = random.choices(range(24), weights=[1,1,1,1,1,1,5,5,5,5,5,5,5,5,5,8,8,10,12,14,15,15,14,10])[0]
    return dt.replace(hour=hour, minute=random.randint(0, 59))

async def populate_sessions_v2():
    host = os.getenv("DB_HOST")
    port = int(os.getenv("DB_PORT", 6543))
    user = os.getenv("DB_USER")
    database = os.getenv("DB_NAME")
    password = os.getenv("DB_PASSWORD")

    conn = await asyncpg.connect(
        user=user, password=password, database=database,
        host=host, port=port, ssl="require", statement_cache_size=0
    )
    logger.info("Conectado ao Supabase.")

    try:
        # 1. Buscar todos os usuários e jogos existentes
        user_rows = await conn.fetch("SELECT id FROM usuarios")
        game_rows = await conn.fetch("SELECT id FROM jogos")

        user_ids = [r['id'] for r in user_rows]
        all_game_ids = [r['id'] for r in game_rows]
        
        n_users = len(user_ids)
        n_games = len(all_game_ids)
        logger.info(f"Usuários: {n_users} | Jogos: {n_games}")

        # 2. Limpar sessões antigas (para repovoar do zero)
        deleted = await conn.execute("DELETE FROM sessoes_jogo")
        logger.info(f"Sessões antigas removidas.")

        # 3. Definir perfis comportamentais
        # 70% casual (5-10 sessões), 25% médio (20-50), 5% hardcore (50-200)
        random.shuffle(user_ids)
        
        n_casual   = int(n_users * 0.70)
        n_medio    = int(n_users * 0.25)
        n_hardcore = n_users - n_casual - n_medio
        
        # 500 usuários extremos de borda (entre os casual/hardcore redirecionados)
        edge_users  = set(random.sample(user_ids[:int(n_users*0.05)], min(500, len(user_ids))))

        perfil_map = {}
        for uid in user_ids[:n_casual]:
            perfil_map[uid] = 'casual'
        for uid in user_ids[n_casual:n_casual+n_medio]:
            perfil_map[uid] = 'medio'
        for uid in user_ids[n_casual+n_medio:]:
            perfil_map[uid] = 'hardcore'

        sessoes = []
        total_sessions = 0

        logger.info("Gerando sessões realistas por perfil...")
        
        for uid in user_ids:
            perfil = perfil_map.get(uid, 'casual')
            is_edge = uid in edge_users
            
            if is_edge:
                n_sessions = random.randint(150, 300)
                horas_range = (0.1, 24.0)
            elif perfil == 'casual':
                n_sessions = random.randint(5, 10)
                horas_range = (0.1, 3.0)
            elif perfil == 'medio':
                n_sessions = random.randint(20, 50)
                horas_range = (0.5, 8.0)
            else:  # hardcore
                n_sessions = random.randint(50, 200)
                horas_range = (1.0, 20.0)
            
            # Jogos favoritos do usuário (80% sessões em jogos de preferência)
            n_favorites = max(3, n_sessions // 8)
            favorite_games = random.sample(all_game_ids, min(n_favorites, len(all_game_ids)))
            
            for _ in range(n_sessions):
                # 80% das sessões em jogos favoritos, 20% em qualquer jogo
                if random.random() < 0.80:
                    game_id = random.choice(favorite_games)
                else:
                    game_id = random.choice(all_game_ids)
                
                horas = round(random.uniform(*horas_range), 2)
                inicio = random_past_datetime()
                fim = inicio + timedelta(hours=horas)

                sessoes.append((uid, game_id, inicio, fim))
                total_sessions += 1

        # 4. Inserir sessões em batches
        logger.info(f"Inserindo {total_sessions} sessões ({total_sessions/n_users:.1f}/usuário médio)...")
        
        for i in range(0, len(sessoes), 5000):
            batch = sessoes[i:i+5000]
            await conn.executemany(
                "INSERT INTO sessoes_jogo (usuario_id, jogo_id, inicio, fim) VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING",
                batch
            )
            if i % 25000 == 0:
                logger.info(f"  {i:,}/{len(sessoes):,} sessões inseridas...")

        # Resultado final
        actual_count = await conn.fetchval("SELECT COUNT(*) FROM sessoes_jogo")
        actual_users = await conn.fetchval("SELECT COUNT(DISTINCT usuario_id) FROM sessoes_jogo")
        logger.info(f"✅ Ingestão v2 completa: {actual_count} sessões | {actual_users} usuários | média: {actual_count/actual_users:.1f}/usuário")

    except Exception as e:
        logger.error(f"Erro: {e}")
        raise
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(populate_sessions_v2())
