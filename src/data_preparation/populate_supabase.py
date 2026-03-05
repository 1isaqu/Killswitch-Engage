# populate_supabase.py
import asyncio
import asyncpg
import pandas as pd
import numpy as np
import json
import uuid
import random
import re
from datetime import datetime, timedelta
from tqdm import tqdm
import os
from dotenv import load_dotenv
import logging

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Namespace para UUIDs determinísticos
NAMESPACE_BASE = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')

def generate_uuid(name):
    return uuid.uuid5(NAMESPACE_BASE, str(name))

async def populate():
    # Parâmetros de conexão lidos do .env
    host = os.getenv("DB_HOST")
    port = int(os.getenv("DB_PORT", 6543))
    user = os.getenv("DB_USER")
    database = os.getenv("DB_NAME")
    password = os.getenv("DB_PASSWORD")

    if not all([host, user, password]):
        logger.error("Credenciais do banco não encontradas no arquivo .env")
        return

    logger.info(f"Conectando ao Supabase em {host}:{port} (Modo Transacional)...")
    try:
        conn = await asyncpg.connect(
            user=user,
            password=password,
            database=database,
            host=host,
            port=port,
            ssl="require",
            statement_cache_size=0
        )
    except Exception as e:
        logger.error(f"Falha na conexão: {e}")
        return
    
    try:
        # 1. Carregar e Sanitizar dados
        logger.info("Carregando e sanitizando datasets...")
        df_raw = pd.read_csv('data/processed/imputed_dataset_final.csv')
        df = df_raw.copy()
        
        # CORREÇÃO DE DESLOCAMENTO E EXTRAÇÃO DE APPID
        logger.info("Recuperando AppIDs via Regex e corrigindo alinhamento de colunas...")
        # 1. Extrair ID real da URL da imagem ou screenshots
        df['AppID_real'] = df['Header image'].str.extract(r'apps/(\d+)/')[0].apply(pd.to_numeric, errors='coerce')
        # Se falhar no Header image, tentar no Screenshots (col 37)
        mask_nan = df['AppID_real'].isna()
        if mask_nan.any():
            df.loc[mask_nan, 'AppID_real'] = df.loc[mask_nan, 'Screenshots'].str.extract(r'apps/(\d+)/')[0].apply(pd.to_numeric, errors='coerce')
        
        # 2. Corrigir Name (está na coluna AppID) e Release date (está na coluna Name)
        df['Real_Name'] = df['AppID'].fillna('Unknown Game').astype(str)
        df['Real_Release_Date'] = df['Name']
        
        # 3. Aplicar os novos campos
        df['AppID'] = df['AppID_real']
        df['Name'] = df['Real_Name'].str[:200]
        df['Release date'] = df['Real_Release_Date']
        
        # 4. Tratar Desenvolvedores (Normalização para evitar duplicações como Ubisoft vs ubisoft)
        df['Developers'] = df['Developers'].fillna(df['Publishers']).fillna('Unknown Developer').astype(str).str.strip().str.title().str[:200]
        
        # 5. Sanidade numérica
        num_cols = ['AppID', 'Achievements', 'Price', 'Positive', 'Negative', 'Average playtime forever', 'Required age']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Filtro de registros válidos (Agora o AppID deve ser > 0 após a extração)
        df = df[df['AppID'] > 0].copy()
        df['AppID_int'] = df['AppID'].astype(np.int64)
        
        logger.info(f"Dataset pronto: {len(df)} jogos válidos identificados.")
        
        # 2. Inserir Categorias
        logger.info("Processando Categorias...")
        all_categories = set()
        for cat_col in ['Categories', 'Genres', 'Tags']:
            if cat_col in df.columns:
                cats = df[cat_col].fillna('').str.split(',')
                for item in cats:
                    if isinstance(item, list):
                        all_categories.update([c.strip().capitalize()[:200] for c in item if c.strip()])
        
        category_data = [(cat,) for cat in all_categories]
        await conn.executemany("INSERT INTO categorias (nome) VALUES ($1) ON CONFLICT (nome) DO NOTHING", category_data)
        
        cat_rows = await conn.fetch("SELECT id, nome FROM categorias")
        cat_map = {row['nome']: row['id'] for row in cat_rows}

        # 3. Inserir Desenvolvedores
        logger.info("Processando Desenvolvedores...")
        unique_devs = df['Developers'].unique()
        dev_data = [(generate_uuid(dev), dev, 'Unknown') for dev in unique_devs]
        await conn.executemany("INSERT INTO desenvolvedores (id, nome, pais) VALUES ($1, $2, $3) ON CONFLICT (nome) DO NOTHING", dev_data)
        
        dev_rows = await conn.fetch("SELECT id, nome FROM desenvolvedores")
        dev_map = {row['nome']: row['id'] for row in dev_rows}

        # 4. Inserir Jogos
        logger.info("Processando Jogos...")
        df['Rating_5'] = (df['Positive'] / (df['Positive'] + df['Negative'] + 1)) * 5
        
        jogo_data = []
        game_uuid_map = {}
        for _, row in df.iterrows():
            # Usar AppID real como seed para o UUID para garantir unicidade oficial
            g_uuid = generate_uuid(str(int(row['AppID_int'])))
            game_uuid_map[row['AppID_int']] = g_uuid
            jogo_data.append((
                g_uuid,
                row['Name'],
                float(row['Price']),
                float(row['Rating_5']),
                int(row['Positive'] + row['Negative']),
                int(row['Required age']),
                dev_map.get(row['Developers'])
            ))
            
        # Inserção em lotes de 5000 para estabilidade
        for i in range(0, len(jogo_data), 5000):
            batch = jogo_data[i:i+5000]
            await conn.executemany("""
                INSERT INTO jogos (id, titulo, preco_base, avaliacao_media, total_avaliacoes, idade_requerida, desenvolvedor_id) 
                VALUES ($1, $2, $3, $4, $5, $6, $7) 
                ON CONFLICT (id) DO UPDATE SET 
                    titulo = EXCLUDED.titulo,
                    preco_base = EXCLUDED.preco_base,
                    avaliacao_media = EXCLUDED.avaliacao_media
            """, batch)

        # 5. Relacionamentos
        logger.info("Processando Relacionamentos...")
        jc_data = []
        for _, row in df.iterrows():
            g_uuid = game_uuid_map[row['AppID_int']]
            cats = set()
            for col in ['Categories', 'Genres', 'Tags']:
                if pd.notna(row[col]):
                    cats.update([c.strip().capitalize()[:200] for c in str(row[col]).split(',')])
            for c_name in cats:
                if c_name in cat_map:
                    jc_data.append((g_uuid, cat_map[c_name]))

        for i in range(0, len(jc_data), 5000):
            await conn.executemany("INSERT INTO jogos_categorias (jogo_id, categoria_id) VALUES ($1, $2) ON CONFLICT DO NOTHING", jc_data[i:i+5000])

        # 6. Usuários Sintéticos
        logger.info("Garantindo Usuários Sintéticos...")
        # Apenas 5k para evitar estourar o storage do free tier inutilmente, focado em massa de teste
        user_data = []
        for i in range(5000):
            name = f"User_{i+1000}"
            user_data.append((
                name, f"{name.lower()}@gameverse.io",
                random.choice(['BR', 'US', 'DE', 'JP']), random.randint(18, 60), 'premium'
            ))
        await conn.executemany("INSERT INTO usuarios (nome, email, pais, total_horas, tipo_assinatura) VALUES ($1, $2, $3, $4, $5) ON CONFLICT (email) DO NOTHING", user_data)
        
        user_rows = await conn.fetch("SELECT id FROM usuarios LIMIT 5000")
        user_ids = [r['id'] for r in user_rows]

        # 7. Biblioteca e Sessões (Amostra para performance)
        logger.info("Gerando atividade sintética (Amostra)...")
        top_games = list(game_uuid_map.values())[:2000] # Limitar a 2k jogos populares para densidade
        biblioteca_data = []
        for u_id in user_ids:
            # Cada user tem 5-15 jogos
            for g_uuid in random.sample(top_games, random.randint(5, 15)):
                biblioteca_data.append((u_id, g_uuid, 'compra', float(random.randint(0, 100))))
        
        for i in range(0, len(biblioteca_data), 5000):
            await conn.executemany("INSERT INTO biblioteca (usuario_id, jogo_id, tipo_aquisicao, preco_pago) VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING", biblioteca_data[i:i+5000])

        logger.info("Ingestão completa e corrigida!")

    except Exception as e:
        logger.error(f"Erro na população: {e}")
        raise e
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(populate())
