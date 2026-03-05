import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import joblib
import os

# Configurações
SESSION_DATA = "data/ml_ready/interacoes_sessoes.csv"
GAME_DATA = "data/ml_ready/jogos_features.csv"
USER_DATA = "data/ml_ready/usuarios_features.csv"

# FIXED: alinhar caminho do modelo de clustering com nomenclatura KMeans, mantendo compatibilidade
CLUSTERING_MODEL = "scripts/modelos/kmeans_clusters.pkl"
LEGACY_CLUSTERING_MODEL = "scripts/modelos/hdbscan_model.pkl"

MODEL_OUTPUT = "scripts/modelos/lightfm_model.pkl"  # Mantendo nome para integração

def train_ranker() -> None:
    df_sessions = pd.read_csv(SESSION_DATA)
    df_games = pd.read_csv(GAME_DATA)
    df_users = pd.read_csv(USER_DATA)

    # Carregar dados de clusterização apenas se necessário em futuras versões do ranker
    try:
        cluster_data = joblib.load(CLUSTERING_MODEL)
    except FileNotFoundError:
        # FIXED: compatibilidade com versões antigas que ainda usam o nome legado do arquivo
        cluster_data = joblib.load(LEGACY_CLUSTERING_MODEL)
    
    print("Preparando matriz de interações...")
    
    # 1. Matriz de Interações (Usuário x Jogo) ponderada por horas_jogadas
    # Mapear IDs para índices
    user_map = {id: i for i, id in enumerate(df_users['id'])}
    game_map = {id: i for i, id in enumerate(df_games['id'])}
    
    # Filtrar sessões com IDs válidos
    df_sessions = df_sessions[df_sessions['usuario_id'].isin(user_map) & df_sessions['jogo_id'].isin(game_map)]
    
    row = df_sessions['usuario_id'].map(user_map).values
    col = df_sessions['jogo_id'].map(game_map).values
    # fillna(0) pois horas_jogadas pode ser NaN em sessões sem duração registrada (GENERATED column no Supabase)
    horas = df_sessions['horas_jogadas'].fillna(0).values
    data = np.log1p(horas) + 1  # +1 garante que interações sem horas ainda têm peso
    
    interactions = csr_matrix((data, (row, col)), shape=(len(user_map), len(game_map)))
    
    # 2. Treinar SVD (Filtragem Colaborativa)
    print(f"Rodando SVD na matriz {interactions.shape}...")
    n_components = 50
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_embeddings = svd.fit_transform(interactions)
    item_embeddings = svd.components_.T
    
    # 3. Integração de Features de Conteúdo (Simulada para Hibridização)
    # Vamos salvar os mapeamentos e os embeddings para reconstrução no serviço
    
    print(f"Variância explicada total: {svd.explained_variance_ratio_.sum():.4f}")
    
    # 4. Salvar tudo
    os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
    model_bundle = {
        'user_embeddings': user_embeddings,
        'item_embeddings': item_embeddings,
        'user_map': user_map,
        'game_map': game_map,
        'reverse_game_map': {v: k for k, v in game_map.items()},
        'n_components': n_components,
        'svd': svd
    }
    joblib.dump(model_bundle, MODEL_OUTPUT)
    print(f"Modelo salvo em {MODEL_OUTPUT}")

if __name__ == "__main__":
    train_ranker()
