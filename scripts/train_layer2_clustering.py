import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib

# Configurações
USER_DATA = "data/ml_ready/usuarios_features.csv"
SESSION_DATA = "data/ml_ready/interacoes_sessoes.csv"
GAME_DATA = "data/ml_ready/jogos_features.csv"

# FIXED: nome de arquivo mais fiel ao algoritmo usado (KMeans), mantendo saída legada para compatibilidade
MODEL_OUTPUT = "scripts/modelos/kmeans_clusters.pkl"
LEGACY_MODEL_OUTPUT = "scripts/modelos/hdbscan_model.pkl"

# FIXED: extrair magic numbers para constantes configuráveis
MIN_K = 3
MAX_K = 10
SILHOUETTE_SAMPLE_FRAC = 0.2  # fração máxima de pontos usados no cálculo
SILHOUETTE_MAX_SAMPLES = 10000  # limite absoluto de amostras para o silhouette


def _safe_eval_categories(value: str) -> list[str]:
    """Converte representação de lista em Python para lista real de forma segura."""
    # FIXED: substituir eval por parsing mais seguro com fallback simples
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value:
        return []
    try:
        import ast

        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def train_clustering() -> None:
    df_users = pd.read_csv(USER_DATA)
    df_sessions = pd.read_csv(SESSION_DATA)
    df_games = pd.read_csv(GAME_DATA)
    
    print("Agregando features por usuário...")
    
    # 1. Feature Engineering
    # FIXED: parse seguro da lista de categorias e gênero primário por jogo
    df_games["categorias"] = df_games["categorias"].apply(_safe_eval_categories)
    df_games["primary_genre"] = df_games["categorias"].apply(
        lambda cats: cats[0] if cats else "Unknown"
    )
    
    # Join sessões com jogos para pegar gêneros
    df_merged = df_sessions.merge(df_games[['id', 'primary_genre']], left_on='jogo_id', right_on='id')
    
    # Horas por gênero
    user_genre = df_merged.groupby(['usuario_id', 'primary_genre'])['horas_jogadas'].sum().unstack(fill_value=0)
    user_genre_pct = user_genre.div(user_genre.sum(axis=1), axis=0)
    
    # Estatísticas de engajamento
    user_stats = df_sessions.groupby('usuario_id').agg({
        'jogo_id': 'nunique',
        'horas_jogadas': ['sum', 'mean', 'count']
    })
    user_stats.columns = ['total_jogos_unicos', 'total_horas', 'media_horas_sessao', 'total_sessoes']
    
    # Merge
    X_raw = user_genre_pct.merge(user_stats, left_index=True, right_index=True)
    
    # 2. Pré-processamento
    print(f"Dataset: {X_raw.shape}")
    # Preencher NaN que podem surgir para usuários sem sessões em certos gêneros
    X_raw = X_raw.fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # PCA para Redução
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA: {X_pca.shape[1]} componentes.")
    
    # 3. KMeans (Busca por K ideal entre MIN_K e MAX_K)
    print("Buscando K ideal via Silhouette Score...")
    best_k = MIN_K
    best_score = -1
    best_kmeans = None
    n_samples = X_pca.shape[0]

    for k in range(MIN_K, MAX_K + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_pca)

        # FIXED: cálculo de silhouette_score em amostra (O(n) em vez de O(n^2) no full dataset)
        if n_samples <= SILHOUETTE_MAX_SAMPLES:
            sample_idx = np.arange(n_samples)
        else:
            sample_size = int(min(SILHOUETTE_MAX_SAMPLES, n_samples * SILHOUETTE_SAMPLE_FRAC))
            rng = np.random.default_rng(seed=42)
            sample_idx = rng.choice(n_samples, size=sample_size, replace=False)

        score = silhouette_score(X_pca[sample_idx], labels[sample_idx])
        print(f"K={k} | Silhouette (amostra {len(sample_idx)}): {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
            best_kmeans = km
            
    print(f"K escolhido: {best_k} (Score: {best_score:.4f})")
    
    # 4. Salvar Modelo
    os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
    model_data = {
        'model': best_kmeans,
        'scaler': scaler,
        'pca': pca,
        'feature_cols': X_raw.columns.tolist(),
        'n_clusters': best_k,
        'best_score': best_score
    }
    joblib.dump(model_data, MODEL_OUTPUT)
    # FIXED: manter escrita em caminho legado para compatibilidade com versões anteriores
    joblib.dump(model_data, LEGACY_MODEL_OUTPUT)
    print(f"Modelo salvo em {MODEL_OUTPUT} (e cópia legada em {LEGACY_MODEL_OUTPUT})")

if __name__ == "__main__":
    train_clustering()
