import os
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "ml_ready"
MODEL_DIR = ROOT_DIR / "scripts" / "modelos"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"


def load_ranker_bundle():
    """
    Carrega o bundle do ranker SVD (lightfm_model.pkl ou híbrido equivalente).

    Este bundle expõe embeddings de usuário e jogo, além de mapas user_id → índice.
    """
    ranker_path = MODEL_DIR / "lightfm_model.pkl"
    if not ranker_path.exists():
        # fallback para um possível modelo híbrido com a mesma estrutura
        ranker_path = MODEL_DIR / "hybrid_ranker.pkl"
    if not ranker_path.exists():
        raise FileNotFoundError("Nenhum modelo de ranker encontrado em scripts/modelos.")
    return joblib.load(ranker_path)


def load_cluster_bundle():
    """
    Carrega o bundle de clustering de usuários.

    Prioriza kmeans_clusters.pkl, mantendo compatibilidade com hdbscan_model.pkl.
    """
    main = MODEL_DIR / "kmeans_clusters.pkl"
    legacy = MODEL_DIR / "hdbscan_model.pkl"
    if main.exists():
        return joblib.load(main)
    if legacy.exists():
        return joblib.load(legacy)
    raise FileNotFoundError("Nenhum modelo de clustering encontrado (kmeans_clusters.pkl ou hdbscan_model.pkl).")


def build_user_features() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Constrói um DataFrame de features por usuário a partir dos CSVs ml_ready.

    Reaproveita a lógica de train_layer2_clustering (user_genre_pct + user_stats),
    adicionando frequencia relativa, codificação de horário médio e flag de outlier.
    """
    df_users = pd.read_csv(DATA_DIR / "usuarios_features.csv")
    df_sessions = pd.read_csv(DATA_DIR / "interacoes_sessoes.csv")
    df_games = pd.read_csv(DATA_DIR / "jogos_features.csv")

    # ===== Features por gênero (horas_jogadas normalizadas) =====
    df_games["primary_genre"] = df_games["categorias"].apply(
        lambda x: (eval(x)[0] if isinstance(x, str) and len(eval(x)) > 0 else "Unknown")
    )
    df_merged = df_sessions.merge(
        df_games[["id", "primary_genre"]], left_on="jogo_id", right_on="id"
    )
    user_genre = (
        df_merged.groupby(["usuario_id", "primary_genre"])["horas_jogadas"]
        .sum()
        .unstack(fill_value=0)
    )
    user_genre_pct = user_genre.div(user_genre.sum(axis=1), axis=0).fillna(0.0)

    # ===== Estatísticas de engajamento =====
    user_stats = df_sessions.groupby("usuario_id").agg(
        {
            "jogo_id": "nunique",
            "horas_jogadas": ["sum", "mean", "count"],
            "inicio": "min",
        }
    )
    user_stats.columns = [
        "total_jogos_unicos",
        "total_horas",
        "media_horas_sessao",
        "total_sessoes",
        "primeira_sessao",
    ]

    # Frequência aproximada: sessões / (dias desde primeira sessão até agora)
    now = pd.Timestamp.utcnow()
    days_active = (now - pd.to_datetime(user_stats["primeira_sessao"])).dt.days.clip(
        lower=1
    )
    session_freq = (user_stats["total_sessoes"] / days_active).astype(float)

    # Horário médio (circular) baseado em 'inicio'
    df_sessions["hora"] = pd.to_datetime(df_sessions["inicio"], format='mixed', utc=True).dt.hour
    hora_mean = (
        df_sessions.groupby("usuario_id")["hora"]
        .mean()
        .reindex(user_stats.index)
        .fillna(12.0)
    )
    hora_rad = 2 * np.pi * hora_mean / 24.0
    hora_cos = np.cos(hora_rad)
    hora_sin = np.sin(hora_rad)

    # Normalizações básicas
    scaler = MinMaxScaler()
    freq_norm = scaler.fit_transform(session_freq.values.reshape(-1, 1)).flatten()
    jogos_norm = scaler.fit_transform(
        user_stats["total_jogos_unicos"].values.reshape(-1, 1)
    ).flatten()

    # Flag de outlier simples: top 5% de total_horas
    horas = user_stats["total_horas"].fillna(0.0)
    lim_outlier = horas.quantile(0.95)
    is_outlier = (horas >= lim_outlier).astype(int)

    # Junta features base para submeter ao modelo de cluster
    X_raw = user_genre_pct.merge(user_stats, left_index=True, right_index=True).fillna(0.0)
    
    # Integramos as lables de Cluster como condição One-Hot do usuário
    try:
        cluster_bundle = load_cluster_bundle()
        # Garante a ordem exata de features exigida pelo modelo
        X_cluster = X_raw[cluster_bundle['feature_cols']]
        X_scaled = cluster_bundle['scaler'].transform(X_cluster)
        X_pca = cluster_bundle['pca'].transform(X_scaled)
        cluster_labels = cluster_bundle['model'].predict(X_pca)
        
        # One-Hot Encoding dos clusters
        n_clusters = cluster_bundle['n_clusters']
        cluster_ohe = np.eye(n_clusters)[cluster_labels]
        df_clusters = pd.DataFrame(
            cluster_ohe, 
            index=X_raw.index, 
            columns=[f"cluster_{i}" for i in range(n_clusters)]
        )
    except Exception as e:
        print(f"Aviso: Erro ao carregar/aplicar clusters ({e}). Prosseguindo sem clusters.")
        df_clusters = pd.DataFrame(index=X_raw.index)

    # Monta o DataFrame final de condição crua
    user_features = pd.concat([user_genre_pct, df_clusters], axis=1)
    user_features["session_frequency"] = freq_norm
    user_features["mean_play_hour_cos"] = hora_cos.values
    user_features["mean_play_hour_sin"] = hora_sin.values
    user_features["unique_games_norm"] = jogos_norm
    user_features["is_outlier"] = is_outlier.values

    # Alinha com df_users para garantir consistência de ids
    user_features = user_features.reindex(df_users["id"]).fillna(0.0)
    user_features.index.name = "usuario_id"

    return df_users, user_features, user_genre_pct


def compute_best_thresholds(k: int = 10) -> pd.DataFrame:
    """
    Para cada usuário presente no ranker, avalia thresholds candidatos (0.3, 0.5, 0.7)
    e escolhe aquele que maximiza a precision@k, usando jogos já jogados como relevância.
    """
    ranker = load_ranker_bundle()
    df_users = pd.read_csv(DATA_DIR / "usuarios_features.csv")
    df_sessions = pd.read_csv(DATA_DIR / "interacoes_sessoes.csv")

    user_map = ranker["user_map"]
    item_embeddings = ranker["item_embeddings"]
    user_embeddings = ranker["user_embeddings"]
    reverse_game_map = ranker["reverse_game_map"]

    # Constrói ground truth de relevância: avalia se o threshold 
    # otimiza a predição dos últimos 20% jogos da timeline do usuário.
    # Assumimos que qualquer sessão constitui um jogo válido (ground truth).
    df_sessions_sorted = df_sessions.sort_values("inicio")
    
    # Agrupamento otimizado O(N) nativo Python.
    games_by_user = {}
    
    # 1. Agrupar em ordem (pois df já está sorted por início)
    for row in df_sessions_sorted.itertuples(index=False):
        uid = row.usuario_id
        jid = row.jogo_id
        if uid not in games_by_user:
            games_by_user[uid] = []
        games_by_user[uid].append(jid)
        
    played_val = {}
    for uid, games_list in games_by_user.items():
        n_latest = max(1, int(len(games_list) * 0.2))
        played_val[uid] = set(games_list[-n_latest:])


    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # Granularidade maior para a GAN
    rows = []
    
    # Pre-instancia o scaler para não criar loop overhead 
    minmax = MinMaxScaler()

    for uid in df_users["id"]:
        uid_str = str(uid)
        if uid_str not in user_map:
            continue
        if uid not in played_val:
            continue

        user_idx = user_map[uid_str]
        u_vec = user_embeddings[user_idx]
        scores = np.dot(item_embeddings, u_vec)

        # Normalizar scores para [0,1] linearmente usando MinMaxScaler intra-usuário
        # para preservar distâncias geradas pelo SVD sem o squash distorcido do Sigmoid
        scores_norm = minmax.fit_transform(scores.reshape(-1, 1)).flatten()
        
        game_ids = [reverse_game_map[i] for i in range(len(scores))]

        relevant = played_val[uid]
        if not relevant:
            continue
            
        best_t = None
        best_prec = -1.0

        for t in thresholds:
            # Seleciona itens com score >= t e pega top-k
            scores_norm = np.nan_to_num(scores_norm, nan=0.0)
            mask = scores_norm >= t
            if not mask.any():
                continue
            
            candidate_ids = np.array(game_ids)[mask]
            candidate_scores = scores_norm[mask]
            
            if len(candidate_ids) == 0:
                continue
                
            order = np.argsort(-candidate_scores)
            recs = candidate_ids[order][:k]

            if len(recs) == 0:
                continue
            hits = sum(1 for g in recs if g in relevant)
            prec = hits / float(len(recs))

            if prec > best_prec:
                best_prec = prec
                best_t = t

        if best_t is None:
            # fallback: usa threshold médio invés de dropar a variância toda
            best_t = thresholds[len(thresholds) // 2]
            best_prec = 0.0

        rows.append(
            {
                "usuario_id": uid,
                "best_threshold": best_t,
                "best_precision": best_prec,
            }
        )

    df_rows = pd.DataFrame(rows)
    print(f"[{len(df_rows)}] usuários ganharam um target threshold gerado.")
    return df_rows


def main() -> None:
    """
    Orquestra a construção do dataset da cGAN:
      - Features condicionais por usuário
      - Threshold ótimo por usuário (target)
      - Splits train/val/test
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_users, user_features, _ = build_user_features()
    df_targets = compute_best_thresholds(k=10)

    # Garantia defensiva de que df_targets não está vazio
    if df_targets.empty:
        raise ValueError("ERRO: Nenhum usuário obteve um threshold melhor que o aleatório na simulação.")

    if "usuario_id" not in df_targets.columns:
        first_col = df_targets.columns[0]
        df_targets = df_targets.rename(columns={first_col: "usuario_id"})

    df = user_features.reset_index().merge(df_targets, on="usuario_id", how="inner")
    df = df.dropna(subset=["best_threshold"])
    
    if df.empty:
        raise ValueError("ERRO: O merge resultou em um dataframe vazio após a junção entre Features e Targets.")

    # Split train/val/test estratificado por best_threshold (aproximação do perfil)
    from sklearn.model_selection import train_test_split

    train_val, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["best_threshold"]
    )
    train, val = train_test_split(
        train_val,
        test_size=0.125,
        random_state=42,
        stratify=train_val["best_threshold"],
    )

    train.to_parquet(OUTPUT_DIR / "cgan_users_train.parquet", index=False)
    val.to_parquet(OUTPUT_DIR / "cgan_users_val.parquet", index=False)
    test.to_parquet(OUTPUT_DIR / "cgan_users_test.parquet", index=False)


if __name__ == "__main__":
    main()

