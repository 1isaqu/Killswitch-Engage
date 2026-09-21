"""Avalia a camada 2 (KMeans) pelo que ela deveria entregar: recomendação.

MOTIVAÇÃO
---------
`train_layer2_clustering.py` escolhe `k` maximizando silhouette, e usa
`PCA(n_components=0.95)`. Duas objeções, medidas em README §3.1:

1. O PCA quase não reduz — retém 21 de 26 colunas. A matriz de gênero é esparsa
   e composicional (usuário médio toca 4,4 de 22 gêneros), então reter 95% da
   variância preserva direções de ruído.
2. O silhouette é calculado nesse espaço de 21 dimensões. Distâncias se
   concentram em alta dimensão, e a métrica perde poder discriminativo.

Mais grave: silhouette mede coesão geométrica, não utilidade. Um agrupamento com
silhouette alto pode recomendar mal. Este script mede as duas coisas e compara.

COMO A RECOMENDAÇÃO POR CLUSTER FUNCIONA
----------------------------------------
Para o usuário u no cluster c, o score de um item é quantos usuários de c
tocaram aquele item no treino. É "gente parecida com você joga isto" — a forma
mais simples de usar a camada 2 para recomendar, e a que o README descrevia.

Avaliado na tarefa de DESCOBERTA (itens que o usuário nunca tocou), porque é
onde o ranker SVD perde para a popularidade global.

Sem vazamento: as features de cluster e a popularidade por cluster saem apenas
do conjunto de treino.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_ranker import (  # noqa: E402
    K, SEED, avaliar, comparar_pareado, ic_bootstrap, split_temporal,
)

DIMENSOES = [2, 3, 5, 8, 13, 21]
K_RANGE = range(3, 11)
EVAL_SAMPLE = 2_000


def construir_features(treino: pd.DataFrame, jogos: pd.DataFrame) -> pd.DataFrame:
    """Perfil de gênero + estatísticas de engajamento, só a partir do treino."""
    jogos = jogos.copy()
    cats = jogos["categorias"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    jogos["genero"] = cats.apply(lambda c: c[0] if c else "Unknown")

    m = treino.merge(jogos[["id", "genero"]], left_on="jogo_id", right_on="id")
    por_genero = (
        m.groupby(["usuario_id", "genero"])["horas_jogadas"].sum().unstack(fill_value=0)
    )
    pct = por_genero.div(por_genero.sum(axis=1), axis=0)

    stats = treino.groupby("usuario_id").agg(
        {"jogo_id": "nunique", "horas_jogadas": ["sum", "mean", "count"]}
    )
    stats.columns = ["jogos_unicos", "horas_total", "horas_media", "n_sessoes"]

    return pct.merge(stats, left_index=True, right_index=True).fillna(0.0)


def popularidade_por_cluster(
    treino: pd.DataFrame, rotulos: dict[int, int], n_clusters: int, n_games: int
) -> np.ndarray:
    """Matriz (n_clusters, n_games): usuários distintos do cluster que tocaram o item."""
    mat = np.zeros((n_clusters, n_games))
    df = treino[["usuario_id", "jogo_id"]].drop_duplicates()
    c = df["usuario_id"].map(rotulos)
    ok = c.notna()
    np.add.at(mat, (c[ok].astype(int).values, df["jogo_id"][ok].values), 1.0)
    return mat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default="data/ml_ready")
    args = ap.parse_args()

    sess = pd.read_csv(os.path.join(args.csv_dir, "interacoes_sessoes.csv"))
    sess = sess.rename(columns={"inicio": "t"})
    sess["t"] = pd.to_datetime(sess["t"], format="mixed", utc=True).astype("int64")
    jogos = pd.read_csv(os.path.join(args.csv_dir, "jogos_features.csv"))

    treino, verdade_val, _ = split_temporal(sess)
    n_games = int(sess["jogo_id"].max()) + 1

    X_raw = construir_features(treino, jogos)
    print(f"Features por usuário: {X_raw.shape[1]} colunas, {X_raw.shape[0]:,} usuários")
    genero_cols = [c for c in X_raw.columns if c not in
                   {"jogos_unicos", "horas_total", "horas_media", "n_sessoes"}]
    toca = (X_raw[genero_cols] > 0).sum(axis=1).mean()
    print(f"Esparsidade: usuário médio toca {toca:.1f} de {len(genero_cols)} gêneros\n")

    X_scaled = StandardScaler().fit_transform(X_raw)

    rng = np.random.default_rng(SEED)
    elegiveis = np.array(sorted(set(verdade_val) & set(X_raw.index)))
    users = rng.choice(elegiveis, size=min(EVAL_SAMPLE, len(elegiveis)), replace=False)
    vistos = treino.groupby("usuario_id")["jogo_id"].apply(set).to_dict()

    # Referência: popularidade global, o que a camada 2 precisa superar
    pop_global = np.zeros(n_games)
    d = treino[["usuario_id", "jogo_id"]].drop_duplicates()
    np.add.at(pop_global, d["jogo_id"].values, 1.0)
    ref = avaliar("pop global", lambda u: pop_global, users, verdade_val, n_games, vistos=vistos)
    print(f"Baseline — popularidade global (descoberta): P@10 = {ref.precision:.4f}\n")

    print(f"{'PCA dims':>9} {'k':>3} {'silhouette':>11} {'P@10 descoberta':>17} {'IC 95%':>20}")
    print("-" * 68)

    melhor = (None, None, -1.0)
    melhor_res = None
    for n_dim in DIMENSOES:
        pca = PCA(n_components=n_dim, random_state=SEED)
        X_pca = pca.fit_transform(X_scaled)

        # Escolhe k por silhouette, como o projeto faz
        melhor_k, melhor_sil, melhor_lab = None, -1.0, None
        for k in K_RANGE:
            km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
            lab = km.fit_predict(X_pca)
            sil = silhouette_score(X_pca, lab, sample_size=5_000, random_state=SEED)
            if sil > melhor_sil:
                melhor_k, melhor_sil, melhor_lab = k, sil, lab

        rotulos = dict(zip(X_raw.index, melhor_lab))
        mat = popularidade_por_cluster(treino, rotulos, melhor_k, n_games)
        res = avaliar(
            f"pca{n_dim}",
            lambda u: mat[rotulos[u]],
            users,
            verdade_val,
            n_games,
            vistos=vistos,
        )
        lo, hi = ic_bootstrap(res.por_usuario)
        marca = "  <-- melhor" if res.precision > melhor[2] else ""
        if res.precision > melhor[2]:
            melhor = (n_dim, melhor_k, res.precision)
            melhor_res = res
        print(
            f"{n_dim:>9} {melhor_k:>3} {melhor_sil:>11.4f} {res.precision:>17.4f}"
            f"  [{lo:.4f}, {hi:.4f}]{marca}"
        )

    print(f"\nMelhor: PCA={melhor[0]} dims, k={melhor[1]}, P@10 = {melhor[2]:.4f}")
    print(f"Popularidade global:  P@10 = {ref.precision:.4f}")
    razao = melhor[2] / ref.precision if ref.precision > 0 else float("nan")
    print(f"Razão: {razao:.2f}x")

    d, lo, hi = comparar_pareado(melhor_res, ref)
    sig = "significativa" if (lo > 0 or hi < 0) else "NÃO significativa"
    print("\nMelhor cluster - popularidade global (bootstrap pareado):")
    print(f"  diferença {d:+.4f}  IC 95% [{lo:+.4f}, {hi:+.4f}]  -> {sig}")


if __name__ == "__main__":
    main()
