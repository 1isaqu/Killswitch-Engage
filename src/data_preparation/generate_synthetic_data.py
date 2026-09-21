"""Gera o dataset sintético localmente, em CSV, sem depender do Supabase.

POR QUE ESTE SCRIPT EXISTE
--------------------------
O caminho original era:

    Steam CSV -> populate_supabase*.py -> [Supabase] -> extract_data_for_ml.py -> CSVs

O banco era só um entreposto: os dados entravam e saíam como CSV, e nenhum script
de ML falava com ele. Isso criava uma dependência dura de um serviço hospedado
externo para rodar o pipeline — e projetos gratuitos do Supabase são pausados e
eventualmente removidos por inatividade. Este script produz os mesmos três CSVs
direto em disco.

O QUE MUDOU EM RELAÇÃO A populate_supabase_v2.py
------------------------------------------------
1. SEED. O gerador original chamava `random` sem semear, então o dataset nunca
   foi reproduzível — nem quando o banco estava vivo. Aqui tudo passa por um
   `np.random.Generator` semeado.

2. ESTRUTURA COLABORATIVA COMPARTILHADA. No original (populate_supabase_v2.py:119)
   os favoritos de cada usuário eram sorteados uniformemente do catálogo inteiro,
   de forma independente entre usuários. Consequência medida em
   `scripts/experimentation/evaluate_ranker.py`: 99.9% dos pares de usuários não
   compartilhavam nenhum jogo, e a precisão do ranker sobre itens não-vistos era
   exatamente zero. Filtragem colaborativa precisa de usuários parecidos; não
   havia nenhum.

   Aqui os jogos têm gêneros, cada usuário tem uma afinidade por gênero
   (Dirichlet esparsa) e os favoritos são sorteados proporcionalmente a
   `afinidade_de_genero x popularidade_intrinseca`. Usuários com gostos parecidos
   passam a se sobrepor — que é a premissa que a camada 3 precisa.

3. POPULARIDADE EM LEI DE POTÊNCIA. A popularidade intrínseca vem de uma Zipf, de
   modo que poucos títulos concentram a maior parte das interações, como num
   catálogo real. No original a popularidade saía achatada (top-1% dos jogos com
   ~4.7% das interações, contra 60-80% na Steam real).

Os perfis de engajamento (70% casual / 25% médio / 5% hardcore + 500 usuários de
borda) e a janela temporal de 720 dias seguem o original.

Uso:
    python -m src.data_preparation.generate_synthetic_data
    python -m src.data_preparation.generate_synthetic_data --n-users 2000 --out data/ml_ready
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

SEED = 42
N_USERS = 10_000
N_GAMES = 122_507
N_GENRES = 22
ZIPF_A = 1.15  # expoente da cauda longa de popularidade
DIRICHLET_ALPHA = 0.25  # < 1 => afinidade concentrada em poucos gêneros

GENEROS = [
    "Action", "Adventure", "RPG", "Strategy", "Simulation", "Sports", "Racing",
    "Indie", "Casual", "Puzzle", "Shooter", "Platformer", "Horror", "Survival",
    "Fighting", "Stealth", "Roguelike", "Sandbox", "MMO", "Visual Novel",
    "Rhythm", "Card",
]

PERFIS = {
    "casual": ((5, 10), (0.1, 3.0)),
    "medio": ((20, 50), (0.5, 8.0)),
    "hardcore": ((50, 200), (1.0, 20.0)),
}


def gerar_jogos(rng: np.random.Generator, n_games: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Cria o catálogo com gêneros e popularidade intrínseca em lei de potência."""
    genero_primario = rng.integers(0, N_GENRES, size=n_games)

    # Popularidade intrínseca: Zipf truncada, embaralhada entre os jogos
    pop = 1.0 / np.power(np.arange(1, n_games + 1), ZIPF_A)
    rng.shuffle(pop)

    lancamento = [
        datetime(2010, 1, 1) + timedelta(days=int(d))
        for d in rng.integers(0, 5_600, size=n_games)
    ]

    df = pd.DataFrame(
        {
            "id": np.arange(n_games),
            "titulo": [f"Game {i}" for i in range(n_games)],
            "preco_base": np.round(rng.gamma(2.0, 12.0, size=n_games), 2),
            "avaliacao_media": np.round(np.clip(rng.normal(3.6, 0.8, size=n_games), 0, 5), 2),
            "total_avaliacoes": (pop / pop.max() * 500_000).astype(int),
            "metacritic_score": np.clip(rng.normal(72, 12, size=n_games), 0, 100).astype(int),
            "idade_requerida": rng.choice([0, 12, 16, 18], size=n_games, p=[0.6, 0.2, 0.1, 0.1]),
            "data_lancamento": [d.date().isoformat() for d in lancamento],
            "dev_nome": [f"Dev {i % 4000}" for i in range(n_games)],
            "dev_id": np.arange(n_games) % 4000,
            "categorias": [f"['{GENEROS[g]}']" for g in genero_primario],
        }
    )
    return df, genero_primario, pop


def gerar_sessoes(
    rng: np.random.Generator,
    n_users: int,
    n_games: int,
    genero_primario: np.ndarray,
    pop: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sorteia favoritos por afinidade de gênero x popularidade, e gera as sessões."""
    # Índice: jogos de cada gênero + distribuição acumulada de popularidade dentro dele
    por_genero, cdf_genero = [], []
    for g in range(N_GENRES):
        idx = np.flatnonzero(genero_primario == g)
        w = pop[idx]
        por_genero.append(idx)
        cdf_genero.append(np.cumsum(w / w.sum()))

    user_ids = np.arange(n_users)
    ordem = rng.permutation(user_ids)
    n_casual, n_medio = int(n_users * 0.70), int(n_users * 0.25)
    perfil = {}
    for uid in ordem[:n_casual]:
        perfil[uid] = "casual"
    for uid in ordem[n_casual : n_casual + n_medio]:
        perfil[uid] = "medio"
    for uid in ordem[n_casual + n_medio :]:
        perfil[uid] = "hardcore"
    borda = set(rng.choice(ordem[: int(n_users * 0.05)], size=min(500, n_users // 20), replace=False).tolist())

    # Afinidade esparsa por gênero — é o que faz usuários parecidos se sobreporem
    afinidade = rng.dirichlet(np.full(N_GENRES, DIRICHLET_ALPHA), size=n_users)

    u_col, g_col, h_col, t_col = [], [], [], []
    for uid in user_ids:
        if uid in borda:
            n_sess, (lo, hi) = int(rng.integers(150, 301)), (0.1, 24.0)
        else:
            (a, b), (lo, hi) = PERFIS[perfil[uid]]
            n_sess = int(rng.integers(a, b + 1))

        # Favoritos: escolhe gênero pela afinidade, depois jogo pela popularidade
        n_fav = max(3, n_sess // 8)
        generos_fav = rng.choice(N_GENRES, size=n_fav, p=afinidade[uid])
        favoritos = np.array(
            [
                por_genero[g][int(np.searchsorted(cdf_genero[g], rng.random()))]
                for g in generos_fav
            ]
        )
        favoritos = np.unique(favoritos)

        # 80% das sessões nos favoritos; 20% exploram o gênero preferido
        usa_fav = rng.random(n_sess) < 0.80
        outros_g = rng.choice(N_GENRES, size=n_sess, p=afinidade[uid])
        outros = np.array(
            [
                por_genero[g][int(np.searchsorted(cdf_genero[g], rng.random()))]
                for g in outros_g
            ]
        )
        jogos = np.where(usa_fav, favoritos[rng.integers(0, len(favoritos), size=n_sess)], outros)

        u_col.append(np.full(n_sess, uid))
        g_col.append(jogos)
        h_col.append(rng.uniform(lo, hi, size=n_sess))
        # janela de 720 dias, com mais massa no passado recente
        t_col.append(720 - rng.beta(1.5, 3.0, size=n_sess) * 720)

    base = datetime(2024, 1, 1)
    dias = np.concatenate(t_col)
    df_sessoes = pd.DataFrame(
        {
            "usuario_id": np.concatenate(u_col),
            "jogo_id": np.concatenate(g_col),
            "horas_jogadas": np.round(np.concatenate(h_col), 2),
            "inicio": [(base + timedelta(days=float(d))).isoformat() for d in dias],
        }
    )

    df_usuarios = pd.DataFrame(
        {
            "id": user_ids,
            "pais": rng.choice(["BR", "US", "DE", "JP"], size=n_users),
            "tipo_assinatura": rng.choice(["free", "plus"], size=n_users, p=[0.8, 0.2]),
        }
    )
    return df_sessoes, df_usuarios


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-users", type=int, default=N_USERS)
    ap.add_argument("--n-games", type=int, default=N_GAMES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="data/ml_ready")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    print(f"Gerando catálogo de {args.n_games:,} jogos (seed={args.seed}) ...")
    df_jogos, genero_primario, pop = gerar_jogos(rng, args.n_games)

    print(f"Gerando {args.n_users:,} usuários e suas sessões ...")
    df_sessoes, df_usuarios = gerar_sessoes(rng, args.n_users, args.n_games, genero_primario, pop)

    df_jogos.to_csv(os.path.join(args.out, "jogos_features.csv"), index=False)
    df_usuarios.to_csv(os.path.join(args.out, "usuarios_features.csv"), index=False)
    df_sessoes.to_csv(os.path.join(args.out, "interacoes_sessoes.csv"), index=False)

    print(
        f"\nOK: {len(df_jogos):,} jogos | {len(df_usuarios):,} usuários | "
        f"{len(df_sessoes):,} sessões -> {args.out}/"
    )
    print("Reproduzível: mesmo --seed produz exatamente o mesmo dataset.")


if __name__ == "__main__":
    main()
