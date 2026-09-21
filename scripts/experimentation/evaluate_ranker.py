"""Avaliação offline honesta da camada 3 (ranker SVD).

O QUE ESTE SCRIPT FAZ DE DIFERENTE de run_experiments.py:
  - O gabarito vem de um SPLIT TEMPORAL real: as últimas 20% das sessões de cada
    usuário são removidas do treino e usadas como verdade. Nada é derivado da
    própria predição.
  - As recomendações vêm do modelo de verdade (TruncatedSVD treinado só no train),
    não de np.random.
  - Há baselines para comparar: popularidade e aleatório. Um número sozinho não
    diz nada.

ORIGEM DOS DADOS (leia antes de citar qualquer número):
  Os dados originais do projeto não estão disponíveis. Este script REGENERA um
  dataset sintético seguindo o mesmo processo generativo de
  `src/data_preparation/populate_supabase_v2.py` (mesma mistura de perfis, mesma
  concentração de 80% em jogos favoritos, mesma janela temporal de 720 dias).

  Portanto os resultados medem o comportamento do ALGORITMO sobre dados com a
  mesma estrutura — NÃO são as métricas do dataset original do autor. Para obter
  aquelas, aponte CSV_DIR para os CSVs reais de data/ml_ready/ e rode de novo.

Uso:
    python scripts/experimentation/evaluate_ranker.py
    python scripts/experimentation/evaluate_ranker.py --csv-dir data/ml_ready
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

SEED = 42
N_USERS = 10_000
N_GAMES = 122_507
N_COMPONENTS = 50
K = 10
HOLDOUT_FRAC = 0.20
EVAL_SAMPLE = 2_000  # usuários avaliados (scoring é O(n_users * n_games))


# ── Geração sintética (espelha populate_supabase_v2.py) ──────────────────────


def gerar_sessoes(rng: np.random.Generator) -> pd.DataFrame:
    """Reproduz a mistura de perfis e a concentração em favoritos do gerador original."""
    user_ids = np.arange(N_USERS)
    rng.shuffle(user_ids)

    n_casual = int(N_USERS * 0.70)
    n_medio = int(N_USERS * 0.25)

    perfil = {}
    for uid in user_ids[:n_casual]:
        perfil[uid] = "casual"
    for uid in user_ids[n_casual : n_casual + n_medio]:
        perfil[uid] = "medio"
    for uid in user_ids[n_casual + n_medio :]:
        perfil[uid] = "hardcore"

    # 500 usuários de borda entre os primeiros 5% embaralhados
    edge = set(rng.choice(user_ids[: int(N_USERS * 0.05)], size=500, replace=False).tolist())

    faixas = {
        "casual": ((5, 10), (0.1, 3.0)),
        "medio": ((20, 50), (0.5, 8.0)),
        "hardcore": ((50, 200), (1.0, 20.0)),
    }

    u_col, g_col, h_col, t_col = [], [], [], []
    for uid in user_ids:
        if uid in edge:
            n_sess = int(rng.integers(150, 301))
            lo, hi = 0.1, 24.0
        else:
            (a, b), (lo, hi) = faixas[perfil[uid]]
            n_sess = int(rng.integers(a, b + 1))

        n_fav = max(3, n_sess // 8)
        favoritos = rng.choice(N_GAMES, size=n_fav, replace=False)

        # 80% das sessões caem nos favoritos do usuário
        usa_fav = rng.random(n_sess) < 0.80
        jogos = np.where(
            usa_fav,
            favoritos[rng.integers(0, n_fav, size=n_sess)],
            rng.integers(0, N_GAMES, size=n_sess),
        )
        horas = rng.uniform(lo, hi, size=n_sess)
        # janela de 720 dias com peso maior no passado recente
        dias = 720 - rng.beta(1.5, 3.0, size=n_sess) * 720

        u_col.append(np.full(n_sess, uid))
        g_col.append(jogos)
        h_col.append(horas)
        t_col.append(dias)

    return pd.DataFrame(
        {
            "usuario_id": np.concatenate(u_col),
            "jogo_id": np.concatenate(g_col),
            "horas_jogadas": np.concatenate(h_col),
            "t": np.concatenate(t_col),  # dias desde o início da janela
        }
    )


# ── Split temporal ───────────────────────────────────────────────────────────


def split_temporal(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, set]]:
    """Segura as últimas HOLDOUT_FRAC sessões de cada usuário como verdade."""
    df = df.sort_values(["usuario_id", "t"], kind="stable")
    n_por_user = df.groupby("usuario_id")["t"].transform("size")
    posicao = df.groupby("usuario_id").cumcount()
    corte = (n_por_user * (1 - HOLDOUT_FRAC)).astype(int).clip(lower=1)

    treino = df[posicao < corte]
    teste = df[posicao >= corte]

    verdade = teste.groupby("usuario_id")["jogo_id"].apply(set).to_dict()
    return treino, verdade


# ── Métricas ─────────────────────────────────────────────────────────────────


def ndcg_at_k(recomendados: np.ndarray, relevantes: set, k: int) -> float:
    dcg = sum(1.0 / np.log2(i + 2) for i, g in enumerate(recomendados[:k]) if g in relevantes)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevantes), k)))
    return dcg / idcg if idcg > 0 else 0.0


def varrer_despopularizacao(U, V, popularidade, users, verdade, n_games, vistos) -> None:
    """Penaliza o score pela popularidade do item e mede o efeito na descoberta.

    Sintoma tratado: o ranker acerta mais reordenando o que o usuário já conhece
    do que revelando novidade — na tarefa de descoberta ele perde para a própria
    baseline de popularidade. O ajuste padrão é penalizar itens populares.

    Forma usada (segura para score negativo, ao contrário de `score / pop**a`):

        score' = minmax(score) - alpha * minmax(log1p(popularidade))

    Ambos os termos ficam em [0, 1], então `alpha` é diretamente interpretável:
    0 = sem penalidade, 1 = popularidade pesa tanto quanto o score do modelo.
    """
    print("\n\nDESPOPULARIZAÇÃO DO SCORE (tarefa de descoberta)")
    print("score' = minmax(score) - alpha * minmax(log1p(pop))\n")

    p_log = np.log1p(popularidade.astype(float))
    lo, hi = p_log.min(), p_log.max()
    p_norm = (p_log - lo) / (hi - lo) if hi > lo else np.zeros_like(p_log)

    # Referência: a baseline de popularidade na MESMA tarefa
    ref = avaliar("pop", lambda u: popularidade, users, verdade, n_games, vistos=vistos)
    print(f"{'alpha':>6} {'P@10':>9} {'NDCG@10':>9} {'Cobertura':>11}   vs baseline pop.")
    print("-" * 62)

    melhor = (None, -1.0)
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:

        def scorer(u, a=alpha):
            s = V @ U[u]
            smin, smax = s.min(), s.max()
            s = (s - smin) / (smax - smin) if smax > smin else np.zeros_like(s)
            return s - a * p_norm

        res = avaliar(f"a={alpha}", scorer, users, verdade, n_games, vistos=vistos)
        razao = res.precision / ref.precision if ref.precision > 0 else float("nan")
        marca = "  <-- melhor" if res.precision > melhor[1] else ""
        if res.precision > melhor[1]:
            melhor = (alpha, res.precision)
        print(
            f"{alpha:>6.1f} {res.precision:>9.4f} {res.ndcg:>9.4f} "
            f"{res.cobertura:>10.2%}   {razao:>6.2f}x{marca}"
        )

    print(f"\nBaseline popularidade na mesma tarefa: P@10 = {ref.precision:.4f}")
    print(f"Melhor alpha = {melhor[0]} (P@10 = {melhor[1]:.4f})")


def diagnosticar_sinal(treino: pd.DataFrame, verdade: dict, rng) -> None:
    """Mede se existe sinal COLABORATIVO nos dados.

    Filtragem colaborativa só funciona se usuários compartilham itens. Se cada
    usuário tem gostos sorteados independentemente, não há o que aprender — e
    nenhum ajuste de modelo resolve, porque o problema está nos dados.
    """
    print("\n\nDIAGNÓSTICO: existe sinal colaborativo nestes dados?\n")

    por_user = {u: set(g) for u, g in treino.groupby("usuario_id")["jogo_id"]}
    users = list(por_user)

    pares = rng.choice(len(users), size=(5_000, 2))
    overlap = [len(por_user[users[a]] & por_user[users[b]]) for a, b in pares if a != b]
    print("1. Jogos em comum entre dois usuários aleatórios:")
    print(
        f"   média = {np.mean(overlap):.3f} | "
        f"pares com qualquer sobreposição = {100 * np.mean([o > 0 for o in overlap]):.1f}%"
    )

    contagem = pd.Series(treino.jogo_id.values).value_counts()
    amostra = rng.choice(list(verdade), size=min(2_000, len(verdade)), replace=False)
    proprio = [
        len(verdade[u] & por_user.get(u, set())) / len(verdade[u]) for u in amostra if verdade[u]
    ]
    print(f"\n2. Itens do holdout que o PRÓPRIO usuário já jogou no treino: {100 * np.mean(proprio):.1f}%")
    print("   (é o único sinal que o SVD consegue explorar aqui)")

    c = contagem.values
    top1 = 100 * c[: max(1, int(len(c) * 0.01))].sum() / c.sum()
    print(f"\n3. Top-1% dos jogos concentram {top1:.1f}% das interações.")
    print("   Num catálogo real da Steam a cauda longa passa de 60-80%.")


@dataclass
class Resultado:
    nome: str
    precision: float
    recall: float
    ndcg: float
    cobertura: float


def avaliar(nome, scorer, users, verdade, n_games, vistos=None) -> Resultado:
    """`scorer(uid) -> vetor de scores por item`. `vistos` remove itens do treino."""
    p = r = n = 0.0
    recomendados_unicos = set()

    for uid in users:
        scores = scorer(uid)
        if vistos is not None and uid in vistos:
            scores = scores.copy()
            scores[list(vistos[uid])] = -np.inf

        top = np.argpartition(-scores, K)[:K]
        top = top[np.argsort(-scores[top])]

        rel = verdade[uid]
        hits = sum(1 for g in top if g in rel)
        p += hits / K
        r += hits / len(rel) if rel else 0.0
        n += ndcg_at_k(top, rel, K)
        recomendados_unicos.update(top.tolist())

    m = len(users)
    return Resultado(nome, p / m, r / m, n / m, len(recomendados_unicos) / n_games)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default=None, help="Usar CSVs reais de data/ml_ready")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)

    if args.csv_dir:
        print(f"Lendo sessões reais de {args.csv_dir} ...")
        df = pd.read_csv(os.path.join(args.csv_dir, "interacoes_sessoes.csv"))
        df = df.rename(columns={"inicio": "t"})
        df["t"] = pd.to_datetime(df["t"], format="mixed", utc=True).astype("int64")
        n_games = int(df["jogo_id"].max()) + 1
    else:
        print("⚠️  DADOS REGENERADOS — não são os CSVs originais do projeto.")
        print("   Medem o algoritmo sobre dados com a mesma estrutura generativa.\n")
        print("Gerando sessões sintéticas ...")
        df = gerar_sessoes(rng)
        n_games = N_GAMES

    print(f"  sessões: {len(df):,} | usuários: {df.usuario_id.nunique():,} | jogos: {n_games:,}")

    treino, verdade = split_temporal(df)
    print(f"  treino: {len(treino):,} | holdout: {len(df) - len(treino):,}\n")

    # Matriz de interações — mesma ponderação do train_layer3_ranker.py
    linhas = treino.usuario_id.values
    colunas = treino.jogo_id.values
    dados = np.log1p(treino.horas_jogadas.values) + 1
    n_users = int(df.usuario_id.max()) + 1
    X = csr_matrix((dados, (linhas, colunas)), shape=(n_users, n_games))

    print(f"Treinando TruncatedSVD({N_COMPONENTS}) em {X.shape} ...")
    svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=SEED)
    U = svd.fit_transform(X)
    V = svd.components_.T
    print(f"  variância explicada: {svd.explained_variance_ratio_.sum():.4f}\n")

    # Popularidade do treino = nº de usuários distintos por jogo
    Xb = X.copy()
    Xb.data = np.ones_like(Xb.data)
    popularidade = np.asarray(Xb.sum(axis=0)).ravel()

    # Usuários avaliados: precisam ter holdout
    elegiveis = np.array(sorted(verdade.keys()))
    users = rng.choice(elegiveis, size=min(EVAL_SAMPLE, len(elegiveis)), replace=False)

    vistos = treino.groupby("usuario_id")["jogo_id"].apply(set).to_dict()

    print(f"Avaliando {len(users):,} usuários (k={K}) ...\n")
    resultados = [
        avaliar("SVD (camada 3)", lambda u: V @ U[u], users, verdade, n_games),
        avaliar(
            "SVD sem itens já vistos", lambda u: V @ U[u], users, verdade, n_games, vistos=vistos
        ),
        avaliar("Baseline: popularidade", lambda u: popularidade, users, verdade, n_games),
        # Comparação justa para a linha "sem itens já vistos": a baseline também
        # precisa ser avaliada na tarefa de descoberta, não na de memorização.
        avaliar(
            "Popularidade sem itens vistos",
            lambda u: popularidade,
            users,
            verdade,
            n_games,
            vistos=vistos,
        ),
        avaliar(
            "Baseline: aleatório",
            lambda u: rng.random(n_games),
            users,
            verdade,
            n_games,
        ),
    ]

    print(f"{'Modelo':<28} {'P@10':>8} {'R@10':>8} {'NDCG@10':>9} {'Cobertura':>11}")
    print("-" * 68)
    for res in resultados:
        print(
            f"{res.nome:<28} {res.precision:>8.4f} {res.recall:>8.4f} "
            f"{res.ndcg:>9.4f} {res.cobertura:>10.2%}"
        )

    # ── Efeito da correção de escala do threshold ────────────────────────────
    print("\n\nEFEITO DA NORMALIZAÇÃO DO THRESHOLD")
    print("Quantos itens passam em cada modo, antes e depois da correção:\n")
    print(f"{'Modo':<14} {'thr':>5} {'SEM normalizar':>18} {'COM normalizar':>18}")
    print("-" * 60)
    amostra = users[:200]
    for modo, thr in [("conservador", 0.7), ("equilibrado", 0.5), ("aventureiro", 0.3)]:
        cru = norm = 0
        for u in amostra:
            s = V @ U[u]
            cru += int((s >= thr).sum())
            lo, hi = s.min(), s.max()
            sn = (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)
            norm += int((sn >= thr).sum())
        print(f"{modo:<14} {thr:>5.1f} {cru / len(amostra):>18,.0f} {norm / len(amostra):>18,.0f}")

    # Sobreposição entre modos (o README alegava 4/10 em comum)
    print("\nSobreposição conservador × aventureiro no top-10 (com normalização):")
    overlap = 0
    for u in amostra:
        s = V @ U[u]
        lo, hi = s.min(), s.max()
        sn = (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)
        tops = {}
        for modo, thr in [("conservador", 0.7), ("aventureiro", 0.3)]:
            validos = np.where(sn >= thr)[0]
            if len(validos) == 0:
                validos = np.argsort(-sn)[: K * 2]
            tops[modo] = set(validos[np.argsort(-sn[validos])][:K].tolist())
        overlap += len(tops["conservador"] & tops["aventureiro"])
    print(f"  {overlap / len(amostra):.1f} de {K} jogos em comum, em média")
    print("  (o threshold só restringe o pool de candidatos; como o top-k é sempre")
    print("   ordenado por score, ele não diversifica por si só — quem diversifica")
    print("   são os slots aleatórios de `exploracao`.)")

    varrer_despopularizacao(U, V, popularidade, users, verdade, n_games, vistos)
    diagnosticar_sinal(treino, verdade, np.random.default_rng(0))


if __name__ == "__main__":
    sys.exit(main())
