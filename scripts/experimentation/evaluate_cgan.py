"""Testa se a camada 4 (cGAN) se sustenta quando dimensionada pelos dados.

O QUE ESTE SCRIPT RESPONDE
--------------------------
A auditoria mediu que o alvo da cGAN (`best_threshold`) tem ~1,45 bits de entropia
e 72% da massa num único valor, enquanto o generator original tem ~57 mil
parâmetros. Duas perguntas caem disso:

1. O mode collapse some com um modelo dimensionado para o alvo?
2. A cGAN bate a baseline honesta — prever sempre a moda — ou só a baseline
   inflada do projeto (constante 0.5)?

A comparação contra 0.5 é o ponto fraco do relatório original: a constante 0.5
erra muito mais que a moda, então o "ganho" media a escolha da baseline, não o
aprendizado.

MÉTODO
------
Split por USUÁRIO (80/20): a tarefa da cGAN é prever o threshold de um usuário a
partir do perfil dele, então generalizar significa acertar usuário não visto. O
alvo vem do ranker k=16 avaliado contra as sessões de validação, reproduzindo
`build_cgan_dataset.compute_best_thresholds`.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "meta_learning"))
from evaluate_clustering import construir_features  # noqa: E402
from evaluate_ranker import K_ESCOLHIDO, SEED, split_temporal  # noqa: E402

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


class Gerador(nn.Module):
    """Generator condicional. Mesma forma do scripts/meta_learning/cgan_model.py."""

    def __init__(self, latent: int, cond: int, hidden: int, camadas: int, dropout: float = 0.2):
        super().__init__()
        blocos, entrada = [], latent + cond
        for _ in range(camadas):
            blocos += [nn.Linear(entrada, hidden), nn.LeakyReLU(0.2), nn.Dropout(dropout), nn.BatchNorm1d(hidden)]
            entrada = hidden
        self.net = nn.Sequential(*blocos)
        self.saida = nn.Sequential(nn.Linear(hidden, 1), nn.Tanh())

    def forward(self, ruido, cond):
        return self.saida(self.net(torch.cat([ruido, cond], -1))) * 0.30 + 0.55


class Discriminador(nn.Module):
    def __init__(self, cond: int, hidden: int, camadas: int, dropout: float = 0.2):
        super().__init__()
        blocos, entrada = [], 1 + cond
        for i in range(camadas):
            saida = max(4, hidden // (2**i))
            blocos += [nn.Linear(entrada, saida), nn.LeakyReLU(0.2), nn.Dropout(dropout)]
            entrada = saida
        self.net = nn.Sequential(*blocos)
        self.saida = nn.Linear(entrada, 1)

    def forward(self, thr, cond):
        return self.saida(self.net(torch.cat([thr, cond], -1)))


def alvo_best_threshold(U, V, verdade, usuarios, k=10) -> dict[int, float]:
    """Reproduz build_cgan_dataset.compute_best_thresholds sobre o ranker."""
    alvos = {}
    for u in usuarios:
        rel = verdade.get(u)
        if not rel:
            continue
        s = V @ U[u]
        lo, hi = s.min(), s.max()
        sn = (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)
        melhor_t, melhor_p = None, -1.0
        for t in THRESHOLDS:
            m = np.where(sn >= t)[0]
            if len(m) == 0:
                continue
            recs = m[np.argsort(-sn[m])][:k]
            p = sum(1 for g in recs if g in rel) / len(recs)
            if p > melhor_p:
                melhor_p, melhor_t = p, t
        alvos[u] = melhor_t if melhor_t is not None else 0.5
    return alvos


def treinar(X_tr, y_tr, latent, hidden, camadas, epocas=200, seed=SEED, verbose=False):
    """Treina a cGAN. Mesma receita do train_cgan.py: BCE adversarial + L1 peso 5."""
    torch.manual_seed(seed)
    cond = X_tr.shape[1]
    g = Gerador(latent, cond, hidden, camadas)
    d = Discriminador(cond, hidden, camadas)
    n_params = sum(p.numel() for p in g.parameters())

    opt_g = torch.optim.Adam(g.parameters(), lr=1e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(d.parameters(), lr=4e-4, betas=(0.5, 0.999))
    bce, l1 = nn.BCEWithLogitsLoss(), nn.L1Loss()

    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(-1)
    n, lote = len(Xt), 256

    for ep in range(epocas):
        perm = torch.randperm(n)
        for i in range(0, n - lote + 1, lote):
            idx = perm[i : i + lote]
            c, real = Xt[idx], yt[idx]
            uns = torch.ones(len(idx), 1)
            zeros = torch.zeros(len(idx), 1)

            for _ in range(2):  # n_critic
                opt_d.zero_grad()
                fake = g(torch.randn(len(idx), latent), c)
                perda_d = (bce(d(real, c), uns) + bce(d(fake.detach(), c), zeros)) / 2
                perda_d.backward()
                nn.utils.clip_grad_norm_(d.parameters(), 1.0)
                opt_d.step()

            opt_g.zero_grad()
            ger = g(torch.randn(len(idx), latent), c)
            perda_g = bce(d(ger, c), uns) * 1.0 + l1(ger, real) * 5.0
            perda_g.backward()
            nn.utils.clip_grad_norm_(g.parameters(), 1.0)
            opt_g.step()

        if verbose and (ep + 1) % 50 == 0:
            print(f"    época {ep + 1}/{epocas}  D={perda_d.item():.4f}  G={perda_g.item():.4f}")

    return g, n_params


def avaliar_modelo(g, X_te, y_te, latent, seed=SEED):
    """MAE no conjunto de teste, mais diagnóstico de colapso."""
    g.eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        pred = g(torch.randn(len(X_te), latent), torch.tensor(X_te, dtype=torch.float32))
    p = pred.squeeze(-1).numpy()
    mae = float(np.abs(p - y_te).mean())
    corr = float(np.corrcoef(p, y_te)[0, 1]) if p.std() > 1e-9 else 0.0
    return mae, float(p.std()), corr


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
    n_users = int(sess["usuario_id"].max()) + 1

    X = csr_matrix(
        (np.log1p(treino.horas_jogadas.values) + 1, (treino.usuario_id.values, treino.jogo_id.values)),
        shape=(n_users, n_games),
    )
    svd = TruncatedSVD(n_components=K_ESCOLHIDO, random_state=SEED)
    U, V = svd.fit_transform(X), None
    V = svd.components_.T

    feats = construir_features(treino, jogos)
    rng = np.random.default_rng(SEED)
    candidatos = sorted(set(verdade_val) & set(feats.index))
    amostra = rng.choice(candidatos, size=min(6000, len(candidatos)), replace=False)

    print(f"Calculando best_threshold para {len(amostra):,} usuários (ranker k={K_ESCOLHIDO}) ...")
    alvos = alvo_best_threshold(U, V, verdade_val, amostra)
    uids = np.array(sorted(alvos))
    y = np.array([alvos[u] for u in uids])
    Xf = StandardScaler().fit_transform(feats.loc[uids].values)

    dist = pd.Series(y).value_counts(normalize=True).sort_index()
    ent = float(-(dist * np.log2(dist)).sum())
    print(f"  alvo: moda={pd.Series(y).mode()[0]} com {100 * dist.max():.1f}% | entropia {ent:.3f} bits\n")

    # Split por usuário
    corte = int(len(uids) * 0.8)
    ordem = rng.permutation(len(uids))
    tr, te = ordem[:corte], ordem[corte:]
    X_tr, y_tr, X_te, y_te = Xf[tr], y[tr], Xf[te], y[te]

    moda = float(pd.Series(y_tr).mode()[0])
    print("BASELINES (no conjunto de teste)")
    print(f"  sempre prever a moda do treino ({moda}):  MAE = {np.abs(y_te - moda).mean():.4f}")
    print(f"  sempre prever 0.5 (baseline do projeto):  MAE = {np.abs(y_te - 0.5).mean():.4f}")
    print(f"  média do treino ({y_tr.mean():.3f}):              MAE = {np.abs(y_te - y_tr.mean()).mean():.4f}\n")

    configs = [
        ("original (latent 32, hidden 128, 3 camadas)", 32, 128, 3),
        ("pequena  (latent 4,  hidden 16,  2 camadas)", 4, 16, 2),
    ]
    print(f"{'configuração':<46} {'params':>8} {'MAE':>8} {'std pred':>10} {'corr':>7}")
    print("-" * 84)
    for nome, latent, hidden, camadas in configs:
        g, n_params = treinar(X_tr, y_tr, latent, hidden, camadas)
        mae, std, corr = avaliar_modelo(g, X_te, y_te, latent)
        print(f"{nome:<46} {n_params:>8,} {mae:>8.4f} {std:>10.4f} {corr:>7.3f}")

    print("\nstd pred ~0 = mode collapse (rede ignora a condição).")
    print("corr = correlação entre predição e alvo. ~0 = sem sinal condicional.")


if __name__ == "__main__":
    main()
