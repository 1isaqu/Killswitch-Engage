"""WARP — Weighted Approximate-Rank Pairwise (Weston et al., 2011).

POR QUE WARP E NÃO BPR
----------------------
BPR amostra um negativo uniforme por positivo. Com densidade de 0,02% esse
negativo é quase sempre trivial de separar, e a perda cai a ~0,039 sem que o
modelo tenha aprendido a ordenar candidatos plausíveis — foi o que
`evaluate_ranker.py` mediu.

WARP ataca exatamente isso:

1. Para o par (usuário u, item positivo i), reamostra negativos até encontrar um
   que **viole** a margem: `score_j + margem > score_i`.
2. O número de tentativas até a violação estima o rank de i. Poucas tentativas =
   muitos itens ruins acima do positivo = erro grave.
3. A atualização é ponderada por `Phi(rank) = sum_{k=1..rank} 1/k`, que cresce
   devagar. Isso concentra o aprendizado no topo da lista, que é o que a métrica
   @10 mede.

É a perda que o LightFM implementa — a biblioteca que o README alegava usar e
que o código nunca importou.

IMPLEMENTAÇÃO
-------------
A contagem de tentativas varia por exemplo, o que é hostil a vetorização. A saída
usada aqui é amostrar em rodadas: a cada rodada só os exemplos ainda sem violação
sorteiam de novo, e registra-se em que rodada cada um violou. Exemplos que passam
`max_tentativas` sem violar já estão bem ordenados e não geram atualização.
"""

from __future__ import annotations

import numpy as np


class WARPRanker:
    """Fatoração matricial treinada com perda WARP.

    Args:
        n_factors: dimensão dos embeddings latentes.
        learning_rate: passo do SGD.
        n_epochs: passadas sobre as interações.
        reg: coeficiente de regularização L2.
        margem: folga exigida entre positivo e negativo.
        max_tentativas: teto de reamostragens por positivo. Mais tentativas
            encontram negativos mais difíceis e custam mais tempo.
        max_norm: teto da norma L2 de cada linha de embedding. WARP pondera a
            atualização por Phi(rank), que aqui chega a ~12 com 122k itens; sem
            essa projeção os embeddings crescem sem limite e o treino diverge
            (overflow no produto escalar, violações viram NaN).
        use_item_bias: inclui viés por item (absorve popularidade).
        batch_size: pares positivos por atualização.
        seed: semente do gerador.
    """

    def __init__(
        self,
        n_factors: int = 16,
        learning_rate: float = 0.05,
        n_epochs: int = 30,
        reg: float = 0.01,
        margem: float = 1.0,
        max_tentativas: int = 50,
        max_norm: float = 1.0,
        use_item_bias: bool = False,
        batch_size: int = 4096,
        seed: int = 42,
    ) -> None:
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.reg = reg
        self.margem = margem
        self.max_tentativas = max_tentativas
        self.max_norm = max_norm
        self.use_item_bias = use_item_bias
        self.batch_size = batch_size
        self.seed = seed
        self.U: np.ndarray | None = None
        self.V: np.ndarray | None = None
        self.b: np.ndarray | None = None

    def fit(self, users, items, n_users: int, n_items: int, verbose: bool = True) -> "WARPRanker":
        rng = np.random.default_rng(self.seed)
        self.U = rng.normal(0, 0.1, size=(n_users, self.n_factors))
        self.V = rng.normal(0, 0.1, size=(n_items, self.n_factors))
        self.b = np.zeros(n_items)

        # Phi(k) = soma harmônica até k, pré-computada para lookup O(1).
        phi = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, n_items + 1))])

        n_obs = len(users)
        lr, reg = self.learning_rate, self.reg

        for epoca in range(self.n_epochs):
            ordem = rng.permutation(n_obs)
            n_violacoes, tent_media, n_lotes = 0, 0.0, 0

            for ini in range(0, n_obs - self.batch_size + 1, self.batch_size):
                idx = ordem[ini : ini + self.batch_size]
                u, i = users[idx], items[idx]

                Uu, Vi = self.U[u], self.V[i]
                s_pos = np.einsum("ij,ij->i", Uu, Vi)
                if self.use_item_bias:
                    s_pos = s_pos + self.b[i]

                # Reamostra em rodadas; só quem ainda não violou sorteia de novo.
                aberto = np.ones(len(u), dtype=bool)
                j_final = np.zeros(len(u), dtype=np.int64)
                tentativas = np.zeros(len(u), dtype=np.int64)

                for t in range(1, self.max_tentativas + 1):
                    if not aberto.any():
                        break
                    ativos = np.flatnonzero(aberto)
                    j = rng.integers(0, n_items, size=len(ativos))
                    s_neg = np.einsum("ij,ij->i", self.U[u[ativos]], self.V[j])
                    if self.use_item_bias:
                        s_neg = s_neg + self.b[j]

                    viola = s_neg + self.margem > s_pos[ativos]
                    venceu = ativos[viola]
                    j_final[venceu] = j[viola]
                    tentativas[venceu] = t
                    aberto[venceu] = False

                viol = tentativas > 0
                if not viol.any():
                    continue

                n_violacoes += int(viol.sum())
                tent_media += float(tentativas[viol].mean())
                n_lotes += 1

                uv, iv, jv = u[viol], i[viol], j_final[viol]
                # rank estimado = itens restantes / tentativas até violar
                rank = np.maximum(1, (n_items - 1) // np.maximum(1, tentativas[viol]))
                peso = phi[np.minimum(rank, n_items)]

                Uuv, Viv, Vjv = self.U[uv], self.V[iv], self.V[jv]
                pc = peso[:, None]

                gU = pc * (Viv - Vjv) - reg * Uuv
                gVi = pc * Uuv - reg * Viv
                gVj = -pc * Uuv - reg * Vjv

                np.add.at(self.U, uv, lr * gU)
                np.add.at(self.V, iv, lr * gVi)
                np.add.at(self.V, jv, lr * gVj)
                if self.use_item_bias:
                    np.add.at(self.b, iv, lr * (peso - reg * self.b[iv]))
                    np.add.at(self.b, jv, lr * (-peso - reg * self.b[jv]))

                # Projeta de volta na bola de raio max_norm as linhas tocadas.
                for M, linhas in ((self.U, uv), (self.V, np.concatenate([iv, jv]))):
                    unicas = np.unique(linhas)
                    normas = np.linalg.norm(M[unicas], axis=1)
                    excede = normas > self.max_norm
                    if excede.any():
                        alvo = unicas[excede]
                        M[alvo] *= (self.max_norm / normas[excede])[:, None]

            if verbose and (epoca % 5 == 0 or epoca == self.n_epochs - 1):
                taxa = n_violacoes / max(1, n_lotes * self.batch_size)
                media = tent_media / max(1, n_lotes)
                print(
                    f"  época {epoca + 1:>3}/{self.n_epochs}  violações {100 * taxa:5.1f}%"
                    f"  tentativas médias {media:5.1f}"
                )

        return self

    def scores(self, user: int) -> np.ndarray:
        s = self.V @ self.U[user]
        return s + self.b if self.use_item_bias else s
