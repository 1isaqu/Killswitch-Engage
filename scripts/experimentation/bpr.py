"""BPR-MF — fatoração matricial com perda de ranqueamento par-a-par.

Motivação (medida em evaluate_ranker.py): o TruncatedSVD da camada 3 otimiza
reconstrução da matriz de interações (erro quadrático sobre células), não
ranqueamento. Na tarefa de descoberta ele perde para a baseline de popularidade.

BPR (Rendle et al., 2009) otimiza diretamente o que importa para recomendação:
para um usuário u, um item observado i deve receber score maior que um item não
observado j. Minimiza

    L = -sum log sigmoid(x_ui - x_uj) + lambda * ||theta||^2

com x_ui = U[u] . V[i] + b[i].

É a mesma família de perda que o LightFM implementa (BPR e WARP) — a biblioteca
que o README alegava usar e que o código nunca importou.
"""

from __future__ import annotations

import numpy as np


class BPRRanker:
    """Fatoração matricial treinada com perda BPR por SGD em minilotes.

    Args:
        n_factors: dimensão dos embeddings latentes.
        learning_rate: passo do SGD.
        n_epochs: passadas sobre o conjunto de interações.
        reg: coeficiente de regularização L2.
        n_negatives: negativos amostrados por positivo.
        use_item_bias: inclui viés por item. O viés absorve popularidade, então
            desligá-lo força o modelo a explicar preferência só pelos fatores.
        batch_size: triplas por atualização.
        seed: semente do gerador.
    """

    def __init__(
        self,
        n_factors: int = 64,
        learning_rate: float = 0.05,
        n_epochs: int = 30,
        reg: float = 0.01,
        n_negatives: int = 1,
        use_item_bias: bool = True,
        batch_size: int = 8192,
        seed: int = 42,
    ) -> None:
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.reg = reg
        self.n_negatives = n_negatives
        self.use_item_bias = use_item_bias
        self.batch_size = batch_size
        self.seed = seed
        self.U: np.ndarray | None = None
        self.V: np.ndarray | None = None
        self.b: np.ndarray | None = None

    def fit(self, users: np.ndarray, items: np.ndarray, n_users: int, n_items: int, verbose: bool = True) -> "BPRRanker":
        """Treina com os pares (usuario, item) observados no conjunto de treino."""
        rng = np.random.default_rng(self.seed)

        # Inicialização pequena e aleatória quebra simetria sem saturar sigmoid.
        self.U = rng.normal(0, 0.1, size=(n_users, self.n_factors))
        self.V = rng.normal(0, 0.1, size=(n_items, self.n_factors))
        self.b = np.zeros(n_items)

        n_obs = len(users)
        lr, reg = self.learning_rate, self.reg

        for epoch in range(self.n_epochs):
            ordem = rng.permutation(n_obs)
            perda_total, n_lotes = 0.0, 0

            for inicio in range(0, n_obs, self.batch_size):
                idx = ordem[inicio : inicio + self.batch_size]
                u = np.repeat(users[idx], self.n_negatives)
                i = np.repeat(items[idx], self.n_negatives)
                # Negativo por amostragem uniforme. Com densidade ~0.02% a chance
                # de sortear um positivo verdadeiro é desprezível, então não
                # filtramos — filtrar custaria uma busca por tripla.
                j = rng.integers(0, n_items, size=len(u))

                Uu, Vi, Vj = self.U[u], self.V[i], self.V[j]
                x = np.einsum("ij,ij->i", Uu, Vi - Vj)
                if self.use_item_bias:
                    x = x + self.b[i] - self.b[j]

                # s = sigmoid(-x) = gradiente da perda. Forma estável.
                s = np.where(x >= 0, np.exp(-x) / (1 + np.exp(-x)), 1 / (1 + np.exp(x)))
                perda_total += float(np.mean(np.logaddexp(0, -x)))
                n_lotes += 1

                sc = s[:, None]
                gU = sc * (Vi - Vj) - reg * Uu
                gVi = sc * Uu - reg * Vi
                gVj = -sc * Uu - reg * Vj

                # np.add.at acumula corretamente quando um índice repete no lote.
                np.add.at(self.U, u, lr * gU)
                np.add.at(self.V, i, lr * gVi)
                np.add.at(self.V, j, lr * gVj)
                if self.use_item_bias:
                    np.add.at(self.b, i, lr * (s - reg * self.b[i]))
                    np.add.at(self.b, j, lr * (-s - reg * self.b[j]))

            if verbose and (epoch % 5 == 0 or epoch == self.n_epochs - 1):
                print(f"  época {epoch + 1:>3}/{self.n_epochs}  perda BPR = {perda_total / n_lotes:.4f}")

        return self

    def scores(self, user: int) -> np.ndarray:
        """Score de todos os itens para um usuário."""
        s = self.V @ self.U[user]
        return s + self.b if self.use_item_bias else s
