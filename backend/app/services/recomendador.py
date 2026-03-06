import joblib
import os
import pandas as pd
import numpy as np
from uuid import UUID
from typing import List, Dict, Any
from app.utils.logging import setup_logging

logger = setup_logging()

class RecomendadorService:
    def __init__(self):
        # Caminhos relativos à raiz do projeto
        self.base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        self.model_dir = os.path.join(self.base_path, "scripts", "modelos")
        
        self.classificador = None
        self.clusterer = None
        self.ranker = None
        
        self.is_loaded = self.load_models()

    def load_models(self) -> bool:
        try:
            c1_path = os.path.join(self.model_dir, "classificador_rf.pkl")

            # FIXED: priorizar novo nome kmeans_clusters.pkl mantendo fallback para hdbscan_model.pkl (compatibilidade)
            cluster_main = os.path.join(self.model_dir, "kmeans_clusters.pkl")
            cluster_legacy = os.path.join(self.model_dir, "hdbscan_model.pkl")
            c2_path = cluster_main if os.path.exists(cluster_main) else cluster_legacy

            c3_path = os.path.join(self.model_dir, "lightfm_model.pkl")

            if not all(os.path.exists(p) for p in [c1_path, c2_path, c3_path]):
                logger.warning("Um ou mais modelos de recomendação não foram encontrados.")
                return False

            self.classificador = joblib.load(c1_path)
            self.clusterer = joblib.load(c2_path)
            self.ranker = joblib.load(c3_path)

            logger.info("Sistema de Recomendação: Todas as camadas carregadas com sucesso.")
            return True
        except Exception as e:  # noqa: BLE001
            logger.error(f"Erro ao carregar modelos de recomendação: {e}")
            return False

    # Mapeamento dos 3 arquétipos de recomendação
    MODOS = {
        'conservador': {'threshold': 0.7, 'exploracao': 0.1},
        'equilibrado':  {'threshold': 0.5, 'exploracao': 0.2},
        'aventureiro':  {'threshold': 0.3, 'exploracao': 0.3},
    }
    MODO_PADRAO = 'equilibrado'

    async def get_recomendacoes(
        self,
        usuario_id: UUID,
        k: int = 10,
        modo: str = None,
    ) -> List[Dict[str, Any]]:
        if not self.is_loaded:
            return []

        # Resolve o modo e seus parâmetros
        modo = modo if modo in self.MODOS else self.MODO_PADRAO
        cfg = self.MODOS[modo]
        threshold = cfg['threshold']
        exploracao = cfg['exploracao']

        item_embeddings = self.ranker['item_embeddings']
        user_map = self.ranker['user_map']

        if str(usuario_id) in user_map:
            user_idx = user_map[str(usuario_id)]
            user_vec = self.ranker['user_embeddings'][user_idx]
            scores = np.dot(item_embeddings, user_vec)
            explicacao_base = "Baseado no seu perfil de jogo"
        else:
            # Cold Start: média global dos embeddings de usuário
            scores = np.mean(self.ranker['user_embeddings'], axis=0) @ item_embeddings.T
            explicacao_base = "Jogo popular recomendado para novos usuários"

        # Filtra pelo threshold do modo escolhido
        mascara = scores >= threshold
        indices_validos = np.where(mascara)[0]

        # Se o threshold filtrou tudo (ex.: cold start com scores baixos), relaxa
        if len(indices_validos) == 0:
            indices_validos = np.argsort(scores)[::-1][:k * 2]

        # Ordena os válidos por score decrescente
        indices_ordenados = indices_validos[np.argsort(scores[indices_validos])[::-1]]

        # Exploração: substitui uma fração dos slots por itens aleatórios dos válidos
        n_explorar = int(k * exploracao)
        n_top = k - n_explorar

        top_indices = indices_ordenados[:n_top]

        if n_explorar > 0 and len(indices_validos) > n_top:
            pool_exploracao = indices_validos[n_top:]  # itens válidos não selecionados
            n_amostrar = min(n_explorar, len(pool_exploracao))
            explorados = np.random.choice(pool_exploracao, size=n_amostrar, replace=False)
            final_indices = np.concatenate([top_indices, explorados])
        else:
            final_indices = top_indices

        recomendacoes = []
        for idx in final_indices:
            game_id = self.ranker['reverse_game_map'][idx]
            recomendacoes.append({
                "id": str(game_id),
                "score": float(scores[idx]),
                "modo": modo,
                "explicacao": explicacao_base,
            })

        return recomendacoes[:k]

    def passa_filtro_qualidade(self, game_id: str) -> bool:
        # Lógica simplificada: se temos o ID e o RF carregado, poderíamos rodar a predição
        # Por simplicidade na primeira integração, assumimos que os embeddings do SVD já capturam qualidade
        return True

recomendador = RecomendadorService()
