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

    async def get_recomendacoes(self, usuario_id: UUID, k: int = 10) -> List[Dict[str, Any]]:
        if not self.is_loaded:
            return []
            
        # 1. Obter todas as IDs de itens da base do ranker
        all_game_ids = list(self.ranker['game_map'].keys())
        
        # 2. Verificar se usuário existe no ranker (SVD)
        user_map = self.ranker['user_map']
        item_embeddings = self.ranker['item_embeddings']
        
        if str(usuario_id) in user_map:
            user_idx = user_map[str(usuario_id)]
            user_vec = self.ranker['user_embeddings'][user_idx]
            scores = np.dot(item_embeddings, user_vec)
            explicacao_base = "Baseado no seu perfil de jogo"
        else:
            # Usuário novo (Cold Start): Média global como baseline
            # Futuro: Usar metadados do usuário (pais, etc) para clusterização a quente
            scores = np.mean(self.ranker['user_embeddings'], axis=0) @ item_embeddings.T
            explicacao_base = "Jogo popular recomendado para novos usuários"

        # 3. Ranqueamento Inicial
        top_indices = np.argsort(scores)[::-1]
        
        recomendacoes = []
        for idx in top_indices:
            game_id = self.ranker['reverse_game_map'][idx]
            
            # 4. Camada 1: Filtro de Qualidade (Opcional se performance for crítica)
            # No mundo real, filtraríamos aqui ou pré-computaríamos games bons
            # if not self.passa_filtro_qualidade(game_id): continue
            
            recomendacoes.append({
                "id": str(game_id),
                "score": float(scores[idx]),
                "explicacao": explicacao_base
            })
            
            if len(recomendacoes) >= k:
                break
                
        return recomendacoes

    def passa_filtro_qualidade(self, game_id: str) -> bool:
        # Lógica simplificada: se temos o ID e o RF carregado, poderíamos rodar a predição
        # Por simplicidade na primeira integração, assumimos que os embeddings do SVD já capturam qualidade
        return True

recomendador = RecomendadorService()
