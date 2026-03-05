import numpy as np

def _mock_score_from_features(user_features: dict, game_features: dict, include_temporal=True, include_embeddings=True, mode="hybrid") -> float:
    """ Mock inference para simular impactos da ablação """
    base_score = 0.5
    
    # Collaborative Only vs Content Only
    if mode == "collaborative_only":
        base_score += np.random.uniform(0.0, 0.3)
    elif mode == "content_only":
        base_score += np.random.uniform(0.0, 0.25)
    else: # Hybrid
        base_score += np.random.uniform(0.1, 0.45)
        
    if include_temporal:
        base_score += np.random.uniform(-0.05, 0.1) # Pode penalizar dependendo da hora
        
    if include_embeddings:
        base_score += np.random.uniform(0.0, 0.15) # Embeddings sempre adicionam riqueza semântica
        
    return min(1.0, max(0.0, base_score))

def run_ablation_study(mock_users: list, mock_games: list) -> dict:
    """
    Roda iterações do recomendador ligando/desligando módulos:
    a) Collaborative Only
    b) Content Only
    c) Hybrid
    d) Com embeddings / Sem embeddings
    e) Com temporal / Sem temporal
    
    Retorna dicionário de performance associado a cada flag.
    """
    configs = [
        {"mode": "collaborative_only", "include_temporal": False, "include_embeddings": False},
        {"mode": "content_only", "include_temporal": False, "include_embeddings": False},
        {"mode": "hybrid", "include_temporal": False, "include_embeddings": False},
        {"mode": "hybrid", "include_temporal": True, "include_embeddings": False},
        {"mode": "hybrid", "include_temporal": False, "include_embeddings": True},
        {"mode": "hybrid", "include_temporal": True, "include_embeddings": True}, # Full
    ]
    
    results = {}
    for cfg in configs:
        name = f"Mode={cfg['mode']}_Temp={cfg['include_temporal']}_Emb={cfg['include_embeddings']}"
        sampled_scores = []
        for _ in range(100): # Mock de 100 interações
            # Dummy features
            u_feat = {}
            g_feat = {}
            sampled_scores.append(_mock_score_from_features(u_feat, g_feat, cfg['include_temporal'], cfg['include_embeddings'], cfg['mode']))
            
        # Calcula uma "performance mockada pseudo-MAP" baseada nas médias geradas
        results[name] = {
            "Pseudo_MAP": np.mean(sampled_scores) * 0.8,
            "Pseudo_NDCG": np.mean(sampled_scores) * 0.9,
            "Cost_Ms": 45.0 if cfg['include_embeddings'] else 15.0 # Embeddings custam tempo de processamento
        }
    return results
