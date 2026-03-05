import numpy as np

# Retenção mediana base realistica da nossa EDA (em minutos)
GENRE_BASE_SESSION_TIME = {
    "MMO": 372.0,
    "RPG": 304.0,
    "Strategy": 294.0,
    "Action": 210.0,
    "Simulation": 250.0,
    "Adventure": 180.0,
    "Casual": 90.0
}

def simulate_ctr(predictions: list) -> float:
    """
    Simula uma taxa de clique média baseada na probabilidade de scores em lote.
    Assunções da V2:
    - Score > 0.7: 30% chance
    - Score [0.5, 0.7]: 15% chance
    - Score < 0.5: 5% chance
    """
    total_clicks = 0
    total_impressions = len(predictions)
    
    if total_impressions == 0:
        return 0.0

    for score in predictions:
        if score > 0.7:
            prob = 0.30
        elif score >= 0.5:
            prob = 0.15
        else:
            prob = 0.05
            
        # Simulação de Bernoulli para clique individual
        if np.random.rand() <= prob:
            total_clicks += 1
            
    return total_clicks / total_impressions

def simulate_session_time(predictions: list, genres: list) -> float:
    """
    K.E. Fase 5 Online Metric: Calcula a retenção temporal induzida estimada.
    Caso a predição atinga threshold, usa baseline do gênero correspondente, senão 0 min.
    """
    total_minutes = 0.0
    accepted_recs = 0
    
    for score, genre in zip(predictions, genres):
        base_time = GENRE_BASE_SESSION_TIME.get(genre, 120.0) # Defualt 120 min se não for mapeado
        
        # Logica: Se score for aprovado por um usuário fantasma (>0.5), ocorre uma sessão realista
        if score >= 0.5:
            # Variação Aleatória de +/- 20% no tempo real da sessão
            drift = np.random.uniform(0.8, 1.2)
            total_minutes += (base_time * drift)
            accepted_recs += 1
            
    return total_minutes / accepted_recs if accepted_recs > 0 else 0.0
    
def calculate_acceptance_rate(predictions: list) -> float:
    """
    A Taxa de Aceitação (Conversion) crua baseada nos cortes preditivos
    """
    if not predictions:
        return 0.0
        
    accepted = sum(1 for p in predictions if p >= 0.5)
    return accepted / len(predictions)

def evaluate_simulated_online(predictions_dict: dict, genres_dict: dict) -> dict:
    all_scores = []
    all_genres = []
    
    for u, scores in predictions_dict.items():
        all_scores.extend(scores)
        # Assumindo que o vetor de generos correspondentes foi passado pelo Orchestrator na mesma ordem
        all_genres.extend(genres_dict.get(u, ["Casual"] * len(scores)))
        
    avg_ctr = simulate_ctr(all_scores)
    avg_session_time = simulate_session_time(all_scores, all_genres)
    avg_acceptance = calculate_acceptance_rate(all_scores)
    
    return {
        "Simulated_CTR": avg_ctr,
        "Simulated_Session_Time_Mins": avg_session_time,
        "Acceptance_Rate": avg_acceptance
    }
