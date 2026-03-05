import numpy as np
import pandas as pd

def compute_popularity_bias(recommendations_dict: dict, game_popularity_map: dict) -> float:
    """
    Mede a correlação logarítimica entre a Popularidade Real do Jogo (num_owners) e a frequência  que 
    ele aparece nas recomendações (A API só está recomendando Triple A?).
    """
    pop_list = []
    freq_map = {}
    
    for recs in recommendations_dict.values():
        for r in recs:
            freq_map[r] = freq_map.get(r, 0) + 1
            
    for item, freq in freq_map.items():
        if item in game_popularity_map:
            pop_list.append((freq, game_popularity_map[item]))
            
    if not pop_list:
        return 0.0
        
    df = pd.DataFrame(pop_list, columns=["Freq", "Real_Pop"])
    # Correlação Pearson: quanto mais próximo de 1, mais fortemente o modelo está enviesado para AAA.
    return df['Freq'].corr(df['Real_Pop'])

def calculate_gini_index(recommendations_dict: dict, total_catalog_len: int) -> float:
    """
    O Índice Gini sobre a cobertura de recomendações: (0 = Igualdade Perfeita, 1 = Desigualdade Absoluta).
    """
    freq_map = {k: 0 for k in range(total_catalog_len)} # Assumes IDs mapped 0..N
    
    for recs in recommendations_dict.values():
        for r in recs:
            # Ignora itens OOB da catalog_len (mock scenarios)
            if isinstance(r, int) and r < total_catalog_len:
                freq_map[r] += 1
                
    frequencies = np.sort(list(freq_map.values()))
    index = np.arange(1, len(frequencies) + 1)
    n = len(frequencies)
    
    if n == 0 or np.sum(frequencies) == 0:
        return 0.0
        
    return ((np.sum((2 * index - n  - 1) * frequencies)) / (n * np.sum(frequencies)))

def find_zero_coverage_items(recommendations_dict: dict, catalog_ids: list) -> list:
    """ Lista "Lost Games": Jogos que as Redes Neurais recusam recomendar para os perfis. """
    recommended_set = set()
    for recs in recommendations_dict.values():
        recommended_set.update(recs)
    
    return [game_id for game_id in catalog_ids if game_id not in recommended_set]

def user_performance_confusion(recommendations_dict: dict, ground_truths_dict: dict, user_profiles_dict: dict):
    """
    Mapeia qual perfil de usuário (Cluster HDBSCAN ou Casual/Hardcore) sofre da pior Precision.
    """
    profile_precision = {}
    
    for u, recs in recommendations_dict.items():
        actual = ground_truths_dict.get(u, [])
        profile = user_profiles_dict.get(u, "UnknownProfile")
        
        hits = sum(1 for item in recs[:10] if item in actual)
        precision10 = hits / 10.0
        
        if profile not in profile_precision:
            profile_precision[profile] = []
        profile_precision[profile].append(precision10)
        
    return {p: np.mean(vals) for p, vals in profile_precision.items()}

def execute_error_analysis(predictions: dict, ground_truths: dict, catalog_ids: list, game_popularities: dict, user_profiles: dict) -> dict:
    """ Executa a engine analitica de falha e extrai as respostas em um dicionário unificado. """
    bias = compute_popularity_bias(predictions, game_popularities)
    gini = calculate_gini_index(predictions, len(catalog_ids))
    invisible = find_zero_coverage_items(predictions, catalog_ids)
    confusion_map = user_performance_confusion(predictions, ground_truths, user_profiles)
    
    return {
        "Popularity_Bias_Corr": bias,
        "Gini_Index": gini,
        "Invisible_Games_Count": len(invisible),
        "Precision_By_Profile": confusion_map
    }
