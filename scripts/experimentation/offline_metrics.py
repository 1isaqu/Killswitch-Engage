import numpy as np

def precision_at_k(recommended: list, actual: list, k: int) -> float:
    recs_k = recommended[:k]
    hits = sum(1 for item in recs_k if item in actual)
    return hits / k if k > 0 else 0.0

def recall_at_k(recommended: list, actual: list, k: int) -> float:
    if not actual:
        return 0.0
    recs_k = recommended[:k]
    hits = sum(1 for item in recs_k if item in actual)
    return hits / len(actual)

def average_precision_at_k(recommended: list, actual: list, k: int) -> float:
    if not actual:
        return 0.0
    
    score = 0.0
    num_hits = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in actual:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
            
    return score / min(len(actual), k)

def mean_average_precision(recommendations_dict: dict, ground_truths_dict: dict, k: int = 10) -> float:
    users = list(recommendations_dict.keys())
    ap_sum = 0.0
    for u in users:
        actual = ground_truths_dict.get(u, [])
        predicted = recommendations_dict[u]
        ap_sum += average_precision_at_k(predicted, actual, k)
    return ap_sum / len(users) if users else 0.0

def ndcg_at_k(recommended: list, actual: list, k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in actual:
            dcg += 1.0 / np.log2(i + 2) # index+posicao_1_base+1 -> i+2
            
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual), k)))
    return dcg / idcg if idcg > 0 else 0.0

def reciprocal_rank(recommended: list, actual: list) -> float:
    for i, item in enumerate(recommended):
        if item in actual:
            return 1.0 / (i + 1)
    return 0.0

def mrr(recommendations_dict: dict, ground_truths_dict: dict) -> float:
    users = list(recommendations_dict.keys())
    rr_sum = sum(reciprocal_rank(recommendations_dict[u], ground_truths_dict.get(u, [])) for u in users)
    return rr_sum / len(users) if users else 0.0

def calculate_coverage(recommendations_dict: dict, total_catalog_len: int) -> float:
    unique_recommended = set()
    for recs in recommendations_dict.values():
        unique_recommended.update(recs)
    return len(unique_recommended) / total_catalog_len if total_catalog_len > 0 else 0.0

def evaluate_offline_metrics(recommendations_dict: dict, ground_truths_dict: dict, total_catalog_len: int, k_list=[5, 10, 20]) -> dict:
    metrics = {}
    users = list(recommendations_dict.keys())
    n_users = len(users)
    
    if n_users == 0:
        return metrics

    for k in k_list:
        p_at_k = np.mean([precision_at_k(recommendations_dict[u], ground_truths_dict.get(u, []), k) for u in users])
        r_at_k = np.mean([recall_at_k(recommendations_dict[u], ground_truths_dict.get(u, []), k) for u in users])
        ndcg_k = np.mean([ndcg_at_k(recommendations_dict[u], ground_truths_dict.get(u, []), k) for u in users])
        
        metrics[f"Precision@{k}"] = p_at_k
        metrics[f"Recall@{k}"] = r_at_k
        metrics[f"NDCG@{k}"] = ndcg_k
        
    metrics["MAP"] = mean_average_precision(recommendations_dict, ground_truths_dict, max(k_list))
    metrics["MRR"] = mrr(recommendations_dict, ground_truths_dict)
    metrics["Coverage"] = calculate_coverage(recommendations_dict, total_catalog_len)
    
    return metrics
