def calcular_percentil(valores: list, percentil: float):
    """Calcula o percentil de uma lista de valores."""
    if not valores:
        return 0
    sorted_valores = sorted(valores)
    index = (len(sorted_valores) - 1) * percentil
    lower = int(index)
    upper = lower + 1
    weight = index - lower
    if upper >= len(sorted_valores):
        return sorted_valores[lower]
    return sorted_valores[lower] * (1 - weight) + sorted_valores[upper] * weight

def calcular_resumo_estatistico(valores: list):
    """Retorna média, mediana e desvio padrão simples."""
    if not valores:
        return {"media": 0, "mediana": 0, "std": 0}
    media = sum(valores) / len(valores)
    sorted_v = sorted(valores)
    mediana = sorted_v[len(sorted_v)//2]
    variance = sum((x - media)**2 for x in valores) / len(valores)
    std = variance**0.5
    return {"media": media, "mediana": mediana, "std": std}
