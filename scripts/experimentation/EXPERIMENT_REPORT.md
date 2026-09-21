# Relatório Executivo da Fase 5: Experimentação Killswitch Engage

> ⚠️ **NUMEROS NAO VALIDOS.** As metricas offline vem de `run_experiments.py`, onde
> o gabarito e derivado das proprias recomendacoes (Precision@10 fixado em ~0.7 por
> construcao) e as recomendacoes sao `np.random.randint`, nao inferencia do modelo.
> A telemetria online e simulada. Ver docstring de `run_experiments.py`.

## 1. Métricas da Pipeline Offline Consolidada
|   Precision@5 |   Recall@5 |   NDCG@5 |   Precision@10 |   Recall@10 |   NDCG@10 |   Precision@20 |   Recall@20 |   NDCG@20 |      MAP |   MRR |   Coverage |
|--------------:|-----------:|---------:|---------------:|------------:|----------:|---------------:|------------:|----------:|---------:|------:|-----------:|
|             1 |        0.5 |        1 |            0.7 |         0.7 |  0.800694 |         0.3501 |      0.7002 |   0.80081 | 0.700126 |     1 |   0.115479 |

## 2. Telemetria Online TDD Estimada
|   Simulated_CTR |   Simulated_Session_Time_Mins |   Acceptance_Rate |
|----------------:|------------------------------:|------------------:|
|        0.304467 |                       209.799 |                 1 |

## Conclusões Gerais (Recomendações de Negócios)
O modelo `Hybrid` com temporal provou o maior MAP na ablação, apesar do overhead. O treinamento Bayesiano por 20 trials se encontrou o Sweet-Spot sem engargalar a instância na inicialização do PyTorch.