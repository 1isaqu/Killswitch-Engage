# Relatório Executivo da Fase 5: Experimentação Killswitch Engage

## 1. Métricas da Pipeline Offline Consolidada
|   Precision@5 |   Recall@5 |      NDCG@5 |   Precision@10 |   Recall@10 |     NDCG@10 |   Precision@20 |   Recall@20 |     NDCG@20 |         MAP |        MRR |   Coverage |
|--------------:|-----------:|------------:|---------------:|------------:|------------:|---------------:|------------:|------------:|------------:|-----------:|-----------:|
|        0.0006 |     0.0006 | 0.000461717 |         0.0006 |      0.0012 | 0.000810333 |         0.0004 |      0.0016 | 0.000988797 | 0.000289286 | 0.00144643 |     0.7757 |

## 2. Telemetria Online TDD Estimada
|   Simulated_CTR |   Simulated_Session_Time_Mins |   Acceptance_Rate |
|----------------:|------------------------------:|------------------:|
|        0.142867 |                       270.281 |          0.526867 |

## Conclusões Gerais (Recomendações de Negócios)
O modelo `Hybrid` com temporal provou o maior MAP na ablação, apesar do overhead. O treinamento Bayesiano por 20 trials se encontrou o Sweet-Spot sem engargalar a instância na inicialização do PyTorch.