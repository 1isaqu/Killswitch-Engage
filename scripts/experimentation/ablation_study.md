# Estudo de Ablação (Filtros, Temporal, Embeddings)

|                                              |   Pseudo_MAP |   Pseudo_NDCG |   Cost_Ms |
|:---------------------------------------------|-------------:|--------------:|----------:|
| Mode=collaborative_only_Temp=False_Emb=False |     0.535392 |      0.602316 |        15 |
| Mode=content_only_Temp=False_Emb=False       |     0.499242 |      0.561648 |        15 |
| Mode=hybrid_Temp=False_Emb=False             |     0.623845 |      0.701826 |        15 |
| Mode=hybrid_Temp=True_Emb=False              |     0.647847 |      0.728828 |        15 |
| Mode=hybrid_Temp=False_Emb=True              |     0.680744 |      0.765837 |        45 |
| Mode=hybrid_Temp=True_Emb=True               |     0.685957 |      0.771701 |        45 |