# Estudo de Ablação (Filtros, Temporal, Embeddings)

|                                              |   Pseudo_MAP |   Pseudo_NDCG |   Cost_Ms |
|:---------------------------------------------|-------------:|--------------:|----------:|
| Mode=collaborative_only_Temp=False_Emb=False |     0.521713 |      0.586927 |        15 |
| Mode=content_only_Temp=False_Emb=False       |     0.489791 |      0.551015 |        15 |
| Mode=hybrid_Temp=False_Emb=False             |     0.618379 |      0.695677 |        15 |
| Mode=hybrid_Temp=True_Emb=False              |     0.635229 |      0.714633 |        15 |
| Mode=hybrid_Temp=False_Emb=True              |     0.668028 |      0.751532 |        45 |
| Mode=hybrid_Temp=True_Emb=True               |     0.716953 |      0.806573 |        45 |