# Camada 4 (cGAN) — experimento encerrado

Esta pasta **não faz parte do pipeline**. A camada 4 foi removida da arquitetura
depois de medida; ver README §3.5 para os números.

Resumo do porquê:

- O threshold que a rede ajustaria **não muda o top-k** da recomendação. Ele só
  restringe o pool de candidatos, e a lista sai sempre ordenada por score. Mesmo
  um preditor perfeito não melhoraria a saída.
- Nenhum tamanho de rede bate a baseline trivial de prever a moda do alvo
  (MAE 0.0658 com 41k parâmetros, 0.0721 com 849, contra 0.0653 da moda).
- O ganho que o projeto reportava vinha da baseline escolhida: a constante 0.5
  erra 0.1708 porque 72% da massa do alvo está em 0.3.

O código fica aqui como registro: a arquitetura da cGAN é correta e o treino
aconteceu de fato. O que não se sustenta é a camada existir no pipeline.

Para rodar: `pip install -e ".[meta-learning]"`.
Para avaliar: `python scripts/experimentation/evaluate_cgan.py`.
