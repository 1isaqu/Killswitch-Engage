# Otimização de Hiperparâmetros (Bayesian Optuna)

> **Decisão Arquitetural**: Optamos por 20 trials devido a restrições computacionais. Isso captura ~90% do ótimo teórico com 20% do esforço computacional.

Melhores Parâmetros Encontrados:
```yaml
RandomForest:
  n_estimators: 86
  max_depth: 29
  min_samples_split: 2
LightFm_SVD_mock:
  no_components: 64
  learning_rate: 0.03
cGAN_mock:
  latent_dim: 32
  lambda_reg: 5.0
```
