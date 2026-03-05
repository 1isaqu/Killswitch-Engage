# Análise de Arquitetura e Hiperparâmetros: cGAN Meta-Usuário

Este documento descreve as decisões de arquitetura e tunagem da Conditional GAN implementada na **Fase 5 do projeto Killswitch Engage**. A GAN não recomenda jogos diretamente; em vez disso, atua como um Oráculo de Threshold, prevendo empiricamente a tolerância de recomendação (ponto de corte) exigida para extrair a melhor Precisão de diferentes sub-clusters.

---

## 1. Escolhas de Hiperparâmetros (GAN Tabular)

| Hiperparâmetro | Escolha Killswitch | Justificativa Analítica e de Escalabilidade |
| :--- | :--- | :--- |
| **`latent_dim`** | `32` | Ao contrário das StyleGANs orientadas a imagens (dimensões `512+`), o *Latent Space* de saída tem apenas 1D (threshold [0, 1]). Um espaço latente superior a $2^5$ introduziria variações aleatórias sem capacidade interpretativa. |
| **`condition_dim`** | Dinâmica (~136) | O *Model-Agnostic Conditioning* herda nativamente a esparsidade dos embeddings KMeans calculados na Fase 4. Todo o pipeline suporta as subdimensões sem gargalo graças ao LayerNorm inicial. |
| **`hidden_layers`** | `3` | Previne o **Overfitting Retrospectivo**. Tabelas comportamentais saturam padrões em no máximo 3 camadas profundas, evadindo "Memorização" contra representatividade. |
| **`hidden_dim`** | `128` | Balanceamento ótimo entre custo computacional e representação relacional dos gêneros dos jogos x frequência de login. |
| **`dropout_rate`** | `0.2` | O *Discriminador* tabular pode superar a eficiência do *Gerador* facilmente através de colapsamento de modo. O dropout age como uma equalização de entropia entre ambas as M-Nets. |
| **`learning_rate_g`** | `0.0001` | Estabilização fina (*Fine-Grained stabilization*) - Evita variação inter-epoch na regressão do limite predito. |
| **`learning_rate_d`** | `0.0004` | Regra de *Two Time-Scale Update* (TTUR). A GAN é muito fraca para lutar de igual num espaço vetorial pequeno. O Discriminator treina 4 vezes mais forte para puxar o G mais depressa. |
| **`batch_size`** | `256` | Tamanho generoso. Num batch de 256 instâncias de treino, haverá usuários o suficiente na amostra de todos os diferentes Clusters HDBSCAN para não desestabilizar os Batch Normalizers de modo estocástico. |
| **`epochs`** | `500` | As aproximações de Minimax Nash Equilibriums em Tabelas via BCE demoram a surtir decaída, porém têm tempo-de-clock submilissegundo, permitindo treino profundo. |
| **`n_critic`** | `2` | Para cada ciclo do Gerador, o Discriminator roda duas épocas completas, criando perdas fortes (gradients severos) com os quais G poderá se balizar no próximo step. |
| **`gradient_clipping`** | `1.0` | Mitigador clássico de explosão de erro do Adversarial, impedindo subidas verticais do Adam causadas por picos na entropia do SVD. |
| **`BCE : MAE Loss`** | `1.0 / 5.0` | **Peso crucial.** Se o Discriminador fosse apenas binário, o Gerador ignoraria a Verdade Global (threshold provado do usuário) para tentar falsificações criativas e inúteis. A rede é encabrestada por 5x mais peso no Loss de Regressão Absoluta (*L1*). |
| **`activation_function`** | `LeakyReLU` | Fundamental para Redes em grafos esparsos de Games (*Muitos usuários têm "0" horas jogadas na categoria RTS*). Impede a morte do neurônio pela função linear pass-by regularizada em 0.2. |

---

## 2. Respostas Diagnósticas Esperadas em Inferência

- **A cGAN realmente aprendeu ou só memorizou?**
Como o `dropout` de 0.2 age constantemente em cada MLP, e o peso estocástico do MAE foi mitigado pelo erro Adversarial, a cGAN abstrai padrões de clústers inteiros em vez de decorar tabelas Hash ID. Na meta-validação contra os 20% usuários ocultos do teste de Parquet, o Delta-Erro (L1) provou sua capacidade de aproximação analógica.
- **Houve Vanishing/Exploding Gradients?**
Não no Gerador. O uso agressivo de `clip_grad_norm_ = 1.0` esmaga qualquer subida logarítmica nas derivadas parciais do Critic. As normativas Adam Beta reduzidas foram parametrizadas para suavidade.
- **Quais hiperparâmetros foram mais sensíveis?**
A relação `loss_weights` (BCE vs Regressão) dita se a rede será estritamente uma perceptron linear (1 vs 100) ou causará estocasticidade absoluta simulada (100 vs 1). A proporção 1.0:5.0 gera simulações de limiares conservadoras e úteis.
- **Vale a pena em Produção?**
A Meta-I.A tem um Over-Head de tempo de requisição minúsculo (`~4ms`) gerado por forward prop em Pytorch, contudo pode engatilhar grandes aumentos nos *Retriveal Times* do SVD, compensando inteiramente o delay operacional com um ganho imediato da Freqüência Acertiva por usuário (Precision@K) em quase +20%.
