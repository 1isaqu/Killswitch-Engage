# 🤖 O Meta-Usuário: Como a cGAN Personaliza Recomendações

## O que é uma GAN?

Uma **GAN** (*Generative Adversarial Network* — Rede Generativa Adversarial) é um tipo de inteligência artificial composta por **duas redes neurais que treinam em competição**:

| Componente | Papel | Analogia |
|---|---|---|
| **Gerador (G)** | Cria respostas candidatas | Um falsificador de dinheiro |
| **Discriminador (D)** | Avalia se a resposta é boa ou não | Um policial bancário |

O Gerador tenta enganar o Discriminador. O Discriminador tenta não ser enganado. Com o tempo, o Gerador fica tão bom que aprende a gerar respostas de alta qualidade — sem nunca ter visto a resposta certa explicitamente.

---

## Por que usamos uma cGAN aqui?

No Killswitch Engage, a cGAN (**Conditional** GAN) é usada como uma **Camada 4 de Meta-Aprendizado**. Ela não recomenda jogos diretamente. Ela responde uma pergunta mais sutil:

> **"Qual é o threshold de corte ideal para recomendar jogos a *este* usuário específico?"**

O sistema de recomendação usa um "threshold" (0.3 a 0.8) para decidir se um jogo é relevante o suficiente para aparecer na lista. Um threshold baixo = mais jogos, menos precisão. Um alto = menos jogos, mais precisão.

A cGAN aprende automaticamente **qual valor usar para cada perfil de jogador**.

---

## Como funciona na prática?

```
Perfil do Usuário (Cluster KMeans + Estatísticas)
         │
         ▼
 ┌───────────────────┐
 │  ConditionEncoder │  ← Comprime o perfil em um vetor denso
 └───────────┬───────┘
             │
             ├──── + Ruído Aleatório (Z)
             │
             ▼
    ┌─────────────┐
    │  Gerador    │  ← Propõe um threshold (ex: 0.57)
    └──────┬──────┘
           │
           ▼
    ┌──────────────────┐
    │  Discriminador   │  ← "Esse threshold parece ótimo para esse perfil?"
    └──────────────────┘
           │
    ┌──────┴──────┐
    │  Real/Fake? │  ← Compara com os thresholds validados históricos
    └─────────────┘
```

---

## Os 3 Arquétipos de Recomendação

A condição passada à cGAN inclui o modo escolhido pelo usuário:

| Modo | Threshold Tipico | Características |
|---|---|---|
| 🛡️ **Conservador** | ~0.70 | Só acerta, recomenda poucos jogos |
| ⚖️ **Equilibrado** | ~0.55 | Balanço entre precisão e descoberta |
| 🗺️ **Aventureiro** | ~0.35 | Explora muito, aceita mais riscos |

---

## Por que não simplesmente fixar o threshold?

Porque jogadores diferentes têm tolerâncias diferentes. Um jogador veterano de RPG tem um perfil de qualidade altíssimo e rejeita jogos medianos. Um jogador casual quer variedade e aceita recomendações mais especulativas.

A cGAN aprende esses padrões **a partir dos próprios comportamentos históricos de cada cluster**, gerando um threshold personalizado em **~4ms** por requisição.

---

## Resultados Comprovados

- **PR-AUC do sistema completo:** `0.9153`
- **Precision@10 com cGAN:** `0.70` (~7 de 10 recomendações certas)
- **NDCG@10:** `0.8006`
- **Overhead de latência da cGAN:** `~4ms`
- **Ganho projetado de Precision@K:** `+20%` vs threshold fixo

---

## Referências Internas

- Código-fonte: [`scripts/meta_learning/cgan_model.py`](../../scripts/meta_learning/cgan_model.py)
- Hiperparâmetros: [`scripts/meta_learning/hyperparameter_analysis.md`](../../scripts/meta_learning/hyperparameter_analysis.md)
- Treinamento: [`scripts/meta_learning/train_cgan.py`](../../scripts/meta_learning/train_cgan.py)
- Curvas de Treinamento: [`reports/figures/training_curves.png`](../figures/training_curves.png)

---

## 💡 Ideias Futuras (Próxima Iteração)

### cGAN Geradora de Meta-Usuários Sintéticos

A visão original para a Fase 6 era usar a cGAN de forma ainda mais ambiciosa: em vez de apenas calibrar thresholds, ela **geraria perfis sintéticos de usuários** inteiros a partir de ruído latente.

Esses usuários gerados seriam os verdadeiros **"meta-usuários"** — arquétipos aprendidos diretamente da distribuição do dataset real, representando padrões que talvez o KMeans sozinho não consiga capturar:

```
Ruído Aleatório Z  →  [Gerador cGAN]  →  Perfil Sintético de Usuário
                                          (gêneros, horas, frequência, cluster)
                             ↕
                    [Discriminador cGAN]  ←  Perfis Reais do Dataset
```

**Papel do Gerador:**
O Gerador teria como objetivo **generalizar ao máximo** os padrões do dataset real para criar perfis convincentes de usuários. Ele nunca vê os perfis reais diretamente — apenas recebe o sinal de erro do Discriminador e ajusta. O objetivo é produzir até um **máximo de N usuários sintéticos** plausíveis, sem memorizar nenhum perfil existente.

**Papel do Discriminador:**
O Discriminador **não aprova nem repassa** os usuários gerados como contas reais. Seu único papel é **corrigir os erros do Gerador** — detectar quando um perfil sintético é implausível (ex: usuário com 5000h em RPG mas zero conquistas) e punir o Gerador por isso. Quando o treinamento termina, o Discriminador é descartado e apenas o Gerador é usado em produção.

**Por que não foi implementado:**
O banco Supabase (free tier) atingiu o limite de armazenamento com os 10.000 usuários reais e 309k sessões. Adicionar usuários gerados sinteticamente exigiria uma instância maior ou um pipeline de exportação offline.

**Potencial impacto:**
- Enriquecer cold start para novos usuários com perfis sintéticos de alta qualidade
- Testar o recomendador contra padrões de usuário nunca vistos
- Descoberta de segmentos de usuário emergentes que o dataset atual não cobre
