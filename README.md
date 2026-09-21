# 🎮 Killswitch Engage – Sistema Inteligente de Recomendação de Jogos

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Redis](https://img.shields.io/badge/Redis-7.0-DC382D?logo=redis&logoColor=white)](https://redis.io/)
[![Docker](https://img.shields.io/badge/Docker-24.0-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.5-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Optuna](https://img.shields.io/badge/Optuna-3.3-3C64B1)](https://optuna.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 1. Visão Geral

**Killswitch Engage** é um sistema completo de recomendação de jogos, construído do zero com ML em produção como objetivo central. O pipeline abrange desde a ingestão e limpeza de **122.507 jogos da Steam** até uma API em produção com latência < 15ms, passando por modelos de aprendizado de máquina treinados sobre **10.000 usuários sintéticos** com histórico realista de sessões.

### 🔍 O Problema

A Steam possui mais de 50.000 jogos no catálogo. Um usuário novo se perde. Um usuário experiente fica preso em uma "bolha" dos mesmos gêneros. O desafio: recomendar o jogo certo, para a pessoa certa, no momento certo — lidando com **cold start**, **viés de popularidade**, **dados esparsos** e **escalabilidade**.

### 💡 A Solução

Uma arquitetura de **4 camadas em cascata**:

```
[Entrada: Perfil do Usuário]
         │
         ▼
 ┌─────────────────┐
 │  Camada 1: RF   │  → Classificador (RandomForest) filtra jogos relevantes
 │  (Filtro)       │     com base em features de conteúdo (gênero, preço, tags)
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │  Camada 2:      │  → Clustering (KMeans, k escolhido por silhouette)
 │  Clustering     │     de usuário (casual, médio, hardcore)
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │  Camada 3:      │  → TruncatedSVD (filtragem colaborativa) ranqueia os
 │  SVD Ranker     │     candidatos a partir da matriz usuário × jogo
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │  Camada 4: cGAN │  → Meta-aprendizado por modo (conservador/equilibrado/
 │  (Meta-Learner) │     aventureiro) com threshold e exploração calibrados
 └─────────────────┘
          │
          ▼
  [API FastAPI + Cache Redis]
```

> ### ⚠️ Estado real da implementação (leia antes das métricas)
>
> O diagrama acima descreve o **desenho pretendido**. O que hoje roda em inferência
> (`backend/app/services/recomendador.py`) é **apenas a camada 3**: produto escalar
> entre o embedding do usuário e os embeddings de item, filtrado por um threshold fixo.
>
> | Camada | Treinada? | Usada em inferência? |
> |---|---|---|
> | 1 — RandomForest | ✅ sim | ❌ não — `passa_filtro_qualidade()` retorna `True` incondicionalmente |
> | 2 — KMeans | ✅ sim | ❌ não — carregada, nunca consultada |
> | 3 — TruncatedSVD | ✅ sim | ✅ **sim — é o que gera a recomendação** |
> | 4 — cGAN | ✅ sim | ❌ não — os thresholds vêm de um dicionário fixo, não da rede |
>
> Ou seja: as quatro camadas existem como artefatos treinados, mas **a cascata não está
> conectada**. Integrá-las é o próximo passo do projeto, não um recurso entregue.

---

## ✨ 2. Funcionalidades

- ✅ **Recomendações personalizadas** baseadas em perfil completo de usuário
- ✅ **3 modos de recomendação**: Conservador (precisão), Equilibrado (padrão), Aventureiro (exploração)
- ✅ **Cold start** para novos usuários — fallback pela média global dos embeddings de usuário
- ✅ **API rápida** com latência < 15ms e cache Redis nas rotas analíticas (TTL 1h)
- ✅ **Pipeline completo de dados** com imputação inteligente validada (KS-test p = 1.0)
- ⚠️ **MLOps**: MLflow instrumentado e funcionando; a busca com Optuna existe mas rodou sobre
  dados aleatórios (`optimization.py`), então não produziu hiperparâmetros aproveitáveis
- ✅ **Análise de cobertura com Lei de Potência** (R² = 0.9474) — medida de fato pelo
  `scripts/analysis/coverage_regression.py`, que executa o recomendador real
- ✅ **Segurança**: credenciais por variáveis de ambiente, SSL verificado, sem secrets hard-coded

---

## 📊 3. Métricas e Resultados

### 3.1 Performance dos Modelos — retirada

**Esta seção continha uma tabela de Precision/Recall/NDCG/MAP/MRR que não é válida.
Ela foi removida em vez de corrigida, porque os números não medem o modelo.**

O script que os gerou (`scripts/experimentation/run_experiments.py`) tem dois defeitos:

1. **Gabarito vazado** — o ground truth era construído a partir das próprias
   recomendações (`subset_hits = recommended_ids[:7]`), o que fixa Precision@10
   em ~0.7 por construção.
2. **Recomendações não vinham do modelo** — eram `np.random.randint`; os `.pkl`
   eram carregados mas nunca usados para inferir.

O banco `scripts/experimentation/mlflow_experiments.db` preserva as duas execuções
de 05/03/2026 e mostra o efeito:

| Run MLflow | Horário | Precision@10 | MRR |
|---|---|---|---|
| `63dc073a` (sem o vazamento) | 21:27 | **0.0006** | 0.0014 |
| `d17da97c` (com o vazamento) | 22:12 | **0.6999** | 1.000 |

A tabela publicada anteriormente era o segundo run.

### Medição honesta (`scripts/experimentation/evaluate_ranker.py`)

Split temporal em três partes, por usuário e por tempo:

```
|<------- treino 70% ------->|<- validação 15% ->|<- teste 15% ->|
                                                        tempo ->
```

Toda escolha de modelo e hiperparâmetro (capacidade, alpha, viés) é feita na
**validação**. O **teste** é tocado uma vez, no fim, só para reportar.

> **Correção de método.** As primeiras versões desta avaliação usavam só
> treino/teste, e cerca de 12 configurações diferentes foram comparadas no mesmo
> conjunto de teste. Isso é overfitting ao teste: cada decisão tomada olhando
> aquele número vaza informação, e "o melhor no teste" passa a medir quanto se
> garimpou, não desempenho fora da amostra. O split de três partes corrige isso.

Todas as métricas vêm com **intervalo de confiança de 95% por bootstrap** sobre
usuários, e as comparações principais usam **bootstrap pareado** — se o intervalo
da diferença cruza zero, a comparação não sustenta conclusão.

> Os dados originais não existem mais (o projeto Supabase foi pausado por
> inatividade, e o gerador nunca teve seed — então nunca foram reproduzíveis).
> O dataset agora é gerado localmente e de forma determinística por
> `src/data_preparation/generate_synthetic_data.py`.

#### O gerador antigo não continha sinal colaborativo

Rodando a avaliação sobre dados no formato original (favoritos sorteados
uniformemente do catálogo, independentes por usuário — `populate_supabase_v2.py:119`):

| Modelo | P@10 | NDCG@10 |
|---|---|---|
| SVD (camada 3) | 0.0149 | 0.0185 |
| **SVD sem itens já vistos** | **0.0000** | **0.0000** |
| Baseline: popularidade | 0.0004 | 0.0007 |

Removendo os jogos que o usuário já jogou, a precisão ia a **zero exato**: todo o
acerto era re-recomendar o próprio histórico. O diagnóstico explicava — 99.9% dos
pares de usuários não compartilhavam nenhum jogo, e o top-1% dos títulos
concentrava só 4.7% das interações. Filtragem colaborativa é achar usuários
parecidos; não havia nenhum para achar.

#### Com o gerador corrigido

O novo gerador dá gêneros aos jogos, afinidade por gênero aos usuários (Dirichlet
esparsa) e popularidade em lei de potência. Aí sim existe estrutura a descobrir:

| Medida | Gerador antigo | Gerador novo |
|---|---|---|
| Jogos em comum entre 2 usuários | 0.001 | **0.227** |
| Pares com alguma sobreposição | 0.1% | **17.6%** |
| Top-1% dos jogos / interações | 4.7% | **52.9%** (Steam real: 60–80%) |

| Modelo | P@10 | R@10 | NDCG@10 |
|---|---|---|---|
| **SVD (camada 3)** | **0.1371** | 0.3770 | 0.4009 |
| Baseline: popularidade | 0.0655 | 0.2031 | 0.1500 |
| SVD sem itens já vistos | 0.0069 | 0.0193 | 0.0155 |
| **Popularidade sem itens vistos** | **0.0116** | 0.0439 | 0.0305 |
| Baseline: aleatório | 0.0001 | 0.0000 | 0.0001 |

**Leitura honesta das duas metades:**

- Na tarefa completa, o SVD bate popularidade por **2.1×** — a camada colaborativa
  agrega valor real sobre "recomende o mais jogado".
- Na tarefa de **descoberta** (só itens que o usuário nunca tocou), **a popularidade
  vence o SVD** (0.0116 contra 0.0069). O ganho da camada 3 vem majoritariamente de
  reordenar o que a pessoa já conhece, não de revelar coisa nova.

Ou seja: o modelo aprende, mas ainda **não justifica sua complexidade para
descoberta**. Esse é o problema em aberto do projeto, e agora ele tem um número.

> Pendente relacionado: com a escala corrigida, os thresholds fixos (0.3/0.5/0.7)
> ficam mal calibrados — o modo conservador deixa passar só ~2 itens e cai no
> fallback de relaxamento. Thresholds por **percentil** do score do usuário
> resolveriam, em vez de constantes absolutas.

#### Overfitting: capacidade do modelo vs. generalização

O ranker tem 122.507 itens × 50 fatores ≈ **6,6 milhões de parâmetros contra
246 mil interações de treino — 27× mais parâmetros que dados**. Além disso, 41%
dos jogos aparecem em exatamente uma interação, então o vetor latente desses
itens é ajustado a uma única observação.

Varredura de capacidade, medida na validação:

| `n_factors` | P@10 tarefa completa | P@10 descoberta |
|---|---|---|
| 8 | 0.0691 | 0.0072 |
| **16** | 0.0879 | **0.0075** ← melhor para descoberta |
| 32 | 0.0998 | 0.0065 |
| 50 *(config do projeto)* | 0.1081 | 0.0048 |
| 100 | **0.1217** ← melhor para a tarefa completa | 0.0017 |

**As duas tarefas apontam em direções opostas.** Mais capacidade melhora
monotonicamente a tarefa completa e destrói a descoberta — de 0.0075 em k=16 para
0.0017 em k=100, uma queda de 77%. É a assinatura clássica de overfitting: a
capacidade extra é gasta decorando pares usuário-item do treino, o que ajuda a
reordenar o que já se conhece e atrapalha a generalizar para item novo.

O `n_components=50` que o projeto usa **já está muito além do ótimo para
descoberta**. Só reduzir para 16 melhora a descoberta em 56% (0.0048 → 0.0075).

> Isso qualifica a conclusão anterior: parte da derrota do SVD para a popularidade
> na descoberta era overfitting, não limitação do método. Mesmo em k=16 a
> popularidade ainda ganha (0.0075 contra 0.0091 na validação), mas a distância é
> bem menor do que a medida em k=50.

#### Capacidade de cada camada vs. estatística dos dados

A varredura acima levanta a pergunta para as outras camadas: a capacidade está
calibrada pelo que os dados sustentam, ou por hábito?

| Camada | Capacidade | Dados disponíveis | Veredito |
|---|---|---|---|
| 1 — RandomForest | 25 features, `max_depth=10`, 100 árvores → ~120 amostras por folha se saturada | 122.507 jogos, classes 31/69 | **Adequada.** Única camada bem dimensionada. |
| 2 — KMeans + PCA | `PCA(n_components=0.95)` retém **21 de 26** colunas | 10.000 usuários; usuário médio toca **4,4 de 22 gêneros** | **Exigente demais.** |
| 3 — SVD | `n_components=50` → 6,6M parâmetros | 246k interações (**27×**) | **Exigente demais.** Ótimo em k=16. |
| 4 — cGAN | ~57k parâmetros no generator | ~7.000 linhas (**8×**) | **Muito exigente.** |

**Camada 2.** `PCA(n_components=0.95)` praticamente não reduz nada: guarda 21 de 26
dimensões. Como a matriz de gênero é esparsa e composicional (cada usuário toca 4,4
de 22 gêneros, o resto é zero), padronizar e reter 95% da variância preserva
direções que são ruído. Pior, o `silhouette_score` é calculado nesse espaço de 21
dimensões, onde distâncias se concentram e a métrica perde poder discriminativo —
o que torna qualquer silhouette alto reportado nessa configuração pouco confiável.
Fixar poucos componentes (5–8) ou agrupar direto no perfil de gênero seria mais
honesto com a estatística dos dados.

**Camada 4.** Medindo o alvo real (`best_threshold`, reproduzindo
`compute_best_thresholds` sobre o ranker k=16):

| `best_threshold` | massa |
|---|---|
| **0.3** | **72,1%** |
| 0.4 | 9,8% |
| 0.5 | 7,7% |
| 0.6 | 3,4% |
| 0.7 | 4,4% |
| 0.8 | 2,6% |

Entropia do alvo: **1,45 bits** (máximo possível com 6 valores: 2,59). São **57 mil
parâmetros para aprender 1,45 bits.** Com essa folga e a perda L1 pesando 5×, o
mínimo mais fácil de alcançar é emitir a moda constante — que é exatamente o mode
collapse observado em `reports/figures/18_cgan_threshold_dist.png`.

E a baseline fica em perspectiva:

| Estratégia | MAE |
|---|---|
| Sempre prever a moda (0.3) | **0,0659** |
| Sempre prever 0.5 — *a "baseline estática" do projeto* | 0,1741 |

A baseline escolhida é **2,6× pior que o palpite trivial**. Comparar a cGAN contra
ela infla o ganho; contra a moda, o espaço de melhora é muito menor.

#### Tentativa de corrigir a descoberta: despopularização (não funcionou)

A hipótese padrão para "o ranker perde para popularidade na descoberta" é viés de
popularidade: o modelo empurraria itens populares demais. O ajuste clássico é
penalizar o score pela popularidade do item. Testado com

```
score' = minmax(score) - alpha * minmax(log1p(popularidade))
```

Rodado sobre o modelo k=16 escolhido na validação, não sobre o k=50 sobreajustado:

| alpha | P@10 | NDCG@10 | vs. baseline popularidade |
|---|---|---|---|
| **0.0** | **0.0075** | 0.0216 | 0.82× |
| 0.1 | 0.0053 | 0.0178 | 0.59× |
| 0.3 | 0.0022 | 0.0114 | 0.24× |
| 0.5 | 0.0013 | 0.0082 | 0.14× |
| 1.0 | 0.0001 | 0.0010 | 0.01× |

**Piora monotonicamente. O melhor alpha é zero** — ou seja, nenhuma penalização.

A hipótese estava errada: o SVD não está falhando por viés de popularidade. Neste
dataset a popularidade é **genuinamente preditiva** (os favoritos são sorteados
∝ afinidade_de_gênero × popularidade), então penalizá-la joga fora sinal real. É
por isso que a baseline de popularidade é difícil de bater na descoberta.

O problema real é outro: o `TruncatedSVD` otimiza **reconstrução** da matriz de
interações (erro quadrático), não **ranqueamento**. Para feedback implícito, o que
se usa são perdas de ranking par-a-par — BPR ou WARP.

> 🎯 Ironia útil: **LightFM**, a biblioteca que o README alegava usar e que o código
> nunca importou, implementa exatamente WARP, e é a ferramenta indicada para este
> problema.

#### Segunda tentativa: BPR em vez de SVD (também não funcionou)

`scripts/experimentation/bpr.py` implementa BPR-MF (Rendle et al., 2009), que
otimiza ranqueamento par-a-par em vez de reconstrução. Hiperparâmetros: 50
fatores (mesma capacidade do SVD, para isolar o efeito da perda), lr 0.05,
30 épocas, L2 0.01, 1 negativo por positivo.

Medido na validação, em k=50 e k=16:

| Modelo | P@10 completo | P@10 descoberta |
|---|---|---|
| SVD k=50 | 0.1081 | 0.0048 |
| **SVD k=16** | 0.0879 | **0.0075** |
| BPR k=50, sem viés | 0.0710 | 0.0060 |
| BPR k=16, sem viés | 0.0400 | 0.0045 |
| BPR k=16, com viés | 0.0246 | 0.0021 |
| Baseline: popularidade | — | **0.0091** |

BPR **piora** ao reduzir a capacidade, direção oposta à do SVD, e a perda ainda
cai na época 30 (0.0394). BPR está **subajustado**, não sobreajustado: precisa de
mais épocas ou negativos mais difíceis, não de menos fatores.

BPR perdeu para o SVD nas duas tarefas. A causa aparece na curva de perda, que cai
até **0.0397** — perda BPR perto de zero significa que o modelo separa positivo de
negativo aleatório quase sempre. Com densidade de 0.02%, um negativo sorteado
uniformemente é trivialmente fácil. O modelo aprendeu "jogo que eu joguei" contra
"jogo qualquer", que é uma tarefa fácil e inútil para ordenar candidatos plausíveis.

É a fraqueza conhecida do BPR com amostragem uniforme, e é exatamente o que **WARP**
corrige: reamostra até encontrar um negativo que viola o ranking, treinando em
negativos difíceis. A recomendação de usar LightFM/WARP continua de pé — agora com
evidência de *por quê*, não por analogia.

#### Onde isso deixa o projeto

Três tentativas, nenhuma bate a popularidade na descoberta: SVD (0.0069),
despopularização (piora), BPR (0.0060) contra popularidade (0.0116).

Isso aponta para o que **não** foi tentado: sinal de **conteúdo**. As camadas 1
(RandomForest sobre features de jogo) e 2 (KMeans sobre perfil de gênero) existem
treinadas e nunca foram ligadas à inferência. Para um item que ninguém no cluster
do usuário jogou, conteúdo é a única fonte de sinal — nenhum método colaborativo
tem o que usar.

> ⚠️ Ressalva honesta: o gerador sintético sorteia favoritos ∝ afinidade_de_gênero ×
> popularidade. Ou seja, **o próprio gerador determina que features de conteúdo
> funcionariam bem aqui** — testar isso neste dataset seria medir a premissa do
> gerador, não a qualidade do método. A conclusão só vale de verdade sobre dados
> reais de interação.

---

## 💼 4. Impacto de Negócio — não medido

Esta seção apresentava ROI (320–580%), payback, MAU +27%, churn −18%, LTV +31% e
proxy metrics de CTR / tempo de sessão.

**Nenhum desses números foi derivado de dado algum** — não há usuário real, não há
teste A/B, não há receita. Eram premissas escritas à mão com aparência de medição,
e por isso foram removidas em vez de reetiquetadas.

O que o projeto pode honestamente dizer sobre impacto: **nada ainda**. Um sistema de
recomendação só produz números de negócio depois de ir a produção com usuários reais
e um experimento controlado. Este ainda roda sobre 10.000 usuários sintéticos.

---

## 🛠️ 5. Tecnologias Utilizadas

### ML & Data Science

| Tecnologia | Uso no Projeto |
|------------|---------------|
| **TruncatedSVD** (scikit-learn) | Ranking colaborativo — camada 3 |
| **Scikit-learn** | RandomForest (camada 1), KMeans (camada 2), TruncatedSVD (camada 3) |
| **PyTorch** | cGAN meta-learner (camada 4) |
| **Optuna** | Otimização Bayesiana (20 trials por modelo) |
| **MLflow** | Versionamento de experimentos e artefatos |
| **SciPy** | Testes estatísticos (KS-test, regressão log-log) |

### Backend & Infra

| Tecnologia | Uso no Projeto |
|------------|---------------|
| **FastAPI** | API assíncrona de alta performance |
| **asyncpg** | Driver PostgreSQL async (até 3× mais rápido que síncrono) |
| **Redis** | Cache de rotas analíticas (TTL 1h) |
| **PostgreSQL 15** | Banco principal com índices otimizados |
| **Docker + Compose** | Orquestração completa dos serviços |

### Badges

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-F7931E?logo=scikit-learn&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-4169E1?logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-7.0-DC382D?logo=redis&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-24.0-2496ED?logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.5-0194E2?logo=mlflow&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-3.3-3C64B1)

---

## 🚀 6. Como Executar

### Pré-requisitos

- Python 3.11+
- Docker e Docker Compose
- PostgreSQL (ou usar o container via Docker)
- Redis (ou usar o container via Docker)

### Passo a Passo

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/killswitch-engage.git
cd killswitch-engage

# 2. Configure ambiente virtual
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# 3. Instale dependências
pip install -e .            # usa pyproject.toml
# ou, só para a API:  pip install -r backend/requirements.txt

# 4. Configure variáveis de ambiente
cp .env.example .env
# Edite .env com suas configurações de banco, Redis e secrets

# 5. Suba os serviços com Docker
docker-compose up -d

# 6. Popule o banco (opcional — gera 10.000 usuários SINTÉTICOS e suas sessões)
python -m src.data_preparation.populate_supabase
python -m src.data_preparation.populate_supabase_v2

# 7. Execute a API
uvicorn backend.app.main:app --reload

# 8. Acesse
#   API:      http://localhost:8000
#   Docs:     http://localhost:8000/docs
#   MLflow:   http://localhost:5000
```

### Testando a API

```bash
# Recomendação para usuário (modo padrão: equilibrado)
curl "http://localhost:8000/recomendacoes/1?k=10"

# Especificando modo de recomendação
curl "http://localhost:8000/recomendacoes/1?modo=aventureiro&k=10"
curl "http://localhost:8000/recomendacoes/1?modo=conservador&k=10"

# Verificar saúde da API
curl "http://localhost:8000/health"
```

---

## 📁 7. Estrutura do Projeto

```
killswitch-engage/
├── backend/                    # API FastAPI (rotas, config, middlewares)
│   └── app/
│       ├── main.py             # Entry point da aplicação
│       ├── config.py           # Configurações (SSL, DB, Redis)
│       └── routes/             # Endpoints (recomendações, analíticos)
├── src/                        # Código fonte dos modelos e serviços
│   ├── models/                 # Treinadores ML — AUSENTE deste repo (ver nota abaixo)
│   ├── backend/                # RecomendadorService (orquestração das camadas)
│   ├── data_preparation/       # Ingestão, imputação, geração de usuários sintéticos
│   ├── validation/             # Scripts de validação e sanidade
│   └── eda/                    # Análise exploratória
├── scripts/                    # Scripts utilitários e pipelines
│   ├── analysis/               # coverage_regression.py, ablation, etc.
│   ├── training/               # Treino de cada camada (layer1, layer2, layer3)
│   └── experimentation/        # mlflow.db e experimentos versionados
├── data/                       # Dados brutos e processados (ignorado no git)
├── models/                     # Artefatos treinados (.pkl, .pt) (ignorado no git)
├── reports/
│   ├── figures/                # Gráficos gerados (PNG)
│   └── insights/               # Relatórios técnicos (Markdown, CSV)
├── docs/                       # Documentação interna e revisão técnica
├── .env.example                # Template de variáveis de ambiente
├── docker-compose.yml          # Orquestração dos serviços
├── scripts/sql/indices.sql     # Índices SQL recomendados
├── pyproject.toml              # Dependências e configuração de ferramentas
└── README.md                   # Este arquivo
```

> ⚠️ **`src/models/` não está neste repositório.** A regra `models/` do `.gitignore`
> (pensada para artefatos `.pkl`) casava também com `src/models/` e impediu que o
> código dos treinadores fosse versionado. A regra foi corrigida para `/models/`,
> mas os arquivos precisam ser adicionados de volta pelo autor. Enquanto isso,
> `tests/test_models/test_rf_trainer.py` não coleta — ele importa
> `src.models.classifier.rf_trainer`, que não existe aqui.

---

## 📈 8. Experimentação e MLOps

O projeto adota uma abordagem rigorosa de experimentação:

| Aspecto | Estado | Detalhe |
|---------|--------|---------|
| **Versionamento** | ✅ real | MLflow instrumentado; `mlflow_experiments.db` versionado no repo |
| **PR-AUC** | ✅ real | Calculado em `train_layer1_classifier.py:105`; a troca de ROC-AUC por PR-AUC pelo desbalanceamento 73/27 é decisão correta e está no código |
| **Silhouette** | ✅ real | Busca de k por silhouette em amostra (`train_layer2_clustering.py:92-111`). O valor 0.8654 citado antes não é reproduzível a partir deste repo |
| **KS-test** | ✅ real | `src/validation/validate_kstest.py` |
| **Otimização (Optuna)** | ❌ inválido | `optimization.py` roda sobre `np.random.rand(2000, 10)` — ruído, não os dados do projeto |
| **Ablação** | ❌ inválido | `ablation.py` não desliga camada nenhuma; os scores são `np.random.uniform` |
| **Gini = 0.016** | ❌ inválido | Derivado das recomendações aleatórias de `run_experiments.py` |

```bash
# Visualizar todos os experimentos no MLflow UI
mlflow ui --backend-store-uri sqlite:///scripts/experimentation/mlflow.db
```

### Decisões Técnicas Notáveis

| Decisão | Alternativa Rejeitada | Motivo |
|---------|----------------------|--------|
| **asyncpg Pool** | SQLAlchemy Sync | Até 3× mais rápido; suporta 1000+ RPS em hardware modesto |
| **Batch Insert (5.000)** | Inserts unitários | Reduz RTTs: ingestão de 122k jogos caiu de horas para ~90s |
| **PR-AUC como métrica** | ROC-AUC | Dataset desbalanceado (73/27%) — ROC-AUC era enganoso |
| **KMeans** | HDBSCAN | HDBSCAN foi descartado; o clusterizador do projeto é KMeans com k por silhouette. O arquivo `hdbscan_model.pkl` é apenas um nome legado |
| **`equilibrado` como padrão** | `conservador` como padrão | Melhor onboarding sem sacrificar qualidade para novos usuários |

---

## 🤝 9. Como Contribuir

1. Faça um fork do projeto
2. Crie uma branch (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'feat: adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

Leia o arquivo `CONTRIBUTING.md` para mais detalhes sobre o processo de contribuição e padrões de código.

---

**Autor:** Isaque  
**Contato:** [![GitHub](https://img.shields.io/badge/GitHub-seu--usuario-181717?logo=github)](https://github.com/1isaqu) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Isaque-0A66C2?logo=linkedin)](https://linkedin.com/in/seu-usuario)
**Artigo:** [![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@isaquecarvalho2007/como-constru%C3%AD-um-sistema-de-recomenda%C3%A7%C3%A3o-que-entende-jogadores-de-verdade-c0c32aafa470?postPublishedType=repub)
<div align="center">

**⭐ Se este projeto foi útil, considere deixar uma estrela!**

*Killswitch Engage — Do pipeline de dados à API em produção, com ML que funciona de verdade.*

</div>
