# Plano de medição de indicadores de negócio

> ⚠️ **Este documento não contém nenhum número de negócio.** É o inverso de
> `reports/metricas_negocio.md`: aqui está o que **não pode** ser calculado a
> partir dos dados deste repositório, e o que precisaria existir para calculá-lo.
> Onde uma fórmula aparece abaixo, os termos são símbolos — nenhum é preenchido
> aqui. Preenchimentos sob premissa explícita (não medição) ficam em
> `reports/roi_cenarios.md`.

## Por que nenhum destes números existe hoje

O sistema nunca foi a produção. Não há usuário real, não há cobrança, não há
evento de pagamento, não há grupo de controle. `backend/app/routes/recomendacoes.py`
serve uma lista de jogos e retorna — nenhuma chamada é registrada em lugar
nenhum; não existe uma tabela de eventos no schema (`scripts/sql/indices.sql`
e as queries em `backend/app/models/queries.py` cobrem `jogos`, `sessoes_jogo`,
`biblioteca`, `categorias`, `desenvolvedores` — nenhuma tabela de evento,
pagamento, custo de aquisição ou atribuição de experimento). Medir ROI, LTV,
CAC ou churn causal exige três coisas que não existem juntas hoje: **eventos
de receita**, **um grupo de controle** e **tempo de operação real**.

Este plano cobre, para cada indicador: definição operacional, dados
necessários, instrumentação faltante no backend FastAPI/PostgreSQL atual,
desenho experimental (quando a alegação é causal) e horizonte de tempo.

---

## 1. LTV (Lifetime Value)

### Definição operacional

```
LTV_u = Σ (t = mês de aquisição até o fim da relação)  margem_bruta_mensal(u, t)
```

Versão fechada, usada quando a taxa de churn mensal é aproximadamente
constante (processo geométrico):

```
LTV = ARPU_mensal × margem_bruta × vida_util_media_meses
vida_util_media_meses = 1 / taxa_de_churn_mensal
```

Onde:
- **ARPU_mensal** — receita média por usuário ativo, por mês (`receita_do_mês /
  usuários_ativos_no_mês`).
- **margem_bruta** — fração da receita que sobra após custo direto do bem
  vendido (para jogos digitais, tipicamente custo de distribuição/plataforma;
  aqui é um parâmetro a declarar, não um número contábil deste projeto).
- **taxa_de_churn_mensal** — a mesma definição operacional fixada em
  `reports/metricas_negocio.md` §2 (inatividade de N dias), agora medida sobre
  usuários reais, não sobre o gerador sintético.

A fórmula fechada assume risco de churn constante ao longo da vida do
usuário. É uma simplificação — se o risco cai com o tempo (usuários que
sobrevivem ao primeiro mês tendem a ficar mais), o LTV fechado subestima. A
alternativa correta é somar a curva de retenção observada (a mesma curva D0–D30
de `metricas_negocio.md`, estendida a D360+) mês a mês, sem assumir forma
fechada.

### Dados necessários

| Evento | Campos | Granularidade | Retenção mínima |
|---|---|---|---|
| `transacao_criada` | `usuario_id`, `jogo_id`, `valor`, `moeda`, `tipo` (compra/assinatura/DLC/estorno), `criado_em` | por transação | 24 meses (LTV precisa de cohorts maduros) |
| `assinatura_status` | `usuario_id`, `status` (ativa/cancelada/inadimplente), `plano`, `alterado_em` | por mudança de status | 24 meses |
| `sessao_ativa` | já existe (`sessoes_jogo`) | por sessão | já indefinida |

### Instrumentação faltante

O backend não tem **nenhuma** tabela de transação ou pagamento — `biblioteca`
(ver `queries.USER_LIBRARY`) registra posse do jogo, não preço pago nem data
de cobrança. É preciso:

1. Tabela `transacoes` (Postgres): `id, usuario_id, jogo_id, valor_centavos,
   moeda, tipo, gateway_pagamento_id, criado_em`.
2. Tabela `assinaturas` se o modelo de receita incluir recorrência: `id,
   usuario_id, plano, status, inicio, fim, criado_em, atualizado_em`.
3. Integração com um gateway de pagamento (Stripe, Mercado Pago, etc.) via
   webhook assíncrono gravando em `transacoes` — hoje não existe rota de
   pagamento em `backend/app/routes/`.
4. Um job periódico (ou view materializada) que agregue `transacoes` +
   `sessoes_jogo` em ARPU mensal por cohort de aquisição, para acompanhar a
   curva de LTV enquanto ela amadurece.

### Desenho experimental

LTV em si é descritivo (soma de receita observada), não precisa de grupo de
controle. **Mas "o recomendador aumenta o LTV" é uma alegação causal**, e
exige o mesmo desenho de "3. Receita incremental" abaixo — LTV é essa mesma
curva de receita incremental, integrada por mais tempo.

### Horizonte

LTV não é observável em janela curta — por definição, é a soma de receita ao
longo de toda a vida do usuário. Práticas de mercado usam **LTV projetado**
(curva parcial extrapolada), mas isso é projeção, não medição, e carrega o
mesmo aviso da Parte C: a extrapolação assume que o comportamento futuro segue
o padrão do período observado. Para este produto, um mínimo defensável de
observação direta (sem extrapolar) é **12 meses de cohort maduro** — tempo
suficiente para a maior parte do churn de um cohort já ter acontecido, dado
que a distribuição de tempo de vida medida em `metricas_negocio.md` §4 (ainda
que sobre dados sintéticos) tem mediana bem acima de 30 dias.

---

## 2. CAC (Custo de Aquisição de Cliente)

### Definição operacional

```
CAC = gasto_total_de_aquisição_no_período / novos_usuários_adquiridos_no_período
```

`gasto_total_de_aquisição` inclui mídia paga, custo de programas de indicação,
e a fração atribuível de custo de time de growth/marketing — cada termo
precisa de uma regra de atribuição por canal declarada antes de somar.

### Dados necessários

| Evento | Campos | Granularidade | Retenção mínima |
|---|---|---|---|
| `gasto_marketing` | `canal`, `campanha_id`, `data`, `valor_gasto`, `moeda` | diária, por canal/campanha | 24 meses |
| `usuario_criado` | `usuario_id`, `canal_aquisicao`, `campanha_id`, `criado_em` | por cadastro | indefinida |

### Instrumentação faltante

1. `usuarios` (schema atual não é visível neste repositório, mas nenhuma rota
   em `backend/app/routes/usuarios.py` grava canal de aquisição) precisa
   ganhar `canal_aquisicao` e `campanha_id`, capturados no cadastro via UTM ou
   código de indicação.
2. Tabela `gasto_marketing`, alimentada manualmente ou por integração com as
   plataformas de mídia (Google Ads, Meta Ads etc.) — nenhuma existe hoje.
3. Sem atribuição de canal por usuário, CAC só pode ser calculado de forma
   **agregada e grosseira** (gasto total / novos usuários totais no período),
   que mistura canais orgânicos (custo zero) com pagos e sub ou superestima
   dependendo do mix — por isso a atribuição por usuário é a peça que falta,
   não só o log de gasto.

### Desenho experimental

CAC é contábil, não causal, por definição. A pergunta causal adjacente —
"o recomendador reduz CAC via retenção/indicação, permitindo pagar menos por
usuário mantendo o mesmo volume" — não tem desenho direto de A/B (CAC não é
uma métrica por usuário individual); ela se infere indiretamente do efeito do
recomendador sobre retenção (que reduz a pressão de aquisição para manter MAU
constante) e sobre a taxa de indicação orgânica, ambos mensuráveis com o
desenho da seção 3.

### Horizonte

Mínimo de **1 ciclo completo de aquisição** (tipicamente 1 mês) para o
primeiro número, e **3–6 meses** para um CAC estável — canais de mídia paga
têm curva de aprendizado e sazonalidade que um único mês não captura.

---

## 3. Receita incremental atribuída ao recomendador

### Definição operacional

```
receita_incremental = receita_media_por_usuario(tratamento) − receita_media_por_usuario(controle)
```

medida sobre o mesmo período de observação, com `tratamento` e `controle`
definidos no desenho abaixo. **Não** é `receita_total_com_recomendador` — sem
um grupo que não viu o recomendador (ou viu uma versão degradada dele), não
há como separar "receita que teria acontecido de qualquer jeito" de "receita
que o recomendador causou".

### Dados necessários

| Evento | Campos | Granularidade | Retenção mínima |
|---|---|---|---|
| `recomendacao_exibida` | `evento_id`, `usuario_id`, `jogo_id`, `posicao`, `modo`, `score`, `request_id`, `criado_em` | 1 linha por item recomendado, por chamada | 24 meses |
| `recomendacao_interacao` | `evento_id` (FK), `tipo` (clique/adicionado_biblioteca/comprado/ignorado), `criado_em` | por interação | 24 meses |
| `transacao_criada` | ver seção 1, com `recomendacao_evento_id` nullable | por transação | 24 meses |
| `experimento_atribuicao` | `usuario_id`, `experimento_id`, `grupo`, `atribuido_em` | 1 linha por usuário por experimento ativo | duração do experimento + 24 meses |

### Instrumentação faltante

`GET /recomendacoes/{usuario_id}` (`backend/app/routes/recomendacoes.py`)
hoje monta a lista e retorna — nada é persistido. Faltam:

1. Tabela `recomendacao_exibida` (impressão) — gravada de forma assíncrona
   (fire-and-forget, para não adicionar latência à resposta) a cada chamada
   do endpoint, uma linha por item retornado.
2. Tabela `recomendacao_interacao` — precisa de eventos do cliente
   (frontend/app) que hoje não existem no repositório: clique num item
   recomendado, adição à biblioteca a partir de uma recomendação, compra a
   partir de uma recomendação. Isso implica uma rota nova, ex.
   `POST /recomendacoes/{usuario_id}/interacao`, e o cliente precisa carregar
   o `evento_id` da impressão para linkar a interação a ela.
3. Coluna `recomendacao_evento_id` (nullable) em `transacoes` — permite
   atribuir uma compra a uma recomendação específica dentro de uma janela de
   atribuição declarada (ex.: compra em até 7 dias após a impressão).
4. Tabela `experimento_atribuicao` — ver desenho abaixo.

### Desenho experimental

- **Unidade de randomização:** usuário (`usuario_id`), sorteado uma vez na
  primeira chamada a `/recomendacoes/{usuario_id}` e fixado por
  `experimento_atribuicao`. Usuário, não sessão: o produto é single-player, e
  alternar o tratamento sessão a sessão contaminaria o efeito (o usuário veria
  os dois braços e a comparação deixaria de isolar o efeito do recomendador).
- **Grupo de controle:** recomendação por popularidade global (a mesma
  baseline já usada em `evaluate_ranker.py`), não "sem recomendação nenhuma" —
  isso isola o efeito da **camada 3 (SVD)** especificamente, e não o efeito de
  "ter uma lista de jogos na tela", que uma baseline trivial já captura. Um
  terceiro braço "sem lista nenhuma" é opcional, para medir o efeito de ter
  qualquer recomendação versus nenhuma.
- **Métrica primária:** receita por usuário no período do experimento
  (soma de `transacoes.valor_centavos`, incluindo zero para quem não comprou).
  Métricas secundárias: taxa de clique na recomendação, taxa de conversão
  clique→compra, retenção D7/D30 (reaproveitando a definição de
  `metricas_negocio.md`).
- **Efeito mínimo detectável (MDE):** a decidir com o time de produto — é uma
  escolha de negócio (quanto de lift já compensaria o custo do experimento),
  não uma estimativa estatística. O cálculo de tamanho amostral abaixo é
  **ilustrativo**, com um MDE de exemplo, para mostrar o método.
- **Cálculo de tamanho amostral (exemplo ilustrativo — nenhum destes números é medição):**
  Para a métrica primária contínua (receita por usuário), com teste de duas
  amostras e potência de 80% / α = 5% bicaudal:

  ```
  n_por_grupo ≈ 2 × (z_α/2 + z_β)² × σ² / Δ²
              = 2 × (1.96 + 0.84)² × σ² / Δ²
              ≈ 15.7 × (σ/Δ)²
  ```

  Se a receita mensal por usuário tiver desvio-padrão σ hipotético de R$ 20 e
  o lift mínimo que importa for Δ = R$ 4 (20% de um ARPU hipotético de R$ 20):
  `n_por_grupo ≈ 15.7 × (20/4)² ≈ 393` usuários por braço. **σ e Δ aqui são
  hipotéticos** — o projeto não tem nenhuma medição de dispersão de receita
  real; o número existe só para ilustrar a mecânica do cálculo, não para
  dimensionar um experimento real.

  Para a métrica secundária binária (churn sim/não), com proporção-base
  hipotética p₁ e MDE de 5 pontos percentuais:

  ```
  n_por_grupo = (z_α/2 + z_β)² × [p₁(1−p₁) + p₂(1−p₂)] / (p₁−p₂)²
  ```

  Usando p₁ = 0,50 só como **ordem de grandeza** (a taxa de churn mensal
  pooled medida sobre o gerador sintético na janela de 14 dias, em
  `metricas_negocio.md` §2, é 0,52 — usada aqui apenas para calibrar a escala
  do exemplo, **não** como baseline de produção real) e p₂ = 0,45:
  `n_por_grupo = (2.8)² × [0.25 + 0.2475] / 0.0025 ≈ 1.561` usuários por braço.

- **Duração:** função de quantos usuários elegíveis entram no experimento por
  dia/semana — número que este projeto não tem, porque não há tráfego real.
  Regra prática: rodar por pelo menos um ciclo completo do comportamento que a
  métrica primária mede (se a métrica é receita mensal, rodar por no mínimo
  um mês completo por cohort, não cortar no meio de um ciclo de cobrança).

### Horizonte

Mínimo de 1 ciclo de receita completo (ex.: 1 mês, se a monetização for
mensal) para o primeiro resultado; resultados confiáveis sobre efeito de
retenção de médio prazo pedem 3+ ciclos.

---

## 4. Churn causalmente atribuído ao recomendador

### Definição operacional

```
redução_de_churn = taxa_de_churn(controle) − taxa_de_churn(tratamento)
```

usando a mesma definição operacional de churn fixada em
`metricas_negocio.md` §2 (janela de inatividade declarada), agora aplicada
como **desfecho de um experimento controlado**, não como estatística
descritiva sobre todo o log.

### Dados necessários e instrumentação faltante

Os mesmos eventos de sessão que já existem (`sessoes_jogo`) bastam para medir
churn — o que falta é exclusivamente a peça causal: `experimento_atribuicao`
(seção 3) para separar quem viu o recomendador de quem não viu, com
randomização válida.

### Desenho experimental

Idêntico ao da seção 3 (mesmo experimento, métrica secundária promovida a
primária se o objetivo for especificamente churn em vez de receita): unidade
de randomização = usuário, controle = popularidade global, métrica = taxa de
churn no período pela definição operacional já fixada, MDE e tamanho amostral
calculados como no exemplo binário acima — com p₁ vindo de uma medição real
de produção quando ela existir, não do gerador sintético.

### Horizonte

A definição de churn usa uma janela de inatividade (a mesma da Parte A); o
experimento precisa rodar por tempo suficiente para que essa janela se
complete **para todos os usuários do cohort**, mais o próprio período de
observação de churn. Com janela de 14 dias e cohort de 30 dias, o mínimo é
~44 dias corridos antes do primeiro corte confiável.

---

## 5. ROI

### Definição operacional

```
ROI(horizonte) = (receita_incremental_atribuida(horizonte) − custo_total(horizonte)) / custo_total(horizonte)

custo_total = custo_infraestrutura + custo_desenvolvimento + custo_retreino
receita_incremental_atribuida = Σ (t=0..horizonte) receita_incremental(t)   [seção 3]
```

Nenhum termo é preenchido aqui — a estrutura simbólica é o produto desta
seção. `reports/roi_cenarios.md` (Parte C) preenche o lado de custo com preços
públicos ancorados e o lado de receita com cenários explicitamente rotulados
como premissa, nunca como medição.

### Dados necessários

A união de tudo acima: eventos de transação (seção 1), gasto de aquisição e
operação (seção 2), e receita incremental atribuída por experimento (seção
3). Mais os custos diretos de operar o sistema: fatura de infraestrutura
(Postgres gerenciado, Redis, compute da API), horas de desenvolvimento e
custo de retreino periódico dos modelos.

### Instrumentação faltante

Tudo o que falta nas seções 1–3, mais um registro de custo operacional
(fatura de infraestrutura por mês — hoje nenhuma automação captura isso; é
manual, mas simples de manter: `custo_infraestrutura_mensal(mes, categoria,
valor)`).

### Desenho experimental

ROI herda o desenho da seção 3 para o lado de receita. Sem grupo de controle,
"ROI do recomendador" não tem denominador causal — dá para calcular
"custo total de operar o sistema" (contábil, sem experimento), mas não
"retorno", porque retorno pressupõe receita que **não existiria sem** o
sistema, e essa contrafactual só vem de um experimento.

### Horizonte

O maior de todos os horizontes acima — ROI depende de receita incremental
(mínimo 1 ciclo, idealmente 3+) e, se LTV entrar na conta (payback period),
do horizonte de LTV (mínimo 12 meses de cohort maduro). Qualquer ROI
calculado antes disso é necessariamente uma **projeção** com premissas de
extrapolação explícitas, não uma medição — exatamente a distinção que a
auditoria deste projeto cobrou da versão anterior do README.

---

## Resumo

| Indicador | Medível hoje? | Peça que falta primeiro |
|---|---|---|
| LTV | Não | Eventos de transação/pagamento (não existem) |
| CAC | Não | Canal de aquisição por usuário + log de gasto de marketing |
| Receita incremental | Não | Eventos de transação **e** grupo de controle (experimento) |
| Churn causal | Não | Grupo de controle (experimento) — os eventos de sessão já existem |
| ROI | Não | Depende de todos os anteriores |

Nenhum destes cinco indicadores tem uma medição parcial válida hoje. O que
existe de instrumentação (sessões, biblioteca) sustenta as métricas
descritivas de `reports/metricas_negocio.md` — que medem o gerador sintético,
não jogadores — e nada além disso.
