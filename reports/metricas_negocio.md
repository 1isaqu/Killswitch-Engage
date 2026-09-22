# Indicadores de negócio calculáveis a partir dos dados sintéticos

> ⚠️ **Origem dos dados — leia antes de citar qualquer número desta página.**
>
> Os 10.000 usuários e as sessões abaixo são **sintéticos**, gerados por
> `src/data_preparation/generate_synthetic_data.py`. O gerador sorteia o número
> de sessões por perfil de faixas fixas (`randint(5,10)` para casual, e assim
> por diante) e distribui as datas numa janela de 720 dias com uma distribuição
> Beta. **Toda métrica desta página mede esses parâmetros, não comportamento de
> jogadores.** Não existe usuário real, não existe receita, não existe teste
> A/B por trás de nenhum número aqui.


Gerado por `scripts/experimentation/evaluate_business_metrics.py --csv-dir data/ml_ready --seed 42`. Base: 10,000 usuários, 314,806 sessões. Reproduzível: mesmo comando, mesmo resultado.

## 1. Curva de retenção D0–D30

> ⚠️ Mede o **gerador sintético** (parâmetros de `generate_synthetic_data.py`), não comportamento de jogadores reais. Aviso completo no topo da página.


D_n = fração de usuários com ao menos uma sessão na janela [n, n+1) dias após a primeira sessão. D0 = 1,0 por definição (todo usuário tem sessão no dia da própria primeira sessão). Usuários cuja janela [n, n+1) ainda não podia ter sido observada até o fim dos dados são excluídos do denominador de D_n (censura à direita) — por isso `n_elegíveis` cai à medida que n cresce.

| Dia | Retenção | IC 95% (bootstrap) | n elegíveis |
|---:|---:|---:|---:|
| D0 | 1.0000 | [1.0000, 1.0000] | 10,000 |
| D1 | 0.0119 | [0.0098, 0.0140] | 10,000 |
| D2 | 0.0119 | [0.0097, 0.0140] | 10,000 |
| D3 | 0.0127 | [0.0105, 0.0149] | 10,000 |
| D4 | 0.0129 | [0.0107, 0.0151] | 10,000 |
| D5 | 0.0120 | [0.0099, 0.0142] | 10,000 |
| D6 | 0.0121 | [0.0100, 0.0142] | 10,000 |
| D7 | 0.0133 | [0.0110, 0.0157] | 10,000 |
| D8 | 0.0127 | [0.0105, 0.0148] | 10,000 |
| D9 | 0.0119 | [0.0099, 0.0140] | 10,000 |
| D10 | 0.0135 | [0.0113, 0.0158] | 10,000 |
| D11 | 0.0103 | [0.0084, 0.0123] | 10,000 |
| D12 | 0.0131 | [0.0108, 0.0153] | 10,000 |
| D13 | 0.0116 | [0.0095, 0.0137] | 10,000 |
| D14 | 0.0145 | [0.0122, 0.0168] | 10,000 |
| D15 | 0.0124 | [0.0103, 0.0146] | 10,000 |
| D16 | 0.0147 | [0.0124, 0.0170] | 10,000 |
| D17 | 0.0129 | [0.0109, 0.0152] | 10,000 |
| D18 | 0.0137 | [0.0115, 0.0159] | 10,000 |
| D19 | 0.0120 | [0.0097, 0.0141] | 10,000 |
| D20 | 0.0157 | [0.0134, 0.0181] | 10,000 |
| D21 | 0.0140 | [0.0119, 0.0163] | 10,000 |
| D22 | 0.0156 | [0.0133, 0.0181] | 10,000 |
| D23 | 0.0150 | [0.0127, 0.0175] | 10,000 |
| D24 | 0.0172 | [0.0146, 0.0198] | 10,000 |
| D25 | 0.0161 | [0.0139, 0.0186] | 10,000 |
| D26 | 0.0130 | [0.0109, 0.0152] | 10,000 |
| D27 | 0.0154 | [0.0131, 0.0178] | 10,000 |
| D28 | 0.0153 | [0.0130, 0.0177] | 10,000 |
| D29 | 0.0122 | [0.0100, 0.0143] | 10,000 |
| D30 | 0.0163 | [0.0138, 0.0188] | 10,000 |

## 2. Taxa de churn mensal e sensibilidade à janela

> ⚠️ Mede o **gerador sintético** (parâmetros de `generate_synthetic_data.py`), não comportamento de jogadores reais. Aviso completo no topo da página.


**Definição adotada** (janela declarada = 14 dias): um usuário ativo num mês-calendário de 30 dias "churna" nesse mês se não tiver nenhuma sessão nos `janela_dias` seguintes ao fim do mês. A taxa mensal é a fração de pares (usuário, mês) com essa propriedade, entre pares cuja janela de checagem já terminou antes do fim dos dados. `Churn` tem dezenas de definições possíveis na literatura; esta foi escolhida por ser verificável diretamente no log de sessões, sem precisar de evento de cancelamento explícito (que não existe no sistema — ver `reports/plano_medicao_negocio.md`).

**Teste de sensibilidade** — a mesma definição, variando só a janela de inatividade. Se a taxa mudar muito entre 7, 14 e 30 dias, a métrica é frágil a essa escolha arbitrária, e isso conta tanto quanto o número em si:

| Janela (dias) | Taxa mensal (pooled) | Taxa média/usuário | IC 95% | usuários elegíveis | user-meses elegíveis |
|---:|---:|---:|---:|---:|---:|
| 7 | 0.6658 | 0.7777 | [0.7730, 0.7823] | 10,000 | 95,715 |
| 14 ← definição adotada | 0.5227 | 0.6609 | [0.6554, 0.6665] | 10,000 | 95,715 |
| 30 | 0.3413 | 0.4872 | [0.4812, 0.4930] | 10,000 | 90,208 |

Variação entre a janela mais curta e a mais longa testadas: **49%** da taxa pooled mais alta. É grande — trate a taxa de churn mensal como sensível à definição, não como um número único.

## 3. Frequência e intensidade de sessão por coorte de perfil

> ⚠️ Mede o **gerador sintético** (parâmetros de `generate_synthetic_data.py`), não comportamento de jogadores reais. Aviso completo no topo da página.


Coortes reconstruídas a partir da mesma sequência de sorteios de `generate_synthetic_data.py` (não estão no CSV — ver docstring do módulo). Validação: 100.0% dos usuários caem dentro da faixa de sessões que o próprio gerador declara para seu perfil (esperado: ~100%, com possível ambiguidade pontual na fronteira entre hardcore e borda, 150–200 sessões, onde os dois grupos se sobrepõem por desenho do gerador).

| Coorte | n usuários | Sessões/usuário (freq.) | IC 95% | Horas/sessão (intens.) | IC 95% | Sessões/mês de vida | IC 95% |
|---|---:|---:|---:|---:|---:|---:|---:|
| casual | 6,500 | 7.52 | [7.48, 7.57] | 1.55 | [1.54, 1.56] | 0.62 | [0.62, 0.63] |
| medio | 2,500 | 35.44 | [35.08, 35.77] | 4.24 | [4.22, 4.26] | 1.98 | [1.96, 2.00] |
| hardcore | 500 | 125.30 | [121.57, 129.12] | 10.50 | [10.46, 10.55] | 6.19 | [6.02, 6.37] |
| borda | 500 | 229.30 | [225.44, 233.21] | 12.05 | [12.01, 12.09] | 10.93 | [10.75, 11.11] |

As faixas de sessões por perfil são parâmetros fixos do gerador (`{'casual': (5, 10), 'medio': (20, 50), 'hardcore': (50, 200)}`, borda `(150, 300)`), então a diferença entre coortes aqui é, por construção, a diferença entre essas faixas — não uma segmentação descoberta nos dados.

## 4. Distribuição de tempo de vida do usuário

> ⚠️ Mede o **gerador sintético** (parâmetros de `generate_synthetic_data.py`), não comportamento de jogadores reais. Aviso completo no topo da página.


Tempo de vida = dias entre a primeira e a última sessão observadas no log (não é tempo até churn — um usuário pode voltar depois da última sessão observada; os dados só vão até onde vão).

- Média: **449.8 dias**, IC 95% [447.6, 452.3]

| Percentil | Dias |
|---:|---:|
| p10 | 278.8 |
| p25 | 359.0 |
| p50 | 458.2 |
| p75 | 547.3 |
| p90 | 610.6 |

---

Para o que não é calculável a partir destes dados — ROI, LTV, CAC, churn causalmente atribuído ao recomendador — ver [`reports/plano_medicao_negocio.md`](plano_medicao_negocio.md) (o que falta instrumentar) e [`reports/roi_cenarios.md`](roi_cenarios.md) (estimativa sob premissas explícitas, rotulada como projeção, nunca como medição).
