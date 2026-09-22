# Prompt: indicadores de sucesso do Killswitch Engage

Copie tudo abaixo da linha para a outra IA. O prompt é autocontido — não depende
desta conversa.

---

## Contexto

Você vai trabalhar no repositório `1isaqu/Killswitch-Engage`, um sistema de
recomendação de jogos da Steam. Leia o `README.md` inteiro antes de começar, em
especial a seção 3, e também `reports/artigo_medium_corrigido.md`.

O projeto passou por uma auditoria técnica recente que removeu métricas infladas.
O que você precisa saber dela:

- A versão anterior publicava **ROI de 460%, churn −18%, MAU +27%, LTV +31%**.
  Nenhum desses números derivava de dado algum. Foram removidos.
- Publicava também Precision@10 = 0.70 e MRR = 1.0, vindos de um script em que o
  gabarito era construído a partir das próprias recomendações. O banco MLflow do
  repositório guarda a execução honesta da mesma noite: Precision@10 = 0.0006.
- Os **10.000 usuários e 309 mil sessões são sintéticos**, gerados
  proceduralmente por `src/data_preparation/generate_synthetic_data.py`. Só o
  catálogo de 122.507 jogos vem de dados reais da Steam.
- Não existe receita, não existe A/B test, não existe usuário real. O sistema
  nunca foi a produção.

## Sua tarefa

Produzir os indicadores de sucesso de negócio do projeto — retenção, churn, LTV,
ROI — de forma que cada número publicado seja defensável por alguém que abra o
código e verifique.

Isso significa dividir o trabalho em duas partes, porque parte é calculável hoje
e parte não é.

### Parte A — O que os dados sustentam

Crie `scripts/experimentation/evaluate_business_metrics.py` computando, a partir
de `data/ml_ready/interacoes_sessoes.csv`:

1. **Curvas de retenção D1 / D7 / D30.** Para cada usuário, a data da primeira
   sessão é o dia 0. Retenção D_n = fração de usuários com ao menos uma sessão na
   janela [n, n+1). Reporte a curva completa de D0 a D30.
2. **Taxa de churn por definição explícita.** Escolha e declare a janela (ex.:
   sem sessão por 14 dias consecutivos). Calcule a taxa mensal. Teste a
   sensibilidade da métrica a janelas de 7, 14 e 30 dias — se o número muda muito,
   a definição é frágil e isso precisa aparecer no relatório.
3. **Frequência e intensidade de sessão por coorte de perfil** (casual / médio /
   hardcore / borda), já que o gerador cria esses perfis explicitamente.
4. **Distribuição de tempo de vida do usuário**: dias entre primeira e última
   sessão.

Use intervalo de confiança de 95% por bootstrap sobre usuários em toda média
reportada. O repositório já tem `ic_bootstrap` e `comparar_pareado` em
`scripts/experimentation/evaluate_ranker.py` — reutilize.

**Rotulagem obrigatória.** Todo número desta parte mede o **gerador sintético**,
não o comportamento de jogadores. O gerador sorteia número de sessões por perfil
a partir de faixas fixas (`random.randint(5, 10)` para casual, e assim por
diante) e distribui as datas numa janela de 720 dias com uma distribuição Beta.
Portanto uma "retenção D7 de X%" é uma propriedade dos parâmetros que alguém
escolheu, não uma descoberta sobre pessoas.

Escreva isso no topo do script, no topo do relatório gerado e em cada tabela.
Um leitor que veja só a tabela precisa entender o que ela é.

### Parte B — O que exige instrumentação

Para **ROI, LTV, CAC, receita incremental e churn causalmente atribuído ao
recomendador**, não produza número. Produza o plano de medição, em
`reports/plano_medicao_negocio.md`, cobrindo para cada indicador:

- **Definição operacional.** A fórmula exata, com cada termo definido. "Churn" tem
  dezenas de definições; escolha uma e defenda.
- **Dados necessários.** Quais eventos precisam ser registrados, com qual
  granularidade e por quanto tempo. Seja específico: nome do evento, campos,
  frequência.
- **Instrumentação faltante.** O que o sistema atual não registra e precisaria
  registrar. O backend é FastAPI com PostgreSQL; diga o que acrescentar.
- **Desenho experimental.** Para qualquer alegação causal ("o recomendador reduziu
  churn em X%"), especifique: unidade de randomização, grupo de controle, métrica
  primária, efeito mínimo detectável, cálculo de tamanho amostral e duração. Sem
  controle não há atribuição causal, e isso precisa estar dito.
- **Horizonte.** Quanto tempo de operação real até o indicador ter significado.
  LTV, em particular, não é observável em janelas curtas.

Para o ROI, mostre a **estrutura do cálculo** (custos de infraestrutura e
desenvolvimento contra receita incremental atribuível) com as variáveis como
símbolos não preenchidos. Deixe explícito que preencher com estimativas de
benchmark de outras plataformas produz uma projeção, não uma medição — e que
projeção rotulada como medição foi exatamente o erro que a auditoria corrigiu.

### Parte C — Estimativa sob premissas explícitas

Medir ROI é impossível aqui. **Estimar é legítimo**, desde que a estimativa seja
transparentemente uma função de entradas que o leitor possa inspecionar e
contestar. Produza `scripts/experimentation/roi_sensitivity.py` e
`reports/roi_cenarios.md`.

O modelo tem duas metades. Os **custos** são razoavelmente ancoráveis: preço
público de infraestrutura (Postgres gerenciado, Redis, compute da API), horas de
desenvolvimento, custo de retreino. Cite a fonte de cada preço com a data da
consulta.

O **benefício** não tem âncora alguma neste projeto, e é aí que está o trabalho
honesto. A receita incremental depende de:

```
receita_incremental = MAU × lift_retencao × ARPU × margem
```

- `MAU` — parâmetro de cenário, não dado do projeto.
- `lift_retencao` — **o termo crítico**. Quanto a recomendação melhora retenção
  em relação a não ter recomendação. O projeto não mede isso e não tem como
  medir sem A/B. É a premissa dominante.
- `ARPU` e `margem` — parâmetros de cenário.

Faça o seguinte:

1. **Varra `lift_retencao`** num intervalo largo e plausível (por exemplo de 0%
   a 15%) e reporte o ROI resultante como curva, não como ponto.
2. **Análise de sensibilidade.** Calcule a elasticidade do ROI a cada parâmetro e
   ordene por influência. Se `lift_retencao` dominar — e deve dominar —, diga
   isso com o número: "uma variação de X% nessa premissa move o ROI em Y pontos".
3. **Ponto de equilíbrio.** Qual o `lift_retencao` mínimo para o ROI ser positivo
   em 12 meses? Esse é o número mais útil do relatório inteiro, porque é uma
   afirmação verificável: "o sistema se paga se, e somente se, melhorar retenção
   em pelo menos Z%".
4. **Três cenários** (conservador / base / otimista) com todas as premissas
   tabeladas lado a lado, e o ROI de cada um.

Regras de apresentação, não negociáveis:

- O título de toda tabela e figura diz **"projeção"** ou **"cenário"**, nunca
  "resultado" ou "medição".
- Nenhum número desta parte aparece fora de contexto. Se for para o README ou
  para um currículo, vai acompanhado da premissa que o gera.
- A conclusão da seção declara explicitamente que o intervalo de ROI é largo
  porque a premissa dominante não foi medida, e que só um teste A/B a estreita.

O objetivo desta parte não é chegar a um número de ROI. É mostrar **de que o
número depende** e **quanto** ele depende — e, com isso, transformar "não sei o
ROI" em "sei exatamente o que preciso medir para saber, e sei que o projeto se
paga a partir de um lift de Z%".

## Restrições

Estas não são negociáveis:

1. **Nenhum número inventado.** Se um indicador não é calculável a partir dos
   dados do repositório, ele não recebe valor numérico. Escreva "não medido" e
   explique o que faltaria.
2. **Nenhuma estimativa de benchmark apresentada como resultado.** Citar que
   plataformas similares veem X% é legítimo como referência, desde que a fonte
   esteja citada e o texto deixe claro que não é uma medição deste sistema. Vale
   como entrada de cenário na Parte C, nunca como saída medida.
3. **Toda média vem com intervalo de confiança.** Diferença sem IC não sustenta
   comparação.
4. **Nada de vazamento.** Se for calcular qualquer coisa preditiva, o split
   temporal já implementado em `split_temporal` (treino 70% / validação 15% /
   teste 15%) é o padrão do repositório. Não derive gabarito da própria predição.
5. **Determinismo.** `--seed 42`, como o resto dos scripts. Mesmo comando, mesmo
   resultado.

## Entregáveis

1. `scripts/experimentation/evaluate_business_metrics.py` — executável, com
   docstring de módulo explicando a limitação dos dados sintéticos.
2. `reports/metricas_negocio.md` — saída da Parte A, com o aviso de origem dos
   dados no topo.
3. `reports/plano_medicao_negocio.md` — Parte B.
4. Uma seção nova no `README.md` apresentando os indicadores, coerente com o tom
   do resto do arquivo: número quando há medição, "não medido" quando não há.

## Como começar

Se `data/ml_ready/` estiver vazio, gere os dados primeiro:

```bash
python -m src.data_preparation.generate_synthetic_data
```

Depois leia `scripts/experimentation/evaluate_ranker.py` para absorver as
convenções do repositório — split temporal, bootstrap, rotulagem de origem dos
dados — antes de escrever código novo.

## Critério de aceitação

Um revisor cético deve conseguir, para cada número do relatório, abrir o script
que o produziu e confirmar que ele mede o que a legenda diz medir. Para cada
indicador sem número, deve encontrar a explicação do que falta.

Se ao final o relatório tiver mais "não medido" do que números, o trabalho está
certo. Um projeto que nunca foi a produção não tem indicadores de negócio, e
dizer isso com clareza vale mais do que preencher a tabela.
