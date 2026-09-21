# Artigo corrigido — guia de alterações

> **Leia antes de copiar para o Medium.**
>
> Cada seção alterada começa com um bloco `> 🔧 ALTERADO`. Ele diz **o que mudou** e
> **por quê**. Apague esses blocos antes de publicar — eles são para você, não para o leitor.
>
> **Seções mantidas sem alteração:** 02, 03, 04, 05, 06, 07 e "Ferramentas utilizadas".
> São a parte do artigo que sobrevive à auditoria. Não precisa tocar nelas.
>
> **Onde há `[RODAR]`** o número precisa ser medido de novo antes de publicar. Os dados
> originais se perderam, então não existe número honesto para colocar ali hoje. Rode
> `python scripts/experimentation/evaluate_ranker.py --csv-dir data/ml_ready` e use a
> saída. Não reaproveite números de auditoria: aqueles foram medidos sobre um dataset
> regenerado, não sobre o seu.

---

> 🔧 **ALTERADO — título e subtítulo**
> "50 mil jogos" virou 122.507, que é o tamanho real do catálogo (o número aparecia em
> três valores diferentes no texto). Tirei "4 camadas de ML" do subtítulo: são 3 no
> pipeline, e a quarta foi removida depois de medida. Tirei "RF · HDBSCAN · LightFM ·
> cGAN" do cabeçalho — HDBSCAN e LightFM nunca existiram no código.

# Como construí um sistema de recomendação de jogos — e o que descobri ao auditá-lo sete meses depois

### 122 mil jogos da Steam, um viés no Metacritic que ninguém esperava, e métricas minhas que não sobreviveram à própria revisão

📊 12 análises estatísticas
🔬 Random Forest · KMeans · TruncatedSVD
⏱ ~15 min de leitura

---

> 🔧 **ALTERADO — abertura**
> Mantive a premissa, que é boa. Troquei "quatro camadas" por três e ajustei o número de
> jogos. Acrescentei uma linha avisando que o artigo tem duas partes: o que construí e o
> que a auditoria derrubou. Isso prepara o leitor e evita que ele descubra sozinho.

Você já abriu a Steam com vontade de jogar, rolou as recomendações por dez minutos e
fechou sem instalar nada? Esse problema tem nome: paralisia de escolha, e tem causa: o
algoritmo padrão da plataforma recomenda o que é popular, não o que é relevante para você.

Existem padrões escondidos em dados que parecem caóticos. A Steam, com seu catálogo de
dezenas de milhares de jogos e bilhões de horas de gameplay registradas, é um dos
conjuntos de dados mais ricos sobre comportamento humano que existem.

Este projeto nasceu dessa frustração. Construí um pipeline completo de recomendação sobre
122.507 jogos, com análise exploratória rigorosa e testes estatísticos formais.

Este artigo tem duas partes. A primeira é o que encontrei nos dados — e essa parte se
sustenta. A segunda é o que aconteceu quando, sete meses depois, auditei meu próprio
código linha por linha em vez de confiar no que a documentação dizia. Várias das métricas
que publiquei na primeira versão deste artigo não sobreviveram.

---

> 🔧 **REMOVIDO — bloco de métricas do topo (0.70 Precision@10, 1.0 MRR, 0.915 PR-AUC,
> 460% ROI)**
>
> Os quatro números saíram. Motivos, na ordem:
>
> - **Precision@10 = 0.70 e MRR = 1.0** vieram de um script em que o gabarito era
>   construído a partir das próprias recomendações (`truths_dict[u] = recommended_ids[:7]`),
>   o que fixa o valor em 0.7 por construção. As "recomendações" eram `np.random.randint`,
>   não saída de modelo. O banco MLflow do projeto guarda as duas execuções da mesma noite:
>   sem o vazamento, Precision@10 = **0.0006**.
> - **MRR = 1.0** exato é assinatura de vazamento, não de qualidade. Nenhum sistema real
>   acerta a primeira posição sempre.
> - **PR-AUC = 0.915** é calculado de verdade no código, mas o valor não é reproduzível
>   hoje (os dados se perderam). Se quiser mantê-lo, rode de novo e cite a data.
> - **ROI 460%** não deriva de dado nenhum. Sem usuário real e sem A/B, não existe ROI.
>
> Se quiser um bloco de destaque no topo, use os achados de EDA das seções 02–07, que são
> verificáveis. Sugestão abaixo.

**F = 318.99 · p ≈ 0** — preço varia por categoria de forma não aleatória
**H = 1464.03 · p ≈ 10⁻³⁰⁷** — retenção difere por gênero de forma inquestionável
**ρ = 0.332 vs r = 0.018** — conquistas afetam retenção em degraus, não linearmente
**r = 0.2038 · p = 0** — preço prevê nota do Metacritic mesmo após controle estatístico

---

> 🔧 **REESCRITO — Seção 01 (Arquitetura)**
>
> Três mudanças:
> 1. **HDBSCAN → KMeans** e **LightFM → TruncatedSVD.** Não havia um único `import hdbscan`
>    ou `from lightfm` no projeto. Os arquivos `.pkl` só tinham esses nomes; o conteúdo era
>    outro algoritmo.
> 2. **Camada 4 (cGAN) saiu do diagrama.** Ela nunca foi carregada em inferência — zero
>    `load_state_dict` no repositório. Os thresholds vinham de um dicionário fixo.
> 3. **Acrescentei a tabela "treinada vs. usada".** Era a informação que faltava: as camadas
>    existiam como artefatos, mas a cascata não estava ligada.

## 01 — Arquitetura

### Três camadas em sequência, e o que cada uma realmente faz

```
PIPELINE DE RECOMENDAÇÃO
Usuário entra no sistema
   ↓
[1 · Random Forest] → filtra jogos por potencial de engajamento
   ↓
[2 · KMeans]        → posiciona o usuário no mapa comportamental
   ↓
[3 · TruncatedSVD]  → filtragem colaborativa sobre a matriz usuário × jogo
   ↓
Lista personalizada com modo escolhido
```

Uma ressalva que a primeira versão deste artigo não fazia, e que é a diferença entre
descrever o desenho e descrever o sistema:

| Camada | Treinada? | Usada em inferência? |
|---|---|---|
| 1 — Random Forest | sim | **não** — o filtro de qualidade retornava `True` incondicionalmente |
| 2 — KMeans | sim | **não** — carregada e nunca consultada |
| 3 — TruncatedSVD | sim | **sim — é o que gera a recomendação** |

As três camadas existem como modelos treinados. Só uma gera recomendação. O que o
diagrama descreve é a intenção; o que roda é a camada 3 com um threshold fixo.

Manter essa distinção visível importa mais do que parece. Um diagrama de quatro caixas
sugere um sistema que a execução não entrega, e quem for ler o código descobre em cinco
minutos.

### Os 3 modos: devolvendo o controle ao usuário

A maioria dos recomendadores toma decisões silenciosas sobre o trade-off
precisão/cobertura. Aqui esse trade-off é externalizado e a escolha devolvida ao usuário.

Transparência sobre incerteza é uma feature, não fraqueza. Quando o usuário sabe que está
no modo Aventureiro, ele interpreta uma recomendação estranha de forma diferente.

> 🔧 **ACRESCENTADO — ressalva sobre os modos**
> Medi o efeito real dos três modos. O threshold só restringe o conjunto de candidatos, e
> a lista final sai sempre ordenada por score — então o topo é praticamente o mesmo nos
> três. A diversificação que o sistema entrega vem dos slots de exploração aleatória, não
> do threshold. Vale dizer isso, porque é um resultado interessante por si só.

Uma correção sobre esse mecanismo: ao medir os três modos, descobri que o threshold
**quase não muda o topo da lista**. Ele restringe o conjunto de candidatos, mas o ranking
final é sempre ordenado por score, então os primeiros dez itens tendem a ser os mesmos. A
diferença observável entre Conservador e Aventureiro vem dos slots de exploração
aleatória, não do corte por threshold.

---

> ✅ **MANTIDO SEM ALTERAÇÃO — Seções 02 a 07**
>
> Copie do seu texto original: 02 (Preço & Categoria), 03 (Notas & Categoria), 04 (Retenção
> & Gênero), 05 (Conquistas & Retenção), 06 (O Viés do Metacritic), 07 (Correlações & Power
> Law).
>
> São análise estatística real sobre dados reais da Steam, com o teste apropriado para cada
> pergunta e interpretação cuidadosa. A seção 06 em especial — controlar por
> `average_playtime_forever` e reportar que o resultado não mudou — é exatamente o que se
> espera de quem sabe o que está fazendo. Não mexa.
>
> Único ajuste sugerido: onde o texto diz "50 mil jogos", troque por 122.507 para bater com
> o resto.

*[Cole aqui as seções 02 a 07 do original, sem alterações]*

---

> 🔧 **REESCRITO — Seção 08 (Modelo Preditivo)**
>
> Mantive a importância de features e as matrizes de confusão, que saem de código real
> (`train_layer1_classifier.py`).
>
> **Removi o estudo de ablação inteiro.** O script que o gerou (`ablation.py`) não desliga
> camada nenhuma: os scores vêm de `np.random.uniform`, com a faixa maior atribuída a
> "hybrid". O resultado "híbrido vence com +35%" estava embutido no gerador de números
> aleatórios. Não é um número fraco — é um número que não mede nada.
>
> Marquei PR-AUC como `[RODAR]`: o cálculo é legítimo, o valor não é reproduzível hoje.

## 08 — Modelo Preditivo

### Random Forest: o que a IA aprendeu a valorizar

`Recommendations` e `average_playtime_forever` aparecem com importância mais alta que
métricas críticas e de preço. Engajamento real prevê relevância melhor do que nota ou
posicionamento — o que é coerente com o cluster popularidade-engajamento identificado na
matriz de correlação.

### Precision-Recall e as matrizes de confusão

PR-AUC = `[RODAR]`. A escolha de PR-AUC sobre ROC-AUC foi deliberada: o dataset é
desbalanceado (aproximadamente 31/69), e ROC-AUC é otimista demais nesse cenário. Essa
decisão está no código e é uma das que eu manteria sem mudar nada.

As matrizes de confusão para os thresholds 0.3, 0.5 e 0.7 mostram o trade-off operacional
de cada modo.

> 🔧 **SOBRE A ABLAÇÃO**
> Se quiser manter uma seção de ablação, ela precisa ser refeita instanciando o
> recomendador com cada camada desativada e medindo contra um gabarito independente.
> O script antigo não faz isso. Enquanto não for refeito, não existe número de ablação.

---

> 🔧 **ALTERADO — Seção 09 (Dados & Escalabilidade)**
>
> A curva de cobertura é real: `coverage_regression.py` executa o recomendador de verdade e
> faz regressão log-log honesta. Mantive.
>
> Ajustei dois pontos: o texto dizia "10.000 usuários reais" — são **sintéticos**, gerados
> com `random`. E "48% dos jogos invisíveis" não bate com os outros números de cobertura
> citados no artigo (11.5% e 3.08%). Marquei para você reconciliar.

## 09 — Dados & Escalabilidade

### A fundação: qualidade da imputação e curva de cobertura

**Confiança da imputação.** A maior parte dos registros de gênero foi imputada com alta
confiança, combinando dados originais e um sistema especialista; o restante foi tratado via
ensemble. Incerteza conhecida é gerenciável; incerteza oculta é perigosa.

**Cobertura de catálogo.** A cobertura segue uma lei de potência mensurável (R² = 0.947),
com a curva desacelerando claramente após 100k usuários. Cada novo usuário acrescenta menos
itens cobertos que o anterior.

Uma correção importante sobre a base: os **10.000 usuários e 309 mil sessões são
sintéticos**, gerados proceduralmente a partir do catálogo real da Steam. Os jogos são
reais; os jogadores não. A primeira versão deste artigo dizia "usuários reais", e isso
estava errado.

> `[RODAR]` — o percentual de jogos invisíveis precisa ser reconciliado. O artigo cita 48%,
> o relatório do projeto cita cobertura de 11.5% e depois 3.08%. São três números
> incompatíveis para a mesma grandeza.

---

> 🔧 **REESCRITO POR INTEIRO — Seção 10 (Erros & Lições)**
>
> **Erro 01 mantido** — a história do ROC-AUC com vazamento temporal é verdadeira e a
> lição está certa.
>
> **Erro 02 invertido.** O texto dizia que você migrou de KMeans para HDBSCAN. O projeto
> fez o contrário: HDBSCAN foi descartado e KMeans ficou. Além disso, medi que escolher `k`
> maximizando silhouette otimiza **contra** a qualidade de recomendação. Reescrevi com o
> que aconteceu de fato, que é uma lição melhor.
>
> **Erro 03 mantido** com o número marcado para reconciliar.
>
> **Erro 04 é novo** e é o mais importante do artigo.

## 10 — Erros & Lições

Projetos que relatam apenas os acertos mentem por omissão. Quatro erros moldaram este
sistema — e o quarto só apareceu sete meses depois.

### Erro 01 — O ROC-AUC que mentia

Na primeira versão, ROC-AUC atingiu 0.97. Deveria ter gerado suspeita, não celebração.
Investigação revelou vazamento temporal: features de janelas futuras contaminavam o treino.
Com splits temporais estritos, a métrica caiu para patamares honestos.

**Lição:** métricas muito boas são sinal de alerta.

### Erro 02 — O clustering que otimizava a métrica errada

A segmentação usa KMeans, escolhendo `k` pelo valor que maximiza o silhouette score. Parece
rigoroso. Não é.

Ao medir a camada pelo que ela deveria entregar — recomendação, e não coesão geométrica —
encontrei que **silhouette e qualidade de recomendação são anticorrelacionados** nesta base.
O silhouette cai monotonicamente conforme aumenta a dimensionalidade, enquanto a precisão
tem seu pico justamente na faixa de pior silhouette. Escolher `k` por silhouette estava
otimizando contra o objetivo real.

Pior: nenhuma configuração de clustering bateu uma baseline de popularidade global. A
diferença ficou dentro do intervalo de confiança. Com popularidade de cauda pesada, a
popularidade *dentro* de um cluster é dominada pelos mesmos títulos de cabeça — o
agrupamento não muda o topo da lista.

**Lição:** uma métrica interna de qualidade de cluster não é a mesma coisa que utilidade.
Meça a camada pelo trabalho que ela deveria fazer.

### Erro 03 — Os invisíveis

Uma fração grande do catálogo nunca aparece em recomendação — não por ser ruim, mas por
não ter interações suficientes para gerar sinal colaborativo. Otimizar precisão sem
monitorar cobertura cria sistemas que funcionam bem para o popular e ignoram o resto.

**Lição:** cobertura precisa ser métrica de primeira classe, não apêndice.

### Erro 04 — Eu apliquei a lição do Erro 01 em todo lugar, menos nos meus próprios números

Este é o erro que corrói os outros três.

A primeira versão deste artigo abria com Precision@10 = 0.70 e MRR = 1.0. Sete meses
depois, ao auditar o código em vez da documentação, encontrei o script que produziu esses
números. Ele construía o gabarito a partir das próprias recomendações:

```python
subset_hits = recommended_ids[:7]   # "garantindo que 7 em 10 batem"
truths_dict[u] = subset_hits + np.random.randint(0, catalog_ids, size=3).tolist()
```

Isso fixa Precision@10 em 0.7 por construção. E as recomendações avaliadas nem vinham do
modelo — eram `np.random.randint`; os arquivos `.pkl` eram carregados e nunca usados para
inferir.

O banco MLflow do projeto guardou as duas execuções daquela noite, com 45 minutos de
diferença:

| Execução | Horário | Precision@10 | MRR |
|---|---|---|---|
| Sem o vazamento | 21:27 | **0.0006** | 0.0014 |
| Com o vazamento | 22:12 | **0.6999** | 1.000 |

Publiquei a segunda.

O MRR de exatamente 1.0 deveria ter disparado o mesmo alarme que o ROC-AUC de 0.97 disparou
meses antes. Nenhum sistema de recomendação real acerta a primeira posição sempre. Eu tinha
escrito a lição e não a apliquei ao número que mais queria que fosse verdade.

**Lição:** a lição do Erro 01 não vale só para as métricas que você desconfia. Vale
principalmente para as que você celebra.

---

> 🔧 **SEÇÃO NOVA — substitui a Errata**
>
> A errata original tentava salvar a camada 4 e se contradizia: citava MAE 0.0156 num
> parágrafo e 0.0182 no outro, e "94.7% de precisão" e "98.4%" para a mesma coisa, sempre
> com o mesmo ganho de "13×".
>
> Troquei por esta seção, que conta o que a auditoria encontrou. É mais honesta e mais
> interessante de ler.

## 11 — A auditoria: o que a camada 4 realmente fazia

A primeira versão descrevia uma cGAN que calibraria o threshold por usuário, com ganho de
13× sobre o modo estático. Três coisas sobre isso.

**A rede existe e foi treinada de verdade.** Generator e Discriminator condicionados, loss
adversarial, TTUR, gradient clipping. Os checkpoints registram 40.500 passos de treino,
consistentes com as 500 épocas da curva de loss. Essa parte é real e eu a manteria em
qualquer currículo.

**Mas ela sofreu mode collapse.** As saídas colapsam praticamente todas num único valor,
independentemente do usuário condicionado. A loss do discriminador fica presa perto de
ln 2 — ou seja, no acaso — durante as 500 épocas. O componente adversarial não contribuiu;
o que restou foi um regressor que aprendeu a moda do alvo.

**E o "ganho de 13×" era a escolha da baseline.** O alvo (`best_threshold`) tem cerca de
72% da massa num único valor. A baseline de comparação era a constante 0.5, que fica longe
dessa massa e por isso erra muito. Comparada contra a baseline honesta — sempre prever a
moda — a rede não ganha; empata ou perde. Um `return 0.3` de uma linha teria o mesmo erro.

O argumento que encerra a camada, porém, é independente de tudo isso: **o threshold não
muda o top-k.** Como descrito na Seção 01, ele só restringe o conjunto de candidatos, e a
lista sai sempre ordenada por score. Mesmo um preditor de threshold perfeito não mudaria a
saída do sistema.

A camada 4 otimizava um parâmetro que não afetava o resultado. Removi do pipeline.

### E os meta-usuários sintéticos?

A primeira versão dizia que a geração de perfis sintéticos via cGAN não foi implementada
por limite de armazenamento do Supabase. Isso não é exato, e a correção importa.

A cGAN **foi** implementada — mas para prever um número, não para gerar usuários. A saída
do gerador é `nn.Linear(hidden_dim, 1)`: um escalar, o threshold. A docstring da classe a
chamava de "Meta-Usuário", e foi daí que veio minha própria confusão ao escrever o artigo.
O nome afirmava algo que o código não fazia.

Os 10.000 usuários sintéticos do projeto nunca passaram por rede neural nenhuma. Vieram de
um laço com `random.choice` e `random.randint`.

---

> 🔧 **REESCRITO — Próximos Passos**
>
> Tirei "Meta-usuários via cGAN" como próximo passo, porque a premissa estava errada.
> Acrescentei o que a auditoria mediu e não resolveu, que é um roteiro melhor porque cada
> item tem um número atrás.

## 12 — Próximos Passos

A auditoria deixou o problema mais bem posto do que estava antes.

**O problema central: descoberta.** O ranker vence uma baseline de popularidade quando
reordena itens que o usuário já conhece, e **perde** quando o teste é recomendar algo
novo. Testei quatro abordagens para corrigir isso — despopularização do score, BPR, WARP, e
recomendação por cluster. Nenhuma superou a popularidade na tarefa de descoberta.

O padrão é consistente o bastante para uma conclusão: nesta base, sinal colaborativo não
generaliza para itens não vistos, independentemente da função de perda. Com densidade de
interação na casa de 0,02% e uma fração grande dos jogos aparecendo uma única vez, falta
coocorrência para inferir afinidade.

**O caminho não testado é conteúdo** — usar as features do jogo diretamente para itens sem
histórico de interação. Não testei porque os dados são sintéticos e o gerador sorteia
preferências a partir de gênero: medir conteúdo ali seria medir a premissa que eu mesmo
escrevi, não o método. Essa pergunta só se responde com interação real.

**Testes A/B reais.** Todas as métricas são offline. Nenhum número de negócio deste projeto
tem base empírica, e a versão anterior deste artigo afirmava vários como se tivessem.

---

> 🔧 **REESCRITO — Conclusão**
>
> Tirei os números removidos (0.70, 1.0, 0.915, os 35% da ablação). Mantive os achados de
> EDA, que continuam válidos, e fechei com a lição da auditoria.

## Conclusão

Preço é posicionamento, não qualidade. Gênero dita a equação de retenção. Conquistas bem
calibradas multiplicam a sobrevida do jogador, em degraus e não linearmente. A nota da
comunidade carrega viés de categoria. E há um padrão pequeno mas persistente ligando preço
e nota no Metacritic, que sobrevive ao controle estatístico.

Esses achados são a parte do projeto que resistiu à auditoria. Foram medidos com o teste
certo sobre dados reais, e eu os defenderia em qualquer conversa.

A parte de modelagem resistiu menos. Quatro camadas foram construídas, uma gera
recomendação, e três das métricas que publiquei não sobreviveram à revisão do próprio
código. O melhor resultado do projeto acabou não sendo um número, e sim o método que
permitiu descobrir que os números estavam errados: split temporal honesto, baseline
explícita, intervalo de confiança, e a disciplina de medir cada camada pelo trabalho que
ela deveria fazer.

Escrevi na primeira versão que "métricas muito boas são sinal de alerta". Levei sete meses
para aplicar isso às minhas.

---

> ✅ **MANTIDO — Ferramentas utilizadas**
>
> Copie do original sem alterações. Creditar Antigravity, DeepSeek, Claude e NotebookLM
> joga a seu favor, e fica mais coerente ainda depois da seção de auditoria: parte do que
> deu errado veio de aceitar código gerado sem verificar o que ele fazia. Se quiser, vale
> uma frase dizendo isso.

*[Cole aqui a seção "Ferramentas utilizadas" do original]*

---

> 🔧 **REESCRITO — Observações finais**
>
> A versão antiga atribuía ao limite do Supabase features que "já haviam sido tecnicamente
> implementadas e validadas". Isso não se sustenta: a calibração dinâmica de thresholds não
> teria efeito na saída, e a geração de meta-usuários nunca foi construída. Mantive a parte
> sobre o limite de armazenamento, que é verdadeira e interessante.

## Observações

Este projeto é uma prova de conceito, não um sistema em produção.

O plano gratuito do Supabase atingiu o limite de armazenamento — 0.565 GB contra 0.5 GB
disponíveis — depois de 10.000 usuários sintéticos e mais de 309.000 sessões. A partir daí
o banco passou a rejeitar escritas.

Curiosamente, essa limitação reflete um cenário comum de engenharia: decisões arquiteturais
adequadas nos estágios iniciais viram gargalo conforme o sistema cresce. Entender onde um
sistema quebra, e por quê, é tão valioso quanto saber construí-lo.

A auditoria acrescentou uma segunda versão dessa lição, menos confortável. O banco quebrou
de forma visível: uma mensagem de erro, um limite numérico, uma causa óbvia. As métricas
quebraram de forma silenciosa, e continuaram publicadas por sete meses parecendo corretas.

O código corrigido, os scripts de avaliação e os resultados desta auditoria estão no
repositório.
