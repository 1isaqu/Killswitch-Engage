# Relatório: Análise de Relações Condicionais (Steam EDA V2.5)

A análise multivariada profunda foi realizada sobre a base de dados tratada da Steam. Respondemos as maiores perguntas de negócio do usuário com validações estritas (ANOVA, Kruskal-Wallis, Exponencial Logarítmica e Correlação Parcial) prontas para tunar o "Gerador Sintético de Dados Mimetizados".

## 1. PREÇO POR CATEGORIA (ANOVA One-Way)
*   **O que fizemos:** Boxplots identificando variações drásticas.
*   **Ranking (Top):**
    *   Remote Play on Tablet ($41.53 ± 35.24)
    *   Steam Trading Cards ($39.27 ± 35.26)
*   **Ranking (Bottom):**
    *   In-App Purchases ($5.97 ± 18.78)
    *   MMO ($5.42 ± 17.43)
*   **Conclusão (Teste ANOVA):** `F-Stat: 318.99 | p-value: 0.00`.
*   **Interpretação:** Existem disparidades colossais de preços justificadas pela "Categoria" (Multi-player, MMO e In-App purchases tendem agressivamente para o lado Grátis, enquanto suporte a Cloud, Trading Cards ou Controllers tendem a custar caro - geralmente sendo marca base de jogos AAA/Indies Premiums).
*   **Recomendação Sistêmica:** O Price sintético DEVE ser distribuído dinamicamente ponderando o vetor de Categories do app associado.

![Boxplot Preço por Categoria](v2_smart_imputation/price_category_boxplot.png)

## 2. TEMPO MÉDIO POR GÊNERO (Kruskal-Wallis)
*   **O que fizemos:** Rankeamento de engajamento absoluto isolando nicks zerados.
*   **Top 5 Gêneros que mais "Viciam" o Player (Média Alta e Mediana Estável):**
    1.  Massively Multiplayer (Median: 372 mins)
    2.  RPG (Median: 304 mins)
    3.  Strategy (Median: 294 mins)
    4.  Simulation (Median: 238 mins)
    5.  Early Access / Adventure (Median ~220 mins)
*   **Teste de Kruskal-Wallis:** `H-Stat: 1464.03 | p-value: e-307`
*   **Interpretação:** "Estratégia e RPG viciam mais que Action/FPS?". Absolutamente, a diferença é matematicamente comprovada e intransponível. Jogadores de FPS (Action) giram as partidas rápido e largam jogos (Mediana de 210 mins e Média de 560).
*   **Recomendação Sistêmica:** O Gênero dita a equação principal para calcular o `Average playtime` nos modelos!

![Boxplot Tempo x Gênero](v2_smart_imputation/playtime_genre_boxplot.png)

## 3. AS MELHORES (E PIORES) DESENVOLVEDORAS (N ≥ 5)
*   **O que fizemos:** Cruzamento para estabilidade de Quality Score entre devs com mais de 5 jogos criados na Steam.
*   **O Padrão Ouro (Quem acerta quase sempre):** Playtouch (Média Positiva 19.7, N=10), Destructive Creations (Média 12.8, N=6). São casos de desenvolvedores cujos jogos despontam como incrivelmente bem recebidos.
*   **O Fundo do Poço:** "&y", "100 Cozy Games", "11 bit studios". Diversos desses estúdios entregaram volumes grandes de jogos péssimos ou irrelevantes que flertam com Média 0 de Engajamento.

## 4. CONQUISTAS E RETENÇÃO
*   **O que fizemos:** Plot discreto comparando se `Achievements` retém o jogador no game mais do que o normal (Pearson Linear & Spearman de Postos).
*   **Medianas:**
    *   Baixo (1-10 Conq.): 150 mins
    *   Médio (11-50 Conq.): 226 mins
    *   Alto (51-150 Conq.): 450 mins
    *   Extremo (151-500 Conq.): 593 mins
*   **Spearman Correlation = 0.332 (p=0.0)**
*   **Interpretação:** O Coeficiente de Postos de **Spearman provou que NÃO há linearidade perfeita entre jogar mais e ganhar mais Achievements (Pearson pífio de 0.018)**. E sim que o relacionamento ocorre por "Tiers". Há claras escadas motivadoras, onde jogos com "pacotes" grandes de conquistas dobram a mediana de sobrevida.
*   **Recomendação Sistêmica:** "Vale a pena investir em conquistas?" **MUITO.** O modelo de recomendação precisa incentivar jogos da Faixa 51-150 para usuários Hardcore (Eles retêm o jogador garantido acima das 7 horas).

![Barplot Conquistas](v2_smart_imputation/achievements_retention.png)

## 5. PREÇO VS NOTAS DA COMUNIDADE
*   **O que fizemos:** Confirmamos a desconfiança de que o valor monetário do jogo NÃO se reverte em alegria para o gamer e identificaremos se The Witcher 3 se iguala a DOTA na mentalidade comunitária.
*   **Percentual Positivo Médio da Base (The Score):**
    *   Gratuito: 44.9% vs Alto ($20-$60): 63.3% vs AAA ($60+): 68.9%
*   **ANOVA p-val: 0.0**
*   **Interpretação:** "Jogos caros realmente não são melhores?". **Mito derrubado!** Estatisticamente, jogos de maior precificação sofrem _menos hate_ em ratio. Jogos Grátis são bombardeados de Negative Reviews (44% de Score médio), provavelmente por barreira nula de entrada. Um jogo AAA premium garante ~69% de base aprovando-o.
*   **Recomendação Sistêmica:** O Sistema de Recomendação não deve rebaixar os Grátis apenas pela Nota Absoluta. Ele tem que "normalizá-la". O Grátis carrega um peso social tóxico nativo.

![Violinplot Price vs Score](v2_smart_imputation/price_vs_score.png)

## 6. POPULARIDADE VS VOLUME (Lei de Potência)
*   **Interpretação:** Analisamos logs Base10 do Limite de Jogadores versus Base10 das Reviews que eles geram.
*   **A Relação Real:** Calculou-se um R² de `0.5063` com Slope (Inclinação) log-log em `0.7794`.
*   **O que isso significa:** Existe uma forte lei de potência (power law) no engajamento por review da Steam. Quando uma base de usuários cresce em **10x** (ex: 10 mil para 100 mil compras), o volume verbalizado de Reviews recebidas **não escala na mesma velocidade** perfeitamente, assumindo ritmo levemente retardado. 
*   **Recomendador:** Os grandes jogos de massa ("The Last Of Us", "GTA") faturam com milhões de compras de jogadores "fantasmas", que nunca acessam para avaliar.

![Scatterplot de Popularidade por Volume](reports/figures/popularity_volume.png)

## 7. CORRELAÇÃO PARCIAL (Pingouin Multi-Variate)
Usamos o motor analítico para remover "efeitos parasitas" (fatores confundidores de estatística) em cima das features celtas:
*   **1. Preço vs Metacritic Score (CONTROLANDO FATOR "INDIE")**
    *   *Correlação de Pearson Líquida Anulando Viés de Estúdio Grande*: `r = 0.2038`.
```markdown
    *   *O que isso significa*: "Metacritic é paga?" Calma. Não é isso que os dados provam, mas também não é algo que os dados descartam. Ao analisar a correlação entre preço e nota no Metacritic, controlando o fator "indie vs grande estúdio", encontramos uma associação positiva (r ≈ 0.20). Estatisticamente, isso significa que jogos mais caros tendem a receber notas ligeiramente maiores — mesmo quando removemos parte do viés estrutural de orçamento e escala. Isso não demonstra suborno nem manipulação deliberada, mas demonstra uma tendência. Por que jogos AAA parecem sofrer menos penalidade crítica? Será apenas maior polimento técnico ou existe um viés cognitivo inconsciente? Críticos são humanos e expectativas moldam a percepção. Um jogo de 200 milhões de dólares carrega peso cultural, hype e pressão de mercado; inconscientemente, isso pode influenciar o julgamento por contexto psicológico. A estatística não acusa nem inocenta automaticamente, ela apenas mostra um padrão pequeno e consistente acima do ruído. No fim, talvez a pergunta correta não seja "Metacritic é paga?", mas sim: até que ponto expectativas e escala influenciam a crítica — mesmo quando a intenção é ser neutra?
```
*   **2. Conquistas vs Playtime (CONTROLANDO FATOR "METACRITIC SCORE")**
    *   *Correlação*: `r = 0.0026`.
    *   *O que isso significa*: Fazer achievements num jogo ruim e fazer achievements num jogo absurdamente excelente (Nota 90 do Metacritic) GERA rigorosamente o NULO / MESMO efeito linear na vida retida de tela do usuário! Um jogo porco não é ignorado pela falta de achievements se você tirar a métrica pura! O Fator vício é indiferente às qualidades técnicas das features subjacentes.
