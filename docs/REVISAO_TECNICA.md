## 🧪 Revisão Técnica Completa – Killswitch Engage / GameVerse

Data: 2026-02-28  
Escopo: `src/`, `backend/app/`, `scripts/`, pipeline de dados e consultas SQL.

---

## 1. Inconsistências lógicas

### 1.1 Nome dos modelos vs implementação real (Alta)
- **Local**: `scripts/train_layer2_clustering.py` (L16–L88), `backend/app/services/recomendador.py` (L23–L35), `backend/app/routes/recomendacoes.py` (L9–L12)
- **Problema**: 
  - O arquivo de saída do clusterizador se chama `hdbscan_model.pkl` (L14), mas o código usa **KMeans**, não HDBSCAN.
  - A rota de recomendações documenta “Random Forest + KMeans + SVD”, enquanto o serviço carrega `classificador_rf.pkl`, `hdbscan_model.pkl` e `lightfm_model.pkl`.
- **Impacto**: Confusão arquitetural e risco de documentação desatualizada; dificulta debugging e onboarding.
- **Sugestão**: 
  - Padronizar nomes de arquivos, variáveis e docstrings para refletir o algoritmo real (ex.: `kmeans_clusters.pkl` ou renomear para um nome neutro, tipo `user_clusters.pkl`).
  - Atualizar docstring da rota para bater com o stack efetivo.
- **Ganho estimado**: **Alta legibilidade/manutenibilidade**, reduz risco de bugs conceituais.

### 1.2 Uso parcial de camadas planejadas no recomendador (Média)
- **Local**: `backend/app/services/recomendador.py` (L47–L83), `scripts/train_layer3_ranker.py`
- **Problema**: 
  - O serviço de recomendação utiliza apenas o ranker SVD (`lightfm_model.pkl`) e ignora o classificador RF e o clusterizador na lógica de `get_recomendacoes`.
  - O carregamento das três camadas existe (L33–L35), mas a lógica atual não faz reranking ou filtragem por cluster/qualidade.
- **Impacto**: Divergência entre design “multi-camada” e o que é realmente executado; potencial desperdício de treino de modelos auxiliares.
- **Sugestão**: 
  - Documentar explicitamente que a primeira versão está usando apenas a camada de SVD (MVP).
  - Quando for refatorar, incorporar o RF como filtro de qualidade e o clusterizador para personalização fina, ou remover modelos não usados.
- **Ganho estimado**: **Clareza arquitetural** e melhor alinhar expectativas de resultados.

### 1.3 Cluster data carregado e não usado (Baixa)
- **Local**: `scripts/train_layer3_ranker.py` (L19)
- **Problema**: `cluster_data = joblib.load(CLUSTERING_MODEL)` é carregado, mas nunca utilizado.
- **Impacto**: Dependência desnecessária em artefato externo; aumenta acoplamento sem benefício.
- **Sugestão**: 
  - Documentar intenção futura (ex.: “usar clusters como feature”) ou remover o carregamento se não for usado.
- **Ganho estimado**: **Pequena melhoria de clareza**.

### 1.4 Geração de sessões com horas calculadas mas não usadas (Baixa)
- **Local**: `src/data_preparation/populate_supabase_v2.py` (L92–L117, especialmente L116–L119)
- **Problema**: Variável `horas` é calculada (L116) mas não é persistida; apenas `inicio` é inserido em `sessoes_jogo`.
- **Impacto**: Código morto/redundante; pode confundir sobre a origem de `horas_jogadas` (que hoje é tratada como coluna derivada no Supabase).
- **Sugestão**: 
  - Documentar explicitamente que `horas_jogadas` é uma coluna `GENERATED`/derivada no banco, ou ajustar o insert para preencher a coluna se fizer parte do schema.
- **Ganho estimado**: **Melhor entendimento do pipeline de dados**.

### 1.5 Scripts dependentes de paths absolutos locais (Média)
- **Local**:
  - `src/data_preparation/eda_imputation_pipeline.py` (L30–L32, `DATASET_PATH = r"D:\datasets\steam\games.csv"`)
  - `src/data_preparation/preprocess_data.py` (L72–L73)
  - `src/validation/validate_kstest.py` (L104–L105)
  - `src/eda/analyze_games.py` (L50–L52)
- **Problema**: Vários scripts críticos amarrados a um caminho absoluto de Windows (`D:\datasets\steam\games.csv`).
- **Impacto**: Baixa portabilidade e maior fricção ao rodar em outros ambientes (Linux/Docker/colaboradores).
- **Sugestão**: 
  - Centralizar esses paths em variáveis de ambiente ou config (ex.: `.env` ou `config.py`) e usar caminhos relativos à raiz do projeto onde possível.
- **Ganho estimado**: **Média** em manutenibilidade e facilidade de deploy.

### 1.6 Exclusão total de sessões em repovoamento (Média)
- **Local**: `src/data_preparation/populate_supabase_v2.py` (L60–L62)
- **Problema**: `DELETE FROM sessoes_jogo` é executado sem filtros antes de recriar todas as sessões.
- **Impacto**: Perda total de histórico se rodado em ambiente que já acumula dados reais.
- **Sugestão**: 
  - Manter este script explicitamente marcado como “apenas para ambientes de dev/seed” e, em produção, trabalhar com flags ou tabelas de versão.
- **Ganho estimado**: **Alta segurança operacional** em ambientes multi-tenant.

---

## 2. Redundâncias e operações desnecessárias

### 2.1 Paths e lógica repetidos para o mesmo dataset (Baixa)
- **Local**: `preprocess_data.py`, `eda_imputation_pipeline.py`, `validate_kstest.py`, `analyze_games.py`
- **Problema**: Cada script reimplementa carregamento do mesmo CSV, conversões básicas e paths absolutos.
- **Impacto**: Duplicação de conhecimento sobre onde está o dataset e como ele deve ser lido.
- **Sugestão**: 
  - Futuro: extrair uma função utilitária única de “load_raw_steam_dataset(config)” e usá-la em todos os scripts de EDA/validação.
- **Ganho estimado**: **Baixa/média** em DRY e facilidade de mudança.

### 2.2 `cluster_data` e dependência em `hdbscan_model.pkl` (Baixa)
- **Local**: `scripts/train_layer3_ranker.py` (L19)
- **Problema**: Carregamento de um modelo que não participa do pipeline do ranker.
- **Impacto**: I/O desnecessário e acoplamento sem uso prático.
- **Sugestão**: 
  - Ver 1.3 – remover ou justificar via comentário/documentação.

### 2.3 Conversões repetidas com `eval` para listas (Média)
- **Local**:
  - `scripts/train_layer1_classifier.py` (L43–L47)
  - `scripts/train_layer2_clustering.py` (L25–L27)
- **Problema**: Uso de `eval(x)` repetidamente para converter strings em listas de categorias.
- **Impacto**: 
  - Potencial overhead de CPU com evals repetidas em grandes datasets.
  - Superfície de ataque (se o CSV vier de fonte não-confiável).
- **Sugestão**: 
  - Padronizar para um parser seguro (ex.: `ast.literal_eval`) e idealmente salvar esse pré-processamento em disco para reuso.
- **Ganho estimado**: **Média** em segurança e leve ganho de performance em datasets grandes.

---

## 3. Maus hábitos de código

### 3.1 Credenciais hard-coded em script de teste (Crítico)
- **Local**: `src/validation/test_supabase_auth.py` (L10–L13)
- **Problema**: Usuário, senha e host do Supabase embutidos em código.
- **Impacto**: 
  - **Vazamento direto de credenciais** se o projeto for para um repo público.
  - Risco de acesso indevido ao banco real.
- **Sugestão**: 
  - Remover credenciais do código e usar `.env` mesmo em scripts de teste.
  - Tratar esse script como sensível e nunca comitar com secrets reais.
- **Ganho estimado**: **Altíssimo em segurança**.

### 3.2 SSL sem verificação de certificado (Média)
- **Local**: `backend/app/models/database.py` (L14–L24)
- **Problema**: `ctx.check_hostname = False` e `ctx.verify_mode = ssl.CERT_NONE`.
- **Impacto**: Conexão com DB vulnerável a ataques man-in-the-middle quando exposto fora de rede segura.
- **Sugestão**: 
  - Em produção, usar certificado verificado (CA) e checagem de hostname; manter esse modo apenas para dev/local se necessário.
- **Ganho estimado**: **Alta segurança** em ambientes cloud.

### 3.3 Defaults sensíveis em configuração (Baixa/Média)
- **Local**: `backend/app/config.py` (L13–20)
- **Problema**:
  - `DATABASE_URL` default com `postgres:postgres` embutido.
  - `SECRET_KEY` default fixo (`"super-secret-key-321"`).
- **Impacto**: Se variáveis de ambiente não forem definidas corretamente, a API pode subir com credenciais fracas.
- **Sugestão**: 
  - Tornar obrigatórios em produção (`DEBUG=False`) e usar somente defaults em dev.
- **Ganho estimado**: **Média** em boas práticas de segurança.

### 3.4 Testes aceitando 500 como “sucesso” (Baixa)
- **Local**: 
  - `backend/tests/test_usuarios.py` (L6–15)
  - `backend/tests/test_jogos.py` (L11–16)
- **Problema**: Asserções do tipo `assert response.status_code in [200, 500]`.
- **Impacto**: Testes não detectam regressões reais; 500 passa “como ok”.
- **Sugestão**: 
  - Deixar claro que estes são apenas “smoke tests manuais” e criar uma segunda suíte com mocks/overrides que exija 200 em ambiente de CI.
- **Ganho estimado**: **Média** em confiabilidade de testes.

### 3.5 Magic numbers espalhados (Baixa/Média)
- **Exemplos**:
  - `train_layer1_classifier.py`: threshold de `avaliacao_media >= 4.0` (L25).
  - `train_layer2_clustering.py`: range fixo `K=3..10` (L62–L66).
  - `eda_imputation_pipeline.py`: tamanhos de amostra (2000, 5000), bins fixos para faixas.
  - `populate_supabase.py`: amostra de top 2000 jogos, 5–15 jogos por usuário, 5000 batch size.
- **Impacto**: Ajustar comportamento exige caça manual desses valores.
- **Sugestão**: 
  - Promover valores de negócio para constantes nomeadas no topo dos arquivos (ex.: `TOP_GAMES_LIMIT`, `BATCH_SIZE`, `GOOD_GAME_THRESHOLD`).
- **Ganho estimado**: **Média** em legibilidade.

---

## 4. Gargalos de performance e oportunidades

### 4.1 Silhouette score em full dataset no clustering (Alta – offline)
- **Local**: `scripts/train_layer2_clustering.py` (L56–L72)
- **Problema**: 
  - Loop de K=3..10, cada um com `KMeans.fit_predict` + `silhouette_score` usando todos os pontos de `X_pca`.
  - `silhouette_score` é O(n²) em geral; para milhares de usuários, o custo explode.
- **Impacto (antes)**:
  - Treino de clustering pode demorar **minutos a horas** em bases maiores.
- **Sugestão de otimização (conceitual)**:
  - Calcular `silhouette_score` em uma **amostra estratificada** de usuários (ex.: 5–10k) ou usar outras métricas de escolha de K (elbow, BIC/AIC).
  - Persistir `X_pca` e reusar entre experimentos para evitar recomputar.
- **Ganho esperado**: 
  - Redução de tempo de treino do clusterizador em **1 ordem de grandeza+** em datasets médios/grandes.

### 4.2 EDA/imputação rodando tudo em um único script monolítico (Média – offline)
- **Local**: `src/data_preparation/eda_imputation_pipeline.py` (L55–417)
- **Problema**: 
  - Várias etapas pesadas (TF-IDF, KNN, cosine similarity) rodando sequencialmente no dataset inteiro.
  - Não há checkpointing intermediário nem uso de cache para vetores TF-IDF.
- **Impacto (antes)**:
  - Reexecução completa do script é cara mesmo para mudanças pontuais.
- **Sugestão**:
  - Dividir em etapas reutilizáveis (limpeza, imputação de publishers, imputação de genres/tags, ensemble, fallback) com salvamento de artefatos intermediários.
  - Usar representações esparsas/Tfidf com cuidado para não recalcular tudo em re-runs.
- **Ganho esperado**: 
  - Iteração mais rápida em experimentos de EDA/imputação; potencialmente redução de tempo de execução total em **30–50%** com cache simples.

### 4.3 Geração de sessões sintéticas em duplo loop (Média)
- **Local**: `populate_supabase_v2.py` (L88–121)
- **Problema**: Loop sobre todos os usuários, definindo `n_sessions` e depois um loop interno por sessão.
- **Impacto (antes)**:
  - Para 10k usuários * 50–200 sessões (hardcore/edge), volume de iterações pode chegar a milhões; na prática ainda aceitável, mas está próximo do limite se escalar usuários.
- **Sugestão**:
  - Manter como está para o objetivo atual (seed offline), mas considerar:
    - Gerar parte das sessões via vetorização/pandas (por exemplo, sortear jogos e horários via DataFrame).
    - Controlar número máximo de sessões globais via parâmetro.
- **Ganho esperado**: Pequeno/médio se o número de usuários crescer muito.

### 4.4 `validate_kstest` carregando datasets inteiros (Baixa)
- **Local**: `src/validation/validate_kstest.py` (L18–21)
- **Problema**: Carrega datasets completos para comparar distribuições quando o objetivo é apenas sanity check.
- **Sugestão**:
  - Em cenários de produção, considerar amostragem (ex.: 100k linhas) antes do KS-Test.
- **Ganho esperado**: Menor consumo de memória em bases muito grandes.

---

## 5. Segurança

### 5.1 Exposição de credenciais de banco (Crítico)
- Ver item 3.1 – **prioridade máxima** apagar/rotacionar essas credenciais e garantir que nunca vão para repos público.

### 5.2 Defaults sensíveis em config (Média)
- Ver item 3.3.

### 5.3 Logs e dados sensíveis (Baixa)
- **Local**: `populate_supabase.py`, `populate_supabase_v2.py`
- **Problema**: Logs são genéricos, sem PII. Não há evidência de armazenamento de dados sensíveis em logs.
- **Conclusão**: Sem problemas relevantes aqui no estado atual.

---

## 6. Manutenibilidade

### 6.1 Testes de API muito superficiais (Média)
- **Local**: `backend/tests/test_usuarios.py`, `backend/tests/test_jogos.py`
- **Problema**: Testes não cobrem:
  - Rotas analíticas, recomendações, validações de entrada.
  - Cenários de erro mais ricos (ex.: IDs inválidos, usuário sem biblioteca).
- **Sugestão**:
  - Criar uma camada de testes que use `dependency_overrides` do FastAPI para mockar o `Database` e validar contratos de resposta sem exigir DB real.

### 6.2 Scripts de EDA/validação sem docstrings padrão (Baixa)
- **Local**: vários scripts em `src/eda/` e `src/validation/`.
- **Problema**: Falta de docstrings mais formais e breves explicações de parâmetros de linha de comando (quando houver).
- **Sugestão**:
  - Adicionar docstrings em formato curto (`Args/Returns`) nas funções de entrada principais.

### 6.3 Acoplamento fraco entre service layer e modelos (Baixa/Média)
- **Local**: `backend/app/services/recomendador.py`
- **Problema**: Estrutura de dicionário com campos soltos (`user_embeddings`, `item_embeddings`, `user_map` etc.) em vez de uma classe/DTO com tipos fortes.
- **Sugestão**:
  - No futuro, encapsular o ranker em uma classe própria para evitar erros de chave e facilitar trocas de implementação.

---

## 7. Oportunidades de otimização estrutural

### 7.1 Pré-cálculo offline vs runtime
- **Candidatos a pré-computação**:
  - **Embeddings de usuário/jogo** já estão pré-computados (bom).
  - Poderia haver:
    - Tabelas materializadas para TOP jogos por cluster/segmento de usuário.
    - Tabelas resumo para analíticos (`preco_por_categoria`, `tempo_por_genero`) recalculadas diariamente, em vez de calcular sempre via queries pesadas.
- **Ganho esperado**: Menor carga em tempo real e respostas mais rápidas para dashboards.

### 7.2 Cache na camada de API (Média)
- **Local**: `backend/app/routes/analiticos.py`
- **Problema**: Consultas como `/analiticos/preco_por_categoria` e `/analiticos/tempo_por_genero` são naturalmente **lentas mas estáveis no tempo**.
- **Sugestão**:
  - Usar o `REDIS_URL` já configurado em `config.py` para cachear as respostas por alguns minutos/horas.
- **Ganho esperado**: Latência muito menor sob carga repetida.

### 7.3 Índices sugeridos no banco (Alta para produção)
- **Com base nas queries em `queries.py`**:
  - `jogos_categorias(jogo_id, categoria_id)` e `jogos_categorias(categoria_id, jogo_id)`.
  - `sessoes_jogo(usuario_id, jogo_id, fim)` para queries de biblioteca/estatísticas e tendências.
  - `categorias(nome)` para filtro `WHERE c.nome ILIKE $1`.
- **Ganho esperado**: Redução de tempo de queries de segundos para milissegundos em bases maiores.

---

## 8. Checklist de boas práticas

### 8.1 Implementadas
- **Separação de camadas**: backend (FastAPI) + serviços + camada de dados clara.
- **Uso de pools de conexão** (`asyncpg.create_pool`).
- **Consultas parametrizadas** com placeholders (`$1`, `$2`) – risco baixo de SQL injection.
- **Pydantic v2 para schemas de resposta**.
- **RotatingFileHandler** para logs de API (`backend/app/utils/logging.py`).
- **Scripts de validação de schema e KS-Test** para garantir sanidade dos dados imputados.

### 8.2 Pendentes / parcialmente implementadas
- **Segurança**:
  - Remoção de credenciais hard-coded.
  - Fortalecer SSL e defaults de config em produção.
- **Testes**:
  - Aumentar cobertura e deixar 500 explicitamente como falha em testes automatizados.
- **Configuração**:
  - Eliminar paths absolutos e mover para configurações centralizadas.
- **Performance**:
  - Otimizar escolha de K em clustering, amostragem em validações estatísticas, possível caching em rotas analíticas.
- **Documentação**:
  - Expandir docstrings em scripts EDA/validação.

---

### Resumo executivo

O projeto está **bem estruturado** para um ambiente de portfólio/POC avançada: boa separação de camadas, uso consciente de pools e consultas agregadas, e um pipeline de EDA/imputação estatisticamente robusto.  
Os principais pontos de atenção são: **segurança (credenciais hard-coded, SSL afrouxado)**, **performance em scripts offline (clustering/EDA)** e **clareza arquitetural no sistema de recomendação (nomes de modelos vs algoritmos reais e camadas ainda não completamente integradas)**.  
Com poucos ajustes nessas frentes, o projeto fica pronto para ser apresentado como um case sólido de engenharia de dados + backend + ML de produção.
