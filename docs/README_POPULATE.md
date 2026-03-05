# Guia de População: Supabase Cloud 🚀

Este script (`populate_supabase.py`) é o motor de ingestão responsável por transformar nosso dataset estático em um banco de dados vivo e relacional.

## 📋 Pré-requisitos
- Python 3.9+
- Acesso ao PostgreSQL do seu projeto Supabase.
- Dependências instaladas: `pip install asyncpg pandas numpy tqdm python-dotenv`

## ⚙️ Configuração
1. Copie o arquivo `.env.example_populate` para um novo arquivo chamado `.env`.
2. Insira sua URL de conexão (encontrada no Supabase em *Settings > Database > Connection string > URI*).
   > **Nota:** Use o modo de conexão direta (porta 5432) para maior velocidade na ingestão inicial.

## 🧩 O que o script faz?
1. **Dados Reais:** Insere Categorias, Desenvolvedores e Jogos (80k+) diretamente do `imputed_dataset_final.csv`.
2. **Geração Sintética:**
   - Cria **10.000 usuários** com perfis geográficos e etários realistas.
   - Gera uma **Biblioteca** de jogos para cada usuário baseada na popularidade real do jogo na Steam.
   - Simula **Sessões de Jogo** e **Avaliações** (Reviews), calibrando a nota do usuário com base na recepção real do título (Positive/Negative Ratio).
   - Registra conquistas alcançadas com base no tempo de jogo.

## ⏱️ Estimativa de Execução
- **Processamento local:** ~30 segundos.
- **Inserção no Supabase (Remoto):** Entre **15 a 40 minutos**, dependendo da latência da sua conexão e do limite de recursos do seu tier do Supabase. O script usa inserção em lotes (*batch size 500*) para otimizar o tempo.

## 🚀 Execução
```bash
python populate_supabase.py
```
Observe a barra de progresso no terminal para acompanhar o status de cada tabela.
