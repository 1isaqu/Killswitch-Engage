# Killswitch Engage: Steam Intelligence & Recommender 🎮

Bem-vindo ao **Killswitch Engage**, um projeto de ponta focado em Transformação de Dados, Análise Multivariada e Sistemas de Recomendação aplicados ao ecossistema da Steam.

## 🚀 Visão Geral
Este projeto evoluiu de uma análise exploratória básica para uma infraestrutura de produção robusta, integrando Machine Learning para reparo de dados e uma API escalável servindo mais de **120.000 títulos**.

## 🛠️ Stack Tecnológica
- **Data Science:** Python, Pandas, Scikit-Learn (KNN/Naive Bayes), Scipy, Pingouin.
- **Backend:** FastAPI, Uvicorn, Pydantic v2.
- **Database:** Supabase (PostgreSQL Cloud) com Pooler Transacional.
- **Engenharia:** Docker, Git, Regex Avançado.

## 📂 Estrutura do Projeto
- `backend/app/`: Núcleo da API (Rotas, Modelos e Lógica).
- `src/data_preparation/`: Pipeline de Imputação Inteligente (ML).
- `src/eda/`: Scripts de Análise Condicional e Estatística.
- `data/processed/`: Dataset final higienizado.
- `reports/insights/`: Laudos técnicos das descobertas de dados.
- `brain/`: Memória arquitetural e walkthroughs das fases.

## 🏁 Roadmap de Fases
- **[x] Fase 1 & 2: Data Engineering:** Pipeline de imputação e limpeza (`eda_imputation_pipeline.py`).
- **[x] Fase 3: Backend & Cloud:** Integração com Supabase e API funcional (`backend/`).
- **[ ] Fase 4: Inteligência:** Sistema de Recomendação (Filtragem Colaborativa).

## ⚙️ Como Executar

### 1. Requisitos
```bash
pip install -r requirements.txt
```

### 2. Ingestão de Dados (Opcional)
Se precisar popular o banco do zero:
```bash
python src/data_preparation/populate_supabase.py
```

### 3. Subir o Servidor
```bash
cd backend
python -m uvicorn app.main:app --reload
```
Acesse os Docs interativos em: `http://localhost:8000/docs`

## 📊 Documentação Técnica
Para entender as entranhas do projeto, consulte:
- [DECISOES_TECNICAS.md](file:///C:/Users/isaqu/.gemini/antigravity/brain/cf95126f-d5a5-4b09-b260-13a80bf5bb28/DECISOES_TECNICAS.md): Trade-offs e Escolhas.
- [project_evolution.md](file:///C:/Users/isaqu/.gemini/antigravity/brain/cf95126f-d5a5-4b09-b260-13a80bf5bb28/project_evolution.md): História dos Achievements.
- [multivariate_insights.md](file:///c:/Users/isaqu/.gemini/antigravity/scratch/steam_analysis/reports/insights/multivariate_insights.md): Ouro analítico.

---
*Killswitch Engage - Transformando dados brutos em decisões inteligentes.*
