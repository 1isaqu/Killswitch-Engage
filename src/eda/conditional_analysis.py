import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pingouin as pg
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

OUTPUT_DIR = "reports/figures"
DATA_PATH = os.path.join("data/processed", "imputed_dataset_v2.csv")
REPORT_PATH = os.path.join("reports", "multivariate_report.txt")

# Configuração visual do seaborn
sns.set_theme(style="whitegrid", context="talk")

def write_report(text):
    logging.info(text)
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def explode_multivalue(df, column):
    """Separa variáveis conectadas por vírgula em linhas distintas."""
    return df.assign(**{column: df[col].str.split(',') for col in [column]}).explode(column)

def q1_price_by_category(df):
    write_report("\n" + "="*50 + "\n1. PREÇO MÉDIO POR CATEGORIA\n" + "="*50)
    
    # Prepara dataset
    df_cat = explode_multivalue(df[['Categories', 'Price']].dropna(), 'Categories')
    df_cat['Categories'] = df_cat['Categories'].str.strip()
    
    # Remover categorias raras para robustez (< 500 registros)
    counts = df_cat['Categories'].value_counts()
    valid_cats = counts[counts > 500].index
    df_cat = df_cat[df_cat['Categories'].isin(valid_cats)]
    
    # Limita outliers extremos (95th percentile) para não achatar o gráfico
    cap = df_cat['Price'].quantile(0.95)
    df_cat_plot = df_cat[df_cat['Price'] <= cap]

    # Estatísticas
    stats_df = df_cat.groupby('Categories')['Price'].agg(['mean', 'median', 'std', 'count']).sort_values('mean', ascending=False)
    write_report("\n[Estatísticas de Preço por Categoria]")
    write_report(stats_df.round(2).to_string())

    # ANOVA
    groups = [group['Price'].values for name, group in df_cat.groupby('Categories')]
    f_stat, p_val = stats.f_oneway(*groups)
    write_report(f"\n[Teste ANOVA] F-Stat: {f_stat:.2f} | p-value: {p_val:.4e}")
    write_report("Conclusão: Existem diferenças SGNIFICATIVAS nos preços entre categorias." if p_val < 0.05 else "Conclusão: NENHUMA diferença significativa de preço entre as categorias.")

    # Plot
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df_cat_plot, y='Categories', x='Price', order=stats_df.index, fliersize=1)
    plt.title('Distribuição de Preço por Categoria (Top 95%)')
    plt.xlabel('Preço (USD)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "price_category_boxplot.png"))
    plt.close()

def q2_playtime_by_genre(df):
    write_report("\n" + "="*50 + "\n2. TEMPO DE JOGO POR GÊNERO\n" + "="*50)
    
    # Prepara dataset (Avg playtime > 0 para isolar engajamento real)
    df_gen = explode_multivalue(df[['Genres', 'Average playtime forever']].dropna(), 'Genres')
    df_gen['Genres'] = df_gen['Genres'].str.strip()
    df_gen = df_gen[df_gen['Average playtime forever'] > 0]
    
    counts = df_gen['Genres'].value_counts()
    valid_gens = counts[counts > 500].index
    df_gen = df_gen[df_gen['Genres'].isin(valid_gens)]

    # Kruskal-Wallis (Não paramétrico)
    groups = [group['Average playtime forever'].values for name, group in df_gen.groupby('Genres')]
    h_stat, p_val = stats.kruskal(*groups)
    
    stats_df = df_gen.groupby('Genres')['Average playtime forever'].agg(['mean', 'median', 'count']).sort_values('median', ascending=False)
    write_report("\n[Retenção (Em Minutos) por Gênero]")
    write_report(stats_df.round(1).to_string())
    write_report(f"\n[Kruskal-Wallis] H-Stat: {h_stat:.2f} | p-value: {p_val:.4e}")

    # Plot
    cap = df_gen['Average playtime forever'].quantile(0.90)
    df_plot = df_gen[df_gen['Average playtime forever'] <= cap]
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df_plot, y='Genres', x='Average playtime forever', order=stats_df.index[:15], fliersize=1)
    plt.title('Top 15 Gêneros por Mediana de Retenção (Abaixo do P90)')
    plt.xlabel('Minutos Jogados')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "playtime_genre_boxplot.png"))
    plt.close()

def q3_score_by_developer(df):
    write_report("\n" + "="*50 + "\n3. CONSISTÊNCIA DE NOTAS POR DESENVOLVEDOR\n" + "="*50)
    
    dev_stats = df.groupby('Developers')['User score'].agg(['mean', 'std', 'count']).dropna()
    dev_gt_5 = dev_stats[dev_stats['count'] >= 5].copy()
    
    # Rank por consistência (Desvio padrão x Média de Qualidade)
    # Procuramos os que têm alta média E baixo desvio
    top_devs = dev_gt_5.sort_values(['mean', 'std'], ascending=[False, True]).head(10)
    worst_devs = dev_gt_5.sort_values(['mean', 'std'], ascending=[True, True]).head(10)
    
    write_report("\n[Top 10 Melhores Desenvolvedores (Alta Média, N >= 5)]")
    write_report(top_devs.round(2).to_string())
    
    write_report("\n[Top 10 Piores Desenvolvedores (Baixa Média, N >= 5)]")
    write_report(worst_devs.round(2).to_string())

def q4_achievements_retention(df):
    write_report("\n" + "="*50 + "\n4. CONQUISTAS E RETENÇÃO\n" + "="*50)
    
    df_ach = df[['Achievements', 'Average playtime forever']].dropna().copy()
    df_ach = df_ach[(df_ach['Achievements'] > 0) & (df_ach['Achievements'] <= 500)]
    df_ach = df_ach[df_ach['Average playtime forever'] > 0]
    
    pearson_r, p_pearson = stats.pearsonr(df_ach['Achievements'], df_ach['Average playtime forever'])
    spearman_r, p_spearman = stats.spearmanr(df_ach['Achievements'], df_ach['Average playtime forever'])
    
    write_report(f"Correlação de Pearson (Linear): {pearson_r:.3f} (p={p_pearson:.3e})")
    write_report(f"Correlação de Spearman (Posts): {spearman_r:.3f} (p={p_spearman:.3e})")
    
    # Discretizando conquistas
    df_ach['Achiev_Tier'] = pd.cut(df_ach['Achievements'], bins=[0, 10, 50, 150, 500], labels=['Baixo (1-10)', 'Médio (11-50)', 'Alto (51-150)', 'Extremo (151-500)'])
    
    tier_stats = df_ach.groupby('Achiev_Tier')['Average playtime forever'].agg(['median', 'mean'])
    write_report("\n[Tempo de Jogo Médio por Faixa de Conquista]")
    write_report(tier_stats.round(1).to_string())
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_ach, x='Achiev_Tier', y='Average playtime forever', estimator=np.median, errorbar=None)
    plt.title('Mediana de Tempo de Jogo vs Faixa de Conquistas')
    plt.ylabel('Mediana Mins. Jogados')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "achievements_retention.png"))
    plt.close()

def q5_price_vs_score(df):
    write_report("\n" + "="*50 + "\n5. PREÇO vs NOTA GERAL\n" + "="*50)
    
    df_ps = df[['Price', 'Positive', 'Negative', 'User score']].dropna().copy()
    df_ps['Score'] = df_ps['Positive'] / (df_ps['Positive'] + df_ps['Negative']).replace(0, 1)
    
    df_ps['Price_Tier'] = pd.cut(df_ps['Price'], bins=[-1, 0, 5, 20, 60, 500], labels=['Gratuito', 'Baixo (<$5)', 'Médio ($5-20)', 'Alto ($20-60)', 'AAA ($60+)'])
    
    tier_stats = df_ps.groupby('Price_Tier')['Score'].agg(['mean', 'median', 'std', 'count'])
    write_report("\n[Ratio Positivo Médio por Faixa de Preço]")
    write_report(tier_stats.round(4).to_string())
    
    groups = [group['Score'].values for name, group in df_ps.groupby('Price_Tier') if len(group)>0]
    f_stat, p_val = stats.f_oneway(*groups)
    write_report(f"\n[ANOVA Preço/Nota] F-Stat: {f_stat:.2f} | p-value: {p_val:.4e}")
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df_ps, x='Price_Tier', y='Score', inner='quartile')
    plt.title('Distribuição da Nota de Jogadores vs Preço')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "price_vs_score.png"))
    plt.close()

def q6_popularity_vs_volume(df):
    write_report("\n" + "="*50 + "\n6. POPULARIDADE vs VOLUME (LOG-LOG)\n" + "="*50)
    
    df_pop = df[['Estimated owners', 'Positive', 'Negative']].dropna().copy()
    df_pop['Total_Reviews'] = df_pop['Positive'] + df_pop['Negative']
    
    # Já convertemos em Integer na etapa da Pipeline EDA anterior
    df_pop['Owners_Mean'] = pd.to_numeric(df_pop['Estimated owners'], errors='coerce')
    df_pop = df_pop[(df_pop['Total_Reviews'] > 0) & (df_pop['Owners_Mean'] > 0)]
    
    # Transformação logarítmica
    df_pop['Log_Owners'] = np.log10(df_pop['Owners_Mean'])
    df_pop['Log_Reviews'] = np.log10(df_pop['Total_Reviews'])
    
    # Regressão simples
    slope, intercept, r_value, p_value, std_err = stats.linregress(df_pop['Log_Owners'], df_pop['Log_Reviews'])
    
    write_report(f"\nR² Linear no espaço Log: {r_value**2:.4f}")
    write_report(f"Inclinação (Exponencial da relação): {slope:.4f}")
    
    plt.figure(figsize=(8, 8))
    sns.regplot(data=df_pop, x='Log_Owners', y='Log_Reviews', scatter_kws={'alpha': 0.1, 's': 1}, line_kws={'color': 'red'})
    plt.title('Lei de Potência: Donos Estimados vs Reviews Recebidos (Log10)')
    plt.xlabel('Log10(Média de Donos Estimados)')
    plt.ylabel('Log10(Total de Reviews)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "popularity_volume.png"))
    plt.close()

def q7_multivariate_partial(df):
    write_report("\n" + "="*50 + "\n7. ANÁLISE MULTIVARIADA (CORRELAÇÃO PARCIAL)\n" + "="*50)
    
    df_multi = df[['Price', 'Metacritic score', 'Average playtime forever', 'Achievements']].dropna().copy()
    
    # Tratamento imune com object-series 
    df_multi['Is_Indie'] = df.loc[df_multi.index, 'Genres'].astype(str).str.contains('Indie', case=False, na=False).astype(float)
    
    # Controlamos Preço vs Metacritic pelo Fato do Jogo ser Indie ou não. (Partial Covariate must be float)
    pc_price_score = pg.partial_corr(data=df_multi, x='Price', y='Metacritic score', covar='Is_Indie')
    
    # Controlamos Progressão vs Retenção pelo Fator de Qualidade (Score)
    pc_achiev_time = pg.partial_corr(data=df_multi, x='Achievements', y='Average playtime forever', covar='Metacritic score')
    
    write_report("\n[Preço vs Metacritic Score (Controlando: Ser 'Indie')]")
    write_report(pc_price_score.round(4).to_string())
    
    write_report("\n[Conquistas vs Playtime (Controlando: Metacritic Score)]")
    write_report(pc_achiev_time.round(4).to_string())

def execute_analysis():
    # Limpa relatório antigo
    if os.path.exists(REPORT_PATH):
        os.remove(REPORT_PATH)
        
    write_report("RELATÓRIO DE ANÁLISE DE RELACIONAMENTOS (PIPELINE V2)")
    write_report("Arquivo Base: " + DATA_PATH + "\n")
    
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    q1_price_by_category(df)
    q2_playtime_by_genre(df)
    q3_score_by_developer(df)
    q4_achievements_retention(df)
    q5_price_vs_score(df)
    q6_popularity_vs_volume(df)
    q7_multivariate_partial(df)
    
    write_report("\nFIM DO RELATÓRIO.")

if __name__ == "__main__":
    execute_analysis()
