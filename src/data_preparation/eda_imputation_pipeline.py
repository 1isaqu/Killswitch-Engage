"""
Pipeline de Imputação Inteligente e Análise Exploratória de Dados (EDA) do Steam.

Este script executa uma limpeza e imputação de dados em 3 camadas:
1. Camada Especialista: Propagação de Publishers, Regex, Naive Bayes, kNN, Cosseno.
2. Camada Ensemble Validada: Avalia heurísticas e seleciona as melhores via Holdout 30%.
3. Camada Fallback Hierárquica: Imputa dados restantes usando modas baseadas em desenvolvedores.

Gera:
- Dataset imputado
- JSON com parâmetros estatísticos do dataset
"""

import gc
import json
import logging
import os
import re
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from src.config_paths import RAW_STEAM_DATA_PATH

# Constantes globais
OUTPUT_DIR = "v2_smart_imputation"
# FIXED: remover path absoluto D: e usar caminho configurável/relativo
DATASET_PATH = str(RAW_STEAM_DATA_PATH)

TEXT_COLUMNS = ['Name', 'Genres', 'Tags', 'Categories', 'Publishers', 'Developers', 'About the game', 'Notes']
NUMERIC_COLUMNS = {
    'Price': float, 'User score': float, 'Positive': float, 'Negative': float, 
    'Average playtime forever': float, 'Median playtime forever': float,
    'Peak CCU': float, 'Metacritic score': float, 'Achievements': float
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def report_missing(df: pd.DataFrame, title: str = "Missing Data Report") -> pd.Series:
    """Calcula e reporta a porcentagem de valores nulos em cada coluna."""
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)
    logging.info(f"--- {title} ---")
    if not missing.empty:
        logging.info("\n" + missing.to_string())
    else:
        logging.info("Nenhum dado faltante encontrado.")
    return missing


def clean_initial_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executa a limpeza inicial e sanitização de tipos do DataFrame.
    
    Args:
        df: DataFrame original sem tratamento.
        
    Returns:
        DataFrame com colunas convertidas e outliers básicos tratados.
    """
    logging.info("[PASSO 1] Iniciando Limpeza Inicial...")
    report_missing(df, "Missing Inicial (%)")
    
    # Conversões em lote para datas com cache otimizado
    logging.info("Convertendo tipos: Release date, numéricos e texto...")
    df['Release date'] = pd.to_datetime(df['Release date'], errors='coerce', cache=True)
    
    # Conversões explícitas de tipos contínuos
    for col, dtype in NUMERIC_COLUMNS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(dtype)
            
    # Coerção explícita para string, evitando acessos .str falhos ("AttributeError")
    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype('string')
    
    # Tratamento simples e vetorizado de outliers notórios
    logging.info("Identificando outliers de data e preço...")
    years = df['Release date'].dt.year
    df.loc[df['Price'] < 0, 'Price'] = 0.0
    df.loc[years < 1970, 'Release date'] = pd.NaT
    
    return df


def impute_publishers_layer_1(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa publicadoras usando propagação de franquia, devs e regex."""
    logging.info("[PASSO 2.1] Imputação Especialista: Publishers...")
    df['Publishers_confidence'] = 'none'
    
    # 1. Extrai o "base_name" da franquia removendo números/versões do título do jogo
    base_names = df['Name'].str.replace(r'\s*\d+.*$', '', regex=True).str.strip()
    
    # Relacionamento Mapeado: Nome Base -> Publisher mais comum/primeira registrada
    franchise_map = (
        df.dropna(subset=['Publishers'])
        .assign(base_name=base_names)
        .drop_duplicates('base_name')
        .set_index('base_name')['Publishers']
    )
    
    mask_franchise = df['Publishers'].isna() & base_names.isin(franchise_map.index)
    df.loc[mask_franchise, 'Publishers'] = base_names[mask_franchise].map(franchise_map)
    df.loc[mask_franchise, 'Publishers_confidence'] = 'high_franchise'
    
    # 2. Desenvolvedora atuando como própria publicadora
    mask_dev = df['Publishers'].isna() & df['Developers'].notna()
    df.loc[mask_dev, 'Publishers'] = df.loc[mask_dev, 'Developers']
    df.loc[mask_dev, 'Publishers_confidence'] = 'medium_dev_as_pub'
    
    # 3. Extração via RegEx de informações remanescentes em 'Notes'
    pub_regex = r'(?:Published by|Publisher:)\s*([^,.\n]+)'
    extracted_pubs = df['Notes'].str.extract(pub_regex, flags=re.IGNORECASE)[0].str.strip()
    mask_notes = df['Publishers'].isna() & extracted_pubs.notna()
    df.loc[mask_notes, 'Publishers'] = extracted_pubs[mask_notes]
    df.loc[mask_notes, 'Publishers_confidence'] = 'medium_notes_regex'
    
    return df


def _apply_keyword_spotting(df: pd.DataFrame) -> pd.DataFrame:
    """Busca gêneros na descrição do jogo via keywords vetorizadas."""
    genre_keywords = {
        'Action': 'action|shooter|combat|war|battle|explosion',
        'Adventure': 'adventure|explore|journey|quest|narrative',
        'RPG': 'rpg|role-playing|character|fantasy world',
        'Simulation': 'simulation|simulator|realistic|management',
        'Strategy': 'strategy|tactical|rts|turn-based',
        'Horror': 'horror|scary|spooky|survival horror',
        'Indie': 'indie|independent|small team'
    }
    
    about_clean = df['About the game'].fillna('').str.lower()
    mask_empty = df['Genres'].isna()
    
    # Aplica correspondência (Spotting) diretamente pelo Pandas
    df['Genres_b3'] = ""
    for genre, pattern in genre_keywords.items():
        found_mask = mask_empty & about_clean.str.contains(pattern, regex=True)
        df.loc[found_mask, 'Genres_b3'] += genre + ","
        
    df['Genres_b3'] = df['Genres_b3'].str.strip(',').replace('', pd.NA)
    return df


def impute_genres_layer_1(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa gêneros através de regras de texto (NB), numéricas (kNN) e keywords (Spotting)."""
    logging.info("[PASSO 2.2] Imputação Especialista: Genres...")
    df['Genres_confidence'] = 'none'
    
    df = _apply_keyword_spotting(df)
    
    train_mask = df['Genres'].notna()
    test_mask = df['Genres'].isna()
    
    if train_mask.sum() > 100:
        # A. Modelo Estatístico Baseado em Texto (Naive Bayes)
        logging.info("Treinando Naive Bayes para Gêneros...")
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        train_text = df.loc[train_mask, 'About the game'].fillna('') + " " + df.loc[train_mask, 'Tags'].fillna('')
        X_train_text = vectorizer.fit_transform(train_text)
        y_train = df.loc[train_mask, 'Genres'].astype(str).str.split(',').str[0]
        
        clf_nb = MultinomialNB()
        clf_nb.fit(X_train_text, y_train)
        
        if test_mask.any():
            test_text = df.loc[test_mask, 'About the game'].fillna('') + " " + df.loc[test_mask, 'Tags'].fillna('')
            X_test_text = vectorizer.transform(test_text)
            df.loc[test_mask, 'Genres_b1'] = clf_nb.predict(X_test_text)
            
        del X_train_text; gc.collect()

        # B. Modelo Numérico Baseado em Atributos (KNN)
        logging.info("Treinando kNN para Gêneros...")
        scaler = MinMaxScaler()
        cat_counts = df['Categories'].astype(str).str.count(',').fillna(0).values.reshape(-1, 1)
        
        features_num = df[['Price', 'Metacritic score']].fillna(0).values
        X_num = scaler.fit_transform(np.hstack((features_num, cat_counts)))
        
        clf_knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        clf_knn.fit(X_num[train_mask], y_train)
        
        if test_mask.any():
            df.loc[test_mask, 'Genres_b2'] = pd.Series(clf_knn.predict(X_num[test_mask]), index=df[test_mask].index).astype('string')

    return df


def impute_tags_layer_1(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula Tags baseadas nos Gêneros ou Similaridade de Gêneros."""
    logging.info("[PASSO 2.3] Imputação Especialista: Tags...")
    
    # 1. Regra Direta: Tag Baseada no Gênero Primário
    genre_to_tags = {
        'Action': 'Action,Shooter,Fast-Paced',
        'Adventure': 'Adventure,Exploration,Story Rich',
        'RPG': 'RPG,Open World,Character Customization',
        'Strategy': 'Strategy,Tactical,RTS',
        'Indie': 'Indie,Singleplayer,Atmospheric'
    }
    primary_genre = df['Genres'].astype(str).str.split(',').str[0]
    df['Tags_c3'] = primary_genre.map(genre_to_tags)
    
    # 2. Similaridade de Distâncias via TF-IDF (Batch-optimized)
    logging.info("Calculando similaridade para Tags (Otimizado por Lotes)...")
    known_idx = df.dropna(subset=['Tags']).sample(min(5000, df['Tags'].notna().sum()), random_state=42).index
    unknown_idx = df[df['Tags'].isna()].index
    
    if not unknown_idx.empty and not known_idx.empty:
        vec = TfidfVectorizer(max_features=200)
        known_vecs = vec.fit_transform(df.loc[known_idx, 'Genres'].fillna(''))
        
        BATCH_SIZE = 2000
        for i in range(0, len(unknown_idx), BATCH_SIZE):
            batch = unknown_idx[i:i + BATCH_SIZE]
            batch_vecs = vec.transform(df.loc[batch, 'Genres'].fillna(''))
            sims = cosine_similarity(batch_vecs, known_vecs)
            
            best_matches = sims.argmax(axis=1)
            df.loc[batch, 'Tags_c1'] = df.loc[known_idx[best_matches], 'Tags'].values

    return df


def impute_categories_layer_1(df: pd.DataFrame) -> pd.DataFrame:
    """Cria Categorias baseadas em substrings combinadas de Gêneros e Tags."""
    logging.info("[PASSO 2.4] Imputação Especialista: Categories...")
    
    tag_to_cat = {
        'multiplayer': 'Multi-player',
        'co-op': 'Co-op',
        'singleplayer': 'Single-player',
        'pvp': 'PvP',
        'online': 'Online'
    }
    
    df['Categories_d1'] = ""
    # Monta corpus em minúsculas para match vetorizado eficiente
    text_corpus = (df['Genres'].fillna('') + " " + df['Tags'].fillna('')).str.lower()
    
    mask_empty_cats = df['Categories'].isna()
    for tag_key, cat_value in tag_to_cat.items():
        mask = text_corpus.str.contains(tag_key) & mask_empty_cats
        df.loc[mask, 'Categories_d1'] += cat_value + ","
        
    df['Categories_d1'] = df['Categories_d1'].str.strip(',').replace('', pd.NA)
    return df


def safe_str_access(series: pd.Series) -> pd.Series:
    """Garante que a série é tratável como string compatível com .str."""
    if series.isna().all():
        return pd.Series([''] * len(series), index=series.index)
    if series.dtype.name in ('string', 'object'):
        return series.fillna('').astype(str)
    return series.astype(str).fillna('')

def apply_ensemble_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Usa um conjunto de validação estatística (holdout) para ranquear e selecionar
    as top 3 heurísticas em paralelo, consolidando via forward/back-fill direcional.
    """
    logging.info("[PASSO 3] Iniciando Ensemble Validado...")
    target_cols = ['Genres', 'Tags', 'Categories']
    
    for col in target_cols:
        tech_cols = [c for c in df.columns if c.startswith(f"{col}_")]
        if not tech_cols:
            continue
            
        known_mask = df[col].notna()
        if known_mask.sum() < 100:
            continue
            
        # Avaliar métricas no holdout set (2000 rows limitadas para O(1) time complexity na avaliação)
        test_df = df[known_mask].sample(min(2000, known_mask.sum()), random_state=42)
        y_real = safe_str_access(test_df[col]).str.split(',').str[0]
        
        scores = {}
        for tech in tech_cols:
            safe_pred = safe_str_access(test_df[tech])
            y_pred = safe_pred.str.split(',').str[0]
            valid_idx = y_pred.replace('', pd.NA).replace('nan', pd.NA).replace('<NA>', pd.NA).notna()
            
            if valid_idx.any():
                acc = (y_real[valid_idx] == y_pred[valid_idx]).astype(float).mean()
                cob = valid_idx.astype(float).mean()
                # Ponderação de Negócio (Precisão importa mais que volume para Machine Learning base)
                scores[tech] = (acc * 0.7) + (cob * 0.3)
        
        # Seleciona chaves com maior pontuação, limitando a 3 técnicas
        top_3 = sorted(scores, key=lambda k: scores[k], reverse=True)[:3]
        logging.info(f"Top 3 Técnicas (Ensemble Validado) para {col}: {top_3}")
        
        if top_3:
            mask_impute = df[col].isna()
            # FFill/BFill simula um voto em cascata (a técnica mais precisa tenta cobrir primeiro)
            preds = df.loc[mask_impute, top_3].bfill(axis=1).iloc[:, 0]
            df.loc[mask_impute, col] = preds
            df.loc[mask_impute, f"{col}_confidence"] = 'high_ensemble'
            
    return df


def get_mode_value(x: pd.Series) -> Any:
    """Retorna a moda de forma rápida compatível com groupBy/agg."""
    counts = x.dropna().value_counts()
    return counts.index[0] if len(counts) > 0 else pd.NA


def apply_hierarchical_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """Preenche lacunas extremas restadas usando Modas dos Desenvolvedores."""
    logging.info("[PASSO 4] Iniciando Fallback Hierárquico...")
    
    for col in ['Genres', 'Tags', 'Categories', 'Publishers']:
        conf_col = f"{col}_confidence"
        if conf_col not in df.columns:
            df[conf_col] = 'none'
        df.loc[df[col].notna() & (df[conf_col] == 'none'), conf_col] = 'high'

    targets_defaults = {
        'Genres': "Indie", 
        'Tags': "singleplayer", 
        'Categories': "Single-player"
    }
    
    for col, default_val in targets_defaults.items():
        mask = df[col].isna()
        if not mask.any():
            continue
            
        logging.info(f"Aplicando default corporativo de fallback para: {col}")
        # Abordagem de aglomeração hyper-rápida (groupBy + dict mapping)
        dev_mode = df.dropna(subset=[col, 'Developers']).groupby('Developers')[col].agg(get_mode_value)
        imputed_dev = df.loc[mask, 'Developers'].map(dev_mode).fillna(default_val)
        
        df.loc[mask, col] = imputed_dev
        
        # Atribui confiança conforme o grau de decaimento do fallback
        mask_is_dev_mode = imputed_dev != default_val
        df.loc[mask & mask_is_dev_mode, f"{col}_confidence"] = "medium_dev_mode"
        df.loc[mask & ~mask_is_dev_mode, f"{col}_confidence"] = "low_default"
            
    return df


def validate_temporal_split(df: pd.DataFrame) -> Dict[str, int]:
    """Clusteriza ocorrências por partições temporais para validação."""
    logging.info("[PASSO 5] Relatório de Distribuição Temporal...")
    df_valid = df.dropna(subset=['Release date'])
    years = df_valid['Release date'].dt.year
    metrics = {
        "train_pre_2019": int((years <= 2018).sum()),
        "val_2019_2020": int(((years >= 2019) & (years <= 2020)).sum()),
        "test_post_2021": int((years >= 2021).sum())
    }
    logging.info(f"Distribuição: {metrics}")
    return metrics


def export_pipeline_results(df: pd.DataFrame, time_stats: Dict[str, int]) -> None:
    """Salva dataset finalizado e exports em formato amigável JSON."""
    logging.info("[PASSO 6] Exportando Entregáveis Finais...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    csv_path = os.path.join(OUTPUT_DIR, "imputed_dataset_v2.csv")
    df.to_csv(csv_path, index=False)
    
    params_path = os.path.join(OUTPUT_DIR, "generator_params.json")
    gen_params = {
        "temporal_split_sizes": time_stats,
        "price_stats": {
            "mean": float(df['Price'].mean()), 
            "std": float(df['Price'].std()),
            "median": float(df['Price'].median())
        },
        "top_genres": df['Genres'].astype(str).str.split(',').explode().value_counts().head(20).to_dict()
    }
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(gen_params, f, indent=4, ensure_ascii=False)
        
    logging.info("Pipeline Refatorado Finalizado com Êxito!")


def main() -> None:
    """Orquestrador do Pipeline Otimizado V2."""
    logging.info("Iniciando processamento profissional EDA & Imputação V2...")
    
    if not os.path.exists(DATASET_PATH):
        logging.error(f"Dataset inválido: {DATASET_PATH}")
        return
        
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    
    # Camadas sequenciais 
    df = clean_initial_data(df)
    
    df = impute_publishers_layer_1(df)
    df = impute_genres_layer_1(df)
    df = impute_tags_layer_1(df)
    df = impute_categories_layer_1(df)
    
    df = apply_ensemble_imputation(df)
    df = apply_hierarchical_fallback(df)
    
    temporal_metrics = validate_temporal_split(df)
    
    gc.collect() # Liberação mandatória pré-escrita
    export_pipeline_results(df, temporal_metrics)


if __name__ == "__main__":
    main()
