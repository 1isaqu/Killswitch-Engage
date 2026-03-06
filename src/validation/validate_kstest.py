import logging
import os

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config_paths import IMPUTED_DATA_PATH, RAW_STEAM_DATA_PATH

logging.basicConfig(level=logging.INFO, format="%(message)s")


def validate_datasets(original_path: str, imputed_path: str) -> None:
    logging.info("=" * 50)
    logging.info("INICIANDO VALIDAÇÃO DE TIPOS E KS-TEST")
    logging.info("=" * 50)

    if not os.path.exists(original_path) or not os.path.exists(imputed_path):
        logging.error("Arquivos não encontrados.")
        return

    # 1. Carregamento
    df_orig = pd.read_csv(original_path, low_memory=False)
    df_imp = pd.read_csv(imputed_path, low_memory=False)

    # 2. DIAGNÓSTICO DE TIPOS
    logging.info("\n=== DIAGNÓSTICO DE TIPOS (IMPUTADO V2) ===")
    for col in df_imp.columns:
        nulls = df_imp[col].isna().sum()
        amostra = df_imp[col].iloc[0] if len(df_imp) > 0 else "N/A"
        logging.info(f"{col}: {df_imp[col].dtype} | Nulls: {nulls} | Amostra: {amostra}")

    # 3. CORREÇÃO DE TIPOS (Enforcement)
    logging.info("\n=== APLICANDO CORREÇÕES DE TIPO ===")
    type_corrections = {
        "Genres": "string",
        "Tags": "string",
        "Categories": "string",
        "Publishers": "string",
        "Developers": "string",
        "About the game": "string",
        "Notes": "string",
    }

    for col, target_type in type_corrections.items():
        if col in df_imp.columns and df_imp[col].dtype.name != target_type:
            logging.info(f"Corrigindo imputado {col}: {df_imp[col].dtype} -> {target_type}")
            df_imp[col] = df_imp[col].astype(target_type)
        if col in df_orig.columns and df_orig[col].dtype.name != target_type:
            df_orig[col] = df_orig[col].astype(target_type)

    numeric_cols = [
        "Price",
        "Positive",
        "Negative",
        "Average playtime forever",
        "Median playtime forever",
        "Peak CCU",
        "Metacritic score",
        "User score",
        "Achievements",
    ]
    for col in numeric_cols:
        if col in df_imp.columns:
            df_imp[col] = pd.to_numeric(df_imp[col], errors="coerce")
        if col in df_orig.columns:
            df_orig[col] = pd.to_numeric(df_orig[col], errors="coerce")

    # 4. VALIDAÇÃO PRÉ-KS-TEST
    logging.info("\n=== VALIDAÇÃO PRÉ-KS-TEST ===")
    text_cols_for_ks = ["Genres", "Tags", "Categories"]
    for col in text_cols_for_ks:
        if col in df_imp.columns:
            assert df_imp[col].dtype.name == "string", f"{col} não é string! É {df_imp[col].dtype}"
            # Teste rápido de .str
            try:
                test = df_imp[col].head(10).str.split(",").str[0]
                logging.info(f"✅ {col} OK para operações .str")
            except Exception as e:
                logging.error(f"❌ Erro ao usar .str em {col}: {e}")

    # 5. EXECUÇÃO DO KS-TEST
    logging.info("\n=== RESULTADOS DO KS-TEST ===")
    ks_results = {}

    for col in numeric_cols:
        if col in df_orig.columns and col in df_imp.columns:
            # Remover NaN para o teste
            original = df_orig[col].dropna()
            atual = df_imp[col].dropna()

            if len(original) > 0 and len(atual) > 0:
                stat, p = ks_2samp(original, atual)
                ks_results[col] = {
                    "statistic": round(stat, 4),
                    "p_value": round(p, 4),
                    "pass": p > 0.05,
                    "original_n": len(original),
                    "atual_n": len(atual),
                }

    if ks_results:
        ks_df = pd.DataFrame(ks_results).T
        logging.info(ks_df.to_string())

        # Resumo final
        passed = ks_df["pass"].sum()
        total = len(ks_df)
        logging.info(
            f"\nResumo: {passed}/{total} colunas passaram no teste de sanidade (distribuição preservada)."
        )
    else:
        logging.info("Nenhuma coluna numérica disponível para KS-Test.")


if __name__ == "__main__":
    # FIXED: remover paths absolutos D: e usar caminhos configuráveis/relativos
    ORIG_CSV = str(RAW_STEAM_DATA_PATH)
    IMP_CSV = str(IMPUTED_DATA_PATH)
    validate_datasets(ORIG_CSV, IMP_CSV)
