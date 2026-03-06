import gc
import logging
import os

import pandas as pd

from src.config_paths import RAW_STEAM_DATA_PATH

# OPTIMIZED: Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def preprocess_steam_data(file_path: str, output_path: str) -> bool:
    logging.info("=" * 50)
    logging.info("Iniciando Pré-processamento de Dados do Steam (OTIMIZADO)")
    logging.info("=" * 50)

    if not os.path.exists(file_path):
        logging.error(f"Arquivo {file_path} não encontrado.")
        return False

    try:
        # OPTIMIZED: Uso de low_memory=False para evitar warnings e garantir tipos
        df = pd.read_csv(file_path, low_memory=False)
        logging.info(f"Registros iniciais: {len(df)}")

        # 1. Remoção de colunas (Vetorizado)
        # OPTIMIZED: Drop direto sem list comprehension desnecessário
        df.drop(columns=["Movies"], errors="ignore", inplace=True)

        # 2. Tratamento de Valores Ausentes (Categorias)
        cat_cols = ["Genres", "Tags", "Categories", "Developers", "Publishers"]
        # OPTIMIZED: Preenchimento de múltiplas colunas de uma vez
        df[cat_cols] = df[cat_cols].fillna("Unknown")

        # 3. Conversão de Data de Lançamento
        if "Release date" in df.columns:
            logging.info("Processando datas de lançamento (otimizado)...")
            # OPTIMIZED: cache=True acelera conversão de datas repetidas
            df["Release date"] = pd.to_datetime(df["Release date"], errors="coerce", cache=True)

            # OPTIMIZED: Extração direta de atributos dt
            df["Release Year"] = df["Release date"].dt.year.fillna(0).astype(int)
            df["Release Month"] = df["Release date"].dt.month.fillna(0).astype(int)

        # 4. Engenharia de Features (Vetorizada)
        if "Positive" in df.columns and "Negative" in df.columns:
            logging.info("Criando métricas de reviews...")
            # OPTIMIZED: Operações aritméticas diretas em Series
            df["Total Reviews"] = df["Positive"] + df["Negative"]
            # OPTIMIZED: Uso de .clip(lower=1) é mais eficiente que .replace(0, 1) para evitar div/0
            df["Positive Ratio"] = (df["Positive"] / df["Total Reviews"].clip(lower=1)).fillna(0)
            # Garante ratio 0 onde não há reviews
            df.loc[df["Total Reviews"] == 0, "Positive Ratio"] = 0

        # 5. Tratamento de nulos numéricos (Vetorizado)
        # OPTIMIZED: Seleção de tipos e fillna em bloco
        num_cols = df.select_dtypes(include=["number"]).columns
        df[num_cols] = df[num_cols].fillna(0)

        # 6. Salvar e Gerenciar Memória
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        df.to_csv(output_path, index=False)
        logging.info(f"Sucesso! Registros salvos: {len(df)}")

        del df
        gc.collect()
        return True

    except Exception as e:  # noqa: BLE001
        logging.error(f"Erro no pré-processamento: {e}")
        return False


if __name__ == "__main__":
    # FIXED: remover path absoluto D: e usar caminho configurável/relativo
    preprocess_steam_data(str(RAW_STEAM_DATA_PATH), "data_splits/games_processed.csv")
