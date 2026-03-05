import logging
import sys
import pandas as pd
from src.config_paths import RAW_STEAM_DATA_PATH

# Forçar a saída do console para UTF-8 para evitar erros de 'charmap' no Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        # Fallback para versões mais antigas do Python se necessário
        pass

# OPTIMIZED: Configuração de logging para substituir prints
logging.basicConfig(level=logging.INFO, format="%(message)s")


def analyze_steam_data(file_path: str):
    """
    Realiza uma análise exploratória rápida e otimizada do dataset Steam.
    """
    logging.info("="*50)
    logging.info("Análise Exploratória Otimizada")
    logging.info("="*50)

    try:
        # OPTIMIZED: Uso de chunking ou low_memory para eficiência
        df = pd.read_csv(file_path, low_memory=False)
        
        logging.info(f"Formato do dataset: {df.shape}")
        
        # OPTIMIZED: Uso de to_string() para evitar truncamento no log
        cols_to_show = df.columns[:8].tolist()

        # 6. Exemplo de análise - Jogos por Categoria/Gênero (se existir)
        genre_cols = [col for col in df.columns if 'Genre' in col or 'genre' in col or 'Categories' in col]
        if genre_cols:
            g_col = genre_cols[0]
            print(f"\n--- Top 10 {g_col} ---")
            print(df[g_col].value_counts().head(10))
        
        return df # Retorna o DataFrame para uso posterior

    except Exception as e:  # noqa: BLE001
        print(f"Ocorreu um erro durante a análise: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # FIXED: remover path absoluto D: e usar caminho configurável/relativo
    csv_path = str(RAW_STEAM_DATA_PATH)
    df = analyze_steam_data(csv_path)
