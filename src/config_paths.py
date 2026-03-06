import os
from pathlib import Path

# FIXED: centralizar paths de dataset e arquivos imputados em um único módulo de configuração

BASE_DIR = Path(__file__).resolve().parents[1]

# Caminho para o dataset bruto da Steam
RAW_STEAM_DATA_PATH: Path = Path(
    os.getenv("RAW_STEAM_DATA_PATH", BASE_DIR / "data" / "raw" / "games.csv")
)

# Caminho para o dataset imputado gerado pelo pipeline v2
IMPUTED_DATA_PATH: Path = Path(
    os.getenv(
        "IMPUTED_DATA_PATH",
        BASE_DIR / "v2_smart_imputation" / "imputed_dataset_v2.csv",
    )
)
