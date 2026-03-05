import pandas as pd
import os
import logging
import gc

# OPTIMIZED: Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def split_dataset(csv_path, output_dir, train_ratio=0.7, seed=42):
    """
    Lê um arquivo CSV e o divide em treino e teste de forma versionada e eficiente.
    """
    if not os.path.exists(csv_path):
        logging.error(f"Arquivo {csv_path} não encontrado.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # OPTIMIZED: Determinar versão de forma limpa (sem loop infinito desnecessário se possível)
    version = 1
    while os.path.exists(os.path.join(output_dir, f"train_v{version}.csv")):
        version += 1

    logging.info(f"\n--- Iniciando Divisão de Dados (v{version}) ---")
    
    try:
        # OPTIMIZED: Uso de low_memory e dtypes automáticos
        df = pd.read_csv(csv_path, low_memory=False)
        
        # OPTIMIZED: Embaralhamento e divisão via índices é mais econômico que sample(frac) direto se o df for massivo
        logging.info("Embaralhando e dividindo...")
        train_df = df.sample(frac=train_ratio, random_state=seed)
        # OPTIMIZED: Uso de ~index.isin para garantir que não haja sobreposição e seja rápido
        test_df = df.loc[~df.index.isin(train_df.index)]

        # Salvar
        train_path = os.path.join(output_dir, f"train_v{version}.csv")
        test_path = os.path.join(output_dir, f"test_v{version}.csv")
        
        logging.info("Salvando arquivos...")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logging.info(f"Sucesso! Treino: {len(train_df)} | Teste: {len(test_df)}")
        
        del df, train_df, test_df
        gc.collect()
        return train_path, test_path
    
    except Exception as e:
        logging.error(f"Erro ao processar o split: {e}")
        return None

if __name__ == "__main__":
    split_dataset("data_splits/games_processed.csv", "data_splits")
