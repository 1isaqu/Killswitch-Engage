import os
import mlflow
from dotenv import load_dotenv

# Carrega varíaveis locais, assegurando respeito às regras .env e SECURITY_CHECKLIST.txt
load_dotenv()

def setup_mlflow():
    """
    Configura o MLflow conectando-o ao servidor ou repositório de filesystem.
    Também define variáveis de autoria do experimento.
    """
    # Exige variáveis de ambiente, fallback para sqlite local se omitido
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow_experiments.db")
    mlflow.set_tracking_uri(tracking_uri)
    
    print(f"-> MLflow Tracking inicializado protegido em URI local.")

def create_or_set_experiment(experiment_name: str, description: str = "") -> str:
    """
    Cria ou liga-se a um experimento no mlflow.
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name, tags={
                "project": "Killswitch Engage",
                "phase": "Fase 5 - Experimentação Glocal",
                "notes": description
            })
            print(f"*> Novo experimento '{experiment_name}' criado.")
        else:
            experiment_id = experiment.experiment_id
            print(f"-> Reusando experimento '{experiment_name}'.")

        mlflow.set_experiment(experiment_name)
        return experiment_id
    except Exception as e:
        print(f"[!] Erro ao criar experimento: {e}")
        raise e
