import os
import ast
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "data/ml_ready/jogos_features.csv"
MODEL_OUTPUT = "scripts/modelos/classificador_rf.pkl"
FIGURES_DIR = "reports/figures"

# FIXED: extrair magic numbers relevantes para constantes nomeadas
GOOD_GAME_THRESHOLD = 4.0
DEFAULT_AGE_DAYS = 3650
TEST_SIZE = 0.2
RANDOM_STATE = 42

os.makedirs(FIGURES_DIR, exist_ok=True)


def _safe_parse_categories(value):
    """Converte string de categorias em lista de forma segura."""
    # FIXED: substituir eval por ast.literal_eval para parsing mais seguro
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value:
        return []
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def train_classifier_v2() -> None:
    df = pd.read_csv(DATA_PATH)
    
    # 1. Target e Proporção de Classes
    df["target"] = (df["avaliacao_media"] >= GOOD_GAME_THRESHOLD).astype(int)
    
    n_pos = df["target"].sum()
    n_neg = len(df) - n_pos
    ratio = n_pos / len(df)
    print(f"\n=== ANÁLISE DE CLASSES ===")
    print(f"Positivos (avaliacao >= 4.0): {n_pos:,} ({ratio*100:.1f}%)")
    print(f"Negativos:                    {n_neg:,} ({(1-ratio)*100:.1f}%)")
    print(f"Imbalance ratio: {n_pos/n_neg:.2f}:1")

    # 2. Feature Engineering (mesmo do v1)
    df["data_lancamento"] = pd.to_datetime(df["data_lancamento"], errors="coerce")
    df["idade_dias"] = (
        pd.Timestamp.now() - df["data_lancamento"]
    ).dt.days.fillna(DEFAULT_AGE_DAYS)
    df["idade_log"] = np.log1p(df["idade_dias"])
    
    dev_rep = df.groupby("dev_id")["avaliacao_media"].mean().to_dict()
    df["dev_reputacao"] = df["dev_id"].map(dev_rep).fillna(df["avaliacao_media"].mean())

    df["categorias"] = df["categorias"].apply(_safe_parse_categories)
    all_cats = [cat for sublist in df["categorias"] for cat in sublist]
    top_20_cats = pd.Series(all_cats).value_counts().head(20).index.tolist()
    for cat in top_20_cats:
        df[f"cat_{cat}"] = df["categorias"].apply(lambda x: 1 if cat in x else 0)

    features = [
        "preco_base",
        "metacritic_score",
        "idade_log",
        "dev_reputacao",
        "total_avaliacoes",
    ] + [f"cat_{cat}" for cat in top_20_cats]
    X = df[features].fillna(0)
    y = df["target"]

    # 3. Split Temporal (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 4. Treinamento (class_weight balanced para compensar imbalance)
    print("\nTreinando Random Forest com class_weight='balanced'...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    y_prob = rf.predict_proba(X_test)[:, 1]

    # 5. PR-AUC (Average Precision) em vez de ROC-AUC
    pr_auc = average_precision_score(y_test, y_prob)
    print(f"\n=== MÉTRICAS v2 ===")
    print(f"PR-AUC (Average Precision): {pr_auc:.4f}")

    # Curva PR
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision, lw=2, color='royalblue')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Curva Precision-Recall | PR-AUC = {pr_auc:.4f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/precision_recall_curve.png", dpi=150)
    plt.close()
    print(f"Curva P-R salva em {FIGURES_DIR}/precision_recall_curve.png")

    # 6. Matriz de Confusão para 3 thresholds
    print("\n=== MATRIZES DE CONFUSÃO ===")
    thresholds = [0.3, 0.5, 0.7]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for ax, t in zip(axes, thresholds):
        y_pred_t = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_test, y_pred_t)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ruim", "Bom"])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        
        tn, fp, fn, tp = cm.ravel()
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        rec = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        
        ax.set_title(f"Threshold={t}\nF1={f1:.3f} | P={prec:.3f} | R={rec:.3f}")
        print(f"  Threshold {t}: P={prec:.3f} | R={rec:.3f} | F1={f1:.3f}")
    
    plt.suptitle("Matrizes de Confusão por Threshold", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/confusion_matrices.png", dpi=150)
    plt.close()
    print(f"Matrizes salvas em {FIGURES_DIR}/confusion_matrices.png")

    # 7. Salvar Modelo
    os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
    model_data = {
        'model': rf, 'features': features, 'top_20_cats': top_20_cats,
        'dev_rep': dev_rep, 'pr_auc': pr_auc,
        'class_ratio': ratio, 'global_mean_eval': df['avaliacao_media'].mean()
    }
    joblib.dump(model_data, MODEL_OUTPUT)
    print(f"\nModelo v2 salvo em {MODEL_OUTPUT}")

if __name__ == "__main__":
    train_classifier_v2()
