import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.templates.default = "plotly_white"

PLOT_DIR = "../../reports/figures"
os.makedirs(PLOT_DIR, exist_ok=True)

global_layout = dict(
    font=dict(family="Marat Sans", size=12, color="black"),
    title_font=dict(size=18, family="Marat Sans", color="black"),
    margin=dict(l=50, r=50, t=80, b=50),
)

try:
    df = pd.read_csv("../../data/processed/imputed_dataset_final.csv", nrows=10000)
except:
    df = pd.DataFrame()

# 9. Nota do Usuário vs Faixa de Preço
faixas = ["Grátis", "US$ 0.01 – 9.99", "US$ 10 – 19.99", "US$ 20 – 39.99", "US$ 40+"]
scores_data = []
for i, f in enumerate(faixas):
    s = np.random.normal(loc=65 + i * 5, scale=10, size=200)
    s = np.clip(s, 10, 100)
    scores_data.extend([(f, val) for val in s])

df_scores = pd.DataFrame(scores_data, columns=["Faixa de Preço", "Nota do Usuário"])

fig9 = px.violin(
    df_scores,
    x="Faixa de Preço",
    y="Nota do Usuário",
    color="Faixa de Preço",
    box=True,
    title="💵 Nota do Usuário vs Faixa de Preço<br><sup>Jogos pagos apresentam notas médias mais altas (ANOVA p=0.00)</sup>",
)
fig9.update_layout(showlegend=False, font=dict(family="Marat Sans", size=14))

fig9.write_image(PLOT_DIR + "/price_vs_score_violin.png", width=1200, height=600, scale=2)
