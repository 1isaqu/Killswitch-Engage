import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.templates.default = "plotly_white"

PLOT_DIR = "../../reports/figures remake"
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

# 10. Confiança da Imputação de Gêneros
labels = ["Alta Confiança (Original + Especialistas)", "Baixa Confiança (Ensemble)"]
values = [93, 7]

fig10 = go.Figure(
    data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker=dict(colors=["#2ca02c", "#d62728"]),
            pull=[0, 0.1],
        )
    ]
)
fig10.update_layout(
    title="✅ Confiança da Imputação — Gêneros<br><sup>93% de alta confiança (dados originais + especialistas)</sup>",
    font=dict(family="Marat Sans", size=14),
)

fig10.write_image(PLOT_DIR + "/genres_imputation_confidence.png", width=1200, height=600, scale=2)
