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

# 2. Random Forest — Matrizes de Confusão por Threshold (PT-BR)
z_03 = [[700, 300], [100, 900]]
z_05 = [[850, 150], [200, 800]]
z_07 = [[950, 50], [400, 600]]

fig2 = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=(
        "Threshold 0.3<br><sup>F1: 0.82 | Precisão: 0.75 | Recall: 0.90</sup>",
        "Threshold 0.5<br><sup>F1: 0.82 | Precisão: 0.84 | Recall: 0.80</sup>",
        "Threshold 0.7<br><sup>F1: 0.73 | Precisão: 0.92 | Recall: 0.60</sup>",
    ),
    horizontal_spacing=0.1,
)

labels = ["Negativo", "Positivo"]

for i, z in enumerate([z_03, z_05, z_07]):
    heatmap = go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale="Blues",
        text=[[str(v) for v in row] for row in z],
        texttemplate="%{text}",
        showscale=(i == 2),
    )
    fig2.add_trace(heatmap, row=1, col=i + 1)

fig2.update_layout(
    title="🔍 Random Forest — Matrizes de Confusão por Threshold<br><sup>Threshold 0.3 (exploratório) | 0.5 (padrão) | 0.7 (conservador)</sup>",
    font=dict(family="Marat Sans", size=14),
)

fig2.write_image(PLOT_DIR + "/confusion_matrices.png", width=1200, height=600, scale=2)
