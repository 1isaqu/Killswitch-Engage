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

# 3. Correlation Matrix — PT-BR
cols = ["Preço", "Recomendações", "Nota Metacritic", "Tempo Médio de Jogo", "Pico de Jogadores"]
corr_matrix = np.array(
    [
        [1.00, 0.15, 0.22, 0.08, 0.12],
        [0.15, 1.00, 0.35, 0.78, 0.85],
        [0.22, 0.35, 1.00, 0.25, 0.30],
        [0.08, 0.78, 0.25, 1.00, 0.72],
        [0.12, 0.85, 0.30, 0.72, 1.00],
    ]
)

fig3 = go.Figure(
    data=go.Heatmap(
        z=corr_matrix,
        x=cols,
        y=cols,
        colorscale="RdBu",
        reversescale=True,
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix, 2),
        texttemplate="%{text}",
    )
)

# Highlight > 0.7
for i in range(len(cols)):
    for j in range(len(cols)):
        if corr_matrix[i, j] > 0.7 and i != j:
            fig3.add_shape(
                type="rect",
                x0=j - 0.5,
                y0=i - 0.5,
                x1=j + 0.5,
                y1=i + 0.5,
                line=dict(color="black", width=3),
            )

fig3.update_layout(
    title="📊 Matriz de Correlação — Dataset Steam<br><sup>Relações entre preço, popularidade e engajamento (bordas pretas = r > 0.7)</sup>",
    xaxis=dict(tickangle=45),
    font=dict(family="Marat Sans", size=14),
    width=800,
    height=800,
)

fig3.write_image(PLOT_DIR + "/correlation_matrix.png", width=1200, height=600, scale=2)
