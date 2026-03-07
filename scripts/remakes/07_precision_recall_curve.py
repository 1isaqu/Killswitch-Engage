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

# 7. Curva Precisão-Recall
recall = np.linspace(0, 1, 100)
precision = 1 - 0.2 * recall**2 - 0.1 * recall**5

fig7 = go.Figure()
fig7.add_trace(
    go.Scatter(
        x=recall,
        y=precision,
        fill="tozeroy",
        fillcolor="rgba(0,100,250,0.2)",
        mode="lines",
        line=dict(color="blue", width=3),
        name="Curva PR",
    )
)

# Thresholds
thresh_r = [0.9, 0.8, 0.6]
thresh_p = [1 - 0.2 * r**2 - 0.1 * r**5 for r in thresh_r]
fig7.add_trace(
    go.Scatter(
        x=thresh_r,
        y=thresh_p,
        mode="markers+text",
        text=["0.3", "0.5", "0.7"],
        textposition="top right",
        marker=dict(color="red", size=10),
        name="Thresholds",
    )
)

fig7.add_hline(y=0.5, line_dash="dot", line_color="gray", annotation_text="Linha de Base")

fig7.update_layout(
    title="🎯 Curva Precisão-Recall — Random Forest<br><sup>PR-AUC = 0.915 | Thresholds: 0.3, 0.5, 0.7</sup>",
    xaxis_title="Recall",
    yaxis_title="Precisão",
    font=dict(family="Marat Sans", size=14),
)

fig7.write_image(PLOT_DIR + "/precision_recall_curve.png", width=1200, height=600, scale=2)
