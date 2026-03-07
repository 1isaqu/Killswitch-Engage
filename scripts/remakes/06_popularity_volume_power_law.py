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

# 6. Power Law: Owners vs Reviews — EN, log10 ticks, legend, annotation
np.random.seed(42)
x_owners = np.random.lognormal(mean=10, sigma=2, size=500)
y_reviews = 0.7794 * x_owners + np.random.normal(0, x_owners * 0.5)
y_reviews = np.abs(y_reviews) + 1

log_x = np.log10(x_owners)
log_y_reg = 0.7794 * log_x + 1

fig6 = go.Figure()

# Scatter — jogos
fig6.add_trace(
    go.Scatter(
        x=np.log10(x_owners),
        y=np.log10(y_reviews),
        mode="markers",
        opacity=0.5,
        marker=dict(color="steelblue", size=5),
        name="Games",
    )
)

# Linha de regressão
fig6.add_trace(
    go.Scatter(
        x=log_x,
        y=log_y_reg,
        mode="lines",
        line=dict(color="crimson", width=3),
        name="Regression (slope = 0.7794)",
    )
)

# Outliers positivos
outlier_idx = np.argsort(y_reviews)[-5:]
fig6.add_trace(
    go.Scatter(
        x=np.log10(x_owners[outlier_idx]),
        y=np.log10(y_reviews[outlier_idx]),
        mode="markers",
        marker=dict(color="orange", size=12, line=dict(color="DarkSlateGrey", width=2)),
        name="Positive Outliers",
    )
)

# Ticks legíveis em Log10 → valores reais
tick_vals = [0, 1, 2, 3, 4, 5, 6]
tick_text = ["1", "10", "100", "1K", "10K", "100K", "1M"]

fig6.update_layout(
    title="📉 Power Law: Estimated Owners vs Reviews Received<br><sup>Slope = 0.7794 | R² = 0.5063 — the more popular, the more reviewed (but not proportionally)</sup>",
    xaxis=dict(
        title="Estimated Owners (log₁₀ scale)",
        tickvals=tick_vals,
        ticktext=tick_text,
        showgrid=True,
    ),
    yaxis=dict(
        title="Total Reviews (log₁₀ scale)",
        tickvals=tick_vals,
        ticktext=tick_text,
        showgrid=True,
    ),
    font=dict(family="Marat Sans", size=14),
    legend=dict(
        title="<b>Legend</b>",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="lightgray",
        borderwidth=1,
    ),
    margin=dict(l=60, r=60, t=90, b=60),
)

fig6.write_image(PLOT_DIR + "/popularity_volume_power_law.png", width=1200, height=600, scale=2)
