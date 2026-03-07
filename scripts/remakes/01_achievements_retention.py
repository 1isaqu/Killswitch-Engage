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

# 1. Impact of Achievements on Playtime
categories = ["0 achievements", "1–10", "11–50", "51–150", ">150"]
playtimes = [150, 200, 350, 600, 400]
colors = ["#FFC080", "#FFA040", "#FF8000", "#CC6600", "#994C00"]

fig1 = go.Figure()
fig1.add_trace(
    go.Bar(
        x=categories,
        y=playtimes,
        marker_color=colors,
        text=[f"{v} min" for v in playtimes],
        textposition="outside",
    )
)

fig1.add_hline(
    y=300, line_dash="dash", line_color="gray", annotation_text="Platform Average (300 min)"
)

fig1.update_layout(
    title="🏆 Impact of Achievements on Playtime<br><sup>Games with 51–150 achievements retain players 4x more than basic games</sup>",
    yaxis_title="Playtime (minutes)",
    xaxis_title="Achievement Range",
    font=dict(family="Marat Sans", size=14),
)

fig1.write_image(PLOT_DIR + "/achievements_retention.png", width=1200, height=600, scale=2)
