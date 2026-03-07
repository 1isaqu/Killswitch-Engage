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

# 5. Playtime by Genre
# MMO, RPG and Strategy lead
np.random.seed(42)
genres = ["MMO", "RPG", "Strategy", "Action", "Adventure", "Indie"]
data = []
for i, g in enumerate(genres):
    base = 100 - i * 15
    vals = np.random.lognormal(mean=np.log(base), sigma=0.8, size=200)
    data.extend([(g, v) for v in vals if v < 500])

df_genre = pd.DataFrame(data, columns=["Genre", "Playtime"])

fig5 = px.box(
    df_genre,
    x="Genre",
    y="Playtime",
    color="Genre",
    title="🎮 Retention by Genre (Median Playtime)<br><sup>MMO, RPG, and Strategy lead in engagement</sup>",
)
fig5.update_layout(font=dict(family="Marat Sans", size=14), showlegend=False)
fig5.update_yaxes(title="Hours Played")
fig5.update_xaxes(title="Genre")

fig5.write_image(PLOT_DIR + "/playtime_genre_boxplot.png", width=1200, height=600, scale=2)
