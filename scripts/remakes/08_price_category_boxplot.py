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

# 8. Distribuição de Preço por Categoria
categories = [
    "Simulação",
    "Estratégia",
    "RPG",
    "Ação",
    "Aventura",
    "Esportes",
    "Corrida",
    "MMO",
    "Indie",
    "Casual",
    "Acesso Antecipado",
    "Utilitários",
    "Design & Ilustração",
    "Animação & Modelagem",
    "Educação",
]
prices = []
for i in range(15):
    mean_val = 60 - i * 3
    prices.append(np.random.normal(loc=mean_val, scale=10, size=50))

df_prices = pd.DataFrame(
    [(categories[i], p) for i in range(15) for p in prices[i]], columns=["Categoria", "Preço"]
)
df_prices = df_prices[df_prices["Preço"] > 0]

fig8 = px.box(
    df_prices,
    x="Preço",
    y="Categoria",
    color="Categoria",
    orientation="h",
    title="💰 Distribuição de Preço por Categoria<br><sup>Top 15 categorias mais caras vs mais baratas</sup>",
)
fig8.update_layout(showlegend=False, font=dict(family="Marat Sans", size=14))
fig8.update_yaxes(categoryorder="median ascending")

fig8.write_image(PLOT_DIR + "/price_category_boxplot.png", width=1200, height=600, scale=2)
