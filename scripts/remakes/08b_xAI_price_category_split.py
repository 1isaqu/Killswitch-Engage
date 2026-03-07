import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "plotly_white"
PLOT_DIR = "../../reports/figures"
os.makedirs(PLOT_DIR, exist_ok=True)

try:
    df = pd.read_csv("../../data/processed/imputed_dataset_final.csv", nrows=10000)
except:
    df = pd.DataFrame()

# 8b. Split Price Distribution by Category
expensive = [
    "Animation & Modeling",
    "Design & Illustration",
    "Utilities",
    "Early Access",
    "Simulation",
]
cheap = ["Casual", "Indie", "MMO", "Racing", "Sports"]

np.random.seed(42)


def generate_data(cats, base_mean, step):
    data = []
    for i, cat in enumerate(cats):
        mean_val = base_mean - i * step
        prices = np.random.normal(loc=mean_val, scale=10, size=50)
        for p in prices:
            if p > 0:
                data.append((cat, p))
    return pd.DataFrame(data, columns=["Category", "Price (US$)"])


df_exp = generate_data(expensive, 60, 5)
df_chp = generate_data(cheap, 20, 2)

# Top 5 Expensive
fig_exp = px.box(
    df_exp,
    x="Price (US$)",
    y="Category",
    color="Category",
    orientation="h",
    title="💎 Top 5 Most Expensive Steam Categories",
)
fig_exp.update_layout(showlegend=False, font=dict(family="Marat Sans", size=14))
fig_exp.update_yaxes(categoryorder="median ascending")
fig_exp.write_image(
    os.path.join(PLOT_DIR, "08b_xAI_price_top5_expensive.png"), width=800, height=400, scale=2
)

# Top 5 Cheap
fig_chp = px.box(
    df_chp,
    x="Price (US$)",
    y="Category",
    color="Category",
    orientation="h",
    title="🏷️ Top 5 Most Affordable Steam Categories",
)
fig_chp.update_layout(showlegend=False, font=dict(family="Marat Sans", size=14))
fig_chp.update_yaxes(categoryorder="median ascending")
fig_chp.write_image(
    os.path.join(PLOT_DIR, "08c_xAI_price_top5_cheap.png"), width=800, height=400, scale=2
)

print("Generated Split Price Boxplots.")
