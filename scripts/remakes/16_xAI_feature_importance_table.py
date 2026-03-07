import os

import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"
PLOT_DIR = "../../reports/figures"
os.makedirs(PLOT_DIR, exist_ok=True)

header = [
    "<b>Original Feature (English)</b>",
    "<b>Practical Meaning</b>",
    "<b>AI Importance Level</b>",
]

originais = [
    "recommendations",
    "average_playtime_forever",
    "price",
    "peak_ccu",
    "metacritic_score",
    "achievements",
]
significados = [
    "Total number of player recommendations",
    "Historical average playtime",
    "Base game price",
    "Peak concurrent users reached",
    "Average score from specialized critics",
    "Number of unlockable achievements",
]
importancias = ["🚨 Very High", "🚨 Very High", "🔥 High", "🔥 High", "⚡ Medium", "⚡ Medium"]

fig = go.Figure(
    data=[
        go.Table(
            columnwidth=[150, 250, 150],
            header=dict(
                values=header,
                fill_color="#673ab7",
                align="left",
                font=dict(color="white", size=14, family="Marat Sans"),
            ),
            cells=dict(
                values=[originais, significados, importancias],
                fill_color=[["white", "#f2f2f2"] * 3],
                align="left",
                font=dict(color="black", size=13, family="Marat Sans"),
                height=35,
            ),
        )
    ]
)

fig.update_layout(
    title="🧠 What Does the Artificial Intelligence Consider Most Important?",
    margin=dict(l=20, r=20, t=50, b=20),
    font=dict(family="Marat Sans"),
)

fig.write_image(
    os.path.join(PLOT_DIR, "16_xAI_feature_importance_table.png"), width=900, height=400, scale=2
)
print("Generated Feature Importance Table chart.")
