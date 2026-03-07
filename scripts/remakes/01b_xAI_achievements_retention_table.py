import os

import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"

PLOT_DIR = "../../reports/figures"
os.makedirs(PLOT_DIR, exist_ok=True)

# Data for the table
faixas = ["0 achievements", "1–10 achievements", "11–50 achievements", "51–150 achievements"]
tempos = ["150 min", "200 min", "350 min", "600 min"]
vs_base = ["➖ Base", "🔺 +50 min", "🔺 +200 min", "🔺 +450 min"]
psicologico = [
    "Casual and Uncommitted Exploration",
    "Initial Engagement (Quick Rewards)",
    "Commitment and Goal Seeking",
    "Completionism and Extreme Loyalty",
]

fig = go.Figure(
    data=[
        go.Table(
            columnwidth=[150, 150, 150, 300],
            header=dict(
                values=[
                    "<b>Achievement Range</b>",
                    "<b>Playtime</b>",
                    "<b>vs Zero Achievements</b>",
                    "<b>Psychological Contract</b>",
                ],
                fill_color="#004080",
                align="left",
                font=dict(color="white", size=14, family="Marat Sans"),
            ),
            cells=dict(
                values=[faixas, tempos, vs_base, psicologico],
                fill_color=[["white", "#f2f2f2", "white", "#f2f2f2"]],
                align="left",
                font=dict(color="black", size=13, family="Marat Sans"),
                height=35,
            ),
        )
    ]
)

fig.update_layout(
    title="🏆 The 'Step' Effect: Retention by Achievement Range<br><sup>Pearson = 0.018 | Spearman = 0.332 (Engagement is not linear, but by psychological steps)</sup>",
    margin=dict(l=20, r=20, t=70, b=20),
    font=dict(family="Marat Sans"),
)

fig.write_image(
    os.path.join(PLOT_DIR, "01b_xAI_achievements_retention_table.png"),
    width=900,
    height=350,
    scale=2,
)
print("Generated Achievements vs Playtime xAI table image.")
