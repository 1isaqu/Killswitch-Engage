import os

import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"

PLOT_DIR = "../../reports/figures"
os.makedirs(PLOT_DIR, exist_ok=True)

# Data for the table
categories = [
    "Adventure",
    "Strategy",
    "Racing",
    "RPG",
    "Casual",
    "MMO",
    "Sports",
    "Indie",
    "Action",
    "Simulation",
]
mean_scores = [95, 90, 90, 85, 85, 85, 85, 80, 80, 72]
vs_avg = ["🔺 +15", "🔺 +10", "🔺 +10", "🔺 +5", "🔺 +5", "🔺 +5", "🔺 +5", "➖ 0", "➖ 0", "🔻 -8"]
highlights = [
    "The beloved category",
    "Strategy pleases",
    "Racing pleases",
    "Solid classic",
    "Surprisingly high",
    "Engaged community",
    "Loyal fans",
    "Platform standard",
    "Stable average",
    "Below average",
]

fig = go.Figure(
    data=[
        go.Table(
            header=dict(
                values=[
                    "<b>Category</b>",
                    "<b>Mean Score</b>",
                    "<b>vs General Average</b>",
                    "<b>Highlight</b>",
                ],
                fill_color="#004080",
                align="left",
                font=dict(color="white", size=14, family="Marat Sans"),
            ),
            cells=dict(
                values=[categories, mean_scores, vs_avg, highlights],
                fill_color=[["white", "#f2f2f2"] * 5],
                align="left",
                font=dict(color="black", size=13, family="Marat Sans"),
                height=30,
            ),
        )
    ]
)

fig.update_layout(
    title="🕹️ Best-Rated Categories (Ranked)",
    margin=dict(l=20, r=20, t=50, b=20),
    font=dict(family="Marat Sans"),
)

fig.write_image(
    os.path.join(PLOT_DIR, "09b_xAI_price_vs_category_ranked_table.png"),
    width=800,
    height=450,
    scale=2,
)
print("Generated Price vs Category ranked table image.")
