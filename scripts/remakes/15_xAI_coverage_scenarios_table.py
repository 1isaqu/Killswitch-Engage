import os

import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"
PLOT_DIR = "../../reports/figures"
os.makedirs(PLOT_DIR, exist_ok=True)

header = [
    "<b>User Base</b>",
    "<b>Optimistic Scenario<br>(Best case)</b>",
    "<b>Realistic Scenario<br>(Expected)</b>",
    "<b>Conservative Scenario<br>(Worst case)</b>",
]

usuarios = ["10,000", "100,000", "500,000", "1,000,000"]
otimista = ["12% Coverage", "15% Coverage", "20% Coverage", "25% Coverage"]
realista = ["8% Coverage", "12% Coverage", "15% Coverage", "18% Coverage"]
conservador = ["5% Coverage", "9% Coverage", "12% Coverage", "14% Coverage"]

fig = go.Figure(
    data=[
        go.Table(
            header=dict(
                values=header,
                fill_color="#4caf50",
                align="center",
                font=dict(color="white", size=14, family="Marat Sans"),
            ),
            cells=dict(
                values=[usuarios, otimista, realista, conservador],
                fill_color=[["white", "#f9f9f9", "white", "#f9f9f9"]],
                align="center",
                font=dict(color="black", size=14, family="Marat Sans"),
                height=35,
            ),
        )
    ]
)

fig.update_layout(
    title="📈 Catalog Coverage Projection vs User Scale",
    margin=dict(l=20, r=20, t=50, b=20),
    font=dict(family="Marat Sans"),
)

fig.write_image(
    os.path.join(PLOT_DIR, "15_xAI_coverage_scenarios_table.png"), width=900, height=300, scale=2
)
print("Generated Coverage Scenarios Table chart.")
