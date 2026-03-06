import os

import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"

PLOT_DIR = "../../reports/figures remake"
os.makedirs(PLOT_DIR, exist_ok=True)

# Data for the table
faixas = ["0 conquistas", "1–10 conquistas", "11–50 conquistas", "51–150 conquistas"]
tempos = ["150 min", "200 min", "350 min", "600 min"]
vs_base = ["➖ Base", "🔺 +50 min", "🔺 +200 min", "🔺 +450 min"]
psicologico = [
    "Exploração Casual e Descompromissada",
    "Engajamento Inicial (Recompensas Rápidas)",
    "Comprometimento e Busca por Metas",
    "Completismo e Fidelidade Extrema",
]

fig = go.Figure(
    data=[
        go.Table(
            columnwidth=[150, 150, 150, 300],
            header=dict(
                values=[
                    "<b>Faixa de Conquistas</b>",
                    "<b>Tempo de Jogo</b>",
                    "<b>vs Zero Conquistas</b>",
                    "<b>Contrato Psicológico</b>",
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
    title="🏆 O Efeito 'Degrau': Retenção por Faixa de Conquistas<br><sup>Pearson = 0.018 | Spearman = 0.332 (O engajamento não é linear, e sim por patamares psicológicos)</sup>",
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
