import os

import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"
PLOT_DIR = "../../reports/figures remake"
os.makedirs(PLOT_DIR, exist_ok=True)

header = [
    "<b>Variável Original (Inglês)</b>",
    "<b>Significado Prático</b>",
    "<b>Grau de Importância na IA</b>",
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
    "Número total de recomendações dos jogadores",
    "Tempo médio histórico de jogo",
    "Preço base de venda do jogo",
    "Pico de jogadores simultâneos atingido",
    "Nota média da crítica especializada",
    "Quantidade de conquistas destraváveis",
]
importancias = ["🚨 Altíssimo", "🚨 Altíssimo", "🔥 Alto", "🔥 Alto", "⚡ Médio", "⚡ Médio"]

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
    title="🧠 O que a Inteligência Artificial Considera Mais Importante?",
    margin=dict(l=20, r=20, t=50, b=20),
    font=dict(family="Marat Sans"),
)

fig.write_image(
    os.path.join(PLOT_DIR, "16_xAI_feature_importance_table.png"), width=900, height=400, scale=2
)
print("Generated Feature Importance Table chart.")
