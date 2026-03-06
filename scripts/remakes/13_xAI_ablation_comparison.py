import os

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"

PLOT_DIR = "../../reports/figures remake"
os.makedirs(PLOT_DIR, exist_ok=True)

# Mocked ablation study data
modelos = ["Modelo Base", "+ Features do Jogador", "+ Features do Jogo", "Modelo Completo"]
scores = [0.65, 0.74, 0.81, 0.88]  # Precisão Média (antigo Pseudo_MAP)

# Distinct colors
colors = ["#ced4da", "#6c757d", "#495057", "#0d6efd"]

fig = go.Figure(
    data=[
        go.Bar(
            x=modelos,
            y=scores,
            marker_color=colors,
            text=[f"{s:.2f}" for s in scores],
            textposition="outside",
        )
    ]
)

fig.update_layout(
    title="🛠️ Estudo de Ablação: Impacto de Cada Conjunto de Dados<br><sup>Como as informações do jogo e do jogador melhoram a recomendação</sup>",
    yaxis_title="Precisão Média",
    font=dict(family="Marat Sans", size=14),
    yaxis=dict(range=[0, 1.0]),
)

fig.write_image(
    os.path.join(PLOT_DIR, "13_xAI_ablation_comparison.png"), width=1000, height=600, scale=2
)
print("Generated Ablation Comparison chart.")
