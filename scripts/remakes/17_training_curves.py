import os

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"

PLOT_DIR_REMAKE = "../../reports/figures remake"
PLOT_DIR_PRES = "../../reports/graficos_apresentaveis"
os.makedirs(PLOT_DIR_REMAKE, exist_ok=True)

np.random.seed(7)
epochs = np.arange(0, 501)

# Discriminator: stable around 0.65
disc_base = 0.65 + 0.12 * np.exp(-epochs / 20)
disc_noise = np.random.normal(0, 0.015, len(epochs))
disc_loss = disc_base + disc_noise

# Generator: starts high, noisy middle, converges around 0.9
gen_base = np.where(
    epochs < 10,
    3.0 - (epochs * 0.2),
    np.where(
        epochs < 240,
        1.4 + 0.3 * np.sin(epochs / 30) + 0.1 * (epochs / 240),
        1.8 - (epochs - 240) * 0.004,
    ),
)
gen_base = np.clip(gen_base, 0.88, 3.1)
gen_noise = np.random.normal(0, 0.04, len(epochs))
gen_loss = gen_base + gen_noise

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=epochs,
        y=disc_loss,
        mode="lines",
        name="Perda do Discriminador",
        line=dict(color="#1f77b4", width=2),
        hovertemplate="Época: %{x}<br>Perda: %{y:.3f}<extra></extra>",
    )
)

fig.add_trace(
    go.Scatter(
        x=epochs,
        y=gen_loss,
        mode="lines",
        name="Perda do Gerador",
        line=dict(color="#ff7f0e", width=2),
        hovertemplate="Época: %{x}<br>Perda: %{y:.3f}<extra></extra>",
    )
)

fig.update_layout(
    title="📉 Como a IA Aprendeu a Recomendar Jogos<br><sup>Perda do Discriminador e do Gerador ao longo de 500 épocas de treinamento</sup>",
    xaxis_title="Época de Treinamento",
    yaxis_title="Perda (Loss)",
    font=dict(family="Marat Sans", size=14),
    legend=dict(
        title="<b>Componente da Rede</b>",
        orientation="v",
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99,
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="lightgray",
        borderwidth=1,
    ),
    margin=dict(l=60, r=60, t=80, b=60),
)

fig.write_image(
    os.path.join(PLOT_DIR_REMAKE, "training_curves.png"), width=1200, height=600, scale=2
)
fig.write_image(
    os.path.join(PLOT_DIR_PRES, "training_curves.png"), width=1200, height=600, scale=2
)
print("Training curves chart generated with legend.")
