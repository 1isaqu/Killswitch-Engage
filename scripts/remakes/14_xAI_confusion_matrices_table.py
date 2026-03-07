import os

import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"
PLOT_DIR = "../../reports/figures"
os.makedirs(PLOT_DIR, exist_ok=True)

header = [
    "<b>Threshold<br>(Rigor)</b>",
    "<b>Precision<br>(Acertos)</b>",
    "<b>Recall<br>(Alcance)</b>",
    "<b>F1 Score<br>(Balanço)</b>",
    "<b>Interpretação Prática</b>",
]

thresholds = ["0.3", "0.5 (Padrão)", "0.7"]
precisions = ["75%", "84%", "92%"]
recalls = ["90%", "80%", "60%"]
f1s = ["0.82", "0.82", "0.73"]
interpretations = [
    "Acha muitos jogos pro usuário, mas envia várias recomendações ruins (foco em descobrir tudo).",
    "Equilíbrio ideal: erra pouco e ainda consegue recomendar uma boa quantidade de jogos.",
    "Só recomenda se tiver certeza absoluta. Excelente precisão, mas ignora muitos jogos bons.",
]

fig = go.Figure(
    data=[
        go.Table(
            columnorder=[1, 2, 3, 4, 5],
            columnwidth=[80, 80, 80, 80, 300],
            header=dict(
                values=header,
                fill_color="#004080",
                align=["center", "center", "center", "center", "left"],
                font=dict(color="white", size=14, family="Marat Sans"),
            ),
            cells=dict(
                values=[thresholds, precisions, recalls, f1s, interpretations],
                fill_color=[["white", "#f2f2f2", "white"]],
                align=["center", "center", "center", "center", "left"],
                font=dict(color="black", size=13, family="Marat Sans"),
                height=40,
            ),
        )
    ]
)

fig.update_layout(
    title="🎯 Como o Rigor do Algoritmo Afeta as Recomendações",
    margin=dict(l=20, r=20, t=50, b=20),
    font=dict(family="Marat Sans"),
)

fig.write_image(
    os.path.join(PLOT_DIR, "14_xAI_confusion_matrices_table.png"), width=900, height=350, scale=2
)
print("Generated Confusion Matrices Table chart.")
