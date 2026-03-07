import os

import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"
PLOT_DIR = "../../reports/figures"
os.makedirs(PLOT_DIR, exist_ok=True)

header = [
    "<b>Threshold<br>(Strictness)</b>",
    "<b>Precision<br>(Accuracy)</b>",
    "<b>Recall<br>(Reach)</b>",
    "<b>F1 Score<br>(Balance)</b>",
    "<b>Practical Interpretation</b>",
]

thresholds = ["0.3", "0.5 (Default)", "0.7"]
precisions = ["75%", "84%", "92%"]
recalls = ["90%", "80%", "60%"]
f1s = ["0.82", "0.82", "0.73"]
interpretations = [
    "Finds many games for the user, but sends several bad ones (focus on discovering everything).",
    "Ideal balance: few mistakes while still recommending a good amount of games.",
    "Only recommends if absolutely certain. High precision, but ignores many good games.",
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
    title="🎯 How Algorithm Strictness Affects Recommendations",
    margin=dict(l=20, r=20, t=50, b=20),
    font=dict(family="Marat Sans"),
)

fig.write_image(
    os.path.join(PLOT_DIR, "14_xAI_confusion_matrices_table.png"), width=900, height=350, scale=2
)
print("Generated Confusion Matrices Table chart.")
