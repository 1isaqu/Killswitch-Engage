import os

import nbformat as nbf

notebook_cells = []


def add_notebook_cell(code_str, is_markdown=False):
    if is_markdown:
        notebook_cells.append(nbf.v4.new_markdown_cell(code_str))
    else:
        notebook_cells.append(nbf.v4.new_code_cell(code_str))


add_notebook_cell(
    "# Plotly Charts Remake\n\nThis notebook recreates the requested charts using Plotly, Medium-ready.",
    True,
)

# Read all 10 scripts
scripts = [
    "01_achievements_retention.py",
    "02_confusion_matrices.py",
    "03_correlation_matrix.py",
    "04_coverage_scalability.py",
    "05_playtime_genre_boxplot.py",
    "06_popularity_volume_power_law.py",
    "07_precision_recall_curve.py",
    "08_price_category_boxplot.py",
    "09_price_vs_score_violin.py",
    "10_genres_imputation_confidence.py",
]

for s in scripts:
    with open(s, "r", encoding="utf-8") as f:
        content = f.read()
    # To fix paths in notebook, replace PLOT_DIR relative path
    content = content.replace("../../reports/figures", "reports/figures")
    content = content.replace("../../data/", "data/")
    add_notebook_cell(content)

nb = nbf.v4.new_notebook()
nb["cells"] = notebook_cells
with open("../../remake_graficos.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook generated successfully.")
