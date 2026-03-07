import asyncio
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch

from src.backend.services.recomendador import recomendador


async def simulate_thresholds(n=200):
    """Simula a geração de thresholds para N usuários e retorna os dados."""
    print(f"Simulando {n} gerações de threshold via cGAN...")

    thresholds = []
    for _ in range(n):
        uid = uuid4()
        # Acessamos o método privado para teste de estresse do gerador
        t = await recomendador._generate_meta_threshold(uid)
        thresholds.append(t)

    return thresholds


def plot_threshold_distribution(thresholds):
    """Cria um gráfico de distribuição dos thresholds gerados."""
    df = pd.DataFrame(thresholds, columns=["threshold"])

    # Criar histograma com Plotly
    fig = px.histogram(
        df,
        x="threshold",
        nbins=20,
        title="Dynamic Thresholds Distribution (cGAN Meta-Observer)",
        labels={"threshold": "Generated Threshold"},
        marginal="violin",  # Adiciona um violino no topo para ver a densidade
    )

    # Adicionar linhas de referência dos modos estáticos
    fig.add_vline(x=0.3, line_dash="dash", line_color="green", annotation_text="Adventurous (0.3)")
    fig.add_vline(
        x=0.5, line_dash="dash", line_color="orange", annotation_text="Balanced (0.5)"
    )
    fig.add_vline(x=0.7, line_dash="dash", line_color="red", annotation_text="Conservative (0.7)")

    fig.update_layout(
        template="plotly_dark", font_family="Marat Sans", xaxis_range=[0.1, 0.9], showlegend=False
    )

    output_path = Path("reports/figures/18_cgan_threshold_dist.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Tenta salvar como PNG (requere kaleido)
    try:
        fig.write_image(str(output_path), scale=2)
        print(f"Gráfico salvo em: {output_path}")
    except Exception as e:
        print(f"Aviso: Não foi possível salvar PNG ({e}). Salvando HTML...")

    # Salvar também uma versão interativa em HTML
    fig.write_html(str(output_path.with_suffix(".html")))


async def main():
    if not recomendador.is_loaded:
        print("Erro: Modelos não carregados. Certifique-se de que os artefatos existem.")
        return

    thresholds = await simulate_thresholds(200)
    plot_threshold_distribution(thresholds)

    # Estatísticas rápidas
    print(f"Média: {np.mean(thresholds):.4f}")
    print(f"Desvio Padrão: {np.std(thresholds):.4f}")
    print(f"Mínimo: {np.min(thresholds):.4f}")
    print(f"Máximo: {np.max(thresholds):.4f}")


if __name__ == "__main__":
    asyncio.run(main())
