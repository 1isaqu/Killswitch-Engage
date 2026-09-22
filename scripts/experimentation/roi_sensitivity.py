"""ROI do recomendador sob premissas explícitas — projeção, não medição.

LEIA ISTO ANTES DE CITAR QUALQUER NÚMERO DESTE SCRIPT.
========================================================

Medir o ROI do recomendador é impossível com os dados deste repositório — não
há receita real, não há grupo de controle, não há usuário real (ver
`reports/plano_medicao_negocio.md`). Este script NÃO mede nada. Ele calcula o
que o ROI **seria** sob um conjunto de premissas explícitas, declaradas e
com fonte quando existe uma:

  receita_incremental = MAU × lift_retencao × ARPU × margem

- `MAU`, `ARPU`, `margem` são PARÂMETROS DE CENÁRIO, não dados deste projeto.
- `lift_retencao` é a premissa dominante: quanto a recomendação melhora
  retenção sobre não ter recomendação. Ninguém mediu isso aqui — não há como
  medir sem um teste A/B (ver `plano_medicao_negocio.md` §3). É por isso que
  este script varre um intervalo largo em vez de assumir um valor.

O lado de CUSTO é ancorável em preço público (citado com data de consulta,
abaixo). O lado de BENEFÍCIO não tem nenhuma âncora neste projeto — é onde
este script é mais honesto sobre o que não sabe, não onde ele finge saber.

REGRAS DE APRESENTAÇÃO (não negociáveis, ver tarefa original):
- Todo título de tabela/figura diz "projeção" ou "cenário", nunca "resultado"
  nem "medição".
- Nenhum número aparece fora de contexto: cada um vem com a premissa que o
  gera.
- A conclusão declara que o intervalo de ROI é largo porque `lift_retencao`
  não foi medido — só um teste A/B real o estreita.

FONTES DOS PREÇOS DE INFRAESTRUTURA (consultadas em 22/09/2026):
  - Postgres gerenciado (Supabase): plano Pro US$25/mês (inclui US$10 de
    crédito de compute, cobre 1 instância Micro); add-ons de compute Micro
    US$10, Small US$15, Medium US$60, Large US$110/mês.
    Fonte: https://supabase.com/pricing
  - Redis gerenciado (Upstash, pay-as-you-go): US$0,20 por 100 mil comandos +
    US$0,25/GB de armazenamento (1 GB grátis); banda grátis até 200 GB/mês.
    Fonte: https://upstash.com/pricing
  - Compute da API (Render, web service): Starter US$7/mês (0,5 vCPU/512MB),
    Standard US$25/mês (1 vCPU/2GB), Pro US$85/mês (2 vCPU/4GB).
    Fonte: https://render.com/pricing (specs cruzadas com
    https://render.com/docs/compute-plans)
  - Taxa horária de desenvolvimento/retreino: mediana ~US$100/h no mercado de
    freelancers de ML (faixa observada US$50-200/h).
    Fonte: página de custo de contratação de especialistas em ML do Upwork
    (upwork.com/hire/machine-learning-experts/cost)

  Nenhum preço do lado de benefício (ARPU, margem, lift de retenção) tem fonte
  equivalente — ver docstring de `PARAMETROS_BASE` abaixo para o que é
  referência de mercado (citada, mas não medição) e o que é premissa pura.

Uso:
    python scripts/experimentation/roi_sensitivity.py
    python scripts/experimentation/roi_sensitivity.py --seed 42 --horizonte-meses 12
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SEED = 42

# ── Preços de infraestrutura — ancorados, com fonte e data de consulta ──────
# Ver docstring do módulo para as fontes completas. Consultado em 22/09/2026.
CUSTO_POSTGRES_BASE_MENSAL = 25.0  # Supabase Pro
CUSTO_POSTGRES_COMPUTE_SMALL_MENSAL = 15.0  # add-on Small (2GB RAM)
CUSTO_COMPUTE_API_STANDARD_MENSAL = 25.0  # Render Standard (1 vCPU/2GB)
CUSTO_COMPUTE_API_PRO_MENSAL = 85.0  # Render Pro (2 vCPU/4GB) — MAU maior
REDIS_PRECO_POR_100K_COMANDOS = 0.20  # Upstash
REDIS_PRECO_POR_GB_ARMAZENAMENTO = 0.25  # Upstash, acima de 1GB grátis
TAXA_HORARIA_BASE = 100.0  # mediana Upwork ML, ver fonte acima


# ── Parâmetros de cenário — nenhum é medição deste projeto ─────────────────


@dataclass(frozen=True)
class Parametros:
    """Todo campo aqui é premissa de cenário. Ver `formatar_premissas_texto`."""

    mau: float = 10_000  # âncora narrativa: mesma escala do dataset sintético
    arpu_mensal: float = 1.00  # US$/usuário/mês — referência de mercado, ver fonte
    margem: float = 0.70  # fração da receita que é margem bruta — premissa pura
    lift_retencao: float = 0.05  # NÃO ANCORADO — a variável que este script existe para varrer
    redis_comandos_por_usuario_mes: float = 20.0  # premissa de uso do cache
    redis_armazenamento_gb: float = 0.5  # premissa
    horas_manutencao_mensais: float = 8.0  # premissa
    horas_retreino_mensais: float = 4.0  # premissa
    taxa_horaria: float = TAXA_HORARIA_BASE


PARAMETROS_BASE = Parametros()

# Faixas plausíveis declaradas por parâmetro, para a análise de sensibilidade
# por faixa (tornado). Cada uma é (mínimo, máximo) em valor absoluto.
# lift_retencao: 0% a 25%. É a faixa mais larga do conjunto DE PROPÓSITO — não
#   existe nenhuma fonte, interna ou externa, que ancore sequer a ordem de
#   grandeza. As outras faixas abaixo, mesmo sendo também premissas, têm ao
#   menos uma referência (meta de produto ou benchmark de mercado citado).
# MAU: ±30% — faixa de planejamento de produto, não medição.
# ARPU: ±30% em torno de US$1,00 — dentro da faixa US$0,50-2,00/mês relatada
#   para jogos casuais mobile (fonte: relatório de receita de jogos mobile
#   2026, blog.udonis.co, consultado 22/09/2026). Referência de mercado, não
#   deste projeto — a Steam não é F2P mobile, então é só ordem de grandeza.
# margem: ±15% em torno de 70% — sem benchmark específico citado; faixa
#   mantida estreita deliberadamente por não ter âncora nenhuma.
# custo_infra: ±30%/+80% — infraestrutura tende a subir mais que cair com
#   escala; faixa assimétrica reflete isso.
# horas/taxa_horaria: faixa homóloga à dispersão observada no benchmark do
#   Upwork citado (mediana US$100/h, faixa US$50-200/h) — daí o fator 0.5x-2x.
FAIXAS_DECLARADAS = {
    "lift_retencao": (0.0, 0.25),
    "mau": (PARAMETROS_BASE.mau * 0.7, PARAMETROS_BASE.mau * 1.3),
    "arpu_mensal": (PARAMETROS_BASE.arpu_mensal * 0.7, PARAMETROS_BASE.arpu_mensal * 1.3),
    "margem": (PARAMETROS_BASE.margem * 0.85, min(1.0, PARAMETROS_BASE.margem * 1.15)),
    "horas_manutencao_mensais": (
        PARAMETROS_BASE.horas_manutencao_mensais * 0.5,
        PARAMETROS_BASE.horas_manutencao_mensais * 2.0,
    ),
    "horas_retreino_mensais": (
        PARAMETROS_BASE.horas_retreino_mensais * 0.5,
        PARAMETROS_BASE.horas_retreino_mensais * 2.0,
    ),
    "taxa_horaria": (PARAMETROS_BASE.taxa_horaria * 0.5, PARAMETROS_BASE.taxa_horaria * 2.0),
}


# ── Modelo de custo ──────────────────────────────────────────────────────────


def custo_compute_api_mensal(mau: float) -> float:
    """Degrau simples de instância Render pelo tamanho de MAU (ver fonte no topo)."""
    if mau <= 20_000:
        return CUSTO_COMPUTE_API_STANDARD_MENSAL
    return CUSTO_COMPUTE_API_PRO_MENSAL


def custo_redis_mensal(p: Parametros) -> float:
    comandos_mensais = p.mau * p.redis_comandos_por_usuario_mes
    custo_comandos = (comandos_mensais / 100_000) * REDIS_PRECO_POR_100K_COMANDOS
    custo_armazenamento = (
        max(0.0, p.redis_armazenamento_gb - 1.0) * REDIS_PRECO_POR_GB_ARMAZENAMENTO
    )
    return custo_comandos + custo_armazenamento


def custo_infra_mensal(p: Parametros) -> float:
    postgres = CUSTO_POSTGRES_BASE_MENSAL + CUSTO_POSTGRES_COMPUTE_SMALL_MENSAL
    return postgres + custo_compute_api_mensal(p.mau) + custo_redis_mensal(p)


def custo_operacao_mensal(p: Parametros) -> float:
    """Custo de operar o sistema já construído: infra + manutenção + retreino.

    NÃO inclui o custo de desenvolvimento inicial (setup). Escolha de
    modelagem deliberada: a pergunta que o ROI responde aqui é "vale a pena
    CONTINUAR operando", não "valeu a pena construir" — o investimento inicial
    já foi feito e é afundado (sunk cost) para essa decisão.
    """
    mao_de_obra = (p.horas_manutencao_mensais + p.horas_retreino_mensais) * p.taxa_horaria
    return custo_infra_mensal(p) + mao_de_obra


def receita_incremental_mensal(p: Parametros) -> float:
    """receita_incremental = MAU x lift_retencao x ARPU x margem — símbolos, não medição."""
    return p.mau * p.lift_retencao * p.arpu_mensal * p.margem


def roi(p: Parametros, horizonte_meses: int) -> float:
    custo_total = custo_operacao_mensal(p) * horizonte_meses
    receita_total = receita_incremental_mensal(p) * horizonte_meses
    return (receita_total - custo_total) / custo_total


# ── Varredura de lift_retencao ──────────────────────────────────────────────


def varrer_lift(
    p_base: Parametros,
    horizonte_meses: int,
    faixa: tuple[float, float] = (0.0, 0.25),
    passo: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    lifts = np.arange(faixa[0], faixa[1] + passo / 2, passo)
    rois = [roi(replace(p_base, lift_retencao=float(lift)), horizonte_meses) for lift in lifts]
    return lifts, np.array(rois)


def calcular_breakeven(p_base: Parametros, horizonte_meses: int) -> float:
    """lift_retencao mínimo para ROI = 0 no horizonte dado — forma fechada.

    receita_total = custo_total  =>  mau*lift*arpu*margem*h = custo_mensal*h
    => lift* = custo_mensal / (mau * arpu * margem)
    (custo_mensal não depende de lift_retencao neste modelo, então o
    horizonte cancela e a forma fechada vale para qualquer h.)
    """
    custo_mensal = custo_operacao_mensal(p_base)
    denom = p_base.mau * p_base.arpu_mensal * p_base.margem
    return custo_mensal / denom


# ── Elasticidade pontual ────────────────────────────────────────────────────


def elasticidade_pontual(
    p_base: Parametros, horizonte_meses: int, campo: str, eps: float = 0.01
) -> float:
    """Elasticidade de ROI a `campo`, por diferença finita central em torno de p_base.

    Definida só quando ROI(p_base) != 0 (senão a variação percentual não tem
    denominador). Ver ressalva no relatório sobre por que os 4 parâmetros do
    lado de receita saem EMPATADOS aqui — é uma propriedade da fórmula
    multiplicativa, não uma medida de importância relativa.
    """
    base_val = getattr(p_base, campo)
    roi_base = roi(p_base, horizonte_meses)
    if roi_base == 0 or base_val == 0:
        return float("nan")
    p_hi = replace(p_base, **{campo: base_val * (1 + eps)})
    p_lo = replace(p_base, **{campo: base_val * (1 - eps)})
    d_roi = roi(p_hi, horizonte_meses) - roi(p_lo, horizonte_meses)
    return (d_roi / (2 * eps)) / roi_base


# ── Sensibilidade por faixa declarada (tornado) ─────────────────────────────


@dataclass
class SensibilidadeFaixa:
    parametro: str
    roi_lo: float
    roi_hi: float

    @property
    def amplitude(self) -> float:
        return abs(self.roi_hi - self.roi_lo)


def sensibilidade_por_faixa(p_base: Parametros, horizonte_meses: int) -> list[SensibilidadeFaixa]:
    resultados = []
    for campo, (lo, hi) in FAIXAS_DECLARADAS.items():
        r_lo = roi(replace(p_base, **{campo: lo}), horizonte_meses)
        r_hi = roi(replace(p_base, **{campo: hi}), horizonte_meses)
        resultados.append(SensibilidadeFaixa(campo, min(r_lo, r_hi), max(r_lo, r_hi)))
    resultados.sort(key=lambda s: s.amplitude, reverse=True)
    return resultados


# ── Monte Carlo: propagação de incerteza das premissas ─────────────────────


def monte_carlo_roi(p_base: Parametros, horizonte_meses: int, n: int, seed: int) -> np.ndarray:
    """Amostra cada parâmetro uniformemente na sua faixa declarada e calcula o ROI.

    Não é uma simulação estatística de nada observado — é só a propagação
    mecânica da incerteza que nós mesmos declaramos em `FAIXAS_DECLARADAS`
    para dentro do resultado final, com `--seed` para reprodutibilidade.
    """
    rng = np.random.default_rng(seed)
    amostras = {}
    for campo, (lo, hi) in FAIXAS_DECLARADAS.items():
        amostras[campo] = rng.uniform(lo, hi, size=n)

    rois = np.empty(n)
    for i in range(n):
        kwargs = {campo: float(valores[i]) for campo, valores in amostras.items()}
        rois[i] = roi(replace(p_base, **kwargs), horizonte_meses)
    return rois


# ── Cenários nomeados ────────────────────────────────────────────────────────


def construir_cenarios(horizonte_meses: int) -> dict[str, tuple[Parametros, float]]:
    conservador = Parametros(
        mau=PARAMETROS_BASE.mau * 0.7,
        arpu_mensal=PARAMETROS_BASE.arpu_mensal * 0.7,
        margem=0.60,
        lift_retencao=0.01,
        horas_manutencao_mensais=12,
        horas_retreino_mensais=6,
        taxa_horaria=120.0,
    )
    base = PARAMETROS_BASE
    otimista = Parametros(
        mau=PARAMETROS_BASE.mau * 1.5,
        arpu_mensal=1.50,
        margem=0.75,
        lift_retencao=0.15,
        horas_manutencao_mensais=6,
        horas_retreino_mensais=3,
        taxa_horaria=90.0,
    )
    cenarios = {"conservador": conservador, "base": base, "otimista": otimista}
    return {nome: (p, roi(p, horizonte_meses)) for nome, p in cenarios.items()}


# ── Gráficos ──────────────────────────────────────────────────────────────


def gerar_grafico_curva(
    lifts: np.ndarray, rois: np.ndarray, breakeven: float, out_path: str
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lifts * 100, rois * 100, color="#2563eb", linewidth=2)
    ax.axhline(0, color="#6b7280", linewidth=1, linestyle="--")
    ax.axvline(
        breakeven * 100,
        color="#dc2626",
        linewidth=1,
        linestyle=":",
        label=f"ponto de equilíbrio ≈ {breakeven * 100:.1f}%",
    )
    ax.set_xlabel("lift_retencao (%) — premissa varrida, não medida")
    ax.set_ylabel("ROI projetado em 12 meses (%)")
    ax.set_title(
        "Projeção: ROI em função do lift de retenção (cenário base)\n"
        "premissa dominante não medida — ver reports/roi_cenarios.md",
        fontsize=11,
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def gerar_grafico_tornado(sensibilidades: list[SensibilidadeFaixa], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    nomes = [s.parametro for s in sensibilidades][::-1]
    los = np.array([s.roi_lo for s in sensibilidades])[::-1] * 100
    his = np.array([s.roi_hi for s in sensibilidades])[::-1] * 100
    y = np.arange(len(nomes))
    ax.barh(y, his - los, left=los, color="#f59e0b", alpha=0.85)
    ax.axvline(0, color="#6b7280", linewidth=1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(nomes)
    ax.set_xlabel("ROI projetado em 12 meses (%), extremos da faixa declarada")
    ax.set_title(
        "Cenário: sensibilidade do ROI à faixa declarada de cada premissa\n"
        "faixas maiores refletem menos âncora, não maior importância intrínseca",
        fontsize=10,
    )
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def gerar_grafico_monte_carlo(rois_mc: np.ndarray, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rois_mc * 100, bins=60, color="#10b981", alpha=0.8)
    ax.axvline(0, color="#dc2626", linewidth=1.5, linestyle="--", label="ROI = 0")
    ax.set_xlabel("ROI projetado em 12 meses (%)")
    ax.set_ylabel("frequência (10.000 sorteios)")
    ax.set_title(
        "Cenário: distribuição do ROI sob incerteza das premissas declaradas\n"
        "propagação mecânica de FAIXAS_DECLARADAS, não uma medição",
        fontsize=10,
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Relatório ────────────────────────────────────────────────────────────────


def formatar_premissas(p: Parametros) -> str:
    return (
        f"MAU={p.mau:,.0f} · ARPU=US${p.arpu_mensal:.2f}/mês · margem={p.margem:.0%} · "
        f"lift_retencao={p.lift_retencao:.1%} · "
        f"manutenção={p.horas_manutencao_mensais:.0f}h/mês · "
        f"retreino={p.horas_retreino_mensais:.0f}h/mês · taxa=US${p.taxa_horaria:.0f}/h"
    )


def gerar_relatorio(
    horizonte_meses: int,
    lifts: np.ndarray,
    rois_curva: np.ndarray,
    breakeven: float,
    elasticidades: list[tuple[str, float]],
    sensibilidades: list[SensibilidadeFaixa],
    cenarios: dict[str, tuple[Parametros, float]],
    rois_mc: np.ndarray,
    seed: int,
) -> str:
    linhas = []
    linhas.append("# ROI do recomendador — projeções sob premissas explícitas\n")
    linhas.append(
        "> ⚠️ **Nada nesta página é medição.** O sistema nunca foi a produção — não há "
        "receita real, não há grupo de controle, não há usuário real. Todo número aqui é "
        "função de premissas declaradas explicitamente (ver `PARAMETROS_BASE` e "
        "`FAIXAS_DECLARADAS` em `scripts/experimentation/roi_sensitivity.py`), e cada "
        'tabela/figura diz "projeção" ou "cenário" no título — nunca "resultado" '
        'ou "medição". Para o que falta instrumentar para medir de verdade, ver '
        "[`reports/plano_medicao_negocio.md`](plano_medicao_negocio.md).\n"
    )
    linhas.append(
        f"\nGerado por `scripts/experimentation/roi_sensitivity.py --seed {seed} "
        f"--horizonte-meses {horizonte_meses}`. Reproduzível: mesmo comando, mesmo "
        "resultado.\n"
    )

    linhas.append("## Modelo\n")
    linhas.append(
        "```\n"
        "receita_incremental = MAU × lift_retencao × ARPU × margem\n"
        "custo_operacao_mensal = custo_infra\n"
        "                         + (horas_manutencao + horas_retreino) × taxa_horaria\n"
        "ROI(horizonte) = (receita_incremental × horizonte − custo_operacao_mensal × horizonte)\n"
        "                  / (custo_operacao_mensal × horizonte)\n"
        "```\n"
        "Custo de desenvolvimento inicial (setup) é tratado como afundado e fica fora — a "
        "pergunta que este ROI responde é se vale a pena **continuar operando**, não se "
        "valeu a pena construir.\n"
    )
    linhas.append(
        "**Lado de custo — ancorado em preço público, fonte e data de consulta no topo do "
        "script.** Lado de benefício — `MAU` e `margem` são parâmetros de cenário sem "
        "medição; `ARPU` tem uma referência de mercado citada (jogos casuais mobile, "
        "US$0,50-2,00/mês); `lift_retencao` não tem nenhuma âncora — é a premissa "
        "dominante.\n"
    )

    linhas.append("## 1. Curva de ROI por lift_retencao (cenário, não resultado)\n")
    linhas.append(
        f"Cenário base: {formatar_premissas(PARAMETROS_BASE)}, horizonte = "
        f"{horizonte_meses} meses. `lift_retencao` varrido de "
        f"{lifts.min():.0%} a {lifts.max():.0%} — acima do exemplo de 0-15% da tarefa "
        "original, estendido só para que o ponto de equilíbrio (abaixo) apareça dentro "
        "do gráfico.\n"
    )
    linhas.append("| lift_retencao | ROI projetado (12 meses) |")
    linhas.append("|---:|---:|")
    for lift, r in zip(lifts[::2], rois_curva[::2]):
        marca = "  ← equilíbrio" if abs(lift - breakeven) < 0.0075 else ""
        linhas.append(f"| {lift:.0%} | {r:+.1%}{marca} |")
    linhas.append(
        f"\n**Ponto de equilíbrio: lift_retencao ≥ {breakeven:.1%}** para ROI positivo em "
        f"{horizonte_meses} meses, no cenário base. É o número mais útil desta página: "
        'uma afirmação verificável — "o sistema se paga se, e somente se, melhorar '
        f'retenção em pelo menos {breakeven:.1%}" — contra parâmetros de custo e '
        "benefício que qualquer leitor pode contestar e recalcular.\n"
    )
    linhas.append("![Projeção: ROI por lift de retenção](figures/roi_sensitivity_lift.png)\n")

    linhas.append("## 2. Elasticidade pontual (cenário base)\n")
    linhas.append(
        "Variação percentual do ROI para +1% em cada parâmetro, no cenário base, "
        "mantendo os demais fixos:\n"
    )
    linhas.append("| Parâmetro | Elasticidade do ROI |")
    linhas.append("|---|---:|")
    for nome, e in elasticidades:
        linhas.append(f"| {nome} | {e:+.2f} |")
    linhas.append(
        "\n**As quatro elasticidades do lado da receita (`mau`, `arpu_mensal`, `margem`, "
        "`lift_retencao`) saem empatadas.** Isso é uma propriedade da fórmula — ela é "
        "multiplicativa nos quatro termos, então 1% em qualquer um produz o mesmo efeito "
        "relativo na receita. **Elasticidade pontual não é o que sustenta a afirmação de "
        "que `lift_retencao` domina** — quem sustenta isso é a análise de faixa a seguir, "
        "porque a incerteza sobre cada premissa não é a mesma.\n"
    )

    linhas.append("## 3. Sensibilidade por faixa declarada (cenário, tornado)\n")
    linhas.append(
        "Para cada parâmetro, ROI nos dois extremos da faixa plausível declarada "
        "(`FAIXAS_DECLARADAS` no script), mantendo os demais no cenário base:\n"
    )
    linhas.append("| Parâmetro | ROI no mínimo da faixa | ROI no máximo da faixa | Amplitude |")
    linhas.append("|---|---:|---:|---:|")
    for s in sensibilidades:
        linhas.append(
            f"| {s.parametro} | {s.roi_lo:+.1%} | {s.roi_hi:+.1%} | {s.amplitude * 100:.0f} p.p. |"
        )
    dominante = sensibilidades[0]
    linhas.append(
        f"\n**`{dominante.parametro}` domina — uma variação de "
        f"{FAIXAS_DECLARADAS[dominante.parametro][0]:.0%} a "
        f"{FAIXAS_DECLARADAS[dominante.parametro][1]:.0%} nessa premissa move o ROI em "
        f"{dominante.amplitude * 100:.0f} pontos percentuais**, mais que qualquer outro "
        "parâmetro. Parte da amplitude reflete a faixa em si ter sido declarada mais "
        "larga — mas essa é exatamente a questão: a faixa de `lift_retencao` é a mais "
        "larga porque é a única sem nenhuma fonte, interna ou externa, ancorando sequer a "
        "ordem de grandeza. As demais faixas, mesmo sendo também premissas, vêm de meta "
        "de produto ou de benchmark de mercado citado.\n"
    )
    linhas.append(
        "![Cenário: sensibilidade por faixa (tornado)](figures/roi_sensitivity_tornado.png)\n"
    )

    linhas.append("## 4. Três cenários (conservador / base / otimista)\n")
    linhas.append("| Premissa | Conservador | Base | Otimista |\n" "|---|---:|---:|---:|")
    campos = [
        ("mau", "{:,.0f}"),
        ("arpu_mensal", "US${:.2f}"),
        ("margem", "{:.0%}"),
        ("lift_retencao", "{:.0%}"),
        ("horas_manutencao_mensais", "{:.0f}h"),
        ("horas_retreino_mensais", "{:.0f}h"),
        ("taxa_horaria", "US${:.0f}/h"),
    ]
    for campo, fmt in campos:
        linha = f"| `{campo}` "
        for nome in ("conservador", "base", "otimista"):
            p, _ = cenarios[nome]
            linha += f"| {fmt.format(getattr(p, campo))} "
        linhas.append(linha + "|")
    linha_roi = "| **ROI projetado ({} meses)** ".format(horizonte_meses)
    for nome in ("conservador", "base", "otimista"):
        _, r = cenarios[nome]
        linha_roi += f"| **{r:+.1%}** "
    linhas.append(linha_roi + "|")
    linhas.append(
        '\nTítulo desta tabela é "cenário", não "resultado": os três pontos são '
        "combinações de premissas escolhidas para ilustrar o intervalo, não uma previsão "
        "central com dois desvios.\n"
    )

    linhas.append("## 5. Incerteza propagada (Monte Carlo sobre as faixas declaradas)\n")
    p_positivo = float((rois_mc > 0).mean())
    linhas.append(
        f"10.000 sorteios uniformes dentro de `FAIXAS_DECLARADAS`, seed={seed}. "
        f"Mediana do ROI projetado: **{np.median(rois_mc):+.1%}**. "
        f"P(ROI > 0) sob estas faixas: **{p_positivo:.1%}**. "
        f"Percentis 10/90: [{np.percentile(rois_mc, 10):+.1%}, "
        f"{np.percentile(rois_mc, 90):+.1%}].\n"
    )
    linhas.append(
        "![Cenário: distribuição do ROI sob incerteza](figures/roi_sensitivity_montecarlo.png)\n"
    )

    linhas.append("## Conclusão\n")
    linhas.append(
        "O intervalo de ROI apresentado aqui é largo — de fortemente negativo a fortemente "
        "positivo entre os cenários conservador e otimista — **porque a premissa dominante "
        "(`lift_retencao`) nunca foi medida**, não porque o modelo de custo seja incerto: o "
        "lado de custo está ancorado em preço público (§ fontes, topo desta página) e varia "
        "pouco entre cenários comparado ao lado de benefício. Só um teste A/B real (desenho "
        "em `reports/plano_medicao_negocio.md` §3) estreita esse intervalo. Até lá, a "
        'afirmação defensável não é "o ROI é X%" — é "o sistema se paga a partir de um '
        f"lift de retenção de {breakeven:.1%}, e isso é o que precisa ser medido para saber "
        'se o projeto vale a pena".'
    )

    return "\n".join(linhas) + "\n"


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizonte-meses", type=int, default=12)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="reports/roi_cenarios.md")
    ap.add_argument("--figures-dir", default="reports/figures")
    ap.add_argument("--mc-amostras", type=int, default=10_000)
    args = ap.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print("Varrendo lift_retencao ...")
    lifts, rois_curva = varrer_lift(PARAMETROS_BASE, args.horizonte_meses)
    breakeven = calcular_breakeven(PARAMETROS_BASE, args.horizonte_meses)
    print(f"  ponto de equilíbrio: lift_retencao = {breakeven:.2%}")

    print("Calculando elasticidade pontual ...")
    campos_elasticidade = [
        "lift_retencao",
        "mau",
        "arpu_mensal",
        "margem",
        "taxa_horaria",
        "horas_manutencao_mensais",
        "horas_retreino_mensais",
    ]
    elasticidades = [
        (c, elasticidade_pontual(PARAMETROS_BASE, args.horizonte_meses, c))
        for c in campos_elasticidade
    ]
    elasticidades.sort(key=lambda t: abs(t[1]), reverse=True)

    print("Calculando sensibilidade por faixa declarada (tornado) ...")
    sensibilidades = sensibilidade_por_faixa(PARAMETROS_BASE, args.horizonte_meses)

    print("Construindo cenários conservador/base/otimista ...")
    cenarios = construir_cenarios(args.horizonte_meses)
    for nome, (_, r) in cenarios.items():
        print(f"  {nome}: ROI = {r:+.1%}")

    print(f"Rodando Monte Carlo ({args.mc_amostras:,} amostras, seed={args.seed}) ...")
    rois_mc = monte_carlo_roi(PARAMETROS_BASE, args.horizonte_meses, args.mc_amostras, args.seed)

    print("Gerando figuras ...")
    gerar_grafico_curva(
        lifts, rois_curva, breakeven, os.path.join(args.figures_dir, "roi_sensitivity_lift.png")
    )
    gerar_grafico_tornado(
        sensibilidades, os.path.join(args.figures_dir, "roi_sensitivity_tornado.png")
    )
    gerar_grafico_monte_carlo(
        rois_mc, os.path.join(args.figures_dir, "roi_sensitivity_montecarlo.png")
    )

    relatorio = gerar_relatorio(
        args.horizonte_meses,
        lifts,
        rois_curva,
        breakeven,
        elasticidades,
        sensibilidades,
        cenarios,
        rois_mc,
        args.seed,
    )
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(relatorio)
    print(f"\nRelatório escrito em {args.out}")


if __name__ == "__main__":
    sys.exit(main())
