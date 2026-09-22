"""Indicadores de negócio calculáveis a partir dos dados do repositório.

LEIA ISTO ANTES DE CITAR QUALQUER NÚMERO DESTE SCRIPT.
========================================================

Os 10.000 usuários e as ~315 mil sessões em `data/ml_ready/interacoes_sessoes.csv`
são sintéticos, gerados proceduralmente por
`src/data_preparation/generate_synthetic_data.py`. Não existe usuário real, não
existe receita, não existe teste A/B. Todo número que este script produz —
retenção, churn, frequência, tempo de vida — é uma PROPRIEDADE DOS PARÂMETROS
QUE O GERADOR USA (`random.randint` por perfil, janela Beta de 720 dias), não uma
descoberta sobre comportamento de jogadores. Uma "retenção D7 de X%" aqui mede o
gerador, não pessoas.

O QUE ESTE SCRIPT NÃO FAZ
--------------------------
Não treina modelo nem prediz nada — é estatística descritiva sobre o log de
sessões. Por isso não usa o split temporal (treino/validação/teste) de
`evaluate_ranker.py`: esse split existe para evitar vazamento em avaliação
preditiva, e aqui não há predição para vazar.

Para ROI, LTV, CAC e qualquer indicador que dependa de causalidade ou receita,
ver `reports/plano_medicao_negocio.md` (o que falta medir) e
`reports/roi_cenarios.md` (estimativa sob premissas explícitas, nunca "medição").

METODOLOGIA
-----------
- Toda média reportada vem com intervalo de confiança de 95% por bootstrap sobre
  USUÁRIOS (reutiliza `ic_bootstrap` de `evaluate_ranker.py`).
- As coortes de perfil (casual / médio / hardcore / borda) não estão no CSV — o
  gerador as usa internamente para sortear o número de sessões e depois descarta
  o rótulo. Este script as RECONSTRÓI reproduzindo, na mesma ordem e com o mesmo
  `--seed`, exatamente as chamadas de `numpy.random.Generator` que
  `generate_synthetic_data.py` faz antes de atribuir perfil e usuário de borda
  (ver `_consumir_estado_gerar_jogos` e `_reconstruir_coortes` abaixo). Isso só
  é válido se `--seed`, `--n-users` e o tamanho do catálogo baterem com o que
  gerou o CSV lido — o script valida isso e AVISA se a reconstrução não bate com
  os dados observados.

Uso:
    python scripts/experimentation/evaluate_business_metrics.py
    python scripts/experimentation/evaluate_business_metrics.py --csv-dir data/ml_ready --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from evaluate_ranker import ic_bootstrap  # reutilizado, ver cabeçalho do módulo

# ── Constantes replicadas de src/data_preparation/generate_synthetic_data.py ──
# (mesmos valores; ver a reconciliação de coortes mais abaixo)
N_GENRES = 22
PERFIS_FAIXAS = {
    "casual": (5, 10),
    "medio": (20, 50),
    "hardcore": (50, 200),
}
BORDA_FAIXA = (150, 300)

RETENCAO_MAX_DIA = 30
JANELAS_CHURN = (7, 14, 30)  # dias de inatividade testados na sensibilidade
JANELA_CHURN_DECLARADA = 14  # janela adotada como definição principal
DURACAO_MES_DIAS = 30
N_REAMOSTRAS_BOOTSTRAP = 2_000


# ── Carregamento ────────────────────────────────────────────────────────────


def carregar_sessoes(csv_dir: str) -> pd.DataFrame:
    caminho = os.path.join(csv_dir, "interacoes_sessoes.csv")
    if not os.path.exists(caminho):
        raise FileNotFoundError(
            f"{caminho} não existe. Gere os dados primeiro com:\n"
            "  python -m src.data_preparation.generate_synthetic_data"
        )
    df = pd.read_csv(caminho)
    df["inicio"] = pd.to_datetime(df["inicio"], format="ISO8601")
    return df


def inferir_n_games(csv_dir: str, default: int) -> int:
    """Lê o tamanho real do catálogo de `jogos_features.csv`, se existir.

    Necessário para reconstruir as coortes: o número de chamadas de RNG que
    `gerar_jogos()` consome depende de `n_games`, e um valor errado desalinha
    todo o estado do gerador a partir daí.
    """
    caminho = os.path.join(csv_dir, "jogos_features.csv")
    if os.path.exists(caminho):
        return int(pd.read_csv(caminho, usecols=["id"]).shape[0])
    print(
        f"⚠️  {caminho} não encontrado — usando --n-games={default}. "
        "Se o catálogo gerado tinha outro tamanho, a reconstrução de coortes fica inválida."
    )
    return default


# ── Reconstrução das coortes de perfil ──────────────────────────────────────


def _consumir_estado_gerar_jogos(rng: np.random.Generator, n_games: int) -> None:
    """Consome o RNG exatamente como `gerar_jogos()` consome, sem montar o catálogo.

    Reproduz, na mesma ordem, os 7 usos de `rng` dentro de
    `generate_synthetic_data.gerar_jogos()`:
      1. genero_primario = rng.integers(0, N_GENRES, size=n_games)
      2. rng.shuffle(pop)                          — consumo depende só do tamanho
      3. lancamento: rng.integers(0, 5_600, size=n_games)
      4. preco_base: rng.gamma(2.0, 12.0, size=n_games)
      5. avaliacao_media: rng.normal(3.6, 0.8, size=n_games)
      6. metacritic_score: rng.normal(72, 12, size=n_games)
      7. idade_requerida: rng.choice([0,12,16,18], size=n_games, p=[.6,.2,.1,.1])

    O conteúdo desses arrays não importa (são descartados) — só o estado do
    gerador depois de consumi-los, que é o que `gerar_sessoes()` usa em seguida
    para sortear perfil e usuários de borda.
    """
    rng.integers(0, N_GENRES, size=n_games)
    dummy = np.empty(n_games)
    rng.shuffle(dummy)
    rng.integers(0, 5_600, size=n_games)
    rng.gamma(2.0, 12.0, size=n_games)
    rng.normal(3.6, 0.8, size=n_games)
    rng.normal(72, 12, size=n_games)
    rng.choice([0, 12, 16, 18], size=n_games, p=[0.6, 0.2, 0.1, 0.1])


def _reconstruir_coortes(n_users: int, n_games: int, seed: int) -> dict[int, str]:
    """Reproduz a atribuição de perfil/borda de `gerar_sessoes()`, sem gerar sessões.

    Espelha, nesta ordem exata, as linhas de
    `generate_synthetic_data.gerar_sessoes()` anteriores ao sorteio de afinidade
    de gênero (que não afeta perfil/borda e por isso não precisa ser reproduzido):

        user_ids = np.arange(n_users)
        ordem = rng.permutation(user_ids)
        n_casual, n_medio = int(n_users*0.70), int(n_users*0.25)
        perfil[...] = "casual" / "medio" / "hardcore"
        borda = set(rng.choice(ordem[:int(n_users*0.05)], size=min(500, n_users//20), ...))
    """
    rng = np.random.default_rng(seed)
    _consumir_estado_gerar_jogos(rng, n_games)

    user_ids = np.arange(n_users)
    ordem = rng.permutation(user_ids)
    n_casual, n_medio = int(n_users * 0.70), int(n_users * 0.25)

    perfil: dict[int, str] = {}
    for uid in ordem[:n_casual]:
        perfil[int(uid)] = "casual"
    for uid in ordem[n_casual : n_casual + n_medio]:
        perfil[int(uid)] = "medio"
    for uid in ordem[n_casual + n_medio :]:
        perfil[int(uid)] = "hardcore"

    borda = set(
        rng.choice(
            ordem[: int(n_users * 0.05)], size=min(500, n_users // 20), replace=False
        ).tolist()
    )

    return {uid: ("borda" if uid in borda else p) for uid, p in perfil.items()}


def reconstruir_coortes_validado(
    df: pd.DataFrame, n_users: int, n_games: int, seed: int
) -> tuple[dict[int, str], float]:
    """Reconstrói coortes e valida contra a contagem de sessões observada no CSV.

    Retorna (coortes, taxa_de_acerto). Se a taxa cair abaixo de 0.9, a
    reconstrução não é confiável para este CSV (seed/--n-users/tamanho do
    catálogo não batem com o que o gerou) e o chamador deve avisar o leitor em
    vez de publicar a tabela por coorte.
    """
    coortes = _reconstruir_coortes(n_users, n_games, seed)
    n_sessoes = df.groupby("usuario_id").size()

    acertos, total = 0, 0
    for uid, coorte in coortes.items():
        if uid not in n_sessoes.index:
            continue
        n = n_sessoes.loc[uid]
        total += 1
        if coorte == "borda":
            lo, hi = BORDA_FAIXA
        else:
            lo, hi = PERFIS_FAIXAS[coorte]
        if lo <= n <= hi:
            acertos += 1

    taxa = acertos / total if total else 0.0
    return coortes, taxa


# ── Parte A.1 — Retenção D0–D30 ─────────────────────────────────────────────


@dataclass
class PontoRetencao:
    dia: int
    retencao: float
    ic_lo: float
    ic_hi: float
    n_elegiveis: int


def calcular_retencao(df: pd.DataFrame, seed: int) -> list[PontoRetencao]:
    """D_n = fração de usuários com >=1 sessão na janela [n, n+1) após a primeira.

    Só entram no denominador de D_n usuários cuja janela [n, n+1) já podia ter
    sido observada antes do fim dos dados (censura à direita) — do contrário um
    usuário adquirido há poucos dias pareceria "churnado" em D30 só porque o
    relógio ainda não chegou lá.
    """
    t0 = df.groupby("usuario_id")["inicio"].transform("min")
    dia_rel = (df["inicio"] - t0).dt.total_seconds() // 86400
    df = df.assign(dia_rel=dia_rel.astype(int))

    primeira_sessao = df.groupby("usuario_id")["inicio"].min()
    data_max = df["inicio"].max()
    dias_observaveis = (data_max - primeira_sessao).dt.total_seconds() // 86400
    dias_observaveis = dias_observaveis.astype(int)

    pontos = []
    for n in range(RETENCAO_MAX_DIA + 1):
        elegiveis = dias_observaveis[dias_observaveis >= n].index
        if len(elegiveis) == 0:
            continue
        ativos_no_dia_n = set(df.loc[df["dia_rel"] == n, "usuario_id"].unique())
        indicador = np.array([1.0 if uid in ativos_no_dia_n else 0.0 for uid in elegiveis])
        retencao = float(indicador.mean())
        lo, hi = ic_bootstrap(indicador, n_reamostras=N_REAMOSTRAS_BOOTSTRAP, seed=seed)
        pontos.append(PontoRetencao(n, retencao, lo, hi, len(elegiveis)))
    return pontos


# ── Parte A.2 — Churn ────────────────────────────────────────────────────────


@dataclass
class ResultadoChurn:
    janela_dias: int
    taxa_pooled: float
    taxa_media_por_usuario: float
    ic_lo: float
    ic_hi: float
    usuarios_elegiveis: int
    user_meses_elegiveis: int


def _dias_ativos_por_usuario(df: pd.DataFrame, data_min: pd.Timestamp) -> dict[int, np.ndarray]:
    dia_abs = ((df["inicio"] - data_min).dt.total_seconds() // 86400).astype(int)
    tmp = df.assign(dia_abs=dia_abs)
    return {uid: np.sort(g["dia_abs"].unique()) for uid, g in tmp.groupby("usuario_id")}


def calcular_churn(df: pd.DataFrame, janela_dias: int, seed: int) -> ResultadoChurn:
    """Taxa de churn mensal: definição adotada e sensibilidade à janela.

    DEFINIÇÃO. Um usuário ativo num mês-calendário de 30 dias "churna" nesse mês
    se não tiver nenhuma sessão nos `janela_dias` seguintes ao fim do mês. A taxa
    mensal é a fração de pares (usuário, mês) com essa propriedade, entre os
    pares em que a janela de checagem já terminou antes do fim dos dados
    (senão não dá para saber se o usuário ainda vai voltar — censura à direita).

    Isto é uma escolha entre várias definições de churn possíveis, por isso a
    sensibilidade a `janela_dias` = 7 / 14 / 30 é reportada lado a lado: se a
    taxa muda muito, a definição é frágil, e a Parte A do relatório é obrigada a
    dizer isso.
    """
    data_min = df["inicio"].min()
    data_max = df["inicio"].max()
    dia_max_absoluto = int((data_max - data_min).total_seconds() // 86400)

    dias_por_usuario = _dias_ativos_por_usuario(df, data_min)

    taxas_por_usuario = []
    eventos_totais = 0
    user_meses_totais = 0

    for uid, dias in dias_por_usuario.items():
        meses_ativos = sorted({int(d // DURACAO_MES_DIAS) for d in dias})
        eventos_usuario = 0
        elegiveis_usuario = 0
        for m in meses_ativos:
            fim_mes = (m + 1) * DURACAO_MES_DIAS
            fim_janela = fim_mes + janela_dias
            if fim_janela > dia_max_absoluto:
                continue  # janela ainda não terminou de ser observada
            elegiveis_usuario += 1
            lo = np.searchsorted(dias, fim_mes, side="right")
            hi = np.searchsorted(dias, fim_janela, side="right")
            if hi == lo:  # nenhuma sessão em (fim_mes, fim_janela]
                eventos_usuario += 1

        if elegiveis_usuario > 0:
            taxas_por_usuario.append(eventos_usuario / elegiveis_usuario)
            eventos_totais += eventos_usuario
            user_meses_totais += elegiveis_usuario

    taxas = np.asarray(taxas_por_usuario)
    taxa_pooled = eventos_totais / user_meses_totais if user_meses_totais else float("nan")
    if len(taxas) > 0:
        lo, hi = ic_bootstrap(taxas, n_reamostras=N_REAMOSTRAS_BOOTSTRAP, seed=seed)
        media_por_usuario = float(taxas.mean())
    else:
        lo = hi = media_por_usuario = float("nan")

    return ResultadoChurn(
        janela_dias, taxa_pooled, media_por_usuario, lo, hi, len(taxas), user_meses_totais
    )


# ── Parte A.3 — Frequência e intensidade por coorte ─────────────────────────


@dataclass
class MetricaCoorte:
    coorte: str
    n_usuarios: int
    freq_media: float
    freq_ic: tuple[float, float]
    intensidade_media: float
    intensidade_ic: tuple[float, float]
    sessoes_por_mes_media: float
    sessoes_por_mes_ic: tuple[float, float]


def calcular_frequencia_intensidade(
    df: pd.DataFrame, coortes: dict[int, str], seed: int
) -> list[MetricaCoorte]:
    n_sessoes = df.groupby("usuario_id").size()
    horas_media_por_usuario = df.groupby("usuario_id")["horas_jogadas"].mean()
    primeira = df.groupby("usuario_id")["inicio"].min()
    ultima = df.groupby("usuario_id")["inicio"].max()
    vida_dias = (ultima - primeira).dt.total_seconds() / 86400.0

    resultados = []
    for coorte in ["casual", "medio", "hardcore", "borda"]:
        uids = [uid for uid, c in coortes.items() if c == coorte and uid in n_sessoes.index]
        if not uids:
            continue
        freq = n_sessoes.loc[uids].to_numpy(dtype=float)
        intensidade = horas_media_por_usuario.loc[uids].to_numpy(dtype=float)
        vida = vida_dias.loc[uids].clip(lower=1.0).to_numpy(dtype=float)
        sess_por_mes = freq / (vida / DURACAO_MES_DIAS)

        freq_lo, freq_hi = ic_bootstrap(freq, n_reamostras=N_REAMOSTRAS_BOOTSTRAP, seed=seed)
        int_lo, int_hi = ic_bootstrap(intensidade, n_reamostras=N_REAMOSTRAS_BOOTSTRAP, seed=seed)
        spm_lo, spm_hi = ic_bootstrap(sess_por_mes, n_reamostras=N_REAMOSTRAS_BOOTSTRAP, seed=seed)

        resultados.append(
            MetricaCoorte(
                coorte,
                len(uids),
                float(freq.mean()),
                (freq_lo, freq_hi),
                float(intensidade.mean()),
                (int_lo, int_hi),
                float(sess_por_mes.mean()),
                (spm_lo, spm_hi),
            )
        )
    return resultados


# ── Parte A.4 — Tempo de vida do usuário ────────────────────────────────────


@dataclass
class DistribuicaoVida:
    media: float
    ic_lo: float
    ic_hi: float
    percentis: dict[int, float] = field(default_factory=dict)


def calcular_tempo_de_vida(df: pd.DataFrame, seed: int) -> DistribuicaoVida:
    primeira = df.groupby("usuario_id")["inicio"].min()
    ultima = df.groupby("usuario_id")["inicio"].max()
    vida_dias = ((ultima - primeira).dt.total_seconds() / 86400.0).to_numpy()

    lo, hi = ic_bootstrap(vida_dias, n_reamostras=N_REAMOSTRAS_BOOTSTRAP, seed=seed)
    percentis = {p: float(np.percentile(vida_dias, p)) for p in (10, 25, 50, 75, 90)}
    return DistribuicaoVida(float(vida_dias.mean()), lo, hi, percentis)


# ── Relatório ────────────────────────────────────────────────────────────────

AVISO_ORIGEM = """\
> ⚠️ **Origem dos dados — leia antes de citar qualquer número desta página.**
>
> Os 10.000 usuários e as sessões abaixo são **sintéticos**, gerados por
> `src/data_preparation/generate_synthetic_data.py`. O gerador sorteia o número
> de sessões por perfil de faixas fixas (`randint(5,10)` para casual, e assim
> por diante) e distribui as datas numa janela de 720 dias com uma distribuição
> Beta. **Toda métrica desta página mede esses parâmetros, não comportamento de
> jogadores.** Não existe usuário real, não existe receita, não existe teste
> A/B por trás de nenhum número aqui.
"""

AVISO_CURTO = (
    "> ⚠️ Mede o **gerador sintético** (parâmetros de `generate_synthetic_data.py`), "
    "não comportamento de jogadores reais. Aviso completo no topo da página.\n"
)


def formatar_relatorio(
    retencao: list[PontoRetencao],
    churns: list[ResultadoChurn],
    coortes_metricas: list[MetricaCoorte],
    taxa_validacao_coortes: float,
    vida: DistribuicaoVida,
    n_usuarios: int,
    n_sessoes: int,
    seed: int,
    csv_dir: str,
) -> str:
    linhas = []
    linhas.append("# Indicadores de negócio calculáveis a partir dos dados sintéticos\n")
    linhas.append(AVISO_ORIGEM)
    linhas.append(
        f"\nGerado por `scripts/experimentation/evaluate_business_metrics.py "
        f"--csv-dir {csv_dir} --seed {seed}`. Base: {n_usuarios:,} usuários, "
        f"{n_sessoes:,} sessões. Reproduzível: mesmo comando, mesmo resultado.\n"
    )

    # Retenção
    linhas.append("## 1. Curva de retenção D0–D30\n")
    linhas.append(AVISO_CURTO)
    linhas.append(
        "\nD_n = fração de usuários com ao menos uma sessão na janela [n, n+1) dias "
        "após a primeira sessão. D0 = 1,0 por definição (todo usuário tem sessão no "
        "dia da própria primeira sessão). Usuários cuja janela [n, n+1) ainda não "
        "podia ter sido observada até o fim dos dados são excluídos do denominador "
        "de D_n (censura à direita) — por isso `n_elegíveis` cai à medida que n cresce.\n"
    )
    linhas.append("| Dia | Retenção | IC 95% (bootstrap) | n elegíveis |")
    linhas.append("|---:|---:|---:|---:|")
    for p in retencao:
        linhas.append(
            f"| D{p.dia} | {p.retencao:.4f} | [{p.ic_lo:.4f}, {p.ic_hi:.4f}] | {p.n_elegiveis:,} |"
        )
    linhas.append("")

    # Churn
    linhas.append("## 2. Taxa de churn mensal e sensibilidade à janela\n")
    linhas.append(AVISO_CURTO)
    linhas.append(
        f"\n**Definição adotada** (janela declarada = {JANELA_CHURN_DECLARADA} dias): um "
        'usuário ativo num mês-calendário de 30 dias "churna" nesse mês se não tiver '
        "nenhuma sessão nos `janela_dias` seguintes ao fim do mês. A taxa mensal é a "
        "fração de pares (usuário, mês) com essa propriedade, entre pares cuja janela de "
        "checagem já terminou antes do fim dos dados. `Churn` tem dezenas de definições "
        "possíveis na literatura; esta foi escolhida por ser verificável diretamente no "
        "log de sessões, sem precisar de evento de cancelamento explícito (que não existe "
        "no sistema — ver `reports/plano_medicao_negocio.md`).\n"
    )
    linhas.append(
        "**Teste de sensibilidade** — a mesma definição, variando só a janela de "
        "inatividade. Se a taxa mudar muito entre 7, 14 e 30 dias, a métrica é frágil a "
        "essa escolha arbitrária, e isso conta tanto quanto o número em si:\n"
    )
    linhas.append(
        "| Janela (dias) | Taxa mensal (pooled) | Taxa média/usuário | IC 95% | "
        "usuários elegíveis | user-meses elegíveis |"
    )
    linhas.append("|---:|---:|---:|---:|---:|---:|")
    for c in churns:
        marca = " ← definição adotada" if c.janela_dias == JANELA_CHURN_DECLARADA else ""
        linhas.append(
            f"| {c.janela_dias}{marca} | {c.taxa_pooled:.4f} | "
            f"{c.taxa_media_por_usuario:.4f} | [{c.ic_lo:.4f}, {c.ic_hi:.4f}] | "
            f"{c.usuarios_elegiveis:,} | {c.user_meses_elegiveis:,} |"
        )
    if len(churns) >= 2:
        taxas = [c.taxa_pooled for c in churns]
        variacao = (max(taxas) - min(taxas)) / max(taxas) if max(taxas) > 0 else 0.0
        linhas.append(
            f"\nVariação entre a janela mais curta e a mais longa testadas: "
            f"**{variacao:.0%}** da taxa pooled mais alta. "
            + (
                "É grande — trate a taxa de churn mensal como sensível à definição, não "
                "como um número único."
                if variacao > 0.30
                else "É moderada; a definição não é a maior fonte de incerteza aqui."
            )
        )
    linhas.append("")

    # Coortes
    linhas.append("## 3. Frequência e intensidade de sessão por coorte de perfil\n")
    linhas.append(AVISO_CURTO)
    if taxa_validacao_coortes < 0.90:
        linhas.append(
            f"\n> ❌ **Reconstrução de coortes falhou** (taxa de validação = "
            f"{taxa_validacao_coortes:.1%}, abaixo do limiar de 90%). O `--seed`, "
            "`--n-users` ou o tamanho do catálogo passados não batem com o que gerou "
            "este CSV. A tabela abaixo NÃO é confiável e não deve ser citada — "
            "regenere os dados com o mesmo `--seed` usado aqui, ou passe o seed correto.\n"
        )
    else:
        linhas.append(
            f"\nCoortes reconstruídas a partir da mesma sequência de sorteios de "
            f"`generate_synthetic_data.py` (não estão no CSV — ver docstring do módulo). "
            f"Validação: {taxa_validacao_coortes:.1%} dos usuários caem dentro da faixa de "
            "sessões que o próprio gerador declara para seu perfil (esperado: ~100%, com "
            "possível ambiguidade pontual na fronteira entre hardcore e borda, 150–200 "
            "sessões, onde os dois grupos se sobrepõem por desenho do gerador).\n"
        )
    linhas.append(
        "| Coorte | n usuários | Sessões/usuário (freq.) | IC 95% | Horas/sessão (intens.) | "
        "IC 95% | Sessões/mês de vida | IC 95% |"
    )
    linhas.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for m in coortes_metricas:
        linhas.append(
            f"| {m.coorte} | {m.n_usuarios:,} | {m.freq_media:.2f} | "
            f"[{m.freq_ic[0]:.2f}, {m.freq_ic[1]:.2f}] | {m.intensidade_media:.2f} | "
            f"[{m.intensidade_ic[0]:.2f}, {m.intensidade_ic[1]:.2f}] | "
            f"{m.sessoes_por_mes_media:.2f} | [{m.sessoes_por_mes_ic[0]:.2f}, "
            f"{m.sessoes_por_mes_ic[1]:.2f}] |"
        )
    linhas.append(
        "\nAs faixas de sessões por perfil são parâmetros fixos do gerador "
        f"(`{PERFIS_FAIXAS}`, borda `{BORDA_FAIXA}`), então a diferença entre coortes "
        "aqui é, por construção, a diferença entre essas faixas — não uma segmentação "
        "descoberta nos dados.\n"
    )

    # Vida
    linhas.append("## 4. Distribuição de tempo de vida do usuário\n")
    linhas.append(AVISO_CURTO)
    linhas.append(
        "\nTempo de vida = dias entre a primeira e a última sessão observadas no log "
        "(não é tempo até churn — um usuário pode voltar depois da última sessão "
        "observada; os dados só vão até onde vão).\n"
    )
    linhas.append(
        f"- Média: **{vida.media:.1f} dias**, IC 95% [{vida.ic_lo:.1f}, {vida.ic_hi:.1f}]\n"
    )
    linhas.append("| Percentil | Dias |")
    linhas.append("|---:|---:|")
    for pct, v in vida.percentis.items():
        linhas.append(f"| p{pct} | {v:.1f} |")
    linhas.append("")

    linhas.append("---\n")
    linhas.append(
        "Para o que não é calculável a partir destes dados — ROI, LTV, CAC, churn "
        "causalmente atribuído ao recomendador — ver "
        "[`reports/plano_medicao_negocio.md`](plano_medicao_negocio.md) (o que falta "
        "instrumentar) e [`reports/roi_cenarios.md`](roi_cenarios.md) (estimativa sob "
        "premissas explícitas, rotulada como projeção, nunca como medição)."
    )

    return "\n".join(linhas) + "\n"


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-dir", default="data/ml_ready")
    ap.add_argument("--out", default="reports/metricas_negocio.md")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--n-users",
        type=int,
        default=None,
        help="Usuários usados para gerar o CSV lido (default: inferido do próprio CSV)",
    )
    ap.add_argument(
        "--n-games",
        type=int,
        default=122_507,
        help="Tamanho do catálogo usado para gerar o CSV (default: N_GAMES do gerador)",
    )
    args = ap.parse_args()

    print(f"Lendo sessões de {args.csv_dir} ...")
    df = carregar_sessoes(args.csv_dir)
    n_users = args.n_users or int(df["usuario_id"].max()) + 1
    n_games = inferir_n_games(args.csv_dir, args.n_games)
    print(
        f"  usuários: {df.usuario_id.nunique():,} | sessões: {len(df):,} | catálogo: {n_games:,}\n"
    )

    print("Calculando retenção D0-D30 ...")
    retencao = calcular_retencao(df, args.seed)

    print("Calculando churn (janelas de 7, 14 e 30 dias) ...")
    churns = [calcular_churn(df, w, args.seed) for w in JANELAS_CHURN]

    print("Reconstruindo coortes de perfil e validando contra o CSV ...")
    coortes, taxa_validacao = reconstruir_coortes_validado(df, n_users, n_games, args.seed)
    print(f"  taxa de validação: {taxa_validacao:.1%}")

    print("Calculando frequência e intensidade por coorte ...")
    coortes_metricas = calcular_frequencia_intensidade(df, coortes, args.seed)

    print("Calculando distribuição de tempo de vida ...")
    vida = calcular_tempo_de_vida(df, args.seed)

    relatorio = formatar_relatorio(
        retencao,
        churns,
        coortes_metricas,
        taxa_validacao,
        vida,
        df.usuario_id.nunique(),
        len(df),
        args.seed,
        args.csv_dir,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(relatorio)
    print(f"\nRelatório escrito em {args.out}")


if __name__ == "__main__":
    sys.exit(main())
