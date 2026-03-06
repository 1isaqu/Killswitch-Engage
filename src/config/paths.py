"""Centralised filesystem paths for all project artefacts.

All paths are derived from ``settings.MODELS_PATH`` and ``settings.DATA_PATH``
so they automatically respect environment overrides via ``.env``.

Usage:
    >>> from src.config.paths import MODEL_PATHS, DATA_PATHS
    >>> model = joblib.load(MODEL_PATHS.classifier)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config.settings import settings

# ── Resolved base directories ───────────────────────────────────────────────

_MODELS_DIR = Path(settings.MODELS_PATH)
_DATA_DIR = Path(settings.DATA_PATH)
_REPORTS_DIR = Path("reports")


@dataclass(frozen=True)
class _ModelPaths:
    """Paths to trained model artefacts."""

    classifier: Path
    cluster: Path
    cluster_legacy: Path  # fallback for hdbscan_model.pkl
    ranker: Path
    cgan_dir: Path

    def resolve_cluster(self) -> Path:
        """Return the cluster model path, falling back to the legacy name.

        Returns:
            Path: Path to the cluster model file that exists on disk.
        """
        return self.cluster if self.cluster.exists() else self.cluster_legacy


@dataclass(frozen=True)
class _DataPaths:
    """Paths to data files used across the pipeline."""

    raw_steam: Path
    imputed: Path
    ml_ready_games: Path
    ml_ready_users: Path


@dataclass(frozen=True)
class _ReportPaths:
    """Paths to report and figure output directories."""

    figures: Path
    insights: Path
    metrics: Path


# ── Public singletons ────────────────────────────────────────────────────────

MODEL_PATHS = _ModelPaths(
    classifier=_MODELS_DIR / "classificador_rf.pkl",
    cluster=_MODELS_DIR / "kmeans_clusters.pkl",
    cluster_legacy=_MODELS_DIR / "hdbscan_model.pkl",
    ranker=_MODELS_DIR / "lightfm_model.pkl",
    cgan_dir=_MODELS_DIR / "cgan",
)

DATA_PATHS = _DataPaths(
    raw_steam=Path(settings.RAW_STEAM_DATA_PATH),
    imputed=Path(settings.IMPUTED_DATA_PATH),
    ml_ready_games=_DATA_DIR / "ml_ready" / "jogos_features.csv",
    ml_ready_users=_DATA_DIR / "ml_ready" / "usuarios_features.csv",
)

REPORT_PATHS = _ReportPaths(
    figures=_REPORTS_DIR / "figures",
    insights=_REPORTS_DIR / "insights",
    metrics=_REPORTS_DIR / "metrics",
)


def ensure_directories() -> None:
    """Create all output directories if they do not already exist.

    Creates: models/, data/ml_ready/, reports/figures/, reports/insights/,
    reports/metrics/.
    """
    for directory in [
        _MODELS_DIR,
        MODEL_PATHS.cgan_dir,
        DATA_PATHS.ml_ready_games.parent,
        REPORT_PATHS.figures,
        REPORT_PATHS.insights,
        REPORT_PATHS.metrics,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
