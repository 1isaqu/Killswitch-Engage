"""Refactored RecomendadorService — Layer orchestrator for the recommendation pipeline.

Loads the three model artefacts (Classifier, Cluster, Ranker) and serves
personalised game recommendations via :meth:`get_recomendacoes`.

Security notes (SECURITY_CHECKLIST § 4):
    - Artefact paths are resolved via ``src.config.paths``; no user-supplied
      paths are accepted.
    - Joblib deserialization executes arbitrary Python: only load artefacts
      produced by trusted training scripts.
    - User IDs are validated as ``UUID`` objects at the API layer (Pydantic),
      so this service receives typed values only.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from uuid import UUID

import joblib
import numpy as np

from src.config.paths import MODEL_PATHS
from src.utils.exceptions import InvalidModeError, ModelNotLoadedError
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Mode configuration ────────────────────────────────────────────────────────

MODES: Dict[str, Dict[str, float]] = {
    "conservador": {"threshold": 0.7, "exploracao": 0.1},
    "equilibrado": {"threshold": 0.5, "exploracao": 0.2},
    "aventureiro": {"threshold": 0.3, "exploracao": 0.3},
}
DEFAULT_MODE: str = "equilibrado"


# ── Service ───────────────────────────────────────────────────────────────────


class RecomendadorService:
    """Orchestrates the multi-layer recommendation pipeline.

    Layers:
        1. ``classificador`` — RandomForest filters candidate items.
        2. ``clusterer``     — KMeans maps users to archetypes.
        3. ``ranker``        — SVD-based collaborative ranker (LightFM bundle).

    The service is instantiated once at application startup. If any artefact
    is missing, ``is_loaded`` is ``False`` and endpoints return ``503``.

    Attributes:
        is_loaded (bool): True when all three model artefacts loaded successfully.
    """

    def __init__(self) -> None:
        self._classificador: Optional[object] = None
        self._clusterer: Optional[object] = None
        self._ranker: Optional[Dict] = None
        self.is_loaded: bool = self._load_models()

    # ── Loading ────────────────────────────────────────────────────────────

    def _load_models(self) -> bool:
        """Attempt to load all model artefacts from disk.

        Returns:
            bool: True if all three artefacts loaded successfully.
        """
        try:
            c1_path = MODEL_PATHS.classifier
            c2_path = MODEL_PATHS.resolve_cluster()
            c3_path = MODEL_PATHS.ranker

            missing = [str(p) for p in (c1_path, c2_path, c3_path) if not p.exists()]
            if missing:
                logger.warning("Missing model artefacts: %s", missing)
                return False

            self._classificador = joblib.load(c1_path)
            self._clusterer = joblib.load(c2_path)
            self._ranker = joblib.load(c3_path)

            n_users = len(self._ranker.get("user_map", {}))
            n_games = len(self._ranker.get("game_map", {}))
            logger.info(
                "All model layers loaded — users in ranker: %d, games in catalogue: %d",
                n_users,
                n_games,
            )
            return True

        except Exception:
            logger.exception("Failed to load recommendation models.")
            return False

    # ── Public API ─────────────────────────────────────────────────────────

    async def get_recomendacoes(
        self,
        usuario_id: UUID,
        k: int = 10,
        modo: Optional[str] = None,
    ) -> List[Dict]:
        """Return personalised game recommendations for a user.

        The recommendation flow:
            1. Resolve ``modo`` and its threshold / exploration fraction.
            2. Look up the user embedding (or fall back to cold-start average).
            3. Score all items via dot product with the user vector.
            4. Filter by threshold; relax if nothing passes.
            5. Fill `k` slots: top-N deterministic + M random exploration slots.

        Args:
            usuario_id (UUID): Target user's UUID.
            k (int): Number of items to return. Must be in [1, 50].
            modo (str | None): Recommendation mode — ``'conservador'``,
                ``'equilibrado'``, or ``'aventureiro'``. Defaults to
                :data:`DEFAULT_MODE`.

        Returns:
            List[Dict]: Each dict has keys ``id``, ``score``, ``modo``,
                ``explicacao``.

        Raises:
            ModelNotLoadedError: If the service is not ready.
            InvalidModeError: If ``modo`` is not one of the valid modes.
        """
        if not self.is_loaded or self._ranker is None:
            raise ModelNotLoadedError("RecomendadorService")

        # Validate mode
        if modo is not None and modo not in MODES:
            raise InvalidModeError(modo)
        modo = modo or DEFAULT_MODE
        cfg = MODES[modo]
        threshold: float = cfg["threshold"]
        exploracao: float = cfg["exploracao"]

        # Embeddings
        item_embeddings: np.ndarray = self._ranker["item_embeddings"]
        user_map: Dict[str, int] = self._ranker["user_map"]
        user_id_str = str(usuario_id)

        if user_id_str in user_map:
            user_idx = user_map[user_id_str]
            user_vec: np.ndarray = self._ranker["user_embeddings"][user_idx]
            explicacao = "Baseado no seu perfil de jogo"
        else:
            # Cold start: global mean embedding
            user_vec = np.mean(self._ranker["user_embeddings"], axis=0)
            explicacao = "Jogo popular recomendado para novos usuários"

        scores: np.ndarray = item_embeddings @ user_vec

        # Filter by threshold
        valid_mask = scores >= threshold
        valid_idx: np.ndarray = np.where(valid_mask)[0]

        # Relax threshold if nothing passes (e.g. cold-start with low scores)
        if len(valid_idx) == 0:
            valid_idx = np.argsort(scores)[::-1][: k * 2]

        ordered_idx = valid_idx[np.argsort(scores[valid_idx])[::-1]]

        # Exploration slots
        n_explore = int(k * exploracao)
        n_top = k - n_explore
        top_idx = ordered_idx[:n_top]

        if n_explore > 0 and len(valid_idx) > n_top:
            pool = valid_idx[n_top:]
            n_sample = min(n_explore, len(pool))
            explored = np.random.choice(pool, size=n_sample, replace=False)
            final_idx = np.concatenate([top_idx, explored])
        else:
            final_idx = top_idx

        reverse_map: Dict[int, str] = self._ranker["reverse_game_map"]
        return [
            {
                "id": str(reverse_map[int(idx)]),
                "score": float(scores[idx]),
                "modo": modo,
                "explicacao": explicacao,
            }
            for idx in final_idx[:k]
        ]


# ── Module-level singleton ────────────────────────────────────────────────────

recomendador = RecomendadorService()
