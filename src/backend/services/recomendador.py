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

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

import joblib
import numpy as np
import pandas as pd  # Moved to top-level for type hint in __init__
import torch

from src.config.paths import MODEL_PATHS
from src.config.settings import (  # Keep import, even if not used in this snippet, for completeness
    settings,
)
from src.models.meta.cgan_model import Generator
from src.utils.exceptions import InvalidModeError, ModelNotLoadedError
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Mode configuration ────────────────────────────────────────────────────────

MODES: Dict[str, Dict[str, float]] = {
    "conservador": {"threshold": 0.7, "exploracao": 0.1},
    "equilibrado": {"threshold": 0.5, "exploracao": 0.2},
    "aventureiro": {"threshold": 0.3, "exploracao": 0.3},
    "meta": {"threshold": 0.0, "exploracao": 0.2},  # threshold será dinâmico
}
DEFAULT_MODE: str = "meta"


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
        self._classificador: Optional[Any] = None  # Changed to Any for broader type
        self._clusterer: Optional[Any] = None  # Changed to Any for broader type
        self._ranker: Optional[Dict] = None
        self._cgan: Optional[Generator] = None
        self._cgan_params: Optional[Dict] = None
        self._user_features_cache: Optional[pd.DataFrame] = None  # pd.DataFrame is now defined
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

            # Use relative paths from project root (CWD in tests should be steam_analysis)
            root = Path(".").resolve()
            c4_path = root / "scripts" / "meta_learning" / "models" / "generator_final.pth"
            config_path = root / "src" / "config" / "cgan_params.json"

            missing = [
                str(p) for p in (c1_path, c2_path, c3_path, c4_path, config_path) if not p.exists()
            ]
            if missing:
                logger.warning("Missing model artefacts: %s", missing)
                return False

            self._classificador = joblib.load(c1_path)
            self._clusterer = joblib.load(c2_path)
            self._ranker = joblib.load(c3_path)

            # Load cGAN Generator
            with open(config_path, "r") as f:
                self._cgan_params = json.load(f)

            self._cgan = Generator(
                latent_dim=32,
                condition_dim=147,  # Ajustado para bater com os 147 features do JSON/checkpoint
                hidden_dim=128,
                dropout_rate=0.2,
            )
            # Security Note: Loading torch state_dict from trusted local file
            self._cgan.load_state_dict(torch.load(c4_path, map_location=torch.device("cpu")))
            self._cgan.eval()

            # Load User Feature Cache (Production "Feature Store" MVP)
            # import pandas as pd # Removed duplicate import

            try:
                # Merge train/test/val for full coverage of the 10k users
                f_train = pd.read_parquet(
                    root / "scripts" / "meta_learning" / "data" / "cgan_users_train.parquet"
                )
                f_test = pd.read_parquet(
                    root / "scripts" / "meta_learning" / "data" / "cgan_users_test.parquet"
                )
                f_val = pd.read_parquet(
                    root / "scripts" / "meta_learning" / "data" / "cgan_users_val.parquet"
                )
                self._user_features_cache = pd.concat([f_train, f_test, f_val]).drop_duplicates(
                    subset=["usuario_id"]
                )
                self._user_features_cache.set_index("usuario_id", inplace=True)
                logger.info(
                    "User feature cache loaded: %d profiles", len(self._user_features_cache)
                )
            except Exception as e:
                logger.warning(
                    "Could not load user feature cache (%s). Falling back to simulation.", e
                )

            n_users = len(self._ranker.get("user_map", {}))
            n_games = len(self._ranker.get("game_map", {}))
            logger.info(
                "All model layers loaded (including cGAN Meta-Observer) — users: %d, games: %d",
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

        # If meta mode, generate dynamic threshold
        if modo == "meta":
            threshold = await self._generate_meta_threshold(usuario_id)
            explicacao_prefix = f"Ajuste meta (threshold={threshold:.2f}): "
            logger.info(
                "Recommendation request [meta] | User: %s | Dynamic Threshold: %.4f",
                usuario_id,
                threshold,
            )
        else:
            explicacao_prefix = ""
            logger.info(
                "Recommendation request [%s] | User: %s | Static Threshold: %.4f",
                modo,
                usuario_id,
                threshold,
            )

        # Embeddings
        item_embeddings: np.ndarray = self._ranker["item_embeddings"]
        user_map: Dict[str, int] = self._ranker["user_map"]
        user_id_str = str(usuario_id)

        if user_id_str in user_map:
            user_idx = user_map[user_id_str]
            user_vec: np.ndarray = self._ranker["user_embeddings"][user_idx]
            explicacao = explicacao_prefix + "Baseado no seu perfil de jogo"
        else:
            # Cold start: global mean embedding
            user_vec = np.mean(self._ranker["user_embeddings"], axis=0)
            explicacao = explicacao_prefix + "Jogo popular recomendado para novos usuários"

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

    async def _generate_meta_threshold(
        self, usuario_id: UUID, features: Optional[torch.Tensor] = None
    ) -> float:
        """Use cGAN Generator to predict the optimal threshold for a user.

        Args:
            usuario_id: User UUID.
            features: Optional pre-computed feature tensor (1, 147). If None,
                fetches from internal simulation/DB.

        Returns:
            float: Predicted threshold in [0.25, 0.75].
        """
        if not self._cgan or not self._cgan_params:
            return 0.5  # Fallback

        try:
            # 1. Feature Assembly (Real Feature Store Lookup)
            if features is not None:
                condition = features
            elif (
                self._user_features_cache is not None
                and str(usuario_id) in self._user_features_cache.index
            ):
                # Get REAL features (147 columns defined in config)
                cols = self._cgan_params["features"]
                feat_row = self._user_features_cache.loc[str(usuario_id), cols].values.astype(
                    np.float32
                )
                condition = torch.from_numpy(feat_row).unsqueeze(0)
            else:
                # Cold Start / Out-of-cache simulation
                # import random # Removed duplicate import
                rng = random.Random(str(usuario_id))  # rng is used, so keep this line
                condition = torch.zeros(1, 147)

                user_map = self._ranker.get("user_map", {})
                if str(usuario_id) in user_map:
                    # User exists in SVD but not in cGAN cache? Use baseline mean
                    if self._user_features_cache is not None:
                        condition = torch.from_numpy(
                            self._user_features_cache[self._cgan_params["features"]]
                            .mean()
                            .values.astype(np.float32)
                        ).unsqueeze(0)
                    else:
                        condition[0, 60] = 1.0  # Indie bias
                else:
                    # Generic cold start
                    condition[0, 127 + 4] = 1.0  # Median cluster

            # 2. Inference
            noise = torch.randn(1, 32)

            with torch.no_grad():
                pred = self._cgan(noise, condition)
                threshold = float(pred.item())

            # 3. Safeguard
            final_t = max(0.25, min(0.75, threshold))
            return final_t

        except Exception as e:
            logger.warning("cGAN threshold generation failed (%s), using fallback 0.5", e)
            return 0.5


# ── Module-level singleton ────────────────────────────────────────────────────

from src.backend.database import db  # Injeção tardia para evitar loop circular

recomendador = RecomendadorService()
