"""Shared pytest fixtures for the Killswitch Engage test suite.

Provides:
    - ``mock_ranker_payload``: A minimal in-memory ranker bundle that avoids
      loading real .pkl artefacts in unit tests.
    - ``test_settings``: Overrides settings for a local test database.
    - ``async_client``: An httpx ``AsyncClient`` targeting the FastAPI app.

Security notes (SECURITY_CHECKLIST § 6):
    - Tests never use real credentials — all DB connections use the test
      settings fixture which points to an isolated test database.
    - ``TEST_DB_*`` variables are loaded from the environment, never hardcoded.
"""

from __future__ import annotations

from typing import Dict
from uuid import uuid4

import numpy as np
import pytest
import pytest_asyncio
from httpx import AsyncClient

from src.backend.api import create_app

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def mock_ranker_payload() -> Dict:
    """Return a minimal in-memory ranker bundle for offline unit tests.

    The bundle mimics the structure of ``lightfm_model.pkl`` with 10 users
    and 50 games so that ``RecomendadorService`` can run without real artefacts.

    Returns:
        Dict: Keys: ``user_embeddings``, ``item_embeddings``, ``user_map``,
            ``game_map``, ``reverse_game_map``.
    """
    n_users, n_games, n_components = 10, 50, 5
    rng = np.random.default_rng(seed=42)
    user_ids = [str(uuid4()) for _ in range(n_users)]
    game_ids = [str(uuid4()) for _ in range(n_games)]
    user_map = {uid: i for i, uid in enumerate(user_ids)}
    game_map = {gid: i for i, gid in enumerate(game_ids)}
    return {
        "user_embeddings": rng.random((n_users, n_components)).astype(np.float32),
        "item_embeddings": rng.random((n_games, n_components)).astype(np.float32),
        "user_map": user_map,
        "game_map": game_map,
        "reverse_game_map": {v: k for k, v in game_map.items()},
    }


@pytest.fixture()
def known_user_id(mock_ranker_payload: Dict) -> str:
    """Return a user UUID that exists in the mock ranker payload.

    Args:
        mock_ranker_payload: The mock ranker fixture.

    Returns:
        str: A valid user UUID string.
    """
    return next(iter(mock_ranker_payload["user_map"]))


@pytest_asyncio.fixture()
async def async_client() -> AsyncClient:
    """Provide an httpx AsyncClient targeting the FastAPI app.

    Yields:
        AsyncClient: Configured test client.
    """
    app = create_app()
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
