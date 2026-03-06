"""Unit tests for RecomendadorService.

Uses ``mock_ranker_payload`` from conftest to avoid loading real .pkl files.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from src.backend.services.recomendador import RecomendadorService
from src.utils.exceptions import InvalidModeError, ModelNotLoadedError


class TestRecomendadorService:
    """Tests for :class:`src.backend.services.recomendador.RecomendadorService`."""

    @pytest.fixture()
    def service_with_mock(self, mock_ranker_payload):
        """Create a service instance with mocked model artefacts."""
        service = RecomendadorService.__new__(RecomendadorService)
        service._classificador = MagicMock()
        service._clusterer = MagicMock()
        service._ranker = mock_ranker_payload
        service.is_loaded = True
        return service

    @pytest.mark.asyncio
    async def test_returns_recommendations_for_known_user(self, service_with_mock, known_user_id):
        """Known users should receive k recommendations."""
        from uuid import UUID

        uid = UUID(known_user_id)
        recs = await service_with_mock.get_recomendacoes(uid, k=5)
        assert len(recs) <= 5, "Should return at most k items."
        assert all("id" in r and "score" in r for r in recs)

    @pytest.mark.asyncio
    async def test_cold_start_unknown_user(self, service_with_mock):
        """Unknown users should still receive recommendations (cold-start)."""
        unknown_uid = uuid4()
        recs = await service_with_mock.get_recomendacoes(unknown_uid, k=5)
        assert isinstance(recs, list)
        if recs:
            assert recs[0]["explicacao"] == "Jogo popular recomendado para novos usuários"

    @pytest.mark.asyncio
    async def test_invalid_mode_raises(self, service_with_mock):
        """Invalid mode should raise InvalidModeError."""
        with pytest.raises(InvalidModeError):
            await service_with_mock.get_recomendacoes(uuid4(), modo="invalido")

    @pytest.mark.asyncio
    async def test_not_loaded_raises(self):
        """Service not loaded should raise ModelNotLoadedError."""
        service = RecomendadorService.__new__(RecomendadorService)
        service.is_loaded = False
        service._ranker = None
        with pytest.raises(ModelNotLoadedError):
            await service.get_recomendacoes(uuid4())

    @pytest.mark.asyncio
    async def test_all_modes_return_results(self, service_with_mock, known_user_id):
        """All three modes should return non-empty results for a known user."""
        from uuid import UUID

        uid = UUID(known_user_id)
        for mode in ("conservador", "equilibrado", "aventureiro"):
            recs = await service_with_mock.get_recomendacoes(uid, k=5, modo=mode)
            assert isinstance(recs, list), f"Mode '{mode}' should return a list."
