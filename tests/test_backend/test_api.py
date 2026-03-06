"""Tests for the /api/v1/recomendacoes endpoints.

Smoke tests requiring a live database are marked ``requires_db``
and excluded from CI with ``pytest -m 'not requires_db'``.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from src.backend.api import create_app


@pytest.mark.asyncio
async def test_health_endpoint():
    """GET /health should return 200 with status=healthy."""
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


@pytest.mark.asyncio
async def test_invalid_mode_returns_400():
    """GET /api/v1/recomendacoes/<uuid>?modo=xyz should return 400."""
    app = create_app()
    uid = uuid.uuid4()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(f"/api/v1/recomendacoes/{uid}?modo=xyz")
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_invalid_uuid_returns_422():
    """GET /api/v1/recomendacoes/not-a-uuid should return 422."""
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/recomendacoes/not-a-uuid")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_k_out_of_range_returns_422():
    """k > 50 should return 422 (Pydantic Query bound validation)."""
    app = create_app()
    uid = uuid.uuid4()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(f"/api/v1/recomendacoes/{uid}?k=999")
    assert response.status_code == 422


@pytest.mark.asyncio
@pytest.mark.requires_db
async def test_recommendations_smoke_real_user():
    """Smoke test — requires a real DB and loaded models. Run manually."""
    pass  # placeholder — implement with a seeded DB fixture
