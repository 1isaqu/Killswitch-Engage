import asyncio
from uuid import uuid4

import pytest

from src.backend.services.recomendador import recomendador


@pytest.mark.asyncio
async def test_meta_recommendation():
    """Verify that 'meta' mode generates a valid dynamic threshold and recommendations."""
    # Force loading if not already loaded to see debug output in -s mode
    if not recomendador.is_loaded:
        print("DEBUG: Manually calling _load_models from test")
        recomendador.is_loaded = recomendador._load_models()

    if not recomendador.is_loaded:
        pytest.skip("Models not loaded locally")

    user_id = uuid4()
    # Test 'meta' mode
    recs = await recomendador.get_recomendacoes(user_id, k=5, modo="meta")

    assert len(recs) <= 5
    assert all("id" in r for r in recs)
    assert all("score" in r for r in recs)

    # Check if explanation contains threshold (indicating cGAN was used)
    explanation = recs[0]["explicacao"]
    assert "Ajuste meta" in explanation or "Baseado no seu perfil" in explanation

    # Verify threshold is reported in the first recommendation's explanation if it's meta
    if "Ajuste meta" in explanation:
        import re

        match = re.search(r"threshold=(\d\.\d+)", explanation)
        assert match is not None
        threshold = float(match.group(1))
        assert 0.2 <= threshold <= 0.8


@pytest.mark.asyncio
async def test_invalid_mode():
    """Verify that an invalid mode raises an exception."""
    user_id = uuid4()
    from src.utils.exceptions import InvalidModeError

    with pytest.raises(InvalidModeError):
        await recomendador.get_recomendacoes(user_id, modo="invalid_mode")
