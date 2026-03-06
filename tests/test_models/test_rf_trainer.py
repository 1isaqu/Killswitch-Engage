"""Unit tests for RandomForestTrainer — Layer 1.

These tests use synthetic data and do NOT require the real CSV files or
trained artefacts on disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models.classifier.rf_trainer import RandomForestTrainer


@pytest.fixture()
def synthetic_data():
    """Generate a small synthetic dataset for RF tests."""
    rng = np.random.default_rng(seed=42)
    X = rng.random((200, 5)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)  # linearly separable target
    return X, y


class TestRandomForestTrainer:
    """Tests for :class:`src.models.classifier.rf_trainer.RandomForestTrainer`."""

    def test_train_predict_shape(self, synthetic_data):
        """Fitted model should return probabilities of correct shape."""
        X, y = synthetic_data
        trainer = RandomForestTrainer()
        trainer.train(X, y)
        proba = trainer.predict_proba(X)
        assert proba.shape == (len(X),), "Probability array must have one value per sample."
        assert proba.min() >= 0.0 and proba.max() <= 1.0, "Probabilities must be in [0, 1]."

    def test_train_mismatched_shapes_raises(self):
        """Training with mismatched X / y sizes must raise ValueError."""
        trainer = RandomForestTrainer()
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 90)
        with pytest.raises(ValueError, match="samples"):
            trainer.train(X, y)

    def test_predict_before_train_raises(self):
        """Calling predict_proba before training must raise RuntimeError."""
        trainer = RandomForestTrainer()
        with pytest.raises(RuntimeError, match="not trained"):
            trainer.predict_proba(np.random.rand(10, 5))

    def test_save_and_load(self, synthetic_data, tmp_path: Path):
        """Saved and reloaded model must produce identical predictions."""
        X, y = synthetic_data
        trainer = RandomForestTrainer()
        trainer.train(X, y)
        proba_before = trainer.predict_proba(X)

        artefact = tmp_path / "test_rf.pkl"
        trainer.save(artefact)

        loaded = RandomForestTrainer()
        loaded.load(artefact)
        proba_after = loaded.predict_proba(X)

        np.testing.assert_array_equal(proba_before, proba_after)

    def test_load_missing_file_raises(self, tmp_path: Path):
        """Loading from a non-existent path must raise FileNotFoundError."""
        trainer = RandomForestTrainer()
        with pytest.raises(FileNotFoundError):
            trainer.load(tmp_path / "does_not_exist.pkl")
