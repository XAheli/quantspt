"""Shared fixtures for quantspt tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def rng() -> np.random.Generator:
    """Seeded random generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture()
def sample_weights_2() -> np.ndarray:
    """Simple 2-asset weight vector."""
    return np.array([0.6, 0.4])


@pytest.fixture()
def sample_weights_5(rng: np.random.Generator) -> np.ndarray:
    """Dirichlet-sampled 5-asset weight vector."""
    return rng.dirichlet(np.ones(5))


@pytest.fixture()
def diagonal_cov_2() -> np.ndarray:
    """2×2 diagonal covariance matrix."""
    return np.diag([0.04, 0.09])


@pytest.fixture()
def psd_cov_5(rng: np.random.Generator) -> np.ndarray:
    """Random 5×5 PSD covariance matrix."""
    L = rng.standard_normal((5, 5))
    return L @ L.T + np.eye(5) * 0.01
