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


# ---------------------------------------------------------------------------
# Realistic market weight fixtures (Pareto-distributed, various sizes)
# ---------------------------------------------------------------------------


def _pareto_weights(rng: np.random.Generator, n: int) -> np.ndarray:
    """Generate Pareto-distributed market weights (power-law, top-heavy)."""
    raw = (rng.pareto(1.0, size=n) + 1.0).astype(np.float64)
    return raw / raw.sum()


@pytest.fixture()
def pareto_weights_5() -> np.ndarray:
    """Pareto-distributed 5-asset market weights (top stock ~30-50%)."""
    return _pareto_weights(np.random.default_rng(42), 5)


@pytest.fixture()
def pareto_weights_50() -> np.ndarray:
    """Pareto-distributed 50-asset market weights (top stock ~5-15%)."""
    return _pareto_weights(np.random.default_rng(42), 50)


@pytest.fixture()
def dominant_stock_weights() -> np.ndarray:
    """Edge case: one dominant stock at ~90%."""
    w = np.array([0.90, 0.04, 0.03, 0.02, 0.01])
    return w / w.sum()


@pytest.fixture()
def near_equal_weights_5() -> np.ndarray:
    """Edge case: near-equal weights (close to 1/n)."""
    rng = np.random.default_rng(42)
    w = np.ones(5) / 5.0 + rng.normal(0, 1e-4, 5)
    w = np.abs(w)
    return w / w.sum()


@pytest.fixture()
def near_zero_weights() -> np.ndarray:
    """Edge case: some weights very close to zero."""
    w = np.array([0.50, 0.30, 0.15, 0.04, 1e-6])
    return w / w.sum()
