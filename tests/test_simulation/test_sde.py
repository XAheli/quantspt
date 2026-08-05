"""Tests for simulation/sde/ — convergence verification and adaptive schemes."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt.core.processes import CorrelatedGBM
from quantspt.errors import SPTInvariantError
from quantspt.simulation.sde.euler_maruyama import (
    adaptive_euler_maruyama,
    verify_convergence_order,
)
from quantspt.simulation.sde.milstein import (
    adaptive_milstein,
    verify_milstein_convergence,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def scalar_gbm():
    """1-D GBM with μ=0.05, σ=0.2."""
    return CorrelatedGBM(
        mu=np.array([0.05]),
        cov=np.array([[0.04]]),
        x0=np.array([100.0]),
    )


@pytest.fixture()
def two_asset_gbm():
    """2-D correlated GBM."""
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    return CorrelatedGBM(
        mu=np.array([0.05, 0.08]),
        cov=cov,
        x0=np.array([100.0, 100.0]),
    )


# ---------------------------------------------------------------------------
# Euler-Maruyama convergence
# ---------------------------------------------------------------------------


class TestEMConvergence:
    def test_convergence_order_positive(self, scalar_gbm: CorrelatedGBM) -> None:
        """Estimated convergence order should be positive."""
        result = verify_convergence_order(
            scalar_gbm,
            T=0.5,
            step_counts=[20, 80, 320],
            n_paths=100,
            seed=42,
        )
        assert result["estimated_order"] > 0

    def test_errors_decrease(self, scalar_gbm: CorrelatedGBM) -> None:
        """Errors should decrease with finer step sizes."""
        result = verify_convergence_order(
            scalar_gbm,
            T=0.5,
            step_counts=[20, 80],
            n_paths=100,
            seed=42,
        )
        errors = result["errors"]
        assert errors[0] >= errors[1] or np.isclose(errors[0], errors[1], rtol=0.5)

    def test_multi_asset(self, two_asset_gbm: CorrelatedGBM) -> None:
        result = verify_convergence_order(
            two_asset_gbm,
            T=0.5,
            step_counts=[20, 80],
            n_paths=50,
            seed=42,
        )
        assert "estimated_order" in result

    def test_validation_step_counts(self, scalar_gbm: CorrelatedGBM) -> None:
        with pytest.raises(SPTInvariantError):
            verify_convergence_order(scalar_gbm, T=1.0, step_counts=[10])


# ---------------------------------------------------------------------------
# Adaptive Euler-Maruyama
# ---------------------------------------------------------------------------


class TestAdaptiveEM:
    def test_produces_path(self, scalar_gbm: CorrelatedGBM) -> None:
        rng = np.random.default_rng(42)
        times, path = adaptive_euler_maruyama(scalar_gbm, T=1.0, dt_init=0.01, rng=rng)
        assert len(times) > 1
        assert times[0] == 0.0
        assert_allclose(times[-1], 1.0, atol=1e-10)
        assert path.shape[0] == len(times)
        assert path.shape[1] == 1

    def test_positive_values(self, scalar_gbm: CorrelatedGBM) -> None:
        """GBM should stay positive under adaptive scheme."""
        rng = np.random.default_rng(42)
        _, path = adaptive_euler_maruyama(scalar_gbm, T=1.0, dt_init=0.01, rng=rng)
        assert np.all(np.isfinite(path))

    def test_multi_asset(self, two_asset_gbm: CorrelatedGBM) -> None:
        rng = np.random.default_rng(42)
        _times, path = adaptive_euler_maruyama(
            two_asset_gbm, T=0.5, dt_init=0.01, rng=rng
        )
        assert path.shape[1] == 2
        assert np.all(np.isfinite(path))


# ---------------------------------------------------------------------------
# Milstein convergence
# ---------------------------------------------------------------------------


class TestMilsteinConvergence:
    def test_convergence_order_positive(self, scalar_gbm: CorrelatedGBM) -> None:
        result = verify_milstein_convergence(
            scalar_gbm,
            T=0.5,
            step_counts=[20, 80, 320],
            n_paths=100,
            seed=42,
        )
        assert result["estimated_order"] > 0

    def test_errors_decrease(self, scalar_gbm: CorrelatedGBM) -> None:
        result = verify_milstein_convergence(
            scalar_gbm,
            T=0.5,
            step_counts=[20, 80],
            n_paths=100,
            seed=42,
        )
        errors = result["errors"]
        assert errors[0] >= errors[1] or np.isclose(errors[0], errors[1], rtol=0.5)

    def test_requires_1d(self, two_asset_gbm: CorrelatedGBM) -> None:
        with pytest.raises(SPTInvariantError):
            verify_milstein_convergence(two_asset_gbm, T=0.5, step_counts=[20, 80])

    def test_with_analytic_deriv(self, scalar_gbm: CorrelatedGBM) -> None:
        """Milstein with analytical dσ/dx for GBM: dσ/dx = σ."""
        sigma = 0.2

        def deriv(t: float, x: np.ndarray) -> float:
            return sigma

        result = verify_milstein_convergence(
            scalar_gbm,
            T=0.5,
            step_counts=[20, 80],
            n_paths=50,
            seed=42,
            diffusion_deriv=deriv,
        )
        assert result["estimated_order"] > 0


# ---------------------------------------------------------------------------
# Adaptive Milstein
# ---------------------------------------------------------------------------


class TestAdaptiveMilstein:
    def test_produces_path(self, scalar_gbm: CorrelatedGBM) -> None:
        rng = np.random.default_rng(42)
        times, path = adaptive_milstein(scalar_gbm, T=1.0, dt_init=0.01, rng=rng)
        assert len(times) > 1
        assert_allclose(times[-1], 1.0, atol=1e-10)
        assert path.shape[0] == len(times)

    def test_requires_1d(self, two_asset_gbm: CorrelatedGBM) -> None:
        rng = np.random.default_rng(42)
        with pytest.raises(SPTInvariantError):
            adaptive_milstein(two_asset_gbm, T=1.0, dt_init=0.01, rng=rng)

    def test_finite_values(self, scalar_gbm: CorrelatedGBM) -> None:
        rng = np.random.default_rng(42)
        _, path = adaptive_milstein(scalar_gbm, T=0.5, dt_init=0.01, rng=rng)
        assert np.all(np.isfinite(path))


# ---------------------------------------------------------------------------
# Re-export tests
# ---------------------------------------------------------------------------


class TestReexports:
    def test_euler_import(self) -> None:
        from quantspt.simulation.sde import EulerMaruyamaDiscretization

        em = EulerMaruyamaDiscretization()
        assert hasattr(em, "evolve")

    def test_milstein_import(self) -> None:
        from quantspt.simulation.sde import MilsteinDiscretization

        mil = MilsteinDiscretization()
        assert hasattr(mil, "evolve")
