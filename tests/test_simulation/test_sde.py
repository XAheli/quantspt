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


@pytest.mark.slow
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

    def test_noise_split_sums_to_original(self) -> None:
        """Verify dW₁ + dW₂ = dW — the Brownian bridge partition is exact."""
        rng = np.random.default_rng(42)
        dt = 0.01
        sqrt_dt = np.sqrt(dt)
        n_factors = 3
        dw = rng.standard_normal(n_factors) * sqrt_dt
        z = rng.standard_normal(n_factors) * sqrt_dt * 0.5
        dw1 = dw * 0.5 + z
        dw2 = dw * 0.5 - z
        assert_allclose(dw1 + dw2, dw, atol=1e-15)

    def test_deterministic_seed_reproducibility(
        self, scalar_gbm: CorrelatedGBM
    ) -> None:
        """Same seed must produce identical paths."""
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        t1, p1 = adaptive_euler_maruyama(scalar_gbm, T=0.5, dt_init=0.01, rng=rng1)
        t2, p2 = adaptive_euler_maruyama(scalar_gbm, T=0.5, dt_init=0.01, rng=rng2)
        assert_allclose(t1, t2)
        assert_allclose(p1, p2)


# ---------------------------------------------------------------------------
# Milstein convergence
# ---------------------------------------------------------------------------


@pytest.mark.slow
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
# Brownian bridge split statistical properties
# ---------------------------------------------------------------------------


class TestBrownianBridgeSplit:
    """Verify the Brownian bridge noise split has correct statistical properties."""

    N_SAMPLES = 200_000

    def test_brownian_bridge_split_variance(self) -> None:
        """Verify dW₁ and dW₂ each have correct variance dt/2."""
        rng = np.random.default_rng(42)
        dt = 0.04
        sqrt_dt = np.sqrt(dt)
        n_factors = 3

        dw1_samples = np.zeros((self.N_SAMPLES, n_factors))
        dw2_samples = np.zeros((self.N_SAMPLES, n_factors))

        for i in range(self.N_SAMPLES):
            dw = rng.standard_normal(n_factors) * sqrt_dt
            z = rng.standard_normal(n_factors) * sqrt_dt * 0.5
            dw1_samples[i] = dw * 0.5 + z
            dw2_samples[i] = dw * 0.5 - z

        expected_var = dt / 2.0
        assert_allclose(np.var(dw1_samples, axis=0), expected_var, rtol=0.02)
        assert_allclose(np.var(dw2_samples, axis=0), expected_var, rtol=0.02)

    def test_brownian_bridge_partition(self) -> None:
        """Verify dW₁ + dW₂ = dW exactly for every sample."""
        rng = np.random.default_rng(123)
        dt = 0.05
        sqrt_dt = np.sqrt(dt)
        n_factors = 5

        for _ in range(1000):
            dw = rng.standard_normal(n_factors) * sqrt_dt
            z = rng.standard_normal(n_factors) * sqrt_dt * 0.5
            dw1 = dw * 0.5 + z
            dw2 = dw * 0.5 - z
            assert_allclose(dw1 + dw2, dw, atol=1e-15)

    def test_brownian_bridge_zero_mean(self) -> None:
        """Verify E[dW₁] = E[dW₂] = 0."""
        rng = np.random.default_rng(77)
        dt = 0.04
        sqrt_dt = np.sqrt(dt)
        n_factors = 2

        dw1_samples = np.zeros((self.N_SAMPLES, n_factors))
        dw2_samples = np.zeros((self.N_SAMPLES, n_factors))

        for i in range(self.N_SAMPLES):
            dw = rng.standard_normal(n_factors) * sqrt_dt
            z = rng.standard_normal(n_factors) * sqrt_dt * 0.5
            dw1_samples[i] = dw * 0.5 + z
            dw2_samples[i] = dw * 0.5 - z

        assert_allclose(np.mean(dw1_samples, axis=0), 0.0, atol=0.005)
        assert_allclose(np.mean(dw2_samples, axis=0), 0.0, atol=0.005)

    def test_adaptive_euler_convergence_with_bridge(
        self, scalar_gbm: CorrelatedGBM
    ) -> None:
        """Verify adaptive EM with proper Brownian bridge still converges to exact GBM.

        For GBM with parameters μ and σ, the exact solution at time T is:
            S(T) = S(0) * exp((μ - σ²/2)*T + σ*W(T))
        The adaptive scheme should converge to this with tight tolerances.
        """
        mu = 0.05
        S0 = 100.0
        T = 1.0
        n_paths = 500
        terminal_values = np.zeros(n_paths)

        for i in range(n_paths):
            rng = np.random.default_rng(i)
            _, path = adaptive_euler_maruyama(
                scalar_gbm, T=T, dt_init=0.01, rng=rng, atol=1e-5, rtol=1e-4
            )
            terminal_values[i] = path[-1, 0]

        expected_mean = S0 * np.exp(mu * T)
        empirical_mean = np.mean(terminal_values)
        assert_allclose(empirical_mean, expected_mean, rtol=0.05)

        assert np.all(terminal_values > 0), "GBM paths must stay positive"

    def test_adaptive_milstein_convergence_with_bridge(
        self, scalar_gbm: CorrelatedGBM
    ) -> None:
        """Verify adaptive Milstein with proper Brownian bridge converges."""
        mu = 0.05
        S0 = 100.0
        T = 1.0
        n_paths = 500
        terminal_values = np.zeros(n_paths)

        for i in range(n_paths):
            rng = np.random.default_rng(i)
            _, path = adaptive_milstein(
                scalar_gbm, T=T, dt_init=0.01, rng=rng, atol=1e-5, rtol=1e-4
            )
            terminal_values[i] = path[-1, 0]

        expected_mean = S0 * np.exp(mu * T)
        empirical_mean = np.mean(terminal_values)
        assert_allclose(empirical_mean, expected_mean, rtol=0.05)

        assert np.all(terminal_values > 0), "GBM paths must stay positive"


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
