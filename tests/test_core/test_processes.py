"""Tests for core/processes.py — discretisation schemes and GBM.

Tests verify mathematical convergence properties, not just execution.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt.core.processes import (
    CorrelatedGBM,
    EulerMaruyamaDiscretization,
    ExactGBMDiscretization,
    MilsteinDiscretization,
    simulate_path,
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


@pytest.fixture()
def rng():
    return np.random.default_rng(2024)


# ---------------------------------------------------------------------------
# Tests: CorrelatedGBM protocol compliance
# ---------------------------------------------------------------------------


class TestCorrelatedGBM:
    def test_size_and_factors(self, two_asset_gbm: CorrelatedGBM) -> None:
        assert two_asset_gbm.size() == 2
        assert two_asset_gbm.factors() == 2

    def test_initial_values(self, two_asset_gbm: CorrelatedGBM) -> None:
        x0 = two_asset_gbm.initial_values()
        assert_allclose(x0, [100.0, 100.0])
        x0[0] = 999.0
        assert_allclose(
            two_asset_gbm.initial_values(),
            [100.0, 100.0],
            err_msg="initial_values must return a copy",
        )

    def test_drift_shape(self, two_asset_gbm: CorrelatedGBM) -> None:
        x = np.array([100.0, 100.0])
        mu_val = two_asset_gbm.drift(0.0, x)
        assert mu_val.shape == (2,)

    def test_diffusion_shape(self, two_asset_gbm: CorrelatedGBM) -> None:
        x = np.array([100.0, 100.0])
        sigma_val = two_asset_gbm.diffusion(0.0, x)
        assert sigma_val.shape == (2, 2)

    def test_evolve_stays_positive(
        self, scalar_gbm: CorrelatedGBM, rng: np.random.Generator
    ) -> None:
        """GBM solutions are always positive."""
        x = np.array([100.0])
        for _ in range(1000):
            dw = rng.standard_normal(1) * np.sqrt(0.01)
            x = scalar_gbm.evolve(0.0, x, 0.01, dw)
            assert x[0] > 0

    def test_cholesky_decomposition(self, two_asset_gbm: CorrelatedGBM) -> None:
        """L L^T should reconstruct the covariance matrix."""
        L = two_asset_gbm._cholesky
        assert_allclose(L @ L.T, two_asset_gbm.cov, atol=1e-14)


# ---------------------------------------------------------------------------
# Tests: Euler-Maruyama convergence
# ---------------------------------------------------------------------------


class TestEulerMaruyama:
    def test_converges_to_exact_gbm(self, rng: np.random.Generator) -> None:
        """Euler-Maruyama converges to exact GBM solution as dt → 0.

        We check that halving the step size reduces the strong error,
        consistent with strong order 0.5.
        """
        mu_val, sigma_val = 0.05, 0.2
        T = 1.0
        n_paths = 500

        errors = {}
        for n_steps in [50, 200, 800]:
            dt = T / n_steps
            sqrt_dt = np.sqrt(dt)
            path_errors = []

            for _ in range(n_paths):
                x_euler = 100.0
                x_exact = 100.0
                for _ in range(n_steps):
                    dw_val = rng.standard_normal() * sqrt_dt
                    x_euler = (
                        x_euler + mu_val * x_euler * dt + sigma_val * x_euler * dw_val
                    )
                    x_exact = x_exact * np.exp(
                        (mu_val - 0.5 * sigma_val**2) * dt + sigma_val * dw_val
                    )
                path_errors.append(abs(x_euler - x_exact))

            errors[n_steps] = np.mean(path_errors)

        ratio_1 = errors[50] / errors[200]
        ratio_2 = errors[200] / errors[800]
        assert ratio_1 > 1.2, f"Error should decrease: ratio={ratio_1:.2f}"
        assert ratio_2 > 1.2, f"Error should decrease: ratio={ratio_2:.2f}"

    def test_discretization_class(
        self, scalar_gbm: CorrelatedGBM, rng: np.random.Generator
    ) -> None:
        """EulerMaruyamaDiscretization produces finite results."""
        em = EulerMaruyamaDiscretization()
        _, path = simulate_path(
            scalar_gbm, T=1.0, n_steps=100, rng=rng, discretization=em
        )
        assert np.all(np.isfinite(path))
        assert path.shape == (101, 1)


# ---------------------------------------------------------------------------
# Tests: Milstein convergence
# ---------------------------------------------------------------------------


class TestMilstein:
    def test_better_than_euler_for_gbm(self, rng: np.random.Generator) -> None:
        """Milstein should have smaller strong error than Euler for GBM."""
        mu_val, sigma_val = 0.05, 0.3
        T = 1.0
        n_steps = 100
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        n_paths = 1000

        euler_errors = []
        milstein_errors = []

        for _ in range(n_paths):
            x_euler = 100.0
            x_milstein = 100.0
            x_exact = 100.0

            for _ in range(n_steps):
                dw_val = rng.standard_normal() * sqrt_dt

                x_euler = x_euler + mu_val * x_euler * dt + sigma_val * x_euler * dw_val

                x_milstein = (
                    x_milstein
                    + mu_val * x_milstein * dt
                    + sigma_val * x_milstein * dw_val
                    + 0.5 * sigma_val**2 * x_milstein * (dw_val**2 - dt)
                )

                x_exact = x_exact * np.exp(
                    (mu_val - 0.5 * sigma_val**2) * dt + sigma_val * dw_val
                )

            euler_errors.append(abs(x_euler - x_exact))
            milstein_errors.append(abs(x_milstein - x_exact))

        mean_euler = np.mean(euler_errors)
        mean_milstein = np.mean(milstein_errors)

        assert mean_milstein < mean_euler, (
            f"Milstein ({mean_milstein:.4f}) should be more accurate "
            f"than Euler ({mean_euler:.4f})"
        )

    def test_milstein_class_1d(
        self, scalar_gbm: CorrelatedGBM, rng: np.random.Generator
    ) -> None:
        """MilsteinDiscretization runs on a 1-D process."""
        mil = MilsteinDiscretization()
        _, path = simulate_path(
            scalar_gbm, T=1.0, n_steps=100, rng=rng, discretization=mil
        )
        assert np.all(np.isfinite(path))
        assert np.all(path > 0)


# ---------------------------------------------------------------------------
# Tests: ExactGBMDiscretization
# ---------------------------------------------------------------------------


class TestExactGBM:
    def test_delegates_to_evolve(
        self, scalar_gbm: CorrelatedGBM, rng: np.random.Generator
    ) -> None:
        """Exact discretisation delegates to process.evolve."""
        exact = ExactGBMDiscretization()
        _, path = simulate_path(
            scalar_gbm, T=1.0, n_steps=50, rng=rng, discretization=exact
        )
        assert np.all(path > 0)

    def test_mean_matches_theory(self, rng: np.random.Generator) -> None:
        """E[S(T)] = S(0) exp(μT) for GBM."""
        mu_val = 0.05
        T = 1.0
        gbm = CorrelatedGBM(
            mu=np.array([mu_val]),
            cov=np.array([[0.04]]),
            x0=np.array([100.0]),
        )

        n_paths = 5000
        terminals = []
        for _ in range(n_paths):
            path_rng = np.random.default_rng(rng.integers(0, 2**31))
            _, path = simulate_path(gbm, T=T, n_steps=1, rng=path_rng)
            terminals.append(path[-1, 0])

        expected_mean = 100.0 * np.exp(mu_val * T)
        sample_mean = np.mean(terminals)
        assert_allclose(sample_mean, expected_mean, rtol=0.05)


# ---------------------------------------------------------------------------
# Tests: simulate_path
# ---------------------------------------------------------------------------


class TestSimulatePath:
    def test_path_shape(
        self, two_asset_gbm: CorrelatedGBM, rng: np.random.Generator
    ) -> None:
        times, path = simulate_path(two_asset_gbm, T=1.0, n_steps=50, rng=rng)
        assert times.shape == (51,)
        assert path.shape == (51, 2)
        assert_allclose(times[0], 0.0)
        assert_allclose(times[-1], 1.0)

    def test_initial_value(
        self, two_asset_gbm: CorrelatedGBM, rng: np.random.Generator
    ) -> None:
        _, path = simulate_path(two_asset_gbm, T=1.0, n_steps=10, rng=rng)
        assert_allclose(path[0], [100.0, 100.0])

    def test_all_positive(
        self, two_asset_gbm: CorrelatedGBM, rng: np.random.Generator
    ) -> None:
        """GBM paths stay positive."""
        _, path = simulate_path(two_asset_gbm, T=2.0, n_steps=500, rng=rng)
        assert np.all(path > 0)

    def test_market_weights_sum_to_one(
        self, two_asset_gbm: CorrelatedGBM, rng: np.random.Generator
    ) -> None:
        """Market weights derived from GBM prices sum to 1 at all times."""
        _, path = simulate_path(two_asset_gbm, T=1.0, n_steps=100, rng=rng)
        total_cap = path.sum(axis=1, keepdims=True)
        weights = path / total_cap
        row_sums = weights.sum(axis=1)
        assert_allclose(row_sums, 1.0, atol=1e-14)
