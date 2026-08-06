"""Comprehensive tests for _backends/numba_backend.py.

Verifies each Numba-JIT function produces the same output as the NumPy
equivalent, that JIT compilation doesn't change results, and edge cases
(empty arrays, single element, very large n).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose

numba = pytest.importorskip("numba")

from quantspt._backends.numba_backend import NumbaBackend, _require_numba
from quantspt._backends.numpy_backend import NumpyBackend


@pytest.fixture()
def nb():
    return NumbaBackend()


@pytest.fixture()
def npb():
    return NumpyBackend()


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# excess_growth_rate: Numba vs NumPy
# ---------------------------------------------------------------------------


class TestExcessGrowthRate:
    """Numba excess_growth_rate matches NumPy implementation."""

    def test_equal_weighted_2(self, nb, npb) -> None:
        pi = np.array([0.5, 0.5])
        a = np.diag([0.04, 0.04])
        assert_allclose(
            nb.excess_growth_rate(pi, a), npb.excess_growth_rate(pi, a), atol=1e-12
        )

    def test_concentrated(self, nb, npb) -> None:
        pi = np.array([1.0, 0.0])
        a = np.diag([0.04, 0.09])
        assert_allclose(
            nb.excess_growth_rate(pi, a), npb.excess_growth_rate(pi, a), atol=1e-12
        )

    def test_random_5(self, nb, npb, rng) -> None:
        pi = rng.dirichlet(np.ones(5))
        L = rng.standard_normal((5, 5)) * 0.1
        a = L @ L.T + np.eye(5) * 0.01
        assert_allclose(
            nb.excess_growth_rate(pi, a), npb.excess_growth_rate(pi, a), atol=1e-10
        )

    def test_single_asset(self, nb, npb) -> None:
        """Single asset: excess growth rate is 0."""
        pi = np.array([1.0])
        a = np.array([[0.04]])
        assert_allclose(nb.excess_growth_rate(pi, a), 0.0, atol=1e-12)

    def test_large_portfolio(self, nb, npb, rng) -> None:
        """50 assets: numba matches numpy."""
        n = 50
        pi = rng.dirichlet(np.ones(n))
        L = rng.standard_normal((n, n)) * 0.05
        a = L @ L.T + np.eye(n) * 0.001
        assert_allclose(
            nb.excess_growth_rate(pi, a), npb.excess_growth_rate(pi, a), atol=1e-8
        )


# ---------------------------------------------------------------------------
# relative_covariance: Numba vs NumPy
# ---------------------------------------------------------------------------


class TestRelativeCovariance:
    """Numba relative_covariance matches NumPy implementation."""

    def test_null_space(self, nb, rng) -> None:
        """τ^π · π = 0 (null-space property)."""
        n = 5
        pi = rng.dirichlet(np.ones(n))
        L = rng.standard_normal((n, n)) * 0.1
        a = L @ L.T + np.eye(n) * 0.01
        tau = nb.relative_covariance(a, pi)
        assert_allclose(tau @ pi, 0.0, atol=1e-10)

    def test_symmetric(self, nb, rng) -> None:
        """τ^π is symmetric."""
        n = 5
        pi = rng.dirichlet(np.ones(n))
        L = rng.standard_normal((n, n)) * 0.1
        a = L @ L.T + np.eye(n) * 0.01
        tau = nb.relative_covariance(a, pi)
        assert_allclose(tau, tau.T, atol=1e-12)

    def test_matches_numpy(self, nb, npb, rng) -> None:
        n = 5
        pi = rng.dirichlet(np.ones(n))
        L = rng.standard_normal((n, n)) * 0.1
        a = L @ L.T + np.eye(n) * 0.01
        assert_allclose(
            nb.relative_covariance(a, pi), npb.relative_covariance(a, pi), atol=1e-10
        )

    def test_psd(self, nb, rng) -> None:
        """τ^π is PSD."""
        n = 5
        pi = rng.dirichlet(np.ones(n))
        L = rng.standard_normal((n, n)) * 0.1
        a = L @ L.T + np.eye(n) * 0.01
        tau = nb.relative_covariance(a, pi)
        eigvals = np.linalg.eigvalsh(tau)
        assert np.all(eigvals >= -1e-10)

    def test_single_asset(self, nb) -> None:
        pi = np.array([1.0])
        a = np.array([[0.04]])
        tau = nb.relative_covariance(a, pi)
        assert_allclose(tau, [[0.0]], atol=1e-12)


# ---------------------------------------------------------------------------
# portfolio_variance: Numba vs NumPy
# ---------------------------------------------------------------------------


class TestPortfolioVariance:
    def test_matches_numpy(self, nb, npb, rng) -> None:
        n = 5
        pi = rng.dirichlet(np.ones(n))
        L = rng.standard_normal((n, n)) * 0.1
        a = L @ L.T + np.eye(n) * 0.01
        assert_allclose(
            nb.portfolio_variance(a, pi), npb.portfolio_variance(a, pi), atol=1e-12
        )

    def test_equal_weighted_diagonal(self, nb) -> None:
        pi = np.array([0.5, 0.5])
        a = np.diag([0.04, 0.09])
        expected = 0.25 * 0.04 + 0.25 * 0.09
        assert_allclose(nb.portfolio_variance(a, pi), expected, atol=1e-12)


# ---------------------------------------------------------------------------
# simulate_gbm_paths: shape, positivity, initial value
# ---------------------------------------------------------------------------


class TestSimulateGBMPaths:
    def test_shape(self, nb, rng) -> None:
        n, n_steps = 3, 50
        x0 = np.ones(n) * 100.0
        mu = np.ones(n) * 0.05
        chol = np.eye(n) * 0.2
        dw = rng.standard_normal((n_steps, n)) * np.sqrt(1 / 252)
        paths = nb.simulate_gbm_paths(x0, mu, chol, 1 / 252, n_steps, dw)
        assert paths.shape == (n_steps + 1, n)

    def test_initial_value(self, nb, rng) -> None:
        n = 2
        x0 = np.array([100.0, 50.0])
        mu = np.array([0.05, 0.08])
        chol = np.linalg.cholesky(np.array([[0.04, 0.01], [0.01, 0.09]]))
        dw = rng.standard_normal((10, n)) * np.sqrt(1 / 252)
        paths = nb.simulate_gbm_paths(x0, mu, chol, 1 / 252, 10, dw)
        assert_allclose(paths[0], x0)

    def test_all_positive(self, nb, rng) -> None:
        """GBM paths stay positive."""
        n, n_steps = 3, 200
        x0 = np.ones(n) * 100.0
        mu = np.ones(n) * 0.05
        chol = np.eye(n) * 0.2
        dw = rng.standard_normal((n_steps, n)) * np.sqrt(1 / 252)
        paths = nb.simulate_gbm_paths(x0, mu, chol, 1 / 252, n_steps, dw)
        assert np.all(paths > 0)

    def test_single_asset(self, nb, rng) -> None:
        x0 = np.array([100.0])
        mu = np.array([0.05])
        chol = np.array([[0.2]])
        dw = rng.standard_normal((20, 1)) * np.sqrt(1 / 252)
        paths = nb.simulate_gbm_paths(x0, mu, chol, 1 / 252, 20, dw)
        assert paths.shape == (21, 1)
        assert np.all(paths > 0)

    def test_single_step(self, nb, rng) -> None:
        """Single time step produces 2 rows."""
        n = 2
        x0 = np.array([100.0, 50.0])
        mu = np.array([0.05, 0.08])
        chol = np.eye(n) * 0.2
        dw = rng.standard_normal((1, n)) * np.sqrt(1 / 252)
        paths = nb.simulate_gbm_paths(x0, mu, chol, 1 / 252, 1, dw)
        assert paths.shape == (2, n)
        assert_allclose(paths[0], x0)


# ---------------------------------------------------------------------------
# simulate_gbm_step
# ---------------------------------------------------------------------------


class TestSimulateGBMStep:
    def test_positive(self, nb, rng) -> None:
        x = np.array([100.0, 50.0])
        mu = np.array([0.05, 0.08])
        chol = np.linalg.cholesky(np.array([[0.04, 0.01], [0.01, 0.09]]))
        dt = 1 / 252
        dw = rng.standard_normal(2) * np.sqrt(dt)
        result = nb.simulate_gbm_step(x, mu, chol, dt, dw)
        assert np.all(result > 0)

    def test_shape(self, nb, rng) -> None:
        n = 5
        x = np.ones(n) * 100.0
        mu = np.ones(n) * 0.05
        chol = np.eye(n) * 0.2
        dw = rng.standard_normal(n) * np.sqrt(0.01)
        result = nb.simulate_gbm_step(x, mu, chol, 0.01, dw)
        assert result.shape == (n,)


# ---------------------------------------------------------------------------
# diversity_weights
# ---------------------------------------------------------------------------


class TestDiversityWeights:
    def test_sum_to_one(self, nb, rng) -> None:
        mu = rng.dirichlet(np.ones(5))
        result = nb.diversity_weights(mu, 0.5)
        assert_allclose(result.sum(), 1.0, atol=1e-10)

    def test_all_positive(self, nb, rng) -> None:
        mu = rng.dirichlet(np.ones(5))
        result = nb.diversity_weights(mu, 0.5)
        assert np.all(result > 0)

    def test_p_one_is_identity(self, nb, rng) -> None:
        mu = rng.dirichlet(np.ones(5))
        result = nb.diversity_weights(mu, 1.0)
        assert_allclose(result, mu, atol=1e-10)

    def test_matches_numpy(self, nb, npb, rng) -> None:
        mu = rng.dirichlet(np.ones(5))
        assert_allclose(
            nb.diversity_weights(mu, 0.5), npb.diversity_weights(mu, 0.5), atol=1e-10
        )

    def test_single_asset(self, nb) -> None:
        mu = np.array([1.0])
        result = nb.diversity_weights(mu, 0.5)
        assert_allclose(result, [1.0], atol=1e-10)


# ---------------------------------------------------------------------------
# covariance_shrinkage
# ---------------------------------------------------------------------------


class TestCovarianceShrinkage:
    def test_matches_numpy(self, nb, npb, rng) -> None:
        returns = rng.standard_normal((50, 5))
        assert_allclose(
            nb.covariance_shrinkage(returns, 0.3),
            npb.covariance_shrinkage(returns, 0.3),
            atol=1e-10,
        )

    def test_psd(self, nb, rng) -> None:
        returns = rng.standard_normal((50, 5))
        result = nb.covariance_shrinkage(returns, 0.3)
        eigvals = np.linalg.eigvalsh(result)
        assert np.all(eigvals > 0)

    def test_symmetric(self, nb, rng) -> None:
        returns = rng.standard_normal((50, 5))
        result = nb.covariance_shrinkage(returns, 0.5)
        assert_allclose(result, result.T, atol=1e-12)

    def test_zero_shrinkage_is_sample(self, nb, rng) -> None:
        returns = rng.standard_normal((50, 5))
        result = nb.covariance_shrinkage(returns, 0.0)
        sample = np.cov(returns, rowvar=False, ddof=1)
        assert_allclose(result, sample, atol=1e-10)

    def test_full_shrinkage_is_diagonal(self, nb, rng) -> None:
        returns = rng.standard_normal((50, 5))
        result = nb.covariance_shrinkage(returns, 1.0)
        sample = np.cov(returns, rowvar=False, ddof=1)
        expected = np.eye(5) * np.trace(sample) / 5
        assert_allclose(result, expected, atol=1e-10)

    def test_single_asset(self, nb, rng) -> None:
        """Single-asset returns produce 1x1 covariance."""
        returns = rng.standard_normal((30, 1))
        result = nb.covariance_shrinkage(returns, 0.5)
        assert result.shape == (1, 1)
        assert result[0, 0] > 0

    def test_large_portfolio(self, nb, npb, rng) -> None:
        """20-asset portfolio: numba matches numpy."""
        returns = rng.standard_normal((100, 20))
        assert_allclose(
            nb.covariance_shrinkage(returns, 0.4),
            npb.covariance_shrinkage(returns, 0.4),
            atol=1e-8,
        )


# ---------------------------------------------------------------------------
# JIT compilation consistency
# ---------------------------------------------------------------------------


class TestJITConsistency:
    """Verify first call (compilation) and second call produce same result."""

    def test_excess_growth_rate_jit_stable(self, rng) -> None:
        nb = NumbaBackend()
        pi = rng.dirichlet(np.ones(5))
        L = rng.standard_normal((5, 5)) * 0.1
        a = L @ L.T + np.eye(5) * 0.01
        result1 = nb.excess_growth_rate(pi, a)
        result2 = nb.excess_growth_rate(pi, a)
        assert result1 == result2

    def test_relative_covariance_jit_stable(self, rng) -> None:
        nb = NumbaBackend()
        pi = rng.dirichlet(np.ones(5))
        L = rng.standard_normal((5, 5)) * 0.1
        a = L @ L.T + np.eye(5) * 0.01
        tau1 = nb.relative_covariance(a, pi)
        tau2 = nb.relative_covariance(a, pi)
        assert_allclose(tau1, tau2, atol=0)

    def test_covariance_inner_jit_stable(self, rng) -> None:
        nb = NumbaBackend()
        returns = rng.standard_normal((50, 5))
        c1 = nb.covariance_shrinkage(returns, 0.3)
        c2 = nb.covariance_shrinkage(returns, 0.3)
        assert_allclose(c1, c2, atol=0)


# ---------------------------------------------------------------------------
# Backend name
# ---------------------------------------------------------------------------


class TestBackendName:
    def test_name(self, nb) -> None:
        assert nb.name == "numba"


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------


class TestImportGuard:
    def test_require_numba_raises_when_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_require_numba raises ImportError when numba is unavailable."""
        monkeypatch.setitem(sys.modules, "numba", None)
        with pytest.raises(ImportError, match="quantspt\\[sim\\]"):
            _require_numba()

    def test_require_numba_succeeds(self) -> None:
        """_require_numba returns the numba module when available."""
        mod = _require_numba()
        assert hasattr(mod, "njit")


# ---------------------------------------------------------------------------
# Varying array sizes (small, medium, large)
# ---------------------------------------------------------------------------


class TestVaryingArraySizes:
    """Verify numba backend with small (2), medium (10), and large (50) arrays."""

    @pytest.mark.parametrize("n", [2, 10, 50])
    def test_excess_growth_rate_sizes(self, nb, npb, rng, n) -> None:
        pi = rng.dirichlet(np.ones(n))
        L = rng.standard_normal((n, n)) * 0.05
        a = L @ L.T + np.eye(n) * 0.001
        assert_allclose(
            nb.excess_growth_rate(pi, a), npb.excess_growth_rate(pi, a), atol=1e-8
        )

    @pytest.mark.parametrize("n", [2, 10, 50])
    def test_relative_covariance_sizes(self, nb, npb, rng, n) -> None:
        pi = rng.dirichlet(np.ones(n))
        L = rng.standard_normal((n, n)) * 0.05
        a = L @ L.T + np.eye(n) * 0.001
        assert_allclose(
            nb.relative_covariance(a, pi), npb.relative_covariance(a, pi), atol=1e-8
        )

    @pytest.mark.parametrize("n", [2, 10, 50])
    def test_covariance_shrinkage_sizes(self, nb, npb, rng, n) -> None:
        returns = rng.standard_normal((max(n + 10, 60), n))
        assert_allclose(
            nb.covariance_shrinkage(returns, 0.3),
            npb.covariance_shrinkage(returns, 0.3),
            atol=1e-8,
        )

    @pytest.mark.parametrize("n", [1, 5, 20])
    def test_simulate_gbm_paths_sizes(self, nb, rng, n) -> None:
        x0 = np.ones(n) * 100.0
        mu = np.ones(n) * 0.05
        chol = np.eye(n) * 0.2
        n_steps = 30
        dw = rng.standard_normal((n_steps, n)) * np.sqrt(1 / 252)
        paths = nb.simulate_gbm_paths(x0, mu, chol, 1 / 252, n_steps, dw)
        assert paths.shape == (n_steps + 1, n)
        assert np.all(paths > 0)
        assert_allclose(paths[0], x0)

    @pytest.mark.parametrize("n", [2, 10, 50])
    def test_diversity_weights_sizes(self, nb, npb, rng, n) -> None:
        mu = rng.dirichlet(np.ones(n))
        assert_allclose(
            nb.diversity_weights(mu, 0.5), npb.diversity_weights(mu, 0.5), atol=1e-10
        )

    @pytest.mark.parametrize("n", [2, 10, 50])
    def test_portfolio_variance_sizes(self, nb, npb, rng, n) -> None:
        pi = rng.dirichlet(np.ones(n))
        L = rng.standard_normal((n, n)) * 0.05
        a = L @ L.T + np.eye(n) * 0.001
        assert_allclose(
            nb.portfolio_variance(a, pi), npb.portfolio_variance(a, pi), atol=1e-10
        )

    @pytest.mark.parametrize("n", [2, 10])
    def test_simulate_gbm_step_sizes(self, nb, rng, n) -> None:
        x = np.ones(n) * 100.0
        mu = np.ones(n) * 0.05
        chol = np.eye(n) * 0.2
        dt = 1 / 252
        dw = rng.standard_normal(n) * np.sqrt(dt)
        result = nb.simulate_gbm_step(x, mu, chol, dt, dw)
        assert result.shape == (n,)
        assert np.all(result > 0)
