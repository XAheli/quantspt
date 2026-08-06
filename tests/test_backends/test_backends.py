"""Tests for _backends — NumPy, JAX, and Numba compute backends."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt._backends import (
    _reset_registry,
    get_backend,
    list_backends,
    register_backend,
    set_backend,
)
from quantspt._backends.numpy_backend import NumpyBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_backend():
    """Reset backend registry between tests."""
    _reset_registry()
    yield
    _reset_registry()


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


@pytest.fixture()
def weights_5(rng):
    return rng.dirichlet(np.ones(5))


@pytest.fixture()
def cov_5(rng):
    L = rng.standard_normal((5, 5))
    return L @ L.T + np.eye(5) * 0.01


@pytest.fixture()
def numpy_backend():
    return NumpyBackend()


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestBackendRegistry:
    def test_default_is_numpy(self) -> None:
        backend = get_backend()
        assert backend.name == "numpy"

    def test_list_backends(self) -> None:
        names = list_backends()
        assert "numpy" in names

    def test_set_backend(self) -> None:
        set_backend("numpy")
        assert get_backend().name == "numpy"

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(KeyError, match="not registered"):
            get_backend("nonexistent")

    def test_register_custom_backend(self, numpy_backend) -> None:
        register_backend("custom", numpy_backend)
        assert get_backend("custom").name == "numpy"


# ---------------------------------------------------------------------------
# NumPy backend tests
# ---------------------------------------------------------------------------


class TestNumpyBackend:
    def test_excess_growth_rate_equal_weighted(self, numpy_backend) -> None:
        pi = np.array([0.5, 0.5])
        a = np.diag([0.04, 0.04])
        result = numpy_backend.excess_growth_rate(pi, a)
        assert_allclose(result, 0.01, atol=1e-10)

    def test_excess_growth_rate_concentrated(self, numpy_backend) -> None:
        pi = np.array([1.0, 0.0])
        a = np.diag([0.04, 0.04])
        result = numpy_backend.excess_growth_rate(pi, a)
        assert_allclose(result, 0.0, atol=1e-10)

    def test_excess_growth_rate_nonnegative(
        self, numpy_backend, weights_5, cov_5
    ) -> None:
        result = numpy_backend.excess_growth_rate(weights_5, cov_5)
        assert result >= -1e-10

    def test_relative_covariance_null_space(
        self, numpy_backend, weights_5, cov_5
    ) -> None:
        tau = numpy_backend.relative_covariance(cov_5, weights_5)
        assert_allclose(tau @ weights_5, 0, atol=1e-10)

    def test_relative_covariance_psd(self, numpy_backend, weights_5, cov_5) -> None:
        tau = numpy_backend.relative_covariance(cov_5, weights_5)
        eigenvalues = np.linalg.eigvalsh(tau)
        assert np.all(eigenvalues >= -1e-10)

    def test_relative_covariance_symmetric(
        self, numpy_backend, weights_5, cov_5
    ) -> None:
        tau = numpy_backend.relative_covariance(cov_5, weights_5)
        assert_allclose(tau, tau.T, atol=1e-12)

    def test_portfolio_variance(self, numpy_backend) -> None:
        pi = np.array([0.6, 0.4])
        a = np.array([[0.04, 0.01], [0.01, 0.09]])
        result = numpy_backend.portfolio_variance(a, pi)
        expected = float(pi @ a @ pi)
        assert_allclose(result, expected, atol=1e-12)

    def test_simulate_gbm_step_positive(self, numpy_backend, rng) -> None:
        x = np.array([100.0, 50.0])
        mu = np.array([0.05, 0.08])
        cholesky = np.linalg.cholesky(np.array([[0.04, 0.01], [0.01, 0.09]]))
        dt = 1.0 / 252.0
        dw = rng.standard_normal(2) * np.sqrt(dt)
        result = numpy_backend.simulate_gbm_step(x, mu, cholesky, dt, dw)
        assert np.all(result > 0)

    def test_simulate_gbm_step_shape(self, numpy_backend, rng) -> None:
        n = 5
        x = np.ones(n) * 100.0
        mu = np.ones(n) * 0.05
        cholesky = np.eye(n) * 0.2
        dt = 0.01
        dw = rng.standard_normal(n) * np.sqrt(dt)
        result = numpy_backend.simulate_gbm_step(x, mu, cholesky, dt, dw)
        assert result.shape == (n,)

    def test_diversity_weights_sum_to_one(self, numpy_backend, weights_5) -> None:
        result = numpy_backend.diversity_weights(weights_5, 0.5)
        assert_allclose(np.sum(result), 1.0, atol=1e-10)

    def test_diversity_weights_positive(self, numpy_backend, weights_5) -> None:
        result = numpy_backend.diversity_weights(weights_5, 0.5)
        assert np.all(result > 0)

    def test_diversity_weights_p_one_is_identity(
        self, numpy_backend, weights_5
    ) -> None:
        result = numpy_backend.diversity_weights(weights_5, 1.0)
        assert_allclose(result, weights_5, atol=1e-10)

    def test_covariance_shrinkage_psd(self, numpy_backend, rng) -> None:
        returns = rng.standard_normal((50, 5))
        result = numpy_backend.covariance_shrinkage(returns, 0.3)
        eigenvalues = np.linalg.eigvalsh(result)
        assert np.all(eigenvalues > 0)

    def test_covariance_shrinkage_symmetric(self, numpy_backend, rng) -> None:
        returns = rng.standard_normal((50, 5))
        result = numpy_backend.covariance_shrinkage(returns, 0.5)
        assert_allclose(result, result.T, atol=1e-12)

    def test_covariance_shrinkage_zero_is_sample(self, numpy_backend, rng) -> None:
        returns = rng.standard_normal((50, 5))
        result = numpy_backend.covariance_shrinkage(returns, 0.0)
        sample = np.cov(returns, rowvar=False, ddof=1)
        assert_allclose(result, sample, atol=1e-10)

    def test_covariance_shrinkage_one_is_diagonal(self, numpy_backend, rng) -> None:
        returns = rng.standard_normal((50, 5))
        result = numpy_backend.covariance_shrinkage(returns, 1.0)
        sample = np.cov(returns, rowvar=False, ddof=1)
        expected = np.eye(5) * np.trace(sample) / 5
        assert_allclose(result, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# JAX backend tests (skipped if JAX not installed)
# ---------------------------------------------------------------------------


jax_available = False
try:
    import jax  # noqa: F401

    jax_available = True
except ImportError:
    pass


@pytest.mark.skipif(not jax_available, reason="JAX not installed")
class TestJaxBackend:
    @pytest.fixture()
    def jax_backend(self):
        from quantspt._backends.jax_backend import JaxBackend

        return JaxBackend()

    def test_excess_growth_rate(self, jax_backend, numpy_backend, weights_5, cov_5):
        jax_result = jax_backend.excess_growth_rate(weights_5, cov_5)
        np_result = numpy_backend.excess_growth_rate(weights_5, cov_5)
        assert_allclose(jax_result, np_result, atol=1e-6)

    def test_relative_covariance(self, jax_backend, numpy_backend, weights_5, cov_5):
        jax_result = jax_backend.relative_covariance(cov_5, weights_5)
        np_result = numpy_backend.relative_covariance(cov_5, weights_5)
        assert_allclose(jax_result, np_result, atol=1e-6)

    def test_portfolio_variance(self, jax_backend, numpy_backend, weights_5, cov_5):
        jax_result = jax_backend.portfolio_variance(cov_5, weights_5)
        np_result = numpy_backend.portfolio_variance(cov_5, weights_5)
        assert_allclose(jax_result, np_result, atol=1e-6)

    def test_diversity_weights(self, jax_backend, numpy_backend, weights_5):
        jax_result = jax_backend.diversity_weights(weights_5, 0.5)
        np_result = numpy_backend.diversity_weights(weights_5, 0.5)
        assert_allclose(jax_result, np_result, atol=1e-6)

    def test_gradient(self, jax_backend):
        import jax.numpy as jnp

        def f(x):
            return jnp.sum(x**2)

        x = np.array([1.0, 2.0, 3.0])
        grad = jax_backend.gradient(f, x)
        assert_allclose(grad, 2.0 * x, atol=1e-5)

    def test_hessian(self, jax_backend):
        import jax.numpy as jnp

        def f(x):
            return jnp.sum(x**2)

        x = np.array([1.0, 2.0, 3.0])
        hess = jax_backend.hessian(f, x)
        assert_allclose(hess, 2.0 * np.eye(3), atol=1e-5)


# ---------------------------------------------------------------------------
# Numba backend tests (skipped if numba not installed)
# ---------------------------------------------------------------------------


numba_available = False
try:
    import numba  # noqa: F401

    numba_available = True
except ImportError:
    pass


@pytest.mark.skipif(not numba_available, reason="Numba not installed")
class TestNumbaBackend:
    @pytest.fixture()
    def numba_backend(self):
        from quantspt._backends.numba_backend import NumbaBackend

        return NumbaBackend()

    def test_excess_growth_rate(self, numba_backend, numpy_backend, weights_5, cov_5):
        nb_result = numba_backend.excess_growth_rate(weights_5, cov_5)
        np_result = numpy_backend.excess_growth_rate(weights_5, cov_5)
        assert_allclose(nb_result, np_result, atol=1e-10)

    def test_relative_covariance(self, numba_backend, numpy_backend, weights_5, cov_5):
        nb_result = numba_backend.relative_covariance(cov_5, weights_5)
        np_result = numpy_backend.relative_covariance(cov_5, weights_5)
        assert_allclose(nb_result, np_result, atol=1e-10)

    def test_portfolio_variance(self, numba_backend, numpy_backend, weights_5, cov_5):
        nb_result = numba_backend.portfolio_variance(cov_5, weights_5)
        np_result = numpy_backend.portfolio_variance(cov_5, weights_5)
        assert_allclose(nb_result, np_result, atol=1e-10)

    def test_simulate_gbm_paths(self, numba_backend, rng):
        x0 = np.array([100.0, 50.0])
        mu = np.array([0.05, 0.08])
        cholesky = np.linalg.cholesky(np.array([[0.04, 0.01], [0.01, 0.09]]))
        dt = 1.0 / 252.0
        n_steps = 100
        dw = rng.standard_normal((n_steps, 2)) * np.sqrt(dt)
        paths = numba_backend.simulate_gbm_paths(x0, mu, cholesky, dt, n_steps, dw)
        assert paths.shape == (101, 2)
        assert np.all(paths > 0)
        assert_allclose(paths[0], x0)

    def test_covariance_shrinkage(self, numba_backend, numpy_backend, rng):
        returns = rng.standard_normal((50, 5))
        nb_result = numba_backend.covariance_shrinkage(returns, 0.3)
        np_result = numpy_backend.covariance_shrinkage(returns, 0.3)
        assert_allclose(nb_result, np_result, atol=1e-10)

    def test_diversity_weights(self, numba_backend, numpy_backend, weights_5):
        nb_result = numba_backend.diversity_weights(weights_5, 0.5)
        np_result = numpy_backend.diversity_weights(weights_5, 0.5)
        assert_allclose(nb_result, np_result, atol=1e-10)
