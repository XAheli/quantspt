"""Tests exercising all JAX backend public methods on CPU."""

from __future__ import annotations

import numpy as np
import pytest

from quantspt._backends.jax_backend import JaxBackend, _require_jax


class TestRequireJax:
    """Test the _require_jax helper."""

    def test_require_jax_succeeds(self) -> None:
        jax, jnp = _require_jax()
        assert hasattr(jax, "jit")
        assert hasattr(jnp, "array")

    def test_require_jax_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "jax", None)
        with pytest.raises(ImportError, match="JAX is required"):
            _require_jax()


class TestJaxBackendMethods:
    """Call every JaxBackend public method with real arrays."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        import jax

        jax.config.update("jax_enable_x64", True)
        self.backend = JaxBackend()
        self.rng = np.random.default_rng(42)

    def _simplex(self, n: int = 5) -> np.ndarray:
        alpha = self.rng.exponential(size=n)
        return (alpha / alpha.sum()).astype(np.float64)

    def _psd_matrix(self, n: int = 5) -> np.ndarray:
        L = self.rng.standard_normal((n, n))
        return (L @ L.T + np.eye(n) * 0.01).astype(np.float64)

    def test_name(self) -> None:
        assert self.backend.name == "jax"

    def test_excess_growth_rate(self) -> None:
        pi = self._simplex()
        a = self._psd_matrix()
        result = self.backend.excess_growth_rate(pi, a)
        assert isinstance(result, float)
        assert np.isfinite(result)

        weighted_var = np.dot(pi, np.diag(a))
        port_var = pi @ a @ pi
        expected = 0.5 * (weighted_var - port_var)
        assert abs(result - expected) < 1e-10

    def test_relative_covariance(self) -> None:
        pi = self._simplex()
        a = self._psd_matrix()
        tau = self.backend.relative_covariance(a, pi)
        assert tau.shape == (5, 5)
        assert tau.dtype == np.float64
        assert np.allclose(tau, tau.T, atol=1e-10)

        a_pi = a @ pi
        a_pipi = pi @ a_pi
        expected = a - (a_pi[:, None] + a_pi[None, :]) + a_pipi
        np.testing.assert_allclose(tau, expected, atol=1e-10)

    def test_portfolio_variance(self) -> None:
        pi = self._simplex()
        a = self._psd_matrix()
        result = self.backend.portfolio_variance(a, pi)
        assert isinstance(result, float)
        expected = float(pi @ a @ pi)
        assert abs(result - expected) < 1e-10

    def test_simulate_gbm_step(self) -> None:
        n = 5
        x = self.rng.uniform(50, 150, size=n).astype(np.float64)
        mu = self.rng.uniform(0.01, 0.1, size=n).astype(np.float64)
        L = np.linalg.cholesky(self._psd_matrix(n))
        dt = 1.0 / 252.0
        dw = self.rng.standard_normal(n).astype(np.float64) * np.sqrt(dt)

        result = self.backend.simulate_gbm_step(x, mu, L, dt, dw)
        assert result.shape == (n,)
        assert result.dtype == np.float64
        assert np.all(result > 0)

    def test_diversity_weights(self) -> None:
        mu = self._simplex()
        p = 0.5
        result = self.backend.diversity_weights(mu, p)
        assert result.shape == (5,)
        assert abs(result.sum() - 1.0) < 1e-10
        assert np.all(result >= 0)

        expected = mu**p / np.sum(mu**p)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_covariance_shrinkage(self) -> None:
        n_obs, n_assets = 100, 5
        returns = self.rng.standard_normal((n_obs, n_assets)).astype(np.float64) * 0.01
        shrinkage = 0.3

        result = self.backend.covariance_shrinkage(returns, shrinkage)
        assert result.shape == (n_assets, n_assets)
        assert result.dtype == np.float64
        assert np.allclose(result, result.T, atol=1e-10)
        eigvals = np.linalg.eigvalsh(result)
        assert np.all(eigvals > -1e-10)

    def test_gradient(self) -> None:
        import jax.numpy as jnp

        def f(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(x**2)

        x = np.array([1.0, 2.0, 3.0])
        grad = self.backend.gradient(f, x)
        assert grad.shape == (3,)
        np.testing.assert_allclose(grad, 2.0 * x, atol=1e-10)

    def test_hessian(self) -> None:
        import jax.numpy as jnp

        def f(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(x**2)

        x = np.array([1.0, 2.0, 3.0])
        H = self.backend.hessian(f, x)
        assert H.shape == (3, 3)
        np.testing.assert_allclose(H, 2.0 * np.eye(3), atol=1e-10)

    def test_excess_growth_rate_matches_numpy(self) -> None:
        """Cross-check against the numpy backend."""
        from quantspt._backends.numpy_backend import NumpyBackend

        np_backend = NumpyBackend()
        pi = self._simplex()
        a = self._psd_matrix()

        jax_result = self.backend.excess_growth_rate(pi, a)
        np_result = np_backend.excess_growth_rate(pi, a)
        assert abs(jax_result - np_result) < 1e-10
