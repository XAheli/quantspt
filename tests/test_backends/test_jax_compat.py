"""Comprehensive JAX function compatibility tests.

Validates that:
- wrap_jax_function with a pure jnp function produces valid GeneratingFunction
- JIT'd function matches non-JIT'd output exactly
- jax.grad inside wrapper matches finite-difference gradient
- jax.hessian inside wrapper matches finite-difference Hessian
- Float64 precision: results match numpy to 1e-12
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(2024)


@pytest.fixture
def simplex_point(rng: np.random.Generator) -> np.ndarray:
    alpha = rng.exponential(size=5)
    return (alpha / alpha.sum()).astype(np.float64)


@pytest.fixture
def simplex_points(rng: np.random.Generator) -> np.ndarray:
    alpha = rng.exponential(size=(20, 5))
    return (alpha / alpha.sum(axis=1, keepdims=True)).astype(np.float64)


def _finite_diff_gradient(func, mu, h=1e-7):
    """Central-difference gradient."""
    n = len(mu)
    grad = np.zeros(n)
    for k in range(n):
        mu_p, mu_m = mu.copy(), mu.copy()
        mu_p[k] += h
        mu_m[k] -= h
        grad[k] = (func(mu_p) - func(mu_m)) / (2 * h)
    return grad


def _finite_diff_hessian(func, mu, h=1e-5):
    """Central-difference Hessian."""
    n = len(mu)
    H = np.zeros((n, n))
    f0 = func(mu)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                mu_p, mu_m = mu.copy(), mu.copy()
                mu_p[i] += h
                mu_m[i] -= h
                H[i, i] = (func(mu_p) - 2 * f0 + func(mu_m)) / h**2
            else:
                mu_pp, mu_pm, mu_mp, mu_mm = (
                    mu.copy(),
                    mu.copy(),
                    mu.copy(),
                    mu.copy(),
                )
                mu_pp[i] += h
                mu_pp[j] += h
                mu_pm[i] += h
                mu_pm[j] -= h
                mu_mp[i] -= h
                mu_mp[j] += h
                mu_mm[i] -= h
                mu_mm[j] -= h
                H[i, j] = (func(mu_pp) - func(mu_pm) - func(mu_mp) + func(mu_mm)) / (
                    4 * h**2
                )
                H[j, i] = H[i, j]
    return H


# ---------------------------------------------------------------------------
# wrap_jax_function produces valid GeneratingFunction
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestWrapJaxFunction:
    """wrap_jax_function with pure jnp function → valid GeneratingFunction."""

    def test_basic_wrapping(self, simplex_point) -> None:
        from quantspt.ml.wrappers import wrap_jax_function

        def my_G(mu):
            return -jnp.sum(mu**2) + 2.0

        wrapper = wrap_jax_function(my_G, n_assets=5)
        val = wrapper.generating_function(simplex_point)
        assert val > 0
        assert np.isfinite(val)

    def test_diversity_function(self, simplex_point) -> None:
        from quantspt.ml.wrappers import wrap_jax_function

        def diversity_G(mu):
            return jnp.sum(mu**0.5)

        wrapper = wrap_jax_function(diversity_G, n_assets=5)
        val = wrapper.generating_function(simplex_point)
        expected = np.sum(simplex_point**0.5)
        assert_allclose(val, expected, rtol=1e-12)

    def test_weights_sum_to_one(self, simplex_point) -> None:
        from quantspt.ml.wrappers import wrap_jax_function

        def my_G(mu):
            return jnp.sum(mu**0.5)

        wrapper = wrap_jax_function(my_G, n_assets=5)
        w = wrapper.weights(simplex_point)
        assert abs(w.sum() - 1.0) < 1e-5

    def test_to_generating_function(self, simplex_point) -> None:
        from quantspt.core.generating_functions import GeneratingFunction
        from quantspt.ml.wrappers import wrap_jax_function

        def my_G(mu):
            return jnp.sum(mu**0.5)

        wrapper = wrap_jax_function(my_G, n_assets=5)
        G = wrapper.to_generating_function()
        assert isinstance(G, GeneratingFunction)


# ---------------------------------------------------------------------------
# JIT'd vs non-JIT'd
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestJITConsistency:
    """JIT'd function matches non-JIT'd output exactly."""

    def test_jit_matches_no_jit(self, simplex_points) -> None:
        from quantspt.ml.wrappers import wrap_jax_function

        def my_G(mu):
            return jnp.sum(mu**0.5)

        wrapper = wrap_jax_function(my_G, n_assets=5)
        for mu in simplex_points:
            jit_val = wrapper.generating_function(mu)
            no_jit_val = float(my_G(jnp.array(mu)))
            assert_allclose(jit_val, no_jit_val, rtol=1e-14)


# ---------------------------------------------------------------------------
# jax.grad vs finite differences
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestJaxGradVsFiniteDiff:
    """jax.grad inside wrapper matches finite-difference gradient."""

    def test_grad_matches_fd_diversity(self, simplex_point) -> None:
        from quantspt.ml.wrappers import wrap_jax_function

        def my_G(mu):
            return jnp.sum(mu**0.5)

        wrapper = wrap_jax_function(my_G, n_assets=5)
        jax_grad = wrapper.log_gradient(simplex_point)

        def log_G_np(mu):
            return np.log(np.sum(mu**0.5))

        fd_grad = _finite_diff_gradient(log_G_np, simplex_point)
        assert_allclose(jax_grad, fd_grad, atol=1e-5, rtol=1e-5)

    def test_grad_matches_fd_quadratic(self, simplex_point) -> None:
        from quantspt.ml.wrappers import wrap_jax_function

        def my_G(mu):
            return -jnp.sum(mu**2) + 2.0

        wrapper = wrap_jax_function(my_G, n_assets=5)
        jax_grad = wrapper.log_gradient(simplex_point)

        def log_G_np(mu):
            return np.log(-np.sum(mu**2) + 2.0)

        fd_grad = _finite_diff_gradient(log_G_np, simplex_point)
        assert_allclose(jax_grad, fd_grad, atol=1e-5, rtol=1e-5)

    def test_grad_on_multiple_points(self, simplex_points) -> None:
        from quantspt.ml.wrappers import wrap_jax_function

        def my_G(mu):
            return jnp.sum(mu**0.5)

        wrapper = wrap_jax_function(my_G, n_assets=5)

        def log_G_np(mu):
            return np.log(np.sum(mu**0.5))

        for mu in simplex_points:
            jax_grad = wrapper.log_gradient(mu)
            fd_grad = _finite_diff_gradient(log_G_np, mu)
            assert_allclose(jax_grad, fd_grad, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# jax.hessian vs finite differences
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestJaxHessianVsFiniteDiff:
    """jax.hessian inside wrapper matches finite-difference Hessian."""

    def test_hessian_matches_fd_quadratic(self, simplex_point) -> None:
        from quantspt.ml.wrappers import wrap_jax_function

        def my_G(mu):
            return -jnp.sum(mu**2) + 2.0

        wrapper = wrap_jax_function(my_G, n_assets=5)
        jax_H = wrapper.hessian(simplex_point)

        def G_np(mu):
            return -np.sum(mu**2) + 2.0

        fd_H = _finite_diff_hessian(G_np, simplex_point)
        assert_allclose(jax_H, fd_H, atol=1e-4, rtol=1e-4)

    def test_hessian_matches_fd_diversity(self, simplex_point) -> None:
        from quantspt.ml.wrappers import wrap_jax_function

        def my_G(mu):
            return jnp.sum(mu**0.5)

        wrapper = wrap_jax_function(my_G, n_assets=5)
        jax_H = wrapper.hessian(simplex_point)

        def G_np(mu):
            return np.sum(mu**0.5)

        fd_H = _finite_diff_hessian(G_np, simplex_point)
        assert_allclose(jax_H, fd_H, atol=1e-4, rtol=1e-4)

    def test_hessian_symmetry(self, simplex_points) -> None:
        from quantspt.ml.wrappers import wrap_jax_function

        def my_G(mu):
            return jnp.sum(mu**0.5)

        wrapper = wrap_jax_function(my_G, n_assets=5)
        for mu in simplex_points[:5]:
            H = wrapper.hessian(mu)
            assert_allclose(H, H.T, atol=1e-10)


# ---------------------------------------------------------------------------
# Float64 precision
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestFloat64Precision:
    """Float64: results match numpy to 1e-12."""

    def test_generating_function_matches_numpy(self, simplex_points) -> None:
        from quantspt.ml.wrappers import wrap_jax_function

        def jax_G(mu):
            return jnp.sum(mu**0.5)

        wrapper = wrap_jax_function(jax_G, n_assets=5)
        for mu in simplex_points:
            jax_val = wrapper.generating_function(mu)
            np_val = np.sum(mu**0.5)
            assert_allclose(jax_val, np_val, atol=1e-12, rtol=1e-12)

    def test_gradient_float64(self, simplex_point) -> None:
        from quantspt.ml.wrappers import wrap_jax_function

        def jax_G(mu):
            return jnp.sum(mu**0.5)

        wrapper = wrap_jax_function(jax_G, n_assets=5)
        grad = wrapper.log_gradient(simplex_point)
        assert grad.dtype == np.float64

    def test_hessian_float64(self, simplex_point) -> None:
        from quantspt.ml.wrappers import wrap_jax_function

        def jax_G(mu):
            return jnp.sum(mu**0.5)

        wrapper = wrap_jax_function(jax_G, n_assets=5)
        H = wrapper.hessian(simplex_point)
        assert H.dtype == np.float64
