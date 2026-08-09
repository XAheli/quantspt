"""Additional coverage tests for ml/wrappers.py.

Exercises sklearn FD hessian, JAX wrapper paths, and
torch wrapper branches that existing tests don't cover.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from quantspt.core.generating_functions import GeneratingFunction
from quantspt.errors import SPTInvariantError
from quantspt.ml._protocols import GeneratingFunctionModel
from quantspt.ml.wrappers import (
    SklearnWrapper,
    TorchModelWrapper,
    wrap_callable,
    wrap_jax_function,
    wrap_sklearn_estimator,
    wrap_torch_model,
)

RNG = np.random.default_rng(42)


def _simplex(n: int = 5) -> np.ndarray:
    alpha = RNG.exponential(size=n)
    return (alpha / alpha.sum()).astype(np.float64)


class _SimpleConvex(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x**2).sum(dim=-1)


class _DirectG(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones(x.shape[0]) * 5.0


class TestTorchWrapperFullCoverage:
    """Cover all TorchModelWrapper paths."""

    def test_generating_function_value(self) -> None:
        mu = _simplex()
        wrapper = wrap_torch_model(_SimpleConvex(), n_assets=5, positivity_offset=2.0)
        val = wrapper.generating_function(mu)
        assert val > 0
        assert isinstance(val, float)

    def test_log_gradient_finite(self) -> None:
        mu = _simplex()
        wrapper = wrap_torch_model(_SimpleConvex(), n_assets=5, positivity_offset=2.0)
        grad = wrapper.log_gradient(mu)
        assert grad.shape == (5,)
        assert np.all(np.isfinite(grad))

    def test_hessian_negate_true(self) -> None:
        mu = _simplex()
        wrapper = wrap_torch_model(_SimpleConvex(), n_assets=5, positivity_offset=2.0)
        H = wrapper.hessian(mu)
        assert H.shape == (5, 5)
        assert np.allclose(H, H.T, atol=1e-5)

    def test_hessian_negate_false(self) -> None:
        mu = _simplex()
        wrapper = wrap_torch_model(_DirectG(), n_assets=5, negate=False, validate=True)
        H = wrapper.hessian(mu)
        assert H.shape == (5, 5)
        np.testing.assert_allclose(H, np.zeros((5, 5)), atol=1e-4)

    def test_weights_on_simplex(self) -> None:
        mu = _simplex()
        wrapper = wrap_torch_model(_SimpleConvex(), n_assets=5, positivity_offset=2.0)
        w = wrapper.weights(mu)
        assert w.shape == (5,)
        assert abs(w.sum() - 1.0) < 1e-4
        assert np.all(w >= -1e-6)

    def test_fit_returns_self(self) -> None:
        wrapper = wrap_torch_model(_SimpleConvex(), n_assets=5, positivity_offset=2.0)
        mw = RNG.dirichlet(np.ones(5), size=20)
        result = wrapper.fit(mw)
        assert result is wrapper

    def test_to_generating_function(self) -> None:
        wrapper = wrap_torch_model(_SimpleConvex(), n_assets=5, positivity_offset=2.0)
        gf = wrapper.to_generating_function()
        assert isinstance(gf, GeneratingFunction)

    def test_validate_non_positive_raises(self) -> None:
        with pytest.raises(SPTInvariantError, match="positive"):
            wrap_torch_model(
                _SimpleConvex(), n_assets=5, positivity_offset=0.0, negate=True
            )

    def test_G_value_torch_negate_false(self) -> None:
        wrapper = wrap_torch_model(_DirectG(), n_assets=5, negate=False, validate=True)
        val = wrapper.generating_function(_simplex())
        assert abs(val - 5.0) < 0.1

    def test_implements_protocol(self) -> None:
        wrapper = wrap_torch_model(_SimpleConvex(), n_assets=5, positivity_offset=2.0)
        assert isinstance(wrapper, GeneratingFunctionModel)

    def test_import_error_when_torch_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "torch", None)
        with pytest.raises(ImportError, match="quantspt\\[ml\\]"):
            TorchModelWrapper(model=None, n_assets=5, validate=False)


class TestCallableWrapperFullCoverage:
    """Cover all CallableWrapper paths."""

    def test_generating_function(self) -> None:
        def g(mu: np.ndarray) -> float:
            return float(np.sum(mu**0.5))

        wrapper = wrap_callable(g, n_assets=5)
        mu = _simplex()
        val = wrapper.generating_function(mu)
        assert abs(val - np.sum(mu**0.5)) < 1e-10

    def test_log_gradient(self) -> None:
        def g(mu: np.ndarray) -> float:
            return float(np.sum(mu**0.5))

        wrapper = wrap_callable(g, n_assets=5)
        grad = wrapper.log_gradient(_simplex())
        assert grad.shape == (5,)
        assert np.all(np.isfinite(grad))

    def test_hessian_diagonal_and_offdiag(self) -> None:
        def g(mu: np.ndarray) -> float:
            return float(np.sum(mu**0.5))

        wrapper = wrap_callable(g, n_assets=3)
        mu = np.array([0.4, 0.35, 0.25])
        H = wrapper.hessian(mu)
        assert H.shape == (3, 3)
        assert np.allclose(H, H.T, atol=1e-3)
        assert np.all(np.isfinite(H))

    def test_weights(self) -> None:
        def g(mu: np.ndarray) -> float:
            return float(np.sum(mu**0.5))

        wrapper = wrap_callable(g, n_assets=5)
        w = wrapper.weights(_simplex())
        assert abs(w.sum() - 1.0) < 1e-3
        assert np.all(w >= -1e-6)

    def test_fit_infers_n_assets(self) -> None:
        def g(mu: np.ndarray) -> float:
            return float(np.sum(mu**0.5))

        wrapper = wrap_callable(g, n_assets=None, validate=False)
        mw = RNG.dirichlet(np.ones(7), size=20)
        result = wrapper.fit(mw)
        assert result is wrapper
        assert wrapper._n_assets == 7

    def test_to_generating_function_with_none_n(self) -> None:
        def g(mu: np.ndarray) -> float:
            return float(np.sum(mu**0.5))

        wrapper = wrap_callable(g, n_assets=None, validate=False)
        gf = wrapper.to_generating_function()
        assert isinstance(gf, GeneratingFunction)

    def test_non_positive_raises(self) -> None:
        def bad_g(mu: np.ndarray) -> float:
            return -1.0

        with pytest.raises(SPTInvariantError):
            wrap_callable(bad_g, n_assets=5, validate=True)


class TestSklearnWrapperFullCoverage:
    """End-to-end sklearn wrapper with FD hessian coverage."""

    def _fitted_wrapper(
        self, n: int = 3, target: str = "generating_function"
    ) -> tuple[SklearnWrapper, np.ndarray]:
        from sklearn.kernel_ridge import KernelRidge

        T = 80
        alpha = RNG.exponential(size=(T, n))
        mw = alpha / alpha.sum(axis=1, keepdims=True)
        g_vals = np.sum(mw**0.5, axis=1)
        if target == "log_generating_function":
            g_vals = np.log(np.maximum(g_vals, 1e-30))

        wrapper = wrap_sklearn_estimator(
            KernelRidge(kernel="rbf", alpha=0.1), n_assets=n, target=target
        )
        wrapper.fit(mw, g_values=g_vals)
        return wrapper, mw[0]

    def test_generating_function(self) -> None:
        wrapper, mu = self._fitted_wrapper()
        val = wrapper.generating_function(mu)
        assert val > 0
        assert np.isfinite(val)

    def test_log_gradient(self) -> None:
        wrapper, mu = self._fitted_wrapper()
        grad = wrapper.log_gradient(mu)
        assert grad.shape == (3,)
        assert np.all(np.isfinite(grad))

    def test_hessian_fd(self) -> None:
        """Exercise the finite-difference hessian with both diagonal and off-diagonal."""
        wrapper, mu = self._fitted_wrapper(n=4)
        H = wrapper.hessian(mu)
        assert H.shape == (4, 4)
        assert np.allclose(H, H.T, atol=1e-3)
        assert np.all(np.isfinite(H))

    def test_weights(self) -> None:
        wrapper, mu = self._fitted_wrapper()
        w = wrapper.weights(mu)
        assert w.shape == (3,)
        assert np.all(w >= -1e-6)

    def test_to_generating_function(self) -> None:
        wrapper, _mu = self._fitted_wrapper()
        gf = wrapper.to_generating_function()
        assert isinstance(gf, GeneratingFunction)

    def test_unfitted_raises(self) -> None:
        from sklearn.kernel_ridge import KernelRidge

        wrapper = wrap_sklearn_estimator(KernelRidge())
        with pytest.raises(RuntimeError, match="fitted"):
            wrapper.generating_function(np.array([0.3, 0.3, 0.4]))

    def test_log_generating_function_target(self) -> None:
        wrapper, mu = self._fitted_wrapper(target="log_generating_function")
        val = wrapper.generating_function(mu)
        assert val > 0
        assert np.isfinite(val)

    def test_fit_auto_generates_targets_gf(self) -> None:
        from sklearn.kernel_ridge import KernelRidge

        mw = RNG.dirichlet(np.ones(3), size=80)
        wrapper = wrap_sklearn_estimator(
            KernelRidge(kernel="rbf", alpha=0.1), target="generating_function"
        )
        wrapper.fit(mw)
        val = wrapper.generating_function(mw[0])
        assert val > 0

    def test_fit_auto_generates_targets_log(self) -> None:
        from sklearn.kernel_ridge import KernelRidge

        mw = RNG.dirichlet(np.ones(3), size=80)
        wrapper = wrap_sklearn_estimator(
            KernelRidge(kernel="rbf", alpha=0.1), target="log_generating_function"
        )
        wrapper.fit(mw)
        val = wrapper.generating_function(mw[0])
        assert val > 0

    def test_invalid_target_raises(self) -> None:
        from sklearn.kernel_ridge import KernelRidge

        with pytest.raises(ValueError, match="target must be"):
            wrap_sklearn_estimator(KernelRidge(), target="bogus")

    def test_to_generating_function_unfitted(self) -> None:
        from sklearn.kernel_ridge import KernelRidge

        wrapper = wrap_sklearn_estimator(KernelRidge(), n_assets=None)
        gf = wrapper.to_generating_function()
        assert isinstance(gf, GeneratingFunction)

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    def test_gaussian_process_hessian(self) -> None:
        """GaussianProcessRegressor end-to-end with hessian."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF

        T, n = 80, 3
        alpha = RNG.exponential(size=(T, n))
        mw = alpha / alpha.sum(axis=1, keepdims=True)
        g_vals = np.sum(mw**0.5, axis=1)

        gpr = GaussianProcessRegressor(kernel=RBF(), alpha=0.1, normalize_y=True)
        wrapper = wrap_sklearn_estimator(gpr, n_assets=n)
        wrapper.fit(mw, g_values=g_vals)

        mu = mw[0]
        H = wrapper.hessian(mu)
        assert H.shape == (n, n)
        assert np.allclose(H, H.T, atol=1e-3)


class TestJaxFunctionWrapperFullCoverage:
    """Cover all JaxFunctionWrapper paths."""

    @pytest.fixture(autouse=True)
    def _enable_x64(self) -> None:
        import jax

        jax.config.update("jax_enable_x64", True)

    def test_generating_function(self) -> None:
        import jax.numpy as jnp

        def my_G(mu: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(mu**0.5)

        wrapper = wrap_jax_function(my_G, n_assets=5)
        mu = _simplex()
        val = wrapper.generating_function(mu)
        assert abs(val - np.sum(mu**0.5)) < 1e-6

    def test_log_gradient(self) -> None:
        import jax.numpy as jnp

        def my_G(mu: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(mu**0.5)

        wrapper = wrap_jax_function(my_G, n_assets=5)
        grad = wrapper.log_gradient(_simplex())
        assert grad.shape == (5,)
        assert np.all(np.isfinite(grad))

    def test_hessian(self) -> None:
        import jax.numpy as jnp

        def my_G(mu: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(mu**0.5)

        wrapper = wrap_jax_function(my_G, n_assets=5)
        H = wrapper.hessian(_simplex())
        assert H.shape == (5, 5)
        assert np.allclose(H, H.T, atol=1e-6)

    def test_weights(self) -> None:
        import jax.numpy as jnp

        def my_G(mu: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(mu**0.5)

        wrapper = wrap_jax_function(my_G, n_assets=5)
        w = wrapper.weights(_simplex())
        assert abs(w.sum() - 1.0) < 1e-3
        assert np.all(w >= -1e-6)

    def test_fit_returns_self(self) -> None:
        import jax.numpy as jnp

        def my_G(mu: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(mu**0.5)

        wrapper = wrap_jax_function(my_G, n_assets=5)
        mw = RNG.dirichlet(np.ones(5), size=20)
        assert wrapper.fit(mw) is wrapper

    def test_to_generating_function(self) -> None:
        import jax.numpy as jnp

        def my_G(mu: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(mu**0.5)

        wrapper = wrap_jax_function(my_G, n_assets=5)
        gf = wrapper.to_generating_function()
        assert isinstance(gf, GeneratingFunction)

    def test_validate_non_positive_raises(self) -> None:
        import jax.numpy as jnp

        def bad_G(mu: jnp.ndarray) -> jnp.ndarray:
            return -jnp.sum(mu**2)

        with pytest.raises(SPTInvariantError, match="positive"):
            wrap_jax_function(bad_G, n_assets=5, validate=True)

    def test_import_error_when_jax_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "jax", None)
        with pytest.raises(ImportError, match="quantspt\\[all\\]"):
            wrap_jax_function(lambda mu: 1.0, n_assets=5)
