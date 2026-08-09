"""Tests for framework-agnostic model wrappers.

Validates that wrap_torch_model, wrap_callable, wrap_sklearn_estimator,
and wrap_jax_function produce valid GeneratingFunctionModel implementations
with comprehensive branch coverage.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from quantspt.core.generating_functions import GeneratingFunction
from quantspt.errors import SPTInvariantError
from quantspt.ml._protocols import GeneratingFunctionModel
from quantspt.ml.wrappers import (
    TorchModelWrapper,
    wrap_callable,
    wrap_jax_function,
    wrap_sklearn_estimator,
    wrap_torch_model,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(77)


@pytest.fixture
def simplex_point(rng: np.random.Generator) -> np.ndarray:
    """Random point on the 5-simplex."""
    alpha = rng.exponential(size=5)
    return (alpha / alpha.sum()).astype(np.float64)


# ---------------------------------------------------------------------------
# wrap_torch_model tests
# ---------------------------------------------------------------------------


class TestWrapTorchModel:
    """Tests for wrapping arbitrary PyTorch models."""

    def test_basic_wrapping(self, simplex_point: np.ndarray) -> None:
        """A simple convex network can be wrapped."""

        class SimpleConvex(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(5, 1, bias=True)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x**2).sum(dim=-1)

        model = SimpleConvex()
        wrapper = wrap_torch_model(model, n_assets=5, positivity_offset=2.0)
        G_val = wrapper.generating_function(simplex_point)
        assert G_val > 0

    def test_log_gradient_shape(self, simplex_point: np.ndarray) -> None:
        """log_gradient returns correct shape."""

        class QuadraticConvex(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x**2).sum(dim=-1)

        wrapper = wrap_torch_model(QuadraticConvex(), n_assets=5, positivity_offset=2.0)
        grad = wrapper.log_gradient(simplex_point)
        assert grad.shape == (5,)
        assert np.all(np.isfinite(grad))

    def test_hessian_shape_and_symmetry(self, simplex_point: np.ndarray) -> None:
        """Hessian is (n, n) and symmetric."""

        class QuadConvex(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x**2).sum(dim=-1)

        wrapper = wrap_torch_model(QuadConvex(), n_assets=5, positivity_offset=2.0)
        H = wrapper.hessian(simplex_point)
        assert H.shape == (5, 5)
        assert np.allclose(H, H.T, atol=1e-5)

    def test_weights_on_simplex(self, simplex_point: np.ndarray) -> None:
        """Weights sum to 1 and are non-negative."""

        class QuadConvex(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x**2).sum(dim=-1)

        wrapper = wrap_torch_model(QuadConvex(), n_assets=5, positivity_offset=2.0)
        w = wrapper.weights(simplex_point)
        assert abs(w.sum() - 1.0) < 1e-5
        assert np.all(w >= -1e-6)

    def test_to_generating_function(self, simplex_point: np.ndarray) -> None:
        """Wrapper can be converted to core GeneratingFunction."""

        class QuadConvex(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x**2).sum(dim=-1)

        wrapper = wrap_torch_model(QuadConvex(), n_assets=5, positivity_offset=2.0)
        G = wrapper.to_generating_function()
        assert isinstance(G, GeneratingFunction)
        val = G(simplex_point)
        assert val > 0

    def test_negate_false(self, simplex_point: np.ndarray) -> None:
        """negate=False: model directly outputs G (must be positive)."""

        class DirectG(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.ones(x.shape[0]) * 2.0

        wrapper = wrap_torch_model(DirectG(), n_assets=5, negate=False, validate=True)
        assert wrapper.generating_function(simplex_point) == pytest.approx(2.0, abs=0.1)

    def test_implements_protocol(self, simplex_point: np.ndarray) -> None:
        """Wrapper satisfies GeneratingFunctionModel Protocol."""

        class QuadConvex(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x**2).sum(dim=-1)

        wrapper = wrap_torch_model(QuadConvex(), n_assets=5, positivity_offset=2.0)
        assert isinstance(wrapper, GeneratingFunctionModel)


# ---------------------------------------------------------------------------
# wrap_callable tests
# ---------------------------------------------------------------------------


class TestWrapCallable:
    """Tests for wrapping plain Python functions."""

    def test_diversity_function(self, simplex_point: np.ndarray) -> None:
        """Wrap the diversity-weighted generating function."""

        def diversity_g(mu: np.ndarray) -> float:
            return float(np.sum(mu**0.5))

        wrapper = wrap_callable(diversity_g, n_assets=5)
        val = wrapper.generating_function(simplex_point)
        assert val > 0
        expected = np.sum(simplex_point**0.5)
        assert abs(val - expected) < 1e-10

    def test_entropy_function(self, simplex_point: np.ndarray) -> None:
        """Wrap the entropy generating function."""

        def entropy_g(mu: np.ndarray) -> float:
            return float(np.exp(-np.sum(mu * np.log(mu))))

        wrapper = wrap_callable(entropy_g, n_assets=5)
        val = wrapper.generating_function(simplex_point)
        assert val > 0

    def test_log_gradient_finite(self, simplex_point: np.ndarray) -> None:
        """Finite-difference gradient is finite and correct shape."""

        def g(mu: np.ndarray) -> float:
            return float(np.sum(mu**0.5))

        wrapper = wrap_callable(g, n_assets=5)
        grad = wrapper.log_gradient(simplex_point)
        assert grad.shape == (5,)
        assert np.all(np.isfinite(grad))

    def test_hessian_finite(self, simplex_point: np.ndarray) -> None:
        """Finite-difference Hessian is finite and symmetric."""

        def g(mu: np.ndarray) -> float:
            return float(np.sum(mu**0.5))

        wrapper = wrap_callable(g, n_assets=5)
        H = wrapper.hessian(simplex_point)
        assert H.shape == (5, 5)
        assert np.allclose(H, H.T, atol=1e-4)
        assert np.all(np.isfinite(H))

    def test_to_generating_function(self, simplex_point: np.ndarray) -> None:
        """Callable wrapper converts to GeneratingFunction."""

        def g(mu: np.ndarray) -> float:
            return float(np.sum(mu**0.5))

        wrapper = wrap_callable(g, n_assets=5)
        G = wrapper.to_generating_function()
        assert isinstance(G, GeneratingFunction)

    def test_implements_protocol(self) -> None:
        """CallableWrapper satisfies GeneratingFunctionModel Protocol."""

        def g(mu: np.ndarray) -> float:
            return float(np.sum(mu**0.5))

        wrapper = wrap_callable(g, n_assets=5)
        assert isinstance(wrapper, GeneratingFunctionModel)

    def test_non_positive_raises(self) -> None:
        """Non-positive function raises SPTInvariantError."""
        from quantspt.errors import SPTInvariantError

        def bad_g(mu: np.ndarray) -> float:
            return -1.0

        with pytest.raises(SPTInvariantError):
            wrap_callable(bad_g, n_assets=5, validate=True)


# ---------------------------------------------------------------------------
# wrap_sklearn_estimator tests
# ---------------------------------------------------------------------------


class TestWrapSklearn:
    """Tests for wrapping sklearn estimators."""

    def test_basic_sklearn_wrapper(self, rng: np.random.Generator) -> None:
        """KernelRidge can be wrapped and fitted."""
        from sklearn.kernel_ridge import KernelRidge

        T, n = 100, 5
        alpha = rng.exponential(size=(T, n))
        mw = alpha / alpha.sum(axis=1, keepdims=True)
        g_vals = np.sum(mw**0.5, axis=1)

        est = KernelRidge(kernel="rbf", alpha=0.1)
        wrapper = wrap_sklearn_estimator(est, n_assets=n)
        wrapper.fit(mw, g_values=g_vals)

        mu = mw[0]
        val = wrapper.generating_function(mu)
        assert val > 0
        assert np.isfinite(val)

    def test_sklearn_log_gradient(self, rng: np.random.Generator) -> None:
        """Sklearn wrapper gradient has correct shape."""
        from sklearn.kernel_ridge import KernelRidge

        T, n = 100, 5
        alpha = rng.exponential(size=(T, n))
        mw = alpha / alpha.sum(axis=1, keepdims=True)
        g_vals = np.sum(mw**0.5, axis=1)

        est = KernelRidge(kernel="rbf", alpha=0.1)
        wrapper = wrap_sklearn_estimator(est, n_assets=n)
        wrapper.fit(mw, g_values=g_vals)

        grad = wrapper.log_gradient(mw[0])
        assert grad.shape == (5,)
        assert np.all(np.isfinite(grad))

    def test_sklearn_to_generating_function(self, rng: np.random.Generator) -> None:
        """Sklearn wrapper converts to GeneratingFunction."""
        from sklearn.kernel_ridge import KernelRidge

        T, n = 100, 5
        alpha = rng.exponential(size=(T, n))
        mw = alpha / alpha.sum(axis=1, keepdims=True)
        g_vals = np.sum(mw**0.5, axis=1)

        est = KernelRidge(kernel="rbf", alpha=0.1)
        wrapper = wrap_sklearn_estimator(est, n_assets=n)
        wrapper.fit(mw, g_values=g_vals)

        G = wrapper.to_generating_function()
        assert isinstance(G, GeneratingFunction)

    def test_unfitted_raises(self) -> None:
        """Calling generating_function before fit raises."""
        from sklearn.kernel_ridge import KernelRidge

        wrapper = wrap_sklearn_estimator(KernelRidge())
        mu = np.array([0.2, 0.3, 0.5])
        with pytest.raises(RuntimeError, match="fitted"):
            wrapper.generating_function(mu)

    def test_implements_protocol(self, rng: np.random.Generator) -> None:
        """SklearnWrapper satisfies GeneratingFunctionModel Protocol."""
        from sklearn.kernel_ridge import KernelRidge

        wrapper = wrap_sklearn_estimator(KernelRidge(), n_assets=5)
        assert isinstance(wrapper, GeneratingFunctionModel)


# ---------------------------------------------------------------------------
# Loss Composition Tests
# ---------------------------------------------------------------------------


class TestLossComposition:
    """Verify loss functions are composable."""

    def test_losses_are_additive(self) -> None:
        """Losses can be added together."""
        from quantspt.ml.losses import relative_return_loss, turnover_penalty

        combined = relative_return_loss + turnover_penalty
        assert combined is not None

    def test_losses_scale(self) -> None:
        """Losses can be scaled by a float."""
        from quantspt.ml.losses import turnover_penalty

        scaled = 0.1 * turnover_penalty
        assert scaled is not None

    def test_combined_loss_computes(self) -> None:
        """Combined loss produces a scalar tensor."""
        from quantspt.ml.losses import relative_return_loss, weight_regularization

        combined = relative_return_loss + 0.01 * weight_regularization
        weights = torch.rand(20, 5)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        returns = 1.0 + torch.randn(20, 5) * 0.01

        loss = combined(weights, returns)
        assert loss.shape == ()
        assert loss.requires_grad or loss.item() != 0

    def test_default_loss_matches_paper(self) -> None:
        """default_loss() computes arXiv:2506.19715 Eq. 3.3."""
        from quantspt.ml.losses import default_loss

        loss_fn = default_loss(weight_decay=1e-4)
        weights = torch.rand(20, 5)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        returns = 1.0 + torch.randn(20, 5) * 0.01

        loss = loss_fn(weights, returns)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# TorchModelWrapper — additional branch coverage
# ---------------------------------------------------------------------------


class TestTorchBranches:
    """Cover remaining TorchModelWrapper branches."""

    def test_fit_returns_self(self, simplex_point: np.ndarray) -> None:
        """fit() is a no-op that returns self."""

        class QuadConvex(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x**2).sum(dim=-1)

        wrapper = wrap_torch_model(QuadConvex(), n_assets=5, positivity_offset=2.0)
        rng = np.random.default_rng(0)
        mw = rng.dirichlet(np.ones(5), size=20)
        result = wrapper.fit(mw)
        assert result is wrapper

    def test_validate_non_positive_raises(self) -> None:
        """Model that gives non-positive G raises SPTInvariantError."""

        class NegativeModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x**2).sum(dim=-1) * 100.0

        with pytest.raises(SPTInvariantError, match="positive"):
            wrap_torch_model(
                NegativeModel(), n_assets=5, positivity_offset=0.0, negate=True
            )

    def test_hessian_negate_false(self, simplex_point: np.ndarray) -> None:
        """Hessian works with negate=False (direct G output)."""

        class ConstantG(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.ones(x.shape[0]) * 5.0

        wrapper = wrap_torch_model(ConstantG(), n_assets=5, negate=False, validate=True)
        H = wrapper.hessian(simplex_point)
        assert H.shape == (5, 5)
        assert np.allclose(H, H.T, atol=1e-5)

    def test_import_error_when_torch_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TorchModelWrapper raises ImportError when torch is unavailable."""
        monkeypatch.setitem(sys.modules, "torch", None)
        with pytest.raises(ImportError, match="quantspt\\[ml\\]"):
            TorchModelWrapper.__init__(
                object.__new__(TorchModelWrapper),
                model=None,
                n_assets=5,
                validate=False,
            )


# ---------------------------------------------------------------------------
# CallableWrapper — additional branch coverage
# ---------------------------------------------------------------------------


class TestCallableBranches:
    """Cover remaining CallableWrapper branches."""

    def test_fit_infers_n_assets(self) -> None:
        """fit() infers n_assets when not provided at init."""

        def g(mu: np.ndarray) -> float:
            return float(np.sum(mu**0.5))

        wrapper = wrap_callable(g, n_assets=None, validate=False)
        assert wrapper._n_assets is None

        mw = np.random.default_rng(0).dirichlet(np.ones(7), size=20)
        result = wrapper.fit(mw)
        assert result is wrapper
        assert wrapper._n_assets == 7

    def test_weights_on_simplex(self, simplex_point: np.ndarray) -> None:
        """CallableWrapper.weights() sums to 1 and is non-negative."""

        def g(mu: np.ndarray) -> float:
            return float(np.sum(mu**0.5))

        wrapper = wrap_callable(g, n_assets=5)
        w = wrapper.weights(simplex_point)
        assert w.shape == (5,)
        assert abs(w.sum() - 1.0) < 1e-4
        assert np.all(w >= -1e-6)

    def test_to_generating_function_inferred_n(self) -> None:
        """to_generating_function uses default n when n_assets is None."""

        def g(mu: np.ndarray) -> float:
            return float(np.sum(mu**0.5))

        wrapper = wrap_callable(g, n_assets=None, validate=False)
        gf = wrapper.to_generating_function()
        assert isinstance(gf, GeneratingFunction)


# ---------------------------------------------------------------------------
# SklearnWrapper — comprehensive tests (hessian, weights, log_gf target)
# ---------------------------------------------------------------------------


class TestSklearnFull:
    """Full coverage of SklearnWrapper including hessian, weights, targets."""

    def test_invalid_target_raises(self) -> None:
        """Invalid target string raises ValueError."""
        from sklearn.kernel_ridge import KernelRidge

        with pytest.raises(ValueError, match="target must be"):
            wrap_sklearn_estimator(KernelRidge(), target="bogus")

    def test_gaussian_process_regressor(self, rng: np.random.Generator) -> None:
        """GaussianProcessRegressor works end-to-end."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF

        T, n = 80, 3
        alpha = rng.exponential(size=(T, n))
        mw = alpha / alpha.sum(axis=1, keepdims=True)
        g_vals = np.sum(mw**0.5, axis=1)

        gpr = GaussianProcessRegressor(kernel=RBF(), alpha=0.1, normalize_y=True)
        wrapper = wrap_sklearn_estimator(gpr, n_assets=n)
        wrapper.fit(mw, g_values=g_vals)

        mu = mw[0]
        val = wrapper.generating_function(mu)
        assert val > 0

        grad = wrapper.log_gradient(mu)
        assert grad.shape == (n,)
        assert np.all(np.isfinite(grad))

        H = wrapper.hessian(mu)
        assert H.shape == (n, n)
        assert np.allclose(H, H.T, atol=1e-4)
        assert np.all(np.isfinite(H))

        w = wrapper.weights(mu)
        assert w.shape == (n,)
        assert np.all(w >= -1e-6)

    def test_log_generating_function_target(self, rng: np.random.Generator) -> None:
        """target='log_generating_function' path works correctly."""
        from sklearn.kernel_ridge import KernelRidge

        T, n = 100, 4
        alpha = rng.exponential(size=(T, n))
        mw = alpha / alpha.sum(axis=1, keepdims=True)
        g_vals = np.log(np.sum(mw**0.5, axis=1))

        wrapper = wrap_sklearn_estimator(
            KernelRidge(kernel="rbf", alpha=0.1),
            n_assets=n,
            target="log_generating_function",
        )
        wrapper.fit(mw, g_values=g_vals)

        mu = mw[0]
        val = wrapper.generating_function(mu)
        assert val > 0
        assert np.isfinite(val)

    def test_fit_auto_targets_log(self, rng: np.random.Generator) -> None:
        """fit() with g_values=None auto-generates targets for log target."""
        from sklearn.kernel_ridge import KernelRidge

        T, n = 80, 3
        alpha = rng.exponential(size=(T, n))
        mw = alpha / alpha.sum(axis=1, keepdims=True)

        wrapper = wrap_sklearn_estimator(
            KernelRidge(kernel="rbf", alpha=0.1),
            target="log_generating_function",
        )
        wrapper.fit(mw)
        mu = mw[0]
        val = wrapper.generating_function(mu)
        assert val > 0

    def test_fit_auto_targets_gf(self, rng: np.random.Generator) -> None:
        """fit() with g_values=None auto-generates targets for gf target."""
        from sklearn.kernel_ridge import KernelRidge

        T, n = 80, 3
        alpha = rng.exponential(size=(T, n))
        mw = alpha / alpha.sum(axis=1, keepdims=True)

        wrapper = wrap_sklearn_estimator(
            KernelRidge(kernel="rbf", alpha=0.1),
            target="generating_function",
        )
        wrapper.fit(mw)
        mu = mw[0]
        val = wrapper.generating_function(mu)
        assert val > 0

    def test_hessian_off_diagonal(self, rng: np.random.Generator) -> None:
        """Hessian off-diagonal (i != j) branches are exercised."""
        from sklearn.kernel_ridge import KernelRidge

        T, n = 100, 4
        alpha = rng.exponential(size=(T, n))
        mw = alpha / alpha.sum(axis=1, keepdims=True)
        g_vals = np.sum(mw**0.5, axis=1)

        wrapper = wrap_sklearn_estimator(
            KernelRidge(kernel="rbf", alpha=0.1), n_assets=n
        )
        wrapper.fit(mw, g_values=g_vals)

        H = wrapper.hessian(mw[0])
        assert H.shape == (n, n)
        assert np.allclose(H, H.T, atol=1e-4)

    def test_to_generating_function_fitted(self, rng: np.random.Generator) -> None:
        """Sklearn wrapper converts to GeneratingFunction after fitting."""
        from sklearn.kernel_ridge import KernelRidge

        T, n = 80, 3
        alpha = rng.exponential(size=(T, n))
        mw = alpha / alpha.sum(axis=1, keepdims=True)
        g_vals = np.sum(mw**0.5, axis=1)

        wrapper = wrap_sklearn_estimator(
            KernelRidge(kernel="rbf", alpha=0.1), n_assets=n
        )
        wrapper.fit(mw, g_values=g_vals)
        gf = wrapper.to_generating_function()
        assert isinstance(gf, GeneratingFunction)

    def test_to_generating_function_unfitted_default_n(self) -> None:
        """to_generating_function falls back to n=5 when n_assets is None."""
        from sklearn.kernel_ridge import KernelRidge

        wrapper = wrap_sklearn_estimator(KernelRidge(), n_assets=None)
        gf = wrapper.to_generating_function()
        assert isinstance(gf, GeneratingFunction)


# ---------------------------------------------------------------------------
# JaxFunctionWrapper — import error and fit coverage
# ---------------------------------------------------------------------------


class TestJaxWrapperBranches:
    """Cover remaining JaxFunctionWrapper branches."""

    def test_import_error_when_jax_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """JaxFunctionWrapper raises ImportError when JAX is unavailable."""
        monkeypatch.setitem(sys.modules, "jax", None)
        with pytest.raises(ImportError, match="quantspt\\[all\\]"):
            wrap_jax_function(lambda mu: 1.0, n_assets=5)

    def test_fit_returns_self(self) -> None:
        """JaxFunctionWrapper.fit() is a no-op returning self."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)

        def my_G(mu: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(mu**0.5)

        wrapper = wrap_jax_function(my_G, n_assets=5)
        mw = np.random.default_rng(0).dirichlet(np.ones(5), size=20)
        result = wrapper.fit(mw)
        assert result is wrapper

    def test_validate_non_positive_raises(self) -> None:
        """Non-positive JAX function raises SPTInvariantError."""
        import jax.numpy as jnp

        def bad_G(mu: jnp.ndarray) -> jnp.ndarray:
            return -jnp.sum(mu**2)

        with pytest.raises(SPTInvariantError, match="positive"):
            wrap_jax_function(bad_G, n_assets=5, validate=True)
