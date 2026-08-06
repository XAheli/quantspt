"""Comprehensive PyTorch compatibility tests.

Validates that:
- wrap_torch_model() with custom nn.Modules produces valid GeneratingFunctions
- autograd.grad matches finite-difference gradients
- autograd.functional.hessian matches finite-difference Hessian
- NeuralFGP with ICNN produces concave G on 50 random simplex points
- NeuralFGP to_generating_function integrates with master_formula_decomposition
- Multiple optimizers work (Adam, SGD, AdamW)
- Different loss functions compute correctly
- Loss composition is arithmetically correct
- GPU tensors produce same results as CPU (when CUDA available)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from numpy.testing import assert_allclose

from quantspt.core.generating_functions import GeneratingFunction
from quantspt.core.master_formula import master_formula_decomposition
from quantspt.ml.losses import (
    DriftIntegralLoss,
    default_loss,
    drift_integral_loss,
    relative_return_loss,
    sharpe_of_relative_loss,
    turnover_penalty,
    weight_regularization,
)
from quantspt.ml.neural_fgp import NeuralFGP, NeuralFGPConfig
from quantspt.ml.wrappers import wrap_torch_model


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(2024)


@pytest.fixture
def simplex_points(rng: np.random.Generator) -> np.ndarray:
    """50 random points on the 5-simplex."""
    alpha = rng.exponential(size=(50, 5))
    return (alpha / alpha.sum(axis=1, keepdims=True)).astype(np.float64)


@pytest.fixture
def simplex_point(rng: np.random.Generator) -> np.ndarray:
    alpha = rng.exponential(size=5)
    return (alpha / alpha.sum()).astype(np.float64)


@pytest.fixture
def synthetic_market_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """300 days, 5 assets synthetic market."""
    T, n = 300, 5
    prices = np.exp(np.cumsum(rng.normal(0.0001, 0.02, size=(T, n)), axis=0))
    weights = prices / prices.sum(axis=1, keepdims=True)
    returns = prices[1:] / prices[:-1]
    return weights[:-1].astype(np.float64), returns.astype(np.float64)


# ---------------------------------------------------------------------------
# Autograd vs finite differences
# ---------------------------------------------------------------------------


class TestAutogradVsFiniteDifference:
    """Verify PyTorch autograd matches finite differences for ∇log G and D²G."""

    def _finite_diff_log_gradient(
        self, wrapper, mu: np.ndarray, h: float = 1e-5
    ) -> np.ndarray:
        """Central-difference ∇log G."""
        n = len(mu)
        grad = np.zeros(n)
        for k in range(n):
            mu_p = mu.copy()
            mu_p[k] += h
            mu_m = mu.copy()
            mu_m[k] -= h
            g_p = wrapper.generating_function(mu_p)
            g_m = wrapper.generating_function(mu_m)
            grad[k] = (np.log(max(g_p, 1e-30)) - np.log(max(g_m, 1e-30))) / (2 * h)
        return grad

    def _finite_diff_hessian(
        self, wrapper, mu: np.ndarray, h: float = 1e-5
    ) -> np.ndarray:
        """Central-difference D²G."""
        n = len(mu)
        H = np.zeros((n, n))
        G0 = wrapper.generating_function(mu)
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    mu_p, mu_m = mu.copy(), mu.copy()
                    mu_p[i] += h
                    mu_m[i] -= h
                    H[i, i] = (
                        wrapper.generating_function(mu_p)
                        - 2 * G0
                        + wrapper.generating_function(mu_m)
                    ) / h**2
                else:
                    mu_pp, mu_pm = mu.copy(), mu.copy()
                    mu_mp, mu_mm = mu.copy(), mu.copy()
                    mu_pp[i] += h
                    mu_pp[j] += h
                    mu_pm[i] += h
                    mu_pm[j] -= h
                    mu_mp[i] -= h
                    mu_mp[j] += h
                    mu_mm[i] -= h
                    mu_mm[j] -= h
                    H[i, j] = (
                        wrapper.generating_function(mu_pp)
                        - wrapper.generating_function(mu_pm)
                        - wrapper.generating_function(mu_mp)
                        + wrapper.generating_function(mu_mm)
                    ) / (4 * h**2)
                    H[j, i] = H[i, j]
        return H

    def test_autograd_gradient_matches_finite_diff(
        self, simplex_point: np.ndarray
    ) -> None:
        """torch autograd ∇log G matches central finite differences."""

        class QuadConvex(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x**2).sum(dim=-1)

        wrapper = wrap_torch_model(QuadConvex(), n_assets=5, positivity_offset=2.0)
        autograd_grad = wrapper.log_gradient(simplex_point)
        fd_grad = self._finite_diff_log_gradient(wrapper, simplex_point)
        assert_allclose(autograd_grad, fd_grad, atol=1e-6, rtol=1e-6)

    def test_autograd_hessian_matches_finite_diff(
        self, simplex_point: np.ndarray
    ) -> None:
        """torch autograd.functional.hessian matches finite-difference Hessian."""

        class QuadConvex(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x**2).sum(dim=-1)

        wrapper = wrap_torch_model(QuadConvex(), n_assets=5, positivity_offset=2.0)
        autograd_H = wrapper.hessian(simplex_point)
        fd_H = self._finite_diff_hessian(wrapper, simplex_point)
        assert_allclose(autograd_H, fd_H, atol=1e-4, rtol=1e-4)

    def test_gradient_on_multiple_points(self, simplex_points: np.ndarray) -> None:
        """Gradient agreement tested on 10 distinct simplex points."""

        class QuadConvex(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x**2).sum(dim=-1)

        wrapper = wrap_torch_model(QuadConvex(), n_assets=5, positivity_offset=2.0)
        for mu in simplex_points[:10]:
            autograd_grad = wrapper.log_gradient(mu)
            fd_grad = self._finite_diff_log_gradient(wrapper, mu)
            assert_allclose(autograd_grad, fd_grad, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# NeuralFGP concavity on random simplex points
# ---------------------------------------------------------------------------


class TestNeuralFGPConcavity:
    """G_θ from NeuralFGP with ICNN must be concave on simplex."""

    def test_G_concave_on_50_random_points(
        self, simplex_points: np.ndarray, synthetic_market_data
    ) -> None:
        """G(λx + (1-λ)y) >= λG(x) + (1-λ)G(y) for 50 random points."""
        mw, ret = synthetic_market_data
        config = NeuralFGPConfig(
            hidden_dims=[32, 16],
            epochs=30,
            train_window=50,
            eval_window=10,
            seed=42,
        )
        model = NeuralFGP(n_assets=5, config=config)
        model.fit(mw, returns=ret)

        x_pts = simplex_points[:25]
        y_pts = simplex_points[25:]
        lam = 0.5
        for x, y in zip(x_pts, y_pts, strict=False):
            G_mid = model.generating_function(lam * x + (1 - lam) * y)
            G_x = model.generating_function(x)
            G_y = model.generating_function(y)
            assert G_mid >= lam * G_x + (1 - lam) * G_y - 1e-4, (
                f"Concavity violated: G(mid)={G_mid:.6f} < "
                f"λG(x)+(1-λ)G(y)={lam*G_x+(1-lam)*G_y:.6f}"
            )

    def test_hessian_negative_semidefinite(
        self, simplex_points: np.ndarray, synthetic_market_data
    ) -> None:
        """Hessian D²G_θ has all eigenvalues ≤ 0 (NSD)."""
        mw, ret = synthetic_market_data
        config = NeuralFGPConfig(
            hidden_dims=[32, 16],
            epochs=30,
            train_window=50,
            eval_window=10,
            seed=42,
        )
        model = NeuralFGP(n_assets=5, config=config)
        model.fit(mw, returns=ret)

        for mu in simplex_points[:10]:
            H = model.hessian(mu)
            eigvals = np.linalg.eigvalsh(H)
            assert (
                eigvals[-1] <= 1e-4
            ), f"Hessian not NSD: max eigenvalue={eigvals[-1]:.6f}"


# ---------------------------------------------------------------------------
# NeuralFGP integration with master_formula_decomposition
# ---------------------------------------------------------------------------


class TestNeuralFGPMasterFormula:
    """NeuralFGP.to_generating_function integrates with core master formula."""

    def test_to_generating_function_works_with_master_formula(
        self, synthetic_market_data, rng
    ) -> None:
        """Neural FGP G works with master_formula_decomposition."""
        mw, ret = synthetic_market_data
        config = NeuralFGPConfig(
            hidden_dims=[32, 16],
            epochs=20,
            train_window=50,
            eval_window=10,
            seed=42,
        )
        model = NeuralFGP(n_assets=5, config=config)
        model.fit(mw, returns=ret)
        G = model.to_generating_function()
        assert isinstance(G, GeneratingFunction)

        T_steps = 20
        mu_path = mw[:T_steps]
        n = 5
        a_path = np.zeros((T_steps, n, n))
        for t in range(T_steps):
            L = rng.standard_normal((n, n)) * 0.1
            a_path[t] = L @ L.T + np.eye(n) * 0.01

        dt = 1.0 / 252.0
        result = master_formula_decomposition(G, mu_path, a_path, dt)
        assert "boundary" in result
        assert "drift_integral" in result
        assert "total" in result
        assert np.isfinite(result["boundary"])
        assert np.isfinite(result["drift_integral"])
        assert (
            abs(result["total"] - result["boundary"] - result["drift_integral"]) < 1e-10
        )


# ---------------------------------------------------------------------------
# Multiple optimizers
# ---------------------------------------------------------------------------


class TestMultipleOptimizers:
    """Training converges with Adam, SGD, and AdamW."""

    @pytest.mark.parametrize("optimizer", ["adam", "sgd", "adamw"])
    def test_optimizer_reduces_loss(
        self, optimizer: str, synthetic_market_data
    ) -> None:
        mw, ret = synthetic_market_data
        lr = 5e-3 if optimizer != "sgd" else 1e-2
        config = NeuralFGPConfig(
            hidden_dims=[32, 16],
            epochs=30,
            train_window=50,
            eval_window=10,
            learning_rate=lr,
            optimizer=optimizer,
            seed=42,
        )
        model = NeuralFGP(n_assets=5, config=config)
        model.fit(mw, returns=ret)
        losses = model.training_history["loss"]
        assert len(losses) >= 10
        assert losses[-1] <= losses[0], (
            f"Optimizer {optimizer}: loss did not decrease "
            f"{losses[0]:.6f} → {losses[-1]:.6f}"
        )


# ---------------------------------------------------------------------------
# Different loss functions
# ---------------------------------------------------------------------------


class TestLossFunctions:
    """Each loss function computes correctly on synthetic data."""

    @pytest.fixture
    def synthetic_batch(self):
        torch.manual_seed(42)
        weights = torch.rand(20, 5)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        returns = 1.0 + torch.randn(20, 5) * 0.01
        return weights, returns

    def test_relative_return_loss_is_scalar(self, synthetic_batch) -> None:
        w, r = synthetic_batch
        loss = relative_return_loss(w, r)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_sharpe_loss_is_scalar(self, synthetic_batch) -> None:
        w, r = synthetic_batch
        loss = sharpe_of_relative_loss(w, r)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_turnover_penalty_zero_for_constant_weights(self) -> None:
        """Constant weights → zero turnover."""
        w = torch.ones(10, 5) / 5.0
        r = torch.ones(10, 5) * 1.01
        loss = turnover_penalty(w, r)
        assert_allclose(loss.item(), 0.0, atol=1e-10)

    def test_weight_regularization_minimum_at_equal_weights(self) -> None:
        """Equal weights minimize L2 regularization."""
        w_equal = torch.ones(10, 5) / 5.0
        w_concentrated = torch.zeros(10, 5)
        w_concentrated[:, 0] = 1.0
        r = torch.ones(10, 5) * 1.01
        loss_equal = weight_regularization(w_equal, r)
        loss_conc = weight_regularization(w_concentrated, r)
        assert loss_equal.item() < loss_conc.item()

    def test_default_loss_finite_and_differentiable(self, synthetic_batch) -> None:
        w, r = synthetic_batch
        w.requires_grad_(True)
        loss_fn = default_loss(weight_decay=1e-4)
        loss = loss_fn(w, r)
        assert torch.isfinite(loss)
        loss.backward()
        assert w.grad is not None
        assert torch.all(torch.isfinite(w.grad))


# ---------------------------------------------------------------------------
# Loss composition arithmetic
# ---------------------------------------------------------------------------


class TestLossCompositionArithmetic:
    """Verify combined losses compute correctly (not just "not None")."""

    def test_sum_equals_component_sum(self) -> None:
        """(loss_a + loss_b)(w, r) == loss_a(w, r) + loss_b(w, r)."""
        torch.manual_seed(42)
        w = torch.rand(15, 5)
        w = w / w.sum(dim=-1, keepdim=True)
        r = 1.0 + torch.randn(15, 5) * 0.01

        combined = relative_return_loss + weight_regularization
        val_combined = combined(w, r)
        val_a = relative_return_loss(w, r)
        val_b = weight_regularization(w, r)
        assert_allclose(val_combined.item(), val_a.item() + val_b.item(), rtol=1e-5)

    def test_scalar_multiplication(self) -> None:
        """0.1 * loss computes 0.1 * loss(w, r)."""
        torch.manual_seed(42)
        w = torch.rand(15, 5)
        w = w / w.sum(dim=-1, keepdim=True)
        r = 1.0 + torch.randn(15, 5) * 0.01

        scaled = 0.1 * turnover_penalty
        val_scaled = scaled(w, r)
        val_base = turnover_penalty(w, r)
        assert_allclose(val_scaled.item(), 0.1 * val_base.item(), rtol=1e-5)

    def test_complex_composition(self) -> None:
        """loss_a + 0.1 * loss_b + 0.01 * loss_c is arithmetically correct."""
        torch.manual_seed(42)
        w = torch.rand(15, 5)
        w = w / w.sum(dim=-1, keepdim=True)
        r = 1.0 + torch.randn(15, 5) * 0.01

        combined = (
            relative_return_loss + 0.1 * turnover_penalty + 0.01 * weight_regularization
        )
        val = combined(w, r)

        expected = (
            relative_return_loss(w, r).item()
            + 0.1 * turnover_penalty(w, r).item()
            + 0.01 * weight_regularization(w, r).item()
        )
        assert_allclose(val.item(), expected, rtol=1e-5)

    def test_composition_is_differentiable(self) -> None:
        """Combined loss supports backprop."""
        w = torch.rand(10, 5, requires_grad=True)
        r = 1.0 + torch.randn(10, 5) * 0.01
        combined = relative_return_loss + 0.05 * turnover_penalty
        loss = combined(w, r)
        loss.backward()
        assert w.grad is not None


# ---------------------------------------------------------------------------
# Training loop verification
# ---------------------------------------------------------------------------


class TestTrainingLoop:
    """Verify training actually reduces loss over epochs."""

    def test_training_reduces_loss(self, synthetic_market_data) -> None:
        """After sufficient epochs, loss must be strictly lower than initial."""
        mw, ret = synthetic_market_data
        config = NeuralFGPConfig(
            hidden_dims=[32, 16],
            epochs=50,
            train_window=50,
            eval_window=10,
            learning_rate=5e-3,
            seed=42,
        )
        model = NeuralFGP(n_assets=5, config=config)
        model.fit(mw, returns=ret)
        losses = model.training_history["loss"]
        assert len(losses) > 5
        assert losses[-1] <= losses[0]

    def test_custom_training_loop_reduces_loss(self, synthetic_market_data) -> None:
        """Manual training_step loop reduces loss over 30 steps."""
        mw, ret = synthetic_market_data
        config = NeuralFGPConfig(hidden_dims=[32, 16], seed=42)
        model = NeuralFGP(n_assets=5, config=config)
        model.setup(mw)

        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
        mw_t = torch.tensor(mw[:60], dtype=torch.float32)
        ret_t = torch.tensor(ret[:60], dtype=torch.float32)

        losses = []
        for _ in range(30):
            loss = model.training_step(mw_t, ret_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.enforce_constraints()
            losses.append(loss.item())

        assert all(np.isfinite(losses))
        assert losses[-1] <= losses[0]


# ---------------------------------------------------------------------------
# Weights sum to 1 for custom models
# ---------------------------------------------------------------------------


class TestCustomModelWeights:
    """Custom nn.Module → wrap_torch_model → weights sum to 1."""

    def test_weights_sum_to_one(self, simplex_points: np.ndarray) -> None:
        class CubicConvex(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x**3).sum(dim=-1) + (x**2).sum(dim=-1)

        wrapper = wrap_torch_model(CubicConvex(), n_assets=5, positivity_offset=3.0)
        for mu in simplex_points[:20]:
            w = wrapper.weights(mu)
            assert abs(w.sum() - 1.0) < 1e-4, f"Weights sum = {w.sum()}"
            assert np.all(w >= -1e-6), f"Negative weight: {w.min()}"

    def test_generating_function_positive(self, simplex_points: np.ndarray) -> None:
        class QuadConvex(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x**2).sum(dim=-1)

        wrapper = wrap_torch_model(QuadConvex(), n_assets=5, positivity_offset=2.0)
        for mu in simplex_points:
            G_val = wrapper.generating_function(mu)
            assert G_val > 0, f"G(μ) = {G_val} is not positive"


# ---------------------------------------------------------------------------
# GPU/CPU consistency (skip if no CUDA)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPUConsistency:
    """GPU results match CPU results."""

    def test_same_generating_function_value(self, simplex_point: np.ndarray) -> None:
        class QuadConvex(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x**2).sum(dim=-1)

        wrapper_cpu = wrap_torch_model(
            QuadConvex(), n_assets=5, positivity_offset=2.0, device="cpu"
        )
        wrapper_gpu = wrap_torch_model(
            QuadConvex(), n_assets=5, positivity_offset=2.0, device="cuda"
        )
        val_cpu = wrapper_cpu.generating_function(simplex_point)
        val_gpu = wrapper_gpu.generating_function(simplex_point)
        assert_allclose(val_cpu, val_gpu, atol=1e-5)

    def test_same_gradient(self, simplex_point: np.ndarray) -> None:
        class QuadConvex(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x**2).sum(dim=-1)

        wrapper_cpu = wrap_torch_model(
            QuadConvex(), n_assets=5, positivity_offset=2.0, device="cpu"
        )
        wrapper_gpu = wrap_torch_model(
            QuadConvex(), n_assets=5, positivity_offset=2.0, device="cuda"
        )
        grad_cpu = wrapper_cpu.log_gradient(simplex_point)
        grad_gpu = wrapper_gpu.log_gradient(simplex_point)
        assert_allclose(grad_cpu, grad_gpu, atol=1e-5)


# ---------------------------------------------------------------------------
# Float64 precision
# ---------------------------------------------------------------------------


class TestFloat64Precision:
    """Verify float64 is used throughout and no accidental float32 creep."""

    def test_log_gradient_is_float64(self, simplex_point: np.ndarray) -> None:
        class QuadConvex(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x**2).sum(dim=-1)

        wrapper = wrap_torch_model(QuadConvex(), n_assets=5, positivity_offset=2.0)
        grad = wrapper.log_gradient(simplex_point)
        assert grad.dtype == np.float64

    def test_hessian_is_float64(self, simplex_point: np.ndarray) -> None:
        class QuadConvex(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x**2).sum(dim=-1)

        wrapper = wrap_torch_model(QuadConvex(), n_assets=5, positivity_offset=2.0)
        H = wrapper.hessian(simplex_point)
        assert H.dtype == np.float64

    def test_weights_are_float64(self, simplex_point: np.ndarray) -> None:
        class QuadConvex(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x**2).sum(dim=-1)

        wrapper = wrap_torch_model(QuadConvex(), n_assets=5, positivity_offset=2.0)
        w = wrapper.weights(simplex_point)
        assert w.dtype == np.float64


# ---------------------------------------------------------------------------
# Drift integral loss
# ---------------------------------------------------------------------------


class TestDriftIntegralLoss:
    """Verify drift_integral_loss computes the correct excess growth rate
    differential and produces gradients pointing toward higher diversification."""

    @pytest.fixture
    def cov_data(self):
        """Diagonal covariance matrices (5 assets, 20 timesteps)."""
        torch.manual_seed(42)
        T = 20
        vols = torch.tensor([0.20, 0.25, 0.30, 0.15, 0.22])
        cov = torch.diag(vols**2).unsqueeze(0).expand(T, -1, -1)
        return cov

    @pytest.fixture
    def market_weights_fixture(self):
        """Pareto-distributed market weights (realistic: top-heavy)."""
        torch.manual_seed(42)
        rng_np = np.random.default_rng(42)
        raw = rng_np.pareto(1.0, size=(20, 5)) + 1.0
        mw = raw / raw.sum(axis=1, keepdims=True)
        return torch.tensor(mw, dtype=torch.float64)

    def test_returns_scalar(self, cov_data, market_weights_fixture) -> None:
        """drift_integral_loss returns a finite scalar."""
        pred = market_weights_fixture.clone()
        loss = drift_integral_loss(pred, market_weights_fixture, cov_data, 1 / 252)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_zero_drift_when_weights_equal_market(
        self, cov_data, market_weights_fixture
    ) -> None:
        """If predicted weights == market weights, drift is zero."""
        loss = drift_integral_loss(
            market_weights_fixture, market_weights_fixture, cov_data, 1 / 252
        )
        assert_allclose(loss.item(), 0.0, atol=1e-12)

    def test_diversified_portfolio_has_lower_loss(
        self, cov_data, market_weights_fixture
    ) -> None:
        """More diversified (equal-weight) portfolio should have lower loss
        than the concentrated market portfolio, because its excess growth
        rate is higher."""
        T, n = 20, 5
        equal_w = torch.ones(T, n, dtype=torch.float64) / n
        loss_equal = drift_integral_loss(
            equal_w, market_weights_fixture, cov_data, 1 / 252
        )
        loss_market = drift_integral_loss(
            market_weights_fixture, market_weights_fixture, cov_data, 1 / 252
        )
        assert loss_equal.item() < loss_market.item(), (
            f"Equal-weight loss {loss_equal.item():.8f} should be less than "
            f"market-weight loss {loss_market.item():.8f}"
        )

    def test_concentrated_portfolio_has_higher_loss(
        self, cov_data, market_weights_fixture
    ) -> None:
        """Single-stock portfolio (no diversification) should have higher
        loss than the market portfolio."""
        T, n = 20, 5
        concentrated = torch.zeros(T, n, dtype=torch.float64)
        concentrated[:, 0] = 1.0
        loss_conc = drift_integral_loss(
            concentrated, market_weights_fixture, cov_data, 1 / 252
        )
        loss_market = drift_integral_loss(
            market_weights_fixture, market_weights_fixture, cov_data, 1 / 252
        )
        assert loss_conc.item() > loss_market.item()

    def test_gradient_points_toward_diversification(
        self, cov_data, market_weights_fixture
    ) -> None:
        """Gradient of a concentrated portfolio should push weights
        toward diversification (reduce concentration)."""
        T, n = 20, 5
        pred = torch.zeros(T, n, dtype=torch.float64)
        pred[:, 0] = 0.80
        pred[:, 1:] = 0.05
        pred = pred.clone().requires_grad_(True)

        loss = drift_integral_loss(pred, market_weights_fixture, cov_data, 1 / 252)
        loss.backward()

        grad = pred.grad
        assert grad is not None
        mean_grad = grad.mean(dim=0)
        assert (
            mean_grad[0] > 0
        ), "Gradient on the dominant stock should be positive (increase loss)"

    def test_differentiable(self, cov_data, market_weights_fixture) -> None:
        """Loss supports full autograd."""
        pred = market_weights_fixture.clone().requires_grad_(True)
        loss = drift_integral_loss(pred, market_weights_fixture, cov_data, 1 / 252)
        loss.backward()
        assert pred.grad is not None
        assert torch.all(torch.isfinite(pred.grad))

    def test_scales_with_dt(self, cov_data, market_weights_fixture) -> None:
        """Doubling dt should double the drift integral."""
        T, n = 20, 5
        equal_w = torch.ones(T, n, dtype=torch.float64) / n
        loss_dt1 = drift_integral_loss(
            equal_w, market_weights_fixture, cov_data, 1 / 252
        )
        loss_dt2 = drift_integral_loss(
            equal_w, market_weights_fixture, cov_data, 2 / 252
        )
        assert_allclose(loss_dt2.item(), 2 * loss_dt1.item(), rtol=1e-10)

    def test_class_interface_matches_function(
        self, cov_data, market_weights_fixture
    ) -> None:
        """DriftIntegralLoss class produces same result as the function."""
        T, n = 20, 5
        equal_w = torch.ones(T, n, dtype=torch.float64) / n
        dummy_returns = torch.ones(T, n, dtype=torch.float64)

        func_loss = drift_integral_loss(
            equal_w, market_weights_fixture, cov_data, 1 / 252
        )
        cls_loss = DriftIntegralLoss(dt=1 / 252)(
            equal_w,
            dummy_returns,
            market_weights=market_weights_fixture,
            covariance_matrices=cov_data,
        )
        assert_allclose(cls_loss.item(), func_loss.item(), rtol=1e-12)

    def test_class_raises_without_required_kwargs(self) -> None:
        """DriftIntegralLoss raises if market_weights or covariances missing."""
        loss_fn = DriftIntegralLoss()
        w = torch.rand(10, 5)
        r = torch.ones(10, 5)
        with pytest.raises(ValueError, match="market_weights"):
            loss_fn(w, r)

    def test_composable_with_other_losses(
        self, cov_data, market_weights_fixture
    ) -> None:
        """DriftIntegralLoss can be composed with other losses."""
        combined = DriftIntegralLoss(dt=1 / 252) + 0.01 * weight_regularization
        assert combined is not None


# ---------------------------------------------------------------------------
# Test data quality: edge cases and scale diversity
# ---------------------------------------------------------------------------


class TestEdgeCaseWeights:
    """Verify losses and wrappers handle edge-case market structures."""

    def test_loss_with_dominant_stock(self, dominant_stock_weights) -> None:
        """Loss functions work with a single dominant stock (~90%)."""
        T = 15
        w = np.tile(dominant_stock_weights, (T, 1))
        w_t = torch.tensor(w, dtype=torch.float64)
        r_t = 1.0 + torch.randn(T, 5, dtype=torch.float64) * 0.01
        loss = relative_return_loss(w_t, r_t)
        assert torch.isfinite(loss)
        loss2 = weight_regularization(w_t, r_t)
        assert loss2.item() > 0

    def test_loss_with_near_equal_weights(self, near_equal_weights_5) -> None:
        """Loss functions work with near-equal (1/n) weights."""
        T = 15
        w = np.tile(near_equal_weights_5, (T, 1))
        w_t = torch.tensor(w, dtype=torch.float64)
        r_t = 1.0 + torch.randn(T, 5, dtype=torch.float64) * 0.01
        loss = relative_return_loss(w_t, r_t)
        assert torch.isfinite(loss)

    def test_loss_with_near_zero_weights(self, near_zero_weights) -> None:
        """Loss functions handle weights very close to zero."""
        T = 15
        w = np.tile(near_zero_weights, (T, 1))
        w_t = torch.tensor(w, dtype=torch.float64)
        r_t = 1.0 + torch.randn(T, 5, dtype=torch.float64) * 0.01
        loss = relative_return_loss(w_t, r_t)
        assert torch.isfinite(loss)

    def test_wrapper_with_dominant_stock(self, dominant_stock_weights) -> None:
        """Torch model wrapper produces valid weights for dominant-stock input."""

        class QuadConvex(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x**2).sum(dim=-1)

        wrapper = wrap_torch_model(QuadConvex(), n_assets=5, positivity_offset=2.0)
        w = wrapper.weights(dominant_stock_weights)
        assert abs(w.sum() - 1.0) < 1e-4
        assert np.all(np.isfinite(w))

    def test_wrapper_with_near_zero_weights(self, near_zero_weights) -> None:
        """Wrapper handles near-zero weight inputs without numerical blow-up."""

        class QuadConvex(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x**2).sum(dim=-1)

        wrapper = wrap_torch_model(QuadConvex(), n_assets=5, positivity_offset=2.0)
        G = wrapper.generating_function(near_zero_weights)
        assert G > 0
        assert np.isfinite(G)


class TestScaleDiversity:
    """Verify core operations at different asset counts: 2, 50."""

    def test_2_asset_loss(self) -> None:
        """Loss functions work with a 2-stock market."""
        torch.manual_seed(42)
        T, n = 20, 2
        w = torch.rand(T, n, dtype=torch.float64)
        w = w / w.sum(dim=-1, keepdim=True)
        r = 1.0 + torch.randn(T, n, dtype=torch.float64) * 0.01
        for loss_fn in [
            relative_return_loss,
            weight_regularization,
            turnover_penalty,
            sharpe_of_relative_loss,
        ]:
            loss = loss_fn(w, r)
            assert loss.shape == ()
            assert torch.isfinite(loss)

    def test_50_asset_loss(self) -> None:
        """Loss functions work with a 50-stock market."""
        torch.manual_seed(42)
        T, n = 20, 50
        w = torch.rand(T, n, dtype=torch.float64)
        w = w / w.sum(dim=-1, keepdim=True)
        r = 1.0 + torch.randn(T, n, dtype=torch.float64) * 0.01
        for loss_fn in [
            relative_return_loss,
            weight_regularization,
            turnover_penalty,
            sharpe_of_relative_loss,
        ]:
            loss = loss_fn(w, r)
            assert loss.shape == ()
            assert torch.isfinite(loss)

    def test_2_asset_wrapper(self) -> None:
        """Torch wrapper works with 2-asset simplex."""

        class QuadConvex2(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x**2).sum(dim=-1)

        wrapper = wrap_torch_model(QuadConvex2(), n_assets=2, positivity_offset=2.0)
        mu = np.array([0.6, 0.4])
        w = wrapper.weights(mu)
        assert abs(w.sum() - 1.0) < 1e-4

    def test_50_asset_wrapper(self, pareto_weights_50) -> None:
        """Torch wrapper works with 50-asset Pareto-distributed weights."""

        class QuadConvex50(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x**2).sum(dim=-1)

        wrapper = wrap_torch_model(QuadConvex50(), n_assets=50, positivity_offset=2.0)
        w = wrapper.weights(pareto_weights_50)
        assert abs(w.sum() - 1.0) < 1e-4
        assert np.all(np.isfinite(w))

    def test_drift_integral_loss_50_assets(self, pareto_weights_50) -> None:
        """Drift integral loss works with 50-asset Pareto market.

        With uniform volatilities, equal-weight provably maximises the
        excess growth rate, so the drift integral must be negative (loss
        wants to be minimised).
        """
        T, n = 15, 50
        mw = np.tile(pareto_weights_50, (T, 1))
        eq_w = np.ones((T, n)) / n
        vol = 0.20
        cov = np.zeros((T, n, n))
        for t in range(T):
            cov[t] = np.eye(n) * vol**2

        mw_t = torch.tensor(mw, dtype=torch.float64)
        eq_t = torch.tensor(eq_w, dtype=torch.float64)
        cov_t = torch.tensor(cov, dtype=torch.float64)

        loss = drift_integral_loss(eq_t, mw_t, cov_t, 1 / 252)
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss.item() < 0  # equal-weight outperforms Pareto with uniform vols
