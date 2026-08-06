"""Tests for Neural FGP — ICNN convexity, modular training, and protocol integration.

Validates mathematical correctness of the Input Convex Neural Network
and the modular Neural FGP framework against arXiv:2506.19715.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from quantspt.core.generating_functions import (
    GeneratingFunction,
    drift_process,
)
from quantspt.ml._protocols import GeneratingFunctionModel
from quantspt.ml.neural_fgp import (
    InputConvexNN,
    NeuralFGP,
    NeuralFGPConfig,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def simplex_points(rng: np.random.Generator) -> np.ndarray:
    """50 random points on the 5-simplex."""
    alpha = rng.exponential(size=(50, 5))
    return alpha / alpha.sum(axis=1, keepdims=True)


@pytest.fixture
def icnn_5d() -> InputConvexNN:
    """Small ICNN for 5 assets."""
    return InputConvexNN(n_inputs=5, hidden_dims=[32, 16], activation="softplus")


@pytest.fixture
def synthetic_market_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic market weight paths and returns (300 days, 5 assets)."""
    T, n = 300, 5
    prices = np.exp(np.cumsum(rng.normal(0.0001, 0.02, size=(T, n)), axis=0))
    weights = prices / prices.sum(axis=1, keepdims=True)
    returns = prices[1:] / prices[:-1]
    return weights[:-1].astype(np.float64), returns.astype(np.float64)


# ---------------------------------------------------------------------------
# ICNN Convexity Tests
# ---------------------------------------------------------------------------


class TestICNNConvexity:
    """Verify ICNN produces a convex function f(x)."""

    def test_convexity_midpoint(
        self, icnn_5d: InputConvexNN, simplex_points: np.ndarray
    ) -> None:
        """f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y) for convex f."""
        icnn_5d.eval()
        x = torch.tensor(simplex_points[:25], dtype=torch.float32)
        y = torch.tensor(simplex_points[25:], dtype=torch.float32)
        lam = 0.5

        with torch.no_grad():
            f_mid = icnn_5d(lam * x + (1 - lam) * y)
            f_x = icnn_5d(x)
            f_y = icnn_5d(y)

        assert torch.all(f_mid <= lam * f_x + (1 - lam) * f_y + 1e-5)

    def test_convexity_multiple_lambdas(
        self, icnn_5d: InputConvexNN, simplex_points: np.ndarray
    ) -> None:
        """Convexity holds for multiple interpolation weights."""
        icnn_5d.eval()
        x = torch.tensor(simplex_points[:10], dtype=torch.float32)
        y = torch.tensor(simplex_points[10:20], dtype=torch.float32)
        for lam in [0.1, 0.3, 0.7, 0.9]:
            with torch.no_grad():
                f_mid = icnn_5d(lam * x + (1 - lam) * y)
                rhs = lam * icnn_5d(x) + (1 - lam) * icnn_5d(y)
            assert torch.all(f_mid <= rhs + 1e-5)

    def test_G_is_concave(
        self, icnn_5d: InputConvexNN, simplex_points: np.ndarray
    ) -> None:
        """G_θ = -f + offset is concave: G(mid) >= weighted average."""
        icnn_5d.eval()
        offset = 1.0
        x = torch.tensor(simplex_points[:10], dtype=torch.float32)
        y = torch.tensor(simplex_points[10:20], dtype=torch.float32)
        lam = 0.5

        with torch.no_grad():
            G_mid = -icnn_5d(lam * x + (1 - lam) * y) + offset
            G_x = -icnn_5d(x) + offset
            G_y = -icnn_5d(y) + offset

        assert torch.all(G_mid >= lam * G_x + (1 - lam) * G_y - 1e-5)

    def test_hessian_of_f_is_psd(
        self, icnn_5d: InputConvexNN, rng: np.random.Generator
    ) -> None:
        """Hessian of f (convex) must be PSD."""
        icnn_5d.eval()
        alpha = rng.exponential(size=5)
        mu = (alpha / alpha.sum()).astype(np.float64)
        mu_t = torch.tensor(mu, dtype=torch.float64)

        def f_func(x: torch.Tensor) -> torch.Tensor:
            return _icnn_f64(icnn_5d, x.unsqueeze(0)).squeeze(0)

        H = torch.autograd.functional.hessian(f_func, mu_t)
        eigvals = np.linalg.eigvalsh(H.numpy())  # type: ignore[union-attr]
        assert eigvals[0] >= -1e-4

    def test_hessian_of_G_is_nsd(
        self, icnn_5d: InputConvexNN, rng: np.random.Generator
    ) -> None:
        """Hessian of G = -f (concave) must be NSD (all eigenvalues ≤ 0)."""
        icnn_5d.eval()
        alpha = rng.exponential(size=5)
        mu = (alpha / alpha.sum()).astype(np.float64)
        mu_t = torch.tensor(mu, dtype=torch.float64)

        def G_func(x: torch.Tensor) -> torch.Tensor:
            return -_icnn_f64(icnn_5d, x.unsqueeze(0)).squeeze(0) + 1.0

        H = torch.autograd.functional.hessian(G_func, mu_t)
        eigvals = np.linalg.eigvalsh(H.numpy())  # type: ignore[union-attr]
        assert eigvals[-1] <= 1e-4

    def test_weights_always_non_negative(self, icnn_5d: InputConvexNN) -> None:
        """Softplus parametrization ensures Wz and w_out weights are always ≥ 0."""
        for Wz_k in icnn_5d._Wz:
            assert torch.all(Wz_k.weight >= 0), "Wz weights must be non-negative"
        assert torch.all(icnn_5d._w_out.weight >= 0), "w_out must be non-negative"


# ---------------------------------------------------------------------------
# Modular Training Tests
# ---------------------------------------------------------------------------


class TestNeuralFGPTraining:
    """Verify training loop is modular and reduces loss."""

    def test_fit_reduces_loss(
        self, synthetic_market_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Standard fit() should reduce loss over epochs."""
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
        assert losses[-1] <= losses[0], (
            f"Expected loss to not increase: {losses[0]:.6f} → {losses[-1]:.6f}"
        )

    def test_custom_training_loop(
        self, synthetic_market_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Users can run their own training loop via training_step."""
        mw, ret = synthetic_market_data
        config = NeuralFGPConfig(hidden_dims=[32, 16], seed=42)
        model = NeuralFGP(n_assets=5, config=config)
        model.setup(mw)

        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
        mw_t = torch.tensor(mw[:60], dtype=torch.float32)
        ret_t = torch.tensor(ret[:60], dtype=torch.float32)

        losses = []
        for _epoch in range(30):
            loss = model.training_step(mw_t, ret_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.enforce_constraints()
            losses.append(loss.item())

        assert all(np.isfinite(losses))
        assert losses[-1] <= losses[0], (
            f"Expected loss to not increase: {losses[0]:.6f} → {losses[-1]:.6f}"
        )

    def test_custom_loss_function(
        self, synthetic_market_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Users can pass custom loss functions."""
        from quantspt.ml.losses import relative_return_loss, turnover_penalty

        mw, ret = synthetic_market_data
        custom_loss = relative_return_loss + 0.05 * turnover_penalty
        config = NeuralFGPConfig(
            hidden_dims=[32, 16],
            epochs=10,
            train_window=50,
            eval_window=10,
            seed=42,
        )
        model = NeuralFGP(n_assets=5, config=config, loss_fn=custom_loss)
        model.fit(mw, returns=ret)
        assert len(model.training_history["loss"]) > 0

    def test_custom_network_pluggable(
        self, synthetic_market_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Users can plug in their own PyTorch model."""
        mw, ret = synthetic_market_data
        custom_icnn = InputConvexNN(n_inputs=5, hidden_dims=[16, 8], activation="relu")
        config = NeuralFGPConfig(
            epochs=10,
            train_window=50,
            eval_window=10,
            seed=42,
        )
        model = NeuralFGP(n_assets=5, config=config, network=custom_icnn)
        model.fit(mw, returns=ret)
        assert model._fitted

    def test_walk_forward_produces_val_loss(
        self, synthetic_market_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Walk-forward training produces validation loss entries."""
        mw, ret = synthetic_market_data
        config = NeuralFGPConfig(
            hidden_dims=[32, 16],
            epochs=10,
            train_window=50,
            eval_window=10,
            walk_forward=True,
            seed=42,
        )
        model = NeuralFGP(n_assets=5, config=config)
        model.fit(mw, returns=ret)
        assert len(model.training_history["val_loss"]) > 0


# ---------------------------------------------------------------------------
# Mathematical Correctness Tests
# ---------------------------------------------------------------------------


class TestNeuralFGPMath:
    """Validate mathematical properties of fitted model."""

    def test_positive_G_values(
        self,
        synthetic_market_data: tuple[np.ndarray, np.ndarray],
        rng: np.random.Generator,
    ) -> None:
        """G_θ(μ) > 0 for all μ ∈ Δ_n⁺."""
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

        for _ in range(20):
            alpha = rng.exponential(size=5)
            mu = alpha / alpha.sum()
            assert model.generating_function(mu) > 0

    def test_weights_sum_to_one(
        self,
        synthetic_market_data: tuple[np.ndarray, np.ndarray],
        rng: np.random.Generator,
    ) -> None:
        """Portfolio weights must sum to 1."""
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

        for _ in range(10):
            alpha = rng.exponential(size=5)
            mu = alpha / alpha.sum()
            w = model.weights(mu)
            assert abs(w.sum() - 1.0) < 1e-6
            assert np.all(w >= 0)

    def test_hessian_is_symmetric(
        self,
        synthetic_market_data: tuple[np.ndarray, np.ndarray],
        rng: np.random.Generator,
    ) -> None:
        """Hessian D²G_θ must be symmetric."""
        mw, ret = synthetic_market_data
        config = NeuralFGPConfig(
            hidden_dims=[32, 16],
            epochs=15,
            train_window=50,
            eval_window=10,
            seed=42,
        )
        model = NeuralFGP(n_assets=5, config=config)
        model.fit(mw, returns=ret)

        alpha = rng.exponential(size=5)
        mu = alpha / alpha.sum()
        H = model.hessian(mu)
        assert np.allclose(H, H.T, atol=1e-5)

    def test_unfitted_raises(self, rng: np.random.Generator) -> None:
        """Calling eval methods on unfitted model raises."""
        model = NeuralFGP(n_assets=5)
        mu = rng.dirichlet(np.ones(5))
        with pytest.raises(RuntimeError, match="fitted"):
            model.generating_function(mu)
        with pytest.raises(RuntimeError, match="fitted"):
            model.log_gradient(mu)
        with pytest.raises(RuntimeError, match="fitted"):
            model.hessian(mu)
        with pytest.raises(RuntimeError, match="fitted"):
            model.to_generating_function()


# ---------------------------------------------------------------------------
# Protocol Integration Tests
# ---------------------------------------------------------------------------


class TestProtocolIntegration:
    """Verify NeuralFGP integrates with core SPT."""

    def test_implements_protocol(self) -> None:
        """NeuralFGP satisfies GeneratingFunctionModel Protocol."""
        model = NeuralFGP(n_assets=5)
        assert isinstance(model, GeneratingFunctionModel)

    def test_to_generating_function(
        self, synthetic_market_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """to_generating_function() returns a GeneratingFunction."""
        mw, ret = synthetic_market_data
        config = NeuralFGPConfig(
            hidden_dims=[32, 16],
            epochs=15,
            train_window=50,
            eval_window=10,
            seed=42,
        )
        model = NeuralFGP(n_assets=5, config=config)
        model.fit(mw, returns=ret)
        G = model.to_generating_function()
        assert isinstance(G, GeneratingFunction)

    def test_works_with_drift_process(
        self,
        synthetic_market_data: tuple[np.ndarray, np.ndarray],
        rng: np.random.Generator,
    ) -> None:
        """Converted G works with core drift_process()."""
        mw, ret = synthetic_market_data
        config = NeuralFGPConfig(
            hidden_dims=[32, 16],
            epochs=15,
            train_window=50,
            eval_window=10,
            seed=42,
        )
        model = NeuralFGP(n_assets=5, config=config)
        model.fit(mw, returns=ret)
        G = model.to_generating_function()

        mu = rng.dirichlet(np.ones(5))
        tau_mu = rng.uniform(0.01, 0.1, size=(5, 5))
        tau_mu = (tau_mu + tau_mu.T) / 2

        drift = drift_process(G, mu, tau_mu)
        assert np.isfinite(drift)

    def test_weights_on_simplex(
        self,
        synthetic_market_data: tuple[np.ndarray, np.ndarray],
        rng: np.random.Generator,
    ) -> None:
        """Weights from GeneratingFunction are valid."""
        mw, ret = synthetic_market_data
        config = NeuralFGPConfig(
            hidden_dims=[32, 16],
            epochs=15,
            train_window=50,
            eval_window=10,
            seed=42,
        )
        model = NeuralFGP(n_assets=5, config=config)
        model.fit(mw, returns=ret)
        G = model.to_generating_function()

        mu = rng.dirichlet(np.ones(5))
        w = G.weights(mu)
        assert abs(w.sum() - 1.0) < 1e-5
        assert np.all(w >= -1e-6)


# ---------------------------------------------------------------------------
# GPU/CPU Consistency Tests
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestNeuralFGPGPUConsistency:
    """Verify GPU and CPU produce identical results for NeuralFGP."""

    @pytest.fixture
    def trained_model_cpu(
        self, synthetic_market_data: tuple[np.ndarray, np.ndarray]
    ) -> NeuralFGP:
        """Train a NeuralFGP on CPU with fixed seed."""
        mw, ret = synthetic_market_data
        config = NeuralFGPConfig(
            hidden_dims=[32, 16],
            epochs=20,
            train_window=50,
            eval_window=10,
            learning_rate=5e-3,
            seed=42,
            device="cpu",
        )
        model = NeuralFGP(n_assets=5, config=config)
        model.fit(mw, returns=ret)
        return model

    @pytest.fixture
    def trained_model_gpu(
        self, synthetic_market_data: tuple[np.ndarray, np.ndarray]
    ) -> NeuralFGP:
        """Train a NeuralFGP on GPU with fixed seed."""
        mw, ret = synthetic_market_data
        config = NeuralFGPConfig(
            hidden_dims=[32, 16],
            epochs=20,
            train_window=50,
            eval_window=10,
            learning_rate=5e-3,
            seed=42,
            device="cuda",
        )
        model = NeuralFGP(n_assets=5, config=config)
        model.fit(mw, returns=ret)
        return model

    def test_generating_function_cpu_gpu_match(
        self,
        trained_model_cpu: NeuralFGP,
        trained_model_gpu: NeuralFGP,
        rng: np.random.Generator,
    ) -> None:
        """G_θ(μ) must match to float64 precision (Category A evaluation).

        generating_function() promotes model to float64 before evaluation,
        so CPU and GPU must agree to ~1e-12 since all arithmetic is in
        IEEE 754 double precision on both devices.
        """
        for _ in range(20):
            alpha = rng.exponential(size=5)
            mu = alpha / alpha.sum()
            val_cpu = trained_model_cpu.generating_function(mu)
            val_gpu = trained_model_gpu.generating_function(mu)
            assert abs(val_cpu - val_gpu) < 1e-12, (
                f"CPU={val_cpu:.15e}, GPU={val_gpu:.15e}"
            )

    def test_weights_cpu_gpu_match(
        self,
        trained_model_cpu: NeuralFGP,
        trained_model_gpu: NeuralFGP,
        rng: np.random.Generator,
    ) -> None:
        """Portfolio weights must match to float64 precision (Category A).

        weights() calls log_gradient() which promotes to float64.
        Fernholz weight formula involves cancellation (π_i - μ_i terms)
        that requires full double precision.
        """
        for _ in range(10):
            alpha = rng.exponential(size=5)
            mu = alpha / alpha.sum()
            w_cpu = trained_model_cpu.weights(mu)
            w_gpu = trained_model_gpu.weights(mu)
            np.testing.assert_allclose(w_cpu, w_gpu, atol=1e-12)

    def test_hessian_cpu_gpu_match(
        self,
        trained_model_cpu: NeuralFGP,
        trained_model_gpu: NeuralFGP,
        rng: np.random.Generator,
    ) -> None:
        """Hessian must match to float64 precision (Category A).

        D²G feeds into drift process computation where second-order
        differences amplify precision loss. Promotion to float64
        guarantees identical results on both devices.
        """
        alpha = rng.exponential(size=5)
        mu = alpha / alpha.sum()
        H_cpu = trained_model_cpu.hessian(mu)
        H_gpu = trained_model_gpu.hessian(mu)
        np.testing.assert_allclose(H_cpu, H_gpu, atol=1e-12)

    def test_icnn_forward_cpu_gpu_match(self, rng: np.random.Generator) -> None:
        """Verify CPU and GPU produce identical ICNN forward pass.

        Uses float64 to isolate code correctness from precision artifacts.
        Any mismatch > 1e-12 indicates a code bug (different logic paths),
        not a numerical precision difference.
        """
        torch.manual_seed(42)
        icnn = InputConvexNN(n_inputs=5, hidden_dims=[32, 16], activation="softplus")
        icnn.double()

        alpha = rng.exponential(size=(20, 5))
        mu = (alpha / alpha.sum(axis=1, keepdims=True)).astype(np.float64)
        x = torch.tensor(mu, dtype=torch.float64)

        icnn.to("cpu")
        icnn.eval()
        with torch.no_grad():
            out_cpu = icnn(x).numpy()

        icnn.to("cuda")
        with torch.no_grad():
            out_gpu = icnn(x.to("cuda")).cpu().numpy()

        np.testing.assert_allclose(out_cpu, out_gpu, atol=1e-12)

    def test_training_loss_converges_on_gpu(
        self, synthetic_market_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Training on GPU converges (loss decreases)."""
        mw, ret = synthetic_market_data
        config = NeuralFGPConfig(
            hidden_dims=[32, 16],
            epochs=50,
            train_window=50,
            eval_window=10,
            learning_rate=5e-3,
            seed=42,
            device="cuda",
        )
        model = NeuralFGP(n_assets=5, config=config)
        model.fit(mw, returns=ret)
        losses = model.training_history["loss"]
        assert losses[-1] <= losses[0], (
            f"GPU training loss did not decrease: {losses[0]:.6f} → {losses[-1]:.6f}"
        )


# ---------------------------------------------------------------------------
# Coverage: Training Internals (lines 123, 179, 191, 224, 232, 354, 362,
#   367, 405, 409, 453, 460-462, 477-478, 528-545)
# ---------------------------------------------------------------------------


class TestNeuralFGPInternals:
    """Exercise all internal code paths — no mocks, real execution."""

    def test_unknown_optimizer_raises(self) -> None:
        """Line 123: _build_optimizer raises ValueError for unknown name."""
        config = NeuralFGPConfig(
            optimizer="nosuchoptimizer",
            hidden_dims=[16, 8],
            epochs=1,
            train_window=20,
            eval_window=5,
        )
        model = NeuralFGP(n_assets=3, config=config)
        rng = np.random.default_rng(99)
        mw = rng.dirichlet(np.ones(3), size=50).astype(np.float64)
        with pytest.raises(ValueError, match="Unknown optimizer"):
            model.fit(mw)

    def test_icnn_default_hidden_dims(self) -> None:
        """Line 179: InputConvexNN uses [64,64,32] when hidden_dims=None."""
        icnn = InputConvexNN(n_inputs=4, hidden_dims=None)
        assert icnn.hidden_dims == [64, 64, 32]
        x = torch.randn(2, 4)
        out = icnn(x)
        assert out.shape == (2,)

    def test_icnn_unsupported_activation_raises(self) -> None:
        """Line 191: InputConvexNN raises ValueError for bad activation."""
        with pytest.raises(ValueError, match="Unsupported activation"):
            InputConvexNN(n_inputs=3, hidden_dims=[16], activation="elu")

    def test_icnn_module_property(self) -> None:
        """Line 224: .module property returns the underlying nn.Module."""
        import torch.nn as nn

        icnn = InputConvexNN(n_inputs=3, hidden_dims=[16, 8])
        mod = icnn.module
        assert isinstance(mod, nn.Module)
        assert hasattr(mod, "W0")

    def test_icnn_train_method(self) -> None:
        """Line 232: .train() sets the module to training mode."""
        icnn = InputConvexNN(n_inputs=3, hidden_dims=[16, 8])
        icnn.eval()
        assert not icnn._module.training
        icnn.train()
        assert icnn._module.training

    def test_neural_fgp_config_property(self) -> None:
        """Line 354: .config property returns the NeuralFGPConfig."""
        cfg = NeuralFGPConfig(epochs=77)
        model = NeuralFGP(n_assets=3, config=cfg)
        assert model.config is cfg
        assert model.config.epochs == 77

    def test_neural_fgp_n_assets_property(self) -> None:
        """Line 362: .n_assets property returns the asset count."""
        model = NeuralFGP(n_assets=7)
        assert model.n_assets == 7

    def test_parameters_before_setup_raises(self) -> None:
        """Line 367: .parameters() before setup() raises RuntimeError."""
        model = NeuralFGP(n_assets=5)
        with pytest.raises(RuntimeError, match=r"setup.*fit"):
            model.parameters()

    def test_training_step_before_setup_raises(self) -> None:
        """Line 405: training_step() before setup() raises RuntimeError."""
        model = NeuralFGP(n_assets=3)
        mw_t = torch.randn(10, 3)
        ret_t = torch.randn(10, 3)
        with pytest.raises(RuntimeError, match="setup"):
            model.training_step(mw_t, ret_t)

    def test_training_step_single_row_returns_zero(self) -> None:
        """Line 409: training_step with T<=0 returns zero-grad tensor."""
        model = NeuralFGP(n_assets=3, config=NeuralFGPConfig(hidden_dims=[8]))
        model.setup()
        mw_t = torch.tensor([[0.3, 0.4, 0.3]])
        ret_t = torch.tensor([[1.01, 0.99, 1.0]])
        loss = model.training_step(mw_t, ret_t)
        assert loss.item() == pytest.approx(0.0)
        assert loss.requires_grad

    def test_fit_with_loss_fn_override(self) -> None:
        """Line 453: passing loss_fn to fit() overrides the model's loss."""
        from quantspt.ml.losses import sharpe_of_relative_loss

        rng = np.random.default_rng(11)
        mw = rng.dirichlet(np.ones(4), size=100).astype(np.float64)
        ret = mw[1:] / mw[:-1]
        mw = mw[:-1]

        config = NeuralFGPConfig(
            hidden_dims=[16, 8],
            epochs=3,
            train_window=30,
            eval_window=10,
            seed=11,
        )
        model = NeuralFGP(n_assets=4, config=config)
        model.fit(mw, returns=ret, loss_fn=sharpe_of_relative_loss)
        assert len(model.training_history["loss"]) > 0

    def test_fit_derives_returns_when_none(self) -> None:
        """Lines 460-462: fit() computes returns from market_weights when not provided."""
        rng = np.random.default_rng(22)
        mw = rng.dirichlet(np.ones(3), size=80).astype(np.float64)

        config = NeuralFGPConfig(
            hidden_dims=[16, 8],
            epochs=3,
            train_window=20,
            eval_window=10,
            seed=22,
        )
        model = NeuralFGP(n_assets=3, config=config)
        model.fit(mw)  # no `returns=` argument
        assert model._fitted
        assert len(model.training_history["loss"]) > 0

    def test_fit_non_walk_forward_single_split(self) -> None:
        """Lines 477-478, 528-545: walk_forward=False uses _train_single."""
        rng = np.random.default_rng(33)
        mw = rng.dirichlet(np.ones(4), size=120).astype(np.float64)
        ret = mw[1:] / mw[:-1]
        mw = mw[:-1]

        config = NeuralFGPConfig(
            hidden_dims=[16, 8],
            epochs=10,
            walk_forward=False,
            learning_rate=5e-3,
            seed=33,
        )
        model = NeuralFGP(n_assets=4, config=config)
        model.fit(mw, returns=ret, validation_split=0.2)
        assert model._fitted
        losses = model.training_history["loss"]
        val_losses = model.training_history["val_loss"]
        assert len(losses) > 0
        assert len(val_losses) == len(losses)

    @pytest.mark.slow
    def test_full_training_session_all_paths(self) -> None:
        """Full training covering walk-forward, early stopping, and inference."""
        rng = np.random.default_rng(42)
        mw = rng.dirichlet(np.ones(5), size=200).astype(np.float64)

        config = NeuralFGPConfig(
            hidden_dims=[16, 8],
            epochs=30,
            learning_rate=1e-3,
            train_window=50,
            eval_window=20,
            early_stopping_patience=10,
            walk_forward=True,
            seed=42,
        )

        model = NeuralFGP(n_assets=5, config=config)
        model.fit(mw)

        assert model._fitted
        assert len(model.training_history["loss"]) > 0
        assert len(model.training_history["val_loss"]) > 0

        G = model.to_generating_function()
        assert isinstance(G, GeneratingFunction)

        mu = mw[-1]
        g_val = model.generating_function(mu)
        assert g_val > 0

        pi = model.weights(mu)
        assert abs(pi.sum() - 1.0) < 1e-6
        assert np.all(pi >= 0)

    @pytest.mark.slow
    def test_single_split_early_stopping(self) -> None:
        """Lines 528-545: _train_single can trigger early stopping."""
        rng = np.random.default_rng(55)
        mw = rng.dirichlet(np.ones(4), size=200).astype(np.float64)
        ret = mw[1:] / mw[:-1]
        mw = mw[:-1]

        config = NeuralFGPConfig(
            hidden_dims=[32, 16],
            epochs=200,
            walk_forward=False,
            early_stopping_patience=5,
            learning_rate=1e-2,
            seed=55,
        )
        model = NeuralFGP(n_assets=4, config=config)
        model.fit(mw, returns=ret, validation_split=0.2)
        assert model._fitted
        assert len(model.training_history["loss"]) < 200


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _icnn_f64(icnn: InputConvexNN, x: torch.Tensor) -> torch.Tensor:
    """Evaluate ICNN in float64 by promoting weights (Category C boundary)."""
    icnn.double()
    result = icnn(x)
    icnn.float()
    return result
