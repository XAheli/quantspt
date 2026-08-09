"""Coverage tests for ml/losses.py — real tensors, backward, composition."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from quantspt.ml.losses import (
    DriftIntegralLoss,
    _CompositeLoss,
    default_loss,
    drift_integral_loss,
    relative_return_loss,
    sharpe_of_relative_loss,
    turnover_penalty,
    weight_regularization,
)

T, N = 20, 5


def _weights_and_returns() -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(42)
    w = rng.dirichlet(np.ones(N), size=T)
    r = 1.0 + rng.standard_normal((T, N)) * 0.01
    return (
        torch.tensor(w, dtype=torch.float64, requires_grad=True),
        torch.tensor(r, dtype=torch.float64),
    )


class TestRelativeReturnLoss:
    def test_call_produces_scalar(self) -> None:
        w, r = _weights_and_returns()
        loss = relative_return_loss(w, r)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_backward(self) -> None:
        w, r = _weights_and_returns()
        loss = relative_return_loss(w, r)
        loss.backward()
        assert w.grad is not None
        assert w.grad.shape == w.shape


class TestWeightRegularization:
    def test_call_produces_scalar(self) -> None:
        w, r = _weights_and_returns()
        loss = weight_regularization(w, r)
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_backward(self) -> None:
        w, r = _weights_and_returns()
        loss = weight_regularization(w, r)
        loss.backward()
        assert w.grad is not None


class TestTurnoverPenalty:
    def test_call_with_multiple_steps(self) -> None:
        w, r = _weights_and_returns()
        loss = turnover_penalty(w, r)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_single_step_returns_zero(self) -> None:
        w = torch.rand(1, N, dtype=torch.float64)
        r = torch.rand(1, N, dtype=torch.float64)
        loss = turnover_penalty(w, r)
        assert loss.item() == 0.0

    def test_backward(self) -> None:
        w, r = _weights_and_returns()
        loss = turnover_penalty(w, r)
        loss.backward()
        assert w.grad is not None


class TestSharpeRelativeLoss:
    def test_call_produces_scalar(self) -> None:
        w, r = _weights_and_returns()
        loss = sharpe_of_relative_loss(w, r)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_backward(self) -> None:
        w, r = _weights_and_returns()
        loss = sharpe_of_relative_loss(w, r)
        loss.backward()
        assert w.grad is not None


class TestDriftIntegralLoss:
    def test_drift_integral_loss_function(self) -> None:
        rng = np.random.default_rng(42)
        w_pred = torch.tensor(rng.dirichlet(np.ones(N), size=T), dtype=torch.float64)
        mw = torch.tensor(rng.dirichlet(np.ones(N), size=T), dtype=torch.float64)
        L = rng.standard_normal((N, N))
        cov_single = L @ L.T + np.eye(N) * 0.01
        covs = torch.tensor(np.tile(cov_single, (T, 1, 1)), dtype=torch.float64)

        loss = drift_integral_loss(w_pred, mw, covs, dt=1.0 / 252.0)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_drift_integral_loss_with_numpy_inputs(self) -> None:
        """Test conversion from numpy arrays."""
        rng = np.random.default_rng(42)
        w_pred = rng.dirichlet(np.ones(N), size=T)
        mw = rng.dirichlet(np.ones(N), size=T)
        L = rng.standard_normal((N, N))
        cov_single = L @ L.T + np.eye(N) * 0.01
        covs = np.tile(cov_single, (T, 1, 1))

        loss = drift_integral_loss(w_pred, mw, covs, dt=1.0 / 252.0)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_drift_integral_loss_class(self) -> None:
        rng = np.random.default_rng(42)
        w_pred = torch.tensor(rng.dirichlet(np.ones(N), size=T), dtype=torch.float64)
        r = torch.tensor(1.0 + rng.standard_normal((T, N)) * 0.01, dtype=torch.float64)
        mw = rng.dirichlet(np.ones(N), size=T)
        L = rng.standard_normal((N, N))
        cov_single = L @ L.T + np.eye(N) * 0.01
        covs = np.tile(cov_single, (T, 1, 1))

        loss_fn = DriftIntegralLoss(dt=1.0 / 252.0)
        loss = loss_fn(w_pred, r, market_weights=mw, covariance_matrices=covs)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_drift_integral_loss_class_missing_args(self) -> None:
        w, r = _weights_and_returns()
        loss_fn = DriftIntegralLoss()
        with pytest.raises(ValueError, match="market_weights"):
            loss_fn(w, r)

    def test_drift_integral_loss_class_with_tensors(self) -> None:
        """DriftIntegralLoss with market_weights and covs already as tensors."""
        rng = np.random.default_rng(42)
        w_pred = torch.tensor(
            rng.dirichlet(np.ones(N), size=T),
            dtype=torch.float64,
            requires_grad=True,
        )
        r = torch.tensor(1.0 + rng.standard_normal((T, N)) * 0.01, dtype=torch.float64)
        mw = torch.tensor(rng.dirichlet(np.ones(N), size=T), dtype=torch.float64)
        L = rng.standard_normal((N, N))
        cov_single = L @ L.T + np.eye(N) * 0.01
        covs = torch.tensor(np.tile(cov_single, (T, 1, 1)), dtype=torch.float64)

        loss_fn = DriftIntegralLoss()
        loss = loss_fn(w_pred, r, market_weights=mw, covariance_matrices=covs)
        assert torch.isfinite(loss)
        loss.backward()
        assert w_pred.grad is not None


class TestLossComposition:
    """Exercise _CompositeLoss and _BaseLoss arithmetic operators."""

    def test_add_two_base_losses(self) -> None:
        combined = relative_return_loss + weight_regularization
        assert isinstance(combined, _CompositeLoss)
        w, r = _weights_and_returns()
        loss = combined(w, r)
        assert torch.isfinite(loss)

    def test_radd_base_loss(self) -> None:
        combined = turnover_penalty + relative_return_loss
        assert isinstance(combined, _CompositeLoss)
        w, r = _weights_and_returns()
        loss = combined(w, r)
        assert torch.isfinite(loss)

    def test_mul_scalar(self) -> None:
        scaled = 0.5 * turnover_penalty
        assert isinstance(scaled, _CompositeLoss)
        w, r = _weights_and_returns()
        loss = scaled(w, r)
        assert torch.isfinite(loss)

    def test_rmul_scalar(self) -> None:
        scaled = weight_regularization * 0.01
        assert isinstance(scaled, _CompositeLoss)

    def test_composite_add_base(self) -> None:
        c1 = relative_return_loss + weight_regularization
        c2 = c1 + turnover_penalty
        assert isinstance(c2, _CompositeLoss)
        w, r = _weights_and_returns()
        loss = c2(w, r)
        assert torch.isfinite(loss)

    def test_base_add_composite(self) -> None:
        c1 = relative_return_loss + weight_regularization
        c2 = turnover_penalty + c1
        assert isinstance(c2, _CompositeLoss)

    def test_composite_add_composite(self) -> None:
        c1 = relative_return_loss + weight_regularization
        c2 = turnover_penalty + sharpe_of_relative_loss
        c3 = c1 + c2
        assert isinstance(c3, _CompositeLoss)

    def test_composite_radd_composite(self) -> None:
        c1 = 0.5 * relative_return_loss
        c2 = 0.1 * turnover_penalty
        c3 = _CompositeLoss.__radd__(c2, c1)
        assert isinstance(c3, _CompositeLoss)

    def test_composite_radd_non_composite(self) -> None:
        c1 = 0.5 * relative_return_loss
        c2 = _CompositeLoss.__radd__(c1, turnover_penalty)
        assert isinstance(c2, _CompositeLoss)

    def test_composite_mul(self) -> None:
        c1 = relative_return_loss + weight_regularization
        c2 = c1 * 0.5
        assert isinstance(c2, _CompositeLoss)

    def test_composite_rmul(self) -> None:
        c1 = relative_return_loss + weight_regularization
        c2 = 0.5 * c1
        assert isinstance(c2, _CompositeLoss)

    def test_default_loss(self) -> None:
        loss_fn = default_loss(weight_decay=1e-4)
        assert isinstance(loss_fn, _CompositeLoss)
        w, r = _weights_and_returns()
        loss = loss_fn(w, r)
        assert torch.isfinite(loss)

    def test_composite_call_backward(self) -> None:
        combined = relative_return_loss + 0.01 * weight_regularization
        w, r = _weights_and_returns()
        loss = combined(w, r)
        loss.backward()
        assert w.grad is not None

    def test_base_radd_with_composite_other(self) -> None:
        """Directly call _BaseLoss.__radd__ with a _CompositeLoss other."""
        composite = 0.5 * turnover_penalty
        result = relative_return_loss.__radd__(composite)
        assert isinstance(result, _CompositeLoss)

    def test_base_radd_with_non_composite_other(self) -> None:
        """Directly call _BaseLoss.__radd__ with a plain callable other."""

        def dummy_fn(w, r, **kw):
            return w.sum()

        result = relative_return_loss.__radd__(dummy_fn)
        assert isinstance(result, _CompositeLoss)
