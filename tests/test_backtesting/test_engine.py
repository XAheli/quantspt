"""Tests for the backtesting engine.

Validates the event-driven loop, portfolio value tracking, turnover
computation, and integration with rebalancing/execution models.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt.backtesting.engine import BacktestConfig, BacktestEngine, BacktestResult
from quantspt.backtesting.execution import (
    InstantExecution,
    ProportionalCostExecution,
)
from quantspt.backtesting.rebalancing import CalendarRebalancer, Frequency
from quantspt.core.generating_functions import DiversityGenerator
from quantspt.errors import SPTInvariantError

# =========================================================================
# Helpers
# =========================================================================


def _make_constant_returns(n_steps: int, n_assets: int) -> np.ndarray:
    """Returns where all assets return exactly 1 (no change)."""
    return np.ones((n_steps, n_assets))


def _make_simple_returns(n_steps: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate simple 3-asset returns and initial weights."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, 0.01, size=(n_steps, 3))
    returns = 1.0 + raw
    initial_weights = np.array([0.4, 0.35, 0.25])
    return returns, initial_weights


def _identity_weight_func(mu: np.ndarray) -> np.ndarray:
    """Identity weight function: returns input weights unchanged."""
    return mu.copy()


# =========================================================================
# BacktestResult structure
# =========================================================================


class TestBacktestResult:
    """Tests for BacktestResult data structure."""

    def test_n_steps(self) -> None:
        result = BacktestResult(
            portfolio_values=np.ones(11),
            market_values=np.ones(11),
            weights_history=np.ones((11, 3)),
            market_weights_history=np.ones((11, 3)),
            turnover=np.zeros(11),
            costs=np.zeros(11),
            rebalance_steps=[0],
            n_rebalances=1,
            config=BacktestConfig(),
        )
        assert result.n_steps == 10

    def test_log_relative_return(self) -> None:
        pv = np.array([1.0, 1.1])
        mv = np.array([1.0, 1.05])
        result = BacktestResult(
            portfolio_values=pv,
            market_values=mv,
            weights_history=np.ones((2, 2)),
            market_weights_history=np.ones((2, 2)),
            turnover=np.zeros(2),
            costs=np.zeros(2),
            rebalance_steps=[0],
            n_rebalances=1,
            config=BacktestConfig(),
        )
        expected = np.log(1.1 / 1.05)
        assert_allclose(result.log_relative_return(), expected, atol=1e-14)


# =========================================================================
# Engine with constant returns
# =========================================================================


class TestEngineConstantReturns:
    """When returns are identically 1, portfolio value should not change."""

    def test_value_unchanged(self) -> None:
        n_steps, n = 50, 3
        returns = _make_constant_returns(n_steps, n)
        initial_w = np.array([0.4, 0.35, 0.25])
        G = DiversityGenerator(0.5)

        engine = BacktestEngine(
            weight_func=G.weights,
            returns=returns,
            initial_weights=initial_w,
            rebalancer=CalendarRebalancer(Frequency.DAILY),
            execution=InstantExecution(),
        )
        result = engine.run()
        bt = result.data

        assert_allclose(bt.portfolio_values[-1], 1.0, atol=1e-10)
        assert_allclose(bt.market_values[-1], 1.0, atol=1e-10)

    def test_all_values_finite(self) -> None:
        n_steps, n = 100, 3
        returns = _make_constant_returns(n_steps, n)
        initial_w = np.array([0.4, 0.35, 0.25])
        G = DiversityGenerator(0.5)

        engine = BacktestEngine(
            weight_func=G.weights,
            returns=returns,
            initial_weights=initial_w,
        )
        result = engine.run()
        bt = result.data
        assert np.all(np.isfinite(bt.portfolio_values))
        assert np.all(np.isfinite(bt.weights_history))


# =========================================================================
# Engine portfolio value computation
# =========================================================================


class TestEngineValueComputation:
    """Verify portfolio value is computed correctly step-by-step."""

    def test_single_step_value(self) -> None:
        """One-step backtest: V(1) = V(0) * Σ π_i R_i."""
        returns = np.array([[1.05, 0.98]])
        initial_w = np.array([0.5, 0.5])

        engine = BacktestEngine(
            weight_func=_identity_weight_func,
            returns=returns,
            initial_weights=initial_w,
            rebalancer=CalendarRebalancer(Frequency.DAILY),
            execution=InstantExecution(),
        )
        result = engine.run()
        bt = result.data

        expected_v1 = 1.0 * (0.5 * 1.05 + 0.5 * 0.98)
        assert_allclose(bt.portfolio_values[1], expected_v1, atol=1e-12)

    def test_market_value_tracks_buy_and_hold(self) -> None:
        """Market portfolio is buy-and-hold (weights drift with prices)."""
        returns = np.array([[1.1, 0.9], [1.05, 1.05]])
        initial_w = np.array([0.5, 0.5])

        engine = BacktestEngine(
            weight_func=_identity_weight_func,
            returns=returns,
            initial_weights=initial_w,
            rebalancer=CalendarRebalancer(Frequency.DAILY),
            execution=InstantExecution(),
        )
        result = engine.run()
        bt = result.data

        mkt_v1 = 1.0 * (0.5 * 1.1 + 0.5 * 0.9)
        assert_allclose(bt.market_values[1], mkt_v1, atol=1e-12)


# =========================================================================
# Turnover tracking
# =========================================================================


class TestEngineTurnover:
    """Verify turnover is computed correctly."""

    def test_zero_turnover_identity_strategy(self) -> None:
        """Identity strategy (π = μ) with daily rebalancing: ~zero turnover
        after initial rebalance."""
        n_steps = 10
        returns = _make_constant_returns(n_steps, 2)
        initial_w = np.array([0.5, 0.5])

        engine = BacktestEngine(
            weight_func=_identity_weight_func,
            returns=returns,
            initial_weights=initial_w,
            rebalancer=CalendarRebalancer(Frequency.DAILY),
            execution=InstantExecution(),
        )
        result = engine.run()
        bt = result.data
        assert bt.total_turnover() < 1e-10

    def test_nonzero_turnover_diversity_strategy(self) -> None:
        """Diversity strategy should have nonzero turnover on varying returns."""
        returns, initial_w = _make_simple_returns(100, seed=99)
        G = DiversityGenerator(0.5)

        engine = BacktestEngine(
            weight_func=G.weights,
            returns=returns,
            initial_weights=initial_w,
            rebalancer=CalendarRebalancer(Frequency.DAILY),
            execution=InstantExecution(),
        )
        result = engine.run()
        bt = result.data
        assert bt.total_turnover() > 0


# =========================================================================
# Transaction costs
# =========================================================================


class TestEngineTransactionCosts:
    """Verify transaction costs reduce portfolio value."""

    def test_proportional_costs_reduce_value(self) -> None:
        """With proportional costs, final value < zero-cost final value."""
        returns, initial_w = _make_simple_returns(100, seed=123)
        G = DiversityGenerator(0.5)

        engine_free = BacktestEngine(
            weight_func=G.weights,
            returns=returns,
            initial_weights=initial_w,
            rebalancer=CalendarRebalancer(Frequency.DAILY),
            execution=InstantExecution(),
        )
        engine_costly = BacktestEngine(
            weight_func=G.weights,
            returns=returns,
            initial_weights=initial_w,
            rebalancer=CalendarRebalancer(Frequency.DAILY),
            execution=ProportionalCostExecution(cost_bps=50.0),
        )

        free_val = engine_free.run().data.portfolio_values[-1]
        costly_val = engine_costly.run().data.portfolio_values[-1]
        assert costly_val < free_val

    def test_higher_costs_reduce_more(self) -> None:
        """Higher bps → lower terminal value."""
        returns, initial_w = _make_simple_returns(100, seed=456)
        G = DiversityGenerator(0.5)

        def run_with_bps(bps: float) -> float:
            engine = BacktestEngine(
                weight_func=G.weights,
                returns=returns,
                initial_weights=initial_w,
                rebalancer=CalendarRebalancer(Frequency.DAILY),
                execution=ProportionalCostExecution(cost_bps=bps),
            )
            return float(engine.run().data.portfolio_values[-1])

        v10 = run_with_bps(10.0)
        v50 = run_with_bps(50.0)
        assert v50 < v10


# =========================================================================
# Metadata and SPTResult envelope
# =========================================================================


class TestEngineMetadata:
    """Verify the SPTResult envelope is populated correctly."""

    def test_sptresult_fields(self) -> None:
        returns, initial_w = _make_simple_returns(50, seed=789)
        G = DiversityGenerator(0.5)

        engine = BacktestEngine(
            weight_func=G.weights,
            returns=returns,
            initial_weights=initial_w,
        )
        result = engine.run()

        assert result.computation_time_ms > 0
        assert result.metadata["engine"] == "BacktestEngine"
        assert result.metadata["n_steps"] == 50
        assert result.metadata["n_assets"] == 3

    def test_result_validates(self) -> None:
        returns, initial_w = _make_simple_returns(50)
        G = DiversityGenerator(0.5)
        engine = BacktestEngine(
            weight_func=G.weights,
            returns=returns,
            initial_weights=initial_w,
        )
        result = engine.run()
        assert result.validate()


# =========================================================================
# Input validation
# =========================================================================


class TestEngineValidation:
    """Verify input validation."""

    def test_returns_must_be_2d(self) -> None:
        with pytest.raises(SPTInvariantError, match="2-D"):
            BacktestEngine(
                weight_func=_identity_weight_func,
                returns=np.ones(10),
                initial_weights=np.array([0.5, 0.5]),
            )

    def test_weights_returns_shape_mismatch(self) -> None:
        with pytest.raises(SPTInvariantError, match="mismatch"):
            BacktestEngine(
                weight_func=_identity_weight_func,
                returns=np.ones((10, 3)),
                initial_weights=np.array([0.5, 0.5]),
            )

    def test_weights_must_sum_to_one(self) -> None:
        with pytest.raises(SPTInvariantError, match="sum to 1"):
            BacktestEngine(
                weight_func=_identity_weight_func,
                returns=np.ones((10, 2)),
                initial_weights=np.array([0.6, 0.6]),
            )

    def test_weights_must_be_positive(self) -> None:
        with pytest.raises(SPTInvariantError, match="positive"):
            BacktestEngine(
                weight_func=_identity_weight_func,
                returns=np.ones((10, 2)),
                initial_weights=np.array([-0.5, 1.5]),
            )
