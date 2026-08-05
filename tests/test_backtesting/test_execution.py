"""Tests for trade execution models.

Validates that transaction cost models correctly compute costs and
that market impact increases with trade size.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt.backtesting.execution import (
    ExecutionModel,
    InstantExecution,
    MarketImpactExecution,
    ProportionalCostExecution,
)
from quantspt.errors import SPTInvariantError

# =========================================================================
# Protocol conformance
# =========================================================================


class TestExecutionProtocol:
    """All execution models satisfy the ExecutionModel protocol."""

    def test_instant_is_execution_model(self) -> None:
        assert isinstance(InstantExecution(), ExecutionModel)

    def test_proportional_is_execution_model(self) -> None:
        assert isinstance(ProportionalCostExecution(10.0), ExecutionModel)

    def test_market_impact_is_execution_model(self) -> None:
        model = MarketImpactExecution(
            eta=0.1,
            volatilities=np.array([0.2, 0.3]),
            adv=np.array([1e6, 2e6]),
        )
        assert isinstance(model, ExecutionModel)


# =========================================================================
# InstantExecution
# =========================================================================


class TestInstantExecution:
    """Tests for zero-cost instant execution."""

    def test_zero_cost(self) -> None:
        model = InstantExecution()
        result = model.execute(
            np.array([0.5, 0.5]),
            np.array([0.6, 0.4]),
            1000.0,
        )
        assert result.cost == 0.0

    def test_weights_equal_target(self) -> None:
        model = InstantExecution()
        target = np.array([0.3, 0.3, 0.4])
        result = model.execute(np.array([0.5, 0.3, 0.2]), target, 1000.0)
        assert_allclose(result.weights, target)

    def test_no_trade_no_cost(self) -> None:
        model = InstantExecution()
        w = np.array([0.5, 0.5])
        result = model.execute(w, w, 1000.0)
        assert result.cost == 0.0
        assert_allclose(result.weights, w)


# =========================================================================
# ProportionalCostExecution
# =========================================================================


class TestProportionalCostExecution:
    """Tests for proportional transaction cost model."""

    def test_cost_formula(self) -> None:
        """Cost = (bps/10000) * Σ|Δw_i|."""
        model = ProportionalCostExecution(cost_bps=10.0)
        current = np.array([0.5, 0.5])
        target = np.array([0.6, 0.4])
        result = model.execute(current, target, 1000.0)

        turnover = np.sum(np.abs(target - current))  # 0.1 + 0.1 = 0.2
        expected_cost = 10.0 / 10_000.0 * turnover  # 0.0001 * 0.2 = 0.0002
        assert_allclose(result.cost, expected_cost, atol=1e-14)

    def test_zero_trade_zero_cost(self) -> None:
        model = ProportionalCostExecution(cost_bps=50.0)
        w = np.array([0.5, 0.5])
        result = model.execute(w, w, 1000.0)
        assert result.cost == 0.0

    def test_cost_proportional_to_turnover(self) -> None:
        """Doubling turnover doubles cost."""
        model = ProportionalCostExecution(cost_bps=10.0)

        result1 = model.execute(
            np.array([0.5, 0.5]),
            np.array([0.55, 0.45]),
            1000.0,
        )
        result2 = model.execute(
            np.array([0.5, 0.5]),
            np.array([0.60, 0.40]),
            1000.0,
        )
        assert_allclose(result2.cost / result1.cost, 2.0, atol=1e-12)

    def test_cost_proportional_to_bps(self) -> None:
        """Doubling bps doubles cost for same trade."""
        current = np.array([0.5, 0.5])
        target = np.array([0.6, 0.4])

        model10 = ProportionalCostExecution(cost_bps=10.0)
        model20 = ProportionalCostExecution(cost_bps=20.0)

        cost10 = model10.execute(current, target, 1000.0).cost
        cost20 = model20.execute(current, target, 1000.0).cost
        assert_allclose(cost20 / cost10, 2.0, atol=1e-12)

    def test_weights_match_target(self) -> None:
        """Realized weights always match target (cost is separate)."""
        model = ProportionalCostExecution(cost_bps=100.0)
        target = np.array([0.3, 0.3, 0.4])
        result = model.execute(np.array([0.5, 0.3, 0.2]), target, 1000.0)
        assert_allclose(result.weights, target)

    def test_negative_bps_raises(self) -> None:
        with pytest.raises(SPTInvariantError, match="non-negative"):
            ProportionalCostExecution(cost_bps=-5.0)

    def test_known_value_3_assets(self) -> None:
        """Verify against hand-computed cost for 3-asset trade."""
        model = ProportionalCostExecution(cost_bps=20.0)
        current = np.array([0.40, 0.35, 0.25])
        target = np.array([0.33, 0.33, 0.34])

        delta = np.abs(target - current)
        turnover = np.sum(delta)  # 0.07 + 0.02 + 0.09 = 0.18
        expected = 20.0 / 10_000.0 * turnover

        result = model.execute(current, target, 5000.0)
        assert_allclose(result.cost, expected, atol=1e-14)


# =========================================================================
# MarketImpactExecution
# =========================================================================


class TestMarketImpactExecution:
    """Tests for square-root market impact model."""

    def test_impact_increases_with_trade_size(self) -> None:
        """Larger trades should have higher impact costs."""
        model = MarketImpactExecution(
            eta=0.1,
            volatilities=np.array([0.2, 0.3]),
            adv=np.array([1e6, 1e6]),
        )
        current = np.array([0.5, 0.5])

        small_trade = model.execute(current, np.array([0.51, 0.49]), 1e6)
        large_trade = model.execute(current, np.array([0.60, 0.40]), 1e6)
        assert large_trade.cost > small_trade.cost

    def test_zero_trade_zero_impact(self) -> None:
        model = MarketImpactExecution(
            eta=0.1,
            volatilities=np.array([0.2]),
            adv=np.array([1e6]),
        )
        w = np.array([1.0])
        result = model.execute(w, w, 1e6)
        assert_allclose(result.cost, 0.0, atol=1e-14)

    def test_impact_increases_with_eta(self) -> None:
        """Higher eta → more impact."""
        vols = np.array([0.2, 0.3])
        adv = np.array([1e6, 1e6])
        current = np.array([0.5, 0.5])
        target = np.array([0.6, 0.4])

        model_low = MarketImpactExecution(eta=0.05, volatilities=vols, adv=adv)
        model_high = MarketImpactExecution(eta=0.20, volatilities=vols, adv=adv)

        cost_low = model_low.execute(current, target, 1e6).cost
        cost_high = model_high.execute(current, target, 1e6).cost
        assert cost_high > cost_low

    def test_sublinear_in_trade_size(self) -> None:
        """Square-root model: doubling trade size less than doubles cost."""
        model = MarketImpactExecution(
            eta=0.1,
            volatilities=np.array([0.2]),
            adv=np.array([1e6]),
        )
        small = model.execute(np.array([1.0]), np.array([0.95]), 1e6)
        large = model.execute(np.array([1.0]), np.array([0.90]), 1e6)
        ratio = large.cost / small.cost
        assert ratio < 4.0  # would be 4.0 for linear, less for sqrt

    def test_invalid_eta_raises(self) -> None:
        with pytest.raises(SPTInvariantError, match="eta must be positive"):
            MarketImpactExecution(
                eta=-0.1,
                volatilities=np.array([0.2]),
                adv=np.array([1e6]),
            )

    def test_volatility_adv_length_mismatch(self) -> None:
        with pytest.raises(SPTInvariantError, match="same length"):
            MarketImpactExecution(
                eta=0.1,
                volatilities=np.array([0.2, 0.3]),
                adv=np.array([1e6]),
            )
