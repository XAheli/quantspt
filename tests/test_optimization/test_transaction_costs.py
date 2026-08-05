"""Tests for optimization/transaction_costs -- cost models.

Validates proportional costs, market impact, and net growth rate
computation against known analytical values.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt.errors import SPTInvariantError
from quantspt.optimization.transaction_costs import (
    net_growth_rate,
    proportional_cost,
    sqrt_market_impact,
)


class TestProportionalCost:
    """Tests for proportional_cost()."""

    def test_zero_turnover_zero_cost(self) -> None:
        w = np.array([0.5, 0.3, 0.2])
        assert proportional_cost(w, w) == 0.0

    def test_known_cost(self) -> None:
        """10 bps on turnover of 0.4 = 0.4 * 10/10000 = 0.0004."""
        old = np.array([0.5, 0.3, 0.2])
        new = np.array([0.3, 0.5, 0.2])
        turnover = 0.4
        expected = turnover * 10 / 10000
        assert_allclose(proportional_cost(old, new), expected, atol=1e-14)

    def test_full_rebalance_cost(self) -> None:
        """Complete portfolio turnover."""
        old = np.array([1.0, 0.0])
        new = np.array([0.0, 1.0])
        cost = proportional_cost(old, new, cost_bps=50.0)
        assert_allclose(cost, 2.0 * 50 / 10000, atol=1e-14)

    def test_custom_cost_bps(self) -> None:
        old = np.array([0.5, 0.5])
        new = np.array([0.6, 0.4])
        cost_10 = proportional_cost(old, new, cost_bps=10.0)
        cost_20 = proportional_cost(old, new, cost_bps=20.0)
        assert_allclose(cost_20, cost_10 * 2, atol=1e-14)

    def test_length_mismatch(self) -> None:
        with pytest.raises(SPTInvariantError, match="match"):
            proportional_cost(np.array([0.5, 0.5]), np.array([1.0]))

    def test_non_negative(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(10):
            old = rng.dirichlet(np.ones(5))
            new = rng.dirichlet(np.ones(5))
            assert proportional_cost(old, new) >= 0


class TestSqrtMarketImpact:
    """Tests for sqrt_market_impact()."""

    def test_zero_turnover_zero_impact(self) -> None:
        w = np.array([0.5, 0.5])
        vol = np.array([1e6, 1e6])
        assert sqrt_market_impact(w, w, vol, 1e6) == 0.0

    def test_positive_impact(self) -> None:
        old = np.array([0.5, 0.5])
        new = np.array([0.7, 0.3])
        vol = np.array([1e6, 1e6])
        cost = sqrt_market_impact(old, new, vol, 1e6)
        assert cost > 0

    def test_larger_trade_higher_impact(self) -> None:
        """Larger trades should have higher impact."""
        old = np.array([0.5, 0.5])
        small = np.array([0.55, 0.45])
        large = np.array([0.8, 0.2])
        vol = np.array([1e6, 1e6])
        cost_small = sqrt_market_impact(old, small, vol, 1e6)
        cost_large = sqrt_market_impact(old, large, vol, 1e6)
        assert cost_large > cost_small

    def test_higher_volume_lower_impact(self) -> None:
        """More liquid stocks should have lower impact."""
        old = np.array([0.5, 0.5])
        new = np.array([0.7, 0.3])
        cost_illiquid = sqrt_market_impact(old, new, np.array([1e4, 1e4]), 1e6)
        cost_liquid = sqrt_market_impact(old, new, np.array([1e8, 1e8]), 1e6)
        assert cost_liquid < cost_illiquid

    def test_length_mismatch(self) -> None:
        with pytest.raises(SPTInvariantError):
            sqrt_market_impact(
                np.array([0.5, 0.5]),
                np.array([0.7, 0.3]),
                np.array([1e6]),
                1e6,
            )


class TestNetGrowthRate:
    """Tests for net_growth_rate()."""

    def test_no_rebalance_no_cost(self) -> None:
        """No change in weights means no cost deducted."""
        w = np.array([0.5, 0.5])
        net = net_growth_rate(0.10, w, w)
        assert_allclose(net, 0.10, atol=1e-14)

    def test_cost_reduces_growth(self) -> None:
        """Net growth rate must be less than gross after costs."""
        old = np.array([0.5, 0.5])
        new = np.array([0.7, 0.3])
        net = net_growth_rate(0.10, old, new, cost_bps=10.0)
        assert net < 0.10

    def test_frequent_rebalance_higher_cost(self) -> None:
        """More frequent rebalancing = higher annualised cost."""
        old = np.array([0.5, 0.5])
        new = np.array([0.6, 0.4])
        net_monthly = net_growth_rate(0.10, old, new, rebalance_frequency=21)
        net_daily = net_growth_rate(0.10, old, new, rebalance_frequency=1)
        assert net_daily < net_monthly

    def test_known_net_value(self) -> None:
        """Verify against manual calculation."""
        old = np.array([0.5, 0.5])
        new = np.array([0.6, 0.4])
        cost_bps = 10.0
        rebal_freq = 21

        turnover = 0.2
        per_trade = turnover * cost_bps / 10000
        annual_cost = per_trade * (252 / rebal_freq)
        expected_net = 0.08 - annual_cost

        net = net_growth_rate(
            0.08, old, new, cost_bps=cost_bps, rebalance_frequency=rebal_freq
        )
        assert_allclose(net, expected_net, atol=1e-10)
