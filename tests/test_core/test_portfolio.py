"""Tests for core/portfolio.py — portfolio value process and turnover.

Tests verify mathematical properties from F&K Survey §1, §3.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt.core.portfolio import (
    cumulative_turnover,
    drift_of_relative_return,
    holding_drift,
    log_relative_return,
    portfolio_log_return,
    portfolio_value_weights,
    rebalancing_turnover,
    relative_return,
)
from quantspt.errors import SPTInvariantError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def equal_2():
    return np.array([0.5, 0.5])


@pytest.fixture()
def skewed_3():
    return np.array([0.5, 0.3, 0.2])


# ---------------------------------------------------------------------------
# Tests: portfolio_log_return
# ---------------------------------------------------------------------------


class TestPortfolioLogReturn:
    def test_single_stock(self) -> None:
        """100 % in one stock: portfolio return = that stock's return."""
        pi = np.array([1.0, 0.0])
        r = np.array([0.05, -0.02])
        assert_allclose(portfolio_log_return(pi, r), 0.05)

    def test_equal_weighted(self) -> None:
        """Equal-weighted: portfolio return = mean of stock returns."""
        pi = np.array([0.5, 0.5])
        r = np.array([0.04, 0.06])
        assert_allclose(portfolio_log_return(pi, r), 0.05)


# ---------------------------------------------------------------------------
# Tests: portfolio_value_weights (buy-and-hold drift)
# ---------------------------------------------------------------------------


class TestPortfolioValueWeights:
    def test_sum_to_one(self, equal_2: np.ndarray) -> None:
        """Drifted weights must sum to 1."""
        returns = np.array([0.10, -0.05])
        drifted = portfolio_value_weights(equal_2, returns)
        assert_allclose(np.sum(drifted), 1.0, atol=1e-14)

    def test_winner_gets_larger_weight(self) -> None:
        """Stock with higher return gains weight."""
        pi = np.array([0.5, 0.5])
        returns = np.array([0.20, -0.05])
        drifted = portfolio_value_weights(pi, returns)
        assert drifted[0] > drifted[1]

    def test_zero_return_no_drift(self, equal_2: np.ndarray) -> None:
        """Zero returns → weights unchanged."""
        returns = np.zeros(2)
        drifted = portfolio_value_weights(equal_2, returns)
        assert_allclose(drifted, equal_2)

    def test_negative_total_raises(self) -> None:
        """Total portfolio value becoming non-positive is an error."""
        pi = np.array([0.5, 0.5])
        returns = np.array([-2.0, -2.0])
        with pytest.raises(SPTInvariantError, match="non-positive"):
            portfolio_value_weights(pi, returns)


# ---------------------------------------------------------------------------
# Tests: relative returns (F&K Survey §3)
# ---------------------------------------------------------------------------


class TestRelativeReturn:
    def test_starts_at_ratio(self) -> None:
        V_pi = np.array([100.0, 110.0, 105.0])
        V_mu = np.array([100.0, 108.0, 112.0])
        rel = relative_return(V_pi, V_mu)
        assert_allclose(rel[0], 1.0)

    def test_outperformance(self) -> None:
        """If π ends above μ, relative return > 1 at terminal."""
        V_pi = np.array([100.0, 120.0])
        V_mu = np.array([100.0, 110.0])
        rel = relative_return(V_pi, V_mu)
        assert rel[-1] > 1.0

    def test_log_relative_return(self) -> None:
        V_pi = np.array([100.0, 120.0])
        V_mu = np.array([100.0, 110.0])
        expected = np.log(120.0 / 110.0)
        assert_allclose(log_relative_return(V_pi, V_mu), expected)


# ---------------------------------------------------------------------------
# Tests: drift of relative return (F&K Survey Eq. 3.4)
# ---------------------------------------------------------------------------


class TestDriftOfRelativeReturn:
    def test_market_vs_market_is_zero(self) -> None:
        """Relative drift of market vs itself = 0."""
        mu = np.array([0.5, 0.3, 0.2])
        gamma = np.array([0.03, 0.05, 0.02])
        a = np.diag([0.04, 0.09, 0.0625])
        result = drift_of_relative_return(mu, mu, gamma, a)
        assert_allclose(result, 0.0, atol=1e-14)

    def test_more_diversified_has_positive_drift(self) -> None:
        """Equal-weighted vs concentrated: diversification gives positive drift."""
        n = 5
        a = np.eye(n) * 0.04
        gamma = np.zeros(n)
        mu = np.array([0.8, 0.05, 0.05, 0.05, 0.05])
        pi_equal = np.ones(n) / n
        drift_val = drift_of_relative_return(pi_equal, mu, gamma, a)
        assert drift_val > 0


# ---------------------------------------------------------------------------
# Tests: holding drift
# ---------------------------------------------------------------------------


class TestHoldingDrift:
    def test_sums_to_zero(self) -> None:
        """Holding drift on the simplex sums to zero."""
        pi = np.array([0.5, 0.3, 0.2])
        mu = pi.copy()
        a = np.diag([0.04, 0.09, 0.0625])
        drift = holding_drift(pi, mu, a)
        assert_allclose(np.sum(drift), 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# Tests: rebalancing turnover (F&K Survey §1.3)
# ---------------------------------------------------------------------------


class TestRebalancingTurnover:
    def test_zero_when_equal(self) -> None:
        pi = np.array([0.5, 0.5])
        assert_allclose(rebalancing_turnover(pi, pi), 0.0)

    def test_full_turnover(self) -> None:
        """Swapping 100 % from stock 1 to stock 2 = 100 % turnover."""
        pi_from = np.array([1.0, 0.0])
        pi_to = np.array([0.0, 1.0])
        assert_allclose(rebalancing_turnover(pi_to, pi_from), 1.0)

    def test_symmetry(self) -> None:
        """Turnover is symmetric."""
        a = np.array([0.6, 0.4])
        b = np.array([0.4, 0.6])
        assert_allclose(
            rebalancing_turnover(a, b),
            rebalancing_turnover(b, a),
        )

    def test_bounded_01(self) -> None:
        """Turnover is in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            n = rng.integers(2, 10)
            a_wt = rng.dirichlet(np.ones(n))
            b_wt = rng.dirichlet(np.ones(n))
            t = rebalancing_turnover(a_wt, b_wt)
            assert 0.0 <= t <= 1.0 + 1e-14


# ---------------------------------------------------------------------------
# Tests: cumulative turnover
# ---------------------------------------------------------------------------


class TestCumulativeTurnover:
    def test_constant_weights_zero_turnover(self) -> None:
        """Constant target with zero returns → zero turnover."""
        pi = np.array([0.5, 0.5])
        weight_path = np.tile(pi, (5, 1))
        returns = np.zeros((4, 2))
        assert_allclose(cumulative_turnover(weight_path, returns), 0.0)

    def test_no_returns_direct_comparison(self) -> None:
        """Without returns, just compares consecutive rows."""
        weight_path = np.array(
            [
                [0.6, 0.4],
                [0.5, 0.5],
                [0.4, 0.6],
            ]
        )
        expected = rebalancing_turnover(
            weight_path[1], weight_path[0]
        ) + rebalancing_turnover(weight_path[2], weight_path[1])
        assert_allclose(cumulative_turnover(weight_path), expected)
