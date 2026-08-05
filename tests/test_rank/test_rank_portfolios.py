"""Tests for rank/rank_portfolios.py — rank-based portfolio construction.

Validates:
- Top-m portfolio weights are correct (1/m for top m, 0 otherwise)
- Bottom-m portfolio weights are correct
- Rank-weighted portfolio normalises properly
- Leaking portfolio applies diversity weighting to top m
- All portfolios sum to 1
"""

from __future__ import annotations

import numpy as np
import pytest

from quantspt.rank.rank_portfolios import (
    bottom_m_portfolio,
    leaking_portfolio,
    rank_weighted_portfolio,
    top_m_portfolio,
)


class TestTopMPortfolio:
    """Equal weight on the m largest stocks."""

    def test_weights_correct(self) -> None:
        mu = np.array([0.1, 0.4, 0.2, 0.3])
        pi = top_m_portfolio(mu, 2)
        assert pi[1] == pytest.approx(0.5)
        assert pi[3] == pytest.approx(0.5)
        assert pi[0] == 0.0
        assert pi[2] == 0.0

    def test_sums_to_one(self) -> None:
        rng = np.random.default_rng(42)
        mu = rng.dirichlet(np.ones(10))
        pi = top_m_portfolio(mu, 5)
        np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-14)

    def test_m_equals_n(self) -> None:
        mu = np.array([0.3, 0.2, 0.5])
        pi = top_m_portfolio(mu, 3)
        np.testing.assert_allclose(pi, 1.0 / 3, atol=1e-14)

    def test_m_equals_one(self) -> None:
        mu = np.array([0.1, 0.6, 0.3])
        pi = top_m_portfolio(mu, 1)
        assert pi[1] == 1.0
        assert pi[0] == 0.0
        assert pi[2] == 0.0

    def test_exactly_m_nonzero(self) -> None:
        rng = np.random.default_rng(42)
        mu = rng.dirichlet(np.ones(10))
        m = 4
        pi = top_m_portfolio(mu, m)
        assert np.count_nonzero(pi) == m

    def test_rejects_invalid_m(self) -> None:
        from quantspt.errors import SPTInvariantError

        mu = np.array([0.5, 0.5])
        with pytest.raises(SPTInvariantError):
            top_m_portfolio(mu, 0)
        with pytest.raises(SPTInvariantError):
            top_m_portfolio(mu, 3)


class TestBottomMPortfolio:
    """Equal weight on the m smallest stocks."""

    def test_weights_correct(self) -> None:
        mu = np.array([0.1, 0.4, 0.2, 0.3])
        pi = bottom_m_portfolio(mu, 2)
        assert pi[0] == pytest.approx(0.5)
        assert pi[2] == pytest.approx(0.5)
        assert pi[1] == 0.0
        assert pi[3] == 0.0

    def test_sums_to_one(self) -> None:
        rng = np.random.default_rng(42)
        mu = rng.dirichlet(np.ones(8))
        pi = bottom_m_portfolio(mu, 3)
        np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-14)

    def test_exactly_m_nonzero(self) -> None:
        rng = np.random.default_rng(42)
        mu = rng.dirichlet(np.ones(10))
        pi = bottom_m_portfolio(mu, 3)
        assert np.count_nonzero(pi) == 3


class TestRankWeightedPortfolio:
    """General rank-weighted portfolio construction."""

    def test_uniform_weight_func(self) -> None:
        mu = np.array([0.5, 0.3, 0.2])
        pi = rank_weighted_portfolio(mu, lambda k, n: 1.0)
        np.testing.assert_allclose(pi, 1.0 / 3, atol=1e-14)

    def test_top_heavy_weight_func(self) -> None:
        mu = np.array([0.5, 0.3, 0.2])
        pi = rank_weighted_portfolio(mu, lambda k, n: float(n - k))
        np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-14)
        order = np.argsort(-mu)
        assert pi[order[0]] > pi[order[1]] > pi[order[2]]

    def test_sums_to_one(self) -> None:
        rng = np.random.default_rng(42)
        mu = rng.dirichlet(np.ones(5))
        pi = rank_weighted_portfolio(mu, lambda k, n: 1.0 / (k + 1))
        np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-14)

    def test_nonnegative(self) -> None:
        rng = np.random.default_rng(42)
        mu = rng.dirichlet(np.ones(5))
        pi = rank_weighted_portfolio(mu, lambda k, n: 1.0 / (k + 1))
        assert np.all(pi >= 0)


class TestLeakingPortfolio:
    """Diversity-weighted top-m portfolio."""

    def test_sums_to_one(self) -> None:
        rng = np.random.default_rng(42)
        mu = rng.dirichlet(np.ones(10))
        pi = leaking_portfolio(mu, 5, 0.5)
        np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-14)

    def test_exactly_m_nonzero(self) -> None:
        rng = np.random.default_rng(42)
        mu = rng.dirichlet(np.ones(10))
        pi = leaking_portfolio(mu, 4, 0.7)
        assert np.count_nonzero(pi) == 4

    def test_p_near_zero_approaches_equal_weight(self) -> None:
        mu = np.array([0.5, 0.3, 0.15, 0.05])
        pi = leaking_portfolio(mu, 3, 0.01)
        nonzero = pi[pi > 0]
        np.testing.assert_allclose(nonzero, 1.0 / 3, atol=0.05)

    def test_p_near_one_approaches_market_cap(self) -> None:
        mu = np.array([0.5, 0.3, 0.15, 0.05])
        pi = leaking_portfolio(mu, 3, 0.999)
        top3 = np.sort(mu)[::-1][:3]
        expected = top3**0.999
        expected /= expected.sum()
        top_indices = np.argsort(-mu)[:3]
        np.testing.assert_allclose(pi[top_indices], expected, atol=0.01)

    def test_positive_where_nonzero(self) -> None:
        rng = np.random.default_rng(42)
        mu = rng.dirichlet(np.ones(8))
        pi = leaking_portfolio(mu, 5, 0.5)
        assert np.all(pi[pi > 0] > 0)
