"""Tests for universe/criteria.py — individual SPT selection criteria."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantspt.errors import SPTInvariantError
from quantspt.universe.criteria import (
    boundary_risk_score,
    gamma_star_contribution,
    idiosyncratic_volatility,
    liquidity_filter,
    pairwise_correlation_score,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


@pytest.fixture()
def sample_returns(rng):
    """252 days of 10-stock returns with realistic structure."""
    n, T = 10, 252
    tickers = [f"S{i}" for i in range(n)]
    market_factor = rng.normal(0, 0.01, T)
    betas = rng.uniform(0.5, 1.5, n)
    idio = rng.normal(0, 0.02, (T, n))
    raw = market_factor[:, None] * betas[None, :] + idio
    dates = pd.date_range("2023-01-01", periods=T, freq="B")
    return pd.DataFrame(raw, index=dates, columns=tickers)


@pytest.fixture()
def sample_market_return(sample_returns):
    return sample_returns.mean(axis=1)


# ---------------------------------------------------------------------------
# idiosyncratic_volatility
# ---------------------------------------------------------------------------


class TestIdiosyncraticVolatility:
    def test_returns_series_with_correct_index(
        self, sample_returns, sample_market_return
    ):
        result = idiosyncratic_volatility(sample_returns, sample_market_return)
        assert isinstance(result, pd.Series)
        assert set(result.index) == set(sample_returns.columns)

    def test_all_values_positive(self, sample_returns, sample_market_return):
        result = idiosyncratic_volatility(sample_returns, sample_market_return)
        assert (result.dropna() > 0).all()

    def test_idio_vol_less_than_total_vol(self, sample_returns, sample_market_return):
        idio = idiosyncratic_volatility(sample_returns, sample_market_return)
        total = sample_returns.std() * np.sqrt(252)
        for t in sample_returns.columns:
            assert idio[t] <= total[t] * 1.05  # allow small numerical slack

    def test_rejects_short_series(self, sample_market_return):
        short = pd.DataFrame(np.random.randn(10, 3), columns=["A", "B", "C"])
        mkt = sample_market_return.iloc[:10]
        with pytest.raises(SPTInvariantError, match="20"):
            idiosyncratic_volatility(short, mkt)

    def test_length_mismatch_raises(self, sample_returns, sample_market_return):
        with pytest.raises(SPTInvariantError, match="mismatch"):
            idiosyncratic_volatility(sample_returns, sample_market_return.iloc[:100])


# ---------------------------------------------------------------------------
# pairwise_correlation_score
# ---------------------------------------------------------------------------


class TestPairwiseCorrelationScore:
    def test_returns_series(self, sample_returns):
        result = pairwise_correlation_score(sample_returns)
        assert isinstance(result, pd.Series)
        assert len(result) == sample_returns.shape[1]

    def test_bounded_zero_one(self, sample_returns):
        result = pairwise_correlation_score(sample_returns)
        assert (result >= 0).all()
        assert (result <= 1).all()

    def test_independent_stocks_low_correlation(self, rng):
        independent = pd.DataFrame(
            rng.normal(0, 1, (500, 5)),
            columns=[f"S{i}" for i in range(5)],
        )
        result = pairwise_correlation_score(independent)
        assert result.max() < 0.15

    def test_correlated_stocks_high_score(self):
        base = np.random.default_rng(0).normal(0, 1, 500)
        df = pd.DataFrame(
            {
                "A": base + np.random.default_rng(1).normal(0, 0.1, 500),
                "B": base + np.random.default_rng(2).normal(0, 0.1, 500),
                "C": base + np.random.default_rng(3).normal(0, 0.1, 500),
            }
        )
        result = pairwise_correlation_score(df)
        assert result.min() > 0.55

    def test_rejects_single_stock(self):
        df = pd.DataFrame({"A": np.random.randn(50)})
        with pytest.raises(SPTInvariantError, match="2"):
            pairwise_correlation_score(df)


# ---------------------------------------------------------------------------
# gamma_star_contribution
# ---------------------------------------------------------------------------


class TestGammaStarContribution:
    def test_shape_matches(self):
        n = 5
        w = np.ones(n) / n
        cov = np.eye(n) * 0.04
        result = gamma_star_contribution(w, cov)
        assert result.shape == (n,)

    def test_positive_for_equal_weight_diagonal(self):
        n = 5
        w = np.ones(n) / n
        cov = np.eye(n) * 0.04
        result = gamma_star_contribution(w, cov)
        assert (result > 0).all()

    def test_higher_vol_stock_contributes_more(self):
        n = 3
        w = np.ones(n) / n
        cov = np.diag([0.01, 0.04, 0.09])
        result = gamma_star_contribution(w, cov)
        assert result[2] > result[1] > result[0]

    def test_shape_mismatch_raises(self):
        with pytest.raises(SPTInvariantError, match="mismatch"):
            gamma_star_contribution(np.ones(3) / 3, np.eye(4))


# ---------------------------------------------------------------------------
# boundary_risk_score
# ---------------------------------------------------------------------------


class TestBoundaryRiskScore:
    def test_rising_weight_high_score(self):
        T, n = 60, 3
        w = np.ones(n) / n
        hist = pd.DataFrame(
            {
                "A": np.linspace(0.20, 0.60, T),
                "B": np.linspace(0.40, 0.20, T),
                "C": np.linspace(0.40, 0.20, T),
            }
        )
        result = boundary_risk_score(w, hist)
        assert result["A"] > result["B"]
        assert result["A"] > result["C"]

    def test_stable_weights_moderate_score(self):
        T, n = 60, 3
        w = np.ones(n) / n
        hist = pd.DataFrame(
            {
                "A": np.ones(T) / 3 + np.random.default_rng(0).normal(0, 1e-5, T),
                "B": np.ones(T) / 3 + np.random.default_rng(1).normal(0, 1e-5, T),
                "C": np.ones(T) / 3 + np.random.default_rng(2).normal(0, 1e-5, T),
            }
        )
        result = boundary_risk_score(w, hist)
        assert all(0.3 < v < 0.7 for v in result.values)

    def test_returns_series(self):
        T, n = 60, 3
        w = np.ones(n) / n
        hist = pd.DataFrame(np.random.randn(T, n), columns=["A", "B", "C"])
        result = boundary_risk_score(w, hist)
        assert isinstance(result, pd.Series)
        assert set(result.index) == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# liquidity_filter
# ---------------------------------------------------------------------------


class TestLiquidityFilter:
    def test_filters_illiquid(self):
        volumes = pd.DataFrame(
            {
                "AAPL": [5e6, 6e6, 4e6, 5e6],
                "TINY": [1e3, 2e3, 1e3, 1e3],
                "OK": [2e6, 3e6, 2e6, 2e6],
            }
        )
        result = liquidity_filter(volumes, min_daily_volume=1e6)
        assert "AAPL" in result
        assert "OK" in result
        assert "TINY" not in result

    def test_all_pass(self):
        volumes = pd.DataFrame({"A": [5e6] * 10, "B": [3e6] * 10})
        result = liquidity_filter(volumes, min_daily_volume=1e6)
        assert len(result) == 2

    def test_none_pass(self):
        volumes = pd.DataFrame({"A": [100] * 10, "B": [200] * 10})
        result = liquidity_filter(volumes, min_daily_volume=1e6)
        assert len(result) == 0
