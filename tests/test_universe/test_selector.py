"""Tests for universe/selector.py — SPTUniverseSelector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantspt.errors import SPTInvariantError
from quantspt.universe.selector import SPTUniverseSelector

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _synthetic_prices(
    n_stocks: int = 20,
    n_days: int = 252,
    seed: int = 42,
) -> pd.DataFrame:
    """Build synthetic price panel with structure that the selector can exploit.

    Stocks 0..9 are low-vol/low-corr (good for SPT).
    Stocks 10..19 are high-corr mega-caps (bad for SPT).
    """
    rng = np.random.default_rng(seed)
    tickers = [f"S{i:02d}" for i in range(n_stocks)]
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")

    market = rng.normal(0, 0.01, n_days)
    prices_data = np.zeros((n_days, n_stocks))

    for j in range(n_stocks):
        if j < n_stocks // 2:
            # "Good SPT" stocks: moderate vol, low beta, high idio
            beta = rng.uniform(0.3, 0.6)
            idio_vol = rng.uniform(0.015, 0.025)
            base_price = rng.uniform(50, 150)
        else:
            # "Bad SPT" stocks: high corr, mega-cap proxy
            beta = rng.uniform(1.0, 1.5)
            idio_vol = rng.uniform(0.005, 0.010)
            base_price = rng.uniform(200, 800)

        daily_ret = beta * market + rng.normal(0, idio_vol, n_days)
        cum = np.exp(np.cumsum(daily_ret))
        prices_data[:, j] = base_price * cum

    return pd.DataFrame(prices_data, index=dates, columns=tickers)


@pytest.fixture()
def prices_20():
    return _synthetic_prices(20, 252)


@pytest.fixture()
def prices_30():
    return _synthetic_prices(30, 300, seed=99)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default(self):
        sel = SPTUniverseSelector()
        assert sel.n_stocks == 50
        assert sel.rebalance_frequency == "monthly"

    def test_custom(self):
        sel = SPTUniverseSelector(n_stocks=30, max_market_cap_percentile=85)
        assert sel.n_stocks == 30
        assert sel.max_market_cap_percentile == 85.0

    def test_invalid_n_stocks(self):
        with pytest.raises(SPTInvariantError, match="5"):
            SPTUniverseSelector(n_stocks=2)

    def test_invalid_percentiles(self):
        with pytest.raises(SPTInvariantError, match="percentile"):
            SPTUniverseSelector(
                min_market_cap_percentile=90, max_market_cap_percentile=20
            )

    def test_invalid_frequency(self):
        with pytest.raises(SPTInvariantError, match="frequency"):
            SPTUniverseSelector(rebalance_frequency="weekly")


# ---------------------------------------------------------------------------
# select()
# ---------------------------------------------------------------------------


class TestSelect:
    def test_returns_list_of_strings(self, prices_20):
        sel = SPTUniverseSelector(n_stocks=10, min_idiosyncratic_vol=0.0)
        result = sel.select(prices_20)
        assert isinstance(result, list)
        assert all(isinstance(t, str) for t in result)

    def test_respects_n_stocks(self, prices_20):
        sel = SPTUniverseSelector(n_stocks=8, min_idiosyncratic_vol=0.0)
        result = sel.select(prices_20)
        assert len(result) <= 8

    def test_no_duplicates(self, prices_20):
        sel = SPTUniverseSelector(n_stocks=10, min_idiosyncratic_vol=0.0)
        result = sel.select(prices_20)
        assert len(result) == len(set(result))

    def test_prefers_low_corr_high_idio_stocks(self, prices_20):
        sel = SPTUniverseSelector(
            n_stocks=8,
            min_market_cap_percentile=0,
            max_market_cap_percentile=100,
            max_avg_correlation=1.0,
            min_idiosyncratic_vol=0.0,
        )
        result = sel.select(prices_20)
        good_stocks = {f"S{i:02d}" for i in range(10)}
        overlap = set(result) & good_stocks
        assert len(overlap) >= 5, (
            f"Selector should prefer low-corr/high-idio stocks; "
            f"got {result}, overlap with good set = {overlap}"
        )

    def test_rejects_too_few_observations(self):
        prices = pd.DataFrame(np.random.randn(10, 5), columns=list("ABCDE"))
        sel = SPTUniverseSelector(n_stocks=5, min_idiosyncratic_vol=0.0)
        with pytest.raises(SPTInvariantError, match="30"):
            sel.select(prices)


# ---------------------------------------------------------------------------
# score_stocks()
# ---------------------------------------------------------------------------


class TestScoreStocks:
    def test_returns_expected_columns(self, prices_20):
        sel = SPTUniverseSelector(n_stocks=10, min_idiosyncratic_vol=0.0)
        scores = sel.score_stocks(prices_20)
        expected = {
            "idiosyncratic_vol",
            "avg_correlation",
            "gamma_contribution",
            "boundary_risk",
            "spt_score",
        }
        assert expected <= set(scores.columns)

    def test_scores_are_bounded(self, prices_20):
        sel = SPTUniverseSelector(
            n_stocks=10,
            min_market_cap_percentile=0,
            max_market_cap_percentile=100,
            max_avg_correlation=1.0,
            min_idiosyncratic_vol=0.0,
        )
        scores = sel.score_stocks(prices_20)
        valid = scores["spt_score"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_nan_for_filtered_stocks(self, prices_20):
        sel = SPTUniverseSelector(
            n_stocks=5,
            min_market_cap_percentile=50,
            max_market_cap_percentile=90,
            min_idiosyncratic_vol=0.0,
        )
        scores = sel.score_stocks(prices_20)
        n_nan = scores["spt_score"].isna().sum()
        assert n_nan > 0


# ---------------------------------------------------------------------------
# select_timeseries()
# ---------------------------------------------------------------------------


class TestSelectTimeseries:
    def test_returns_dict_of_lists(self, prices_30):
        sel = SPTUniverseSelector(
            n_stocks=10,
            lookback_days=60,
            min_idiosyncratic_vol=0.0,
        )
        result = sel.select_timeseries(prices_30)
        assert isinstance(result, dict)
        assert len(result) > 0
        for _dt, tickers in result.items():
            assert isinstance(tickers, list)
            assert len(tickers) <= 10

    def test_monthly_cadence(self, prices_30):
        sel = SPTUniverseSelector(
            n_stocks=10,
            lookback_days=60,
            rebalance_frequency="monthly",
            min_idiosyncratic_vol=0.0,
        )
        result = sel.select_timeseries(prices_30)
        dates = sorted(result.keys())
        assert len(dates) >= 2

    def test_quarterly_fewer_dates(self, prices_30):
        monthly_sel = SPTUniverseSelector(
            n_stocks=10,
            lookback_days=60,
            rebalance_frequency="monthly",
            min_idiosyncratic_vol=0.0,
        )
        quarterly_sel = SPTUniverseSelector(
            n_stocks=10,
            lookback_days=60,
            rebalance_frequency="quarterly",
            min_idiosyncratic_vol=0.0,
        )
        m = monthly_sel.select_timeseries(prices_30)
        q = quarterly_sel.select_timeseries(prices_30)
        assert len(q) <= len(m)
