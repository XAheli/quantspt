"""Tests for data/schemas.py — MarketPanel, WeightVector, ReturnsMatrix."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from quantspt.data.schemas import MarketPanel, ReturnsMatrix, WeightVector
from quantspt.errors import SPTInvariantError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_prices():
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    return pd.DataFrame(
        {
            "AAPL": np.linspace(100, 110, 10),
            "GOOG": np.linspace(200, 220, 10),
        },
        index=dates,
    )


@pytest.fixture()
def sample_tickers():
    return ["AAPL", "GOOG"]


# ---------------------------------------------------------------------------
# MarketPanel
# ---------------------------------------------------------------------------


class TestMarketPanel:
    def test_basic_construction(self, sample_prices, sample_tickers) -> None:
        panel = MarketPanel(prices=sample_prices, tickers=sample_tickers)
        assert panel.n_assets == 2
        assert panel.n_observations == 10

    def test_date_range(self, sample_prices, sample_tickers) -> None:
        panel = MarketPanel(prices=sample_prices, tickers=sample_tickers)
        start, end = panel.date_range
        assert start == sample_prices.index[0]
        assert end == sample_prices.index[-1]

    def test_frequency_validation(self, sample_prices, sample_tickers) -> None:
        panel = MarketPanel(
            prices=sample_prices, tickers=sample_tickers, frequency="weekly"
        )
        assert panel.frequency == "weekly"

    def test_invalid_frequency(self, sample_prices, sample_tickers) -> None:
        with pytest.raises(SPTInvariantError):
            MarketPanel(
                prices=sample_prices, tickers=sample_tickers, frequency="hourly"
            )

    def test_empty_prices_rejected(self) -> None:
        with pytest.raises(SPTInvariantError):
            MarketPanel(prices=pd.DataFrame(), tickers=[])

    def test_all_nan_ticker_rejected(self) -> None:
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        prices = pd.DataFrame(
            {"A": [1.0, 2.0, 3.0, 4.0, 5.0], "B": [np.nan] * 5}, index=dates
        )
        with pytest.raises(ValueError, match="all-NaN"):
            MarketPanel(prices=prices, tickers=["A", "B"])

    def test_to_weight_vectors(self, sample_prices, sample_tickers) -> None:
        panel = MarketPanel(prices=sample_prices, tickers=sample_tickers)
        weights = panel.to_weight_vectors()
        assert weights.shape == sample_prices.shape
        row_sums = weights.sum(axis=1)
        assert_allclose(row_sums, 1.0, atol=1e-14)

    def test_to_weight_vectors_with_caps(self, sample_prices, sample_tickers) -> None:
        caps = sample_prices * 1e6
        panel = MarketPanel(
            prices=sample_prices, tickers=sample_tickers, market_caps=caps
        )
        weights = panel.to_weight_vectors()
        assert weights.shape == caps.shape


# ---------------------------------------------------------------------------
# WeightVector
# ---------------------------------------------------------------------------


class TestWeightVector:
    def test_valid_weights(self) -> None:
        w = WeightVector(weights=np.array([0.6, 0.4]), tickers=["A", "B"])
        assert len(w.tickers) == 2

    def test_must_sum_to_one(self) -> None:
        with pytest.raises(SPTInvariantError, match="sum to 1"):
            WeightVector(weights=np.array([0.5, 0.3]), tickers=["A", "B"])

    def test_non_negative(self) -> None:
        with pytest.raises(SPTInvariantError, match="non-negative"):
            WeightVector(weights=np.array([-0.1, 1.1]), tickers=["A", "B"])

    def test_length_mismatch(self) -> None:
        with pytest.raises(SPTInvariantError, match="mismatch"):
            WeightVector(weights=np.array([0.5, 0.5]), tickers=["A"])

    def test_must_be_1d(self) -> None:
        with pytest.raises(SPTInvariantError):
            WeightVector(weights=np.array([[0.5, 0.5]]), tickers=["A", "B"])

    def test_timestamp(self) -> None:
        from datetime import datetime

        ts = datetime(2020, 1, 1)
        w = WeightVector(weights=np.array([1.0]), tickers=["A"], timestamp=ts)
        assert w.timestamp == ts


# ---------------------------------------------------------------------------
# ReturnsMatrix
# ---------------------------------------------------------------------------


class TestReturnsMatrix:
    def test_valid_construction(self) -> None:
        returns = np.random.default_rng(42).standard_normal((100, 3))
        rm = ReturnsMatrix(returns=returns, tickers=["A", "B", "C"])
        assert rm.return_type == "simple"

    def test_log_return_type(self) -> None:
        returns = np.random.default_rng(42).standard_normal((50, 2))
        rm = ReturnsMatrix(returns=returns, tickers=["X", "Y"], return_type="log")
        assert rm.return_type == "log"

    def test_invalid_return_type(self) -> None:
        with pytest.raises(SPTInvariantError, match="return_type"):
            ReturnsMatrix(
                returns=np.zeros((10, 2)),
                tickers=["A", "B"],
                return_type="arithmetic",
            )

    def test_column_mismatch(self) -> None:
        with pytest.raises(SPTInvariantError, match="Column count"):
            ReturnsMatrix(returns=np.zeros((10, 3)), tickers=["A", "B"])

    def test_to_numpy(self) -> None:
        data = np.random.default_rng(42).standard_normal((20, 2))
        rm = ReturnsMatrix(returns=data, tickers=["A", "B"])
        arr = rm.to_numpy()
        assert_allclose(arr, data)

    def test_dataframe_input(self) -> None:
        df = pd.DataFrame({"A": [0.01, -0.02], "B": [0.03, 0.00]})
        rm = ReturnsMatrix(returns=df, tickers=["A", "B"])
        arr = rm.to_numpy()
        assert arr.shape == (2, 2)

    def test_must_be_2d(self) -> None:
        with pytest.raises(SPTInvariantError):
            ReturnsMatrix(returns=np.zeros(10), tickers=["A"])
