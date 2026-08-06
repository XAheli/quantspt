"""Tests for the Yahoo Finance data provider.

Uses mocking to avoid network calls while testing all branches
of the TET pipeline (Transform-Extract-Transform).
"""

from __future__ import annotations

import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quantspt.data.providers.yfinance import YFinanceProvider, _require_yfinance
from quantspt.data.schemas import MarketPanel
from quantspt.errors import DataProviderError


@pytest.fixture
def provider() -> YFinanceProvider:
    return YFinanceProvider(auto_adjust=True, progress=False)


@pytest.fixture
def mock_prices() -> pd.DataFrame:
    """Realistic multi-ticker price DataFrame."""
    dates = pd.bdate_range("2023-01-01", periods=50)
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "AAPL": 150.0 + np.cumsum(rng.standard_normal(50)),
            "MSFT": 250.0 + np.cumsum(rng.standard_normal(50)),
            "GOOG": 100.0 + np.cumsum(rng.standard_normal(50)),
        },
        index=dates,
    )


@pytest.fixture
def mock_multi_index_prices(mock_prices: pd.DataFrame) -> pd.DataFrame:
    """Multi-level column index as returned by yfinance for multiple tickers."""
    tuples = [("Close", t) for t in mock_prices.columns]
    df = mock_prices.copy()
    df.columns = pd.MultiIndex.from_tuples(tuples)
    return df


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------


class TestImportGuard:
    def test_require_yfinance_raises_when_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "yfinance", None)
        with pytest.raises(ImportError, match="quantspt\\[data\\]"):
            _require_yfinance()

    def test_require_yfinance_succeeds(self) -> None:
        mod = _require_yfinance()
        assert hasattr(mod, "download")


# ---------------------------------------------------------------------------
# transform_query
# ---------------------------------------------------------------------------


class TestTransformQuery:
    def test_basic_query(self, provider: YFinanceProvider) -> None:
        query = provider.transform_query(
            tickers=["AAPL", "MSFT"],
            start="2023-01-01",
            end="2023-12-31",
        )
        assert query.tickers == ["AAPL", "MSFT"]
        assert query.start == datetime(2023, 1, 1)
        assert query.end == datetime(2023, 12, 31)
        assert query.frequency == "daily"

    def test_datetime_inputs(self, provider: YFinanceProvider) -> None:
        s = datetime(2022, 6, 1)
        e = datetime(2023, 6, 1)
        query = provider.transform_query(tickers=["GOOG"], start=s, end=e)
        assert query.start == s
        assert query.end == e

    def test_defaults(self, provider: YFinanceProvider) -> None:
        query = provider.transform_query(tickers=["AAPL"])
        assert query.frequency == "daily"
        assert query.adjust is True

    def test_empty_tickers_raises(self, provider: YFinanceProvider) -> None:
        from quantspt.errors import SPTInvariantError

        with pytest.raises(SPTInvariantError):
            provider.transform_query(tickers=[])


# ---------------------------------------------------------------------------
# extract_data
# ---------------------------------------------------------------------------


class TestExtractData:
    @patch("quantspt.data.providers.yfinance._require_yfinance")
    def test_successful_download(
        self, mock_yf: MagicMock, provider: YFinanceProvider, mock_prices: pd.DataFrame
    ) -> None:
        mock_mod = MagicMock()
        mock_mod.download.return_value = mock_prices
        mock_yf.return_value = mock_mod

        query = provider.transform_query(
            tickers=["AAPL", "MSFT", "GOOG"],
            start="2023-01-01",
            end="2023-12-31",
        )
        data = provider.extract_data(query)
        assert not data.empty
        mock_mod.download.assert_called_once()

    @patch("quantspt.data.providers.yfinance._require_yfinance")
    def test_empty_response_raises(
        self, mock_yf: MagicMock, provider: YFinanceProvider
    ) -> None:
        mock_mod = MagicMock()
        mock_mod.download.return_value = pd.DataFrame()
        mock_yf.return_value = mock_mod

        query = provider.transform_query(
            tickers=["INVALID_TICKER"],
            start="2023-01-01",
            end="2023-12-31",
        )
        with pytest.raises(DataProviderError, match="No data returned"):
            provider.extract_data(query)

    @patch("quantspt.data.providers.yfinance._require_yfinance")
    def test_none_response_raises(
        self, mock_yf: MagicMock, provider: YFinanceProvider
    ) -> None:
        mock_mod = MagicMock()
        mock_mod.download.return_value = None
        mock_yf.return_value = mock_mod

        query = provider.transform_query(
            tickers=["AAPL"],
            start="2023-01-01",
            end="2023-12-31",
        )
        with pytest.raises(DataProviderError, match="No data returned"):
            provider.extract_data(query)

    @patch("quantspt.data.providers.yfinance._require_yfinance")
    def test_network_error_raises(
        self, mock_yf: MagicMock, provider: YFinanceProvider
    ) -> None:
        mock_mod = MagicMock()
        mock_mod.download.side_effect = ConnectionError("Network down")
        mock_yf.return_value = mock_mod

        query = provider.transform_query(
            tickers=["AAPL"],
            start="2023-01-01",
            end="2023-12-31",
        )
        with pytest.raises(DataProviderError, match="Failed to download"):
            provider.extract_data(query)


# ---------------------------------------------------------------------------
# transform_data
# ---------------------------------------------------------------------------


class TestTransformData:
    def test_flat_dataframe(
        self, provider: YFinanceProvider, mock_prices: pd.DataFrame
    ) -> None:
        query = provider.transform_query(
            tickers=["AAPL", "MSFT", "GOOG"],
            start="2023-01-01",
            end="2023-12-31",
        )
        panel = provider.transform_data(mock_prices, query)
        assert isinstance(panel, MarketPanel)
        assert panel.tickers == ["AAPL", "MSFT", "GOOG"]
        assert len(panel.prices) == 50

    def test_multi_index_close(
        self,
        provider: YFinanceProvider,
        mock_multi_index_prices: pd.DataFrame,
    ) -> None:
        query = provider.transform_query(
            tickers=["AAPL", "MSFT", "GOOG"],
            start="2023-01-01",
            end="2023-12-31",
        )
        panel = provider.transform_data(mock_multi_index_prices, query)
        assert isinstance(panel, MarketPanel)
        assert len(panel.tickers) == 3

    def test_missing_ticker_raises(
        self, provider: YFinanceProvider, mock_prices: pd.DataFrame
    ) -> None:
        query = provider.transform_query(
            tickers=["AAPL", "TSLA"],
            start="2023-01-01",
            end="2023-12-31",
        )
        with pytest.raises(DataProviderError, match="TSLA"):
            provider.transform_data(mock_prices, query)

    def test_handles_nans(self, provider: YFinanceProvider) -> None:
        dates = pd.bdate_range("2023-01-01", periods=10)
        df = pd.DataFrame(
            {"AAPL": [100, np.nan, 102, 103, 104, 105, 106, 107, 108, 109]},
            index=dates,
        )
        query = provider.transform_query(
            tickers=["AAPL"], start="2023-01-01", end="2023-12-31"
        )
        panel = provider.transform_data(df, query)
        assert panel.prices.isna().sum().sum() == 0

    def test_single_ticker_series(self, provider: YFinanceProvider) -> None:
        dates = pd.bdate_range("2023-01-01", periods=10)
        series = pd.Series([100.0 + i for i in range(10)], index=dates, name="AAPL")
        query = provider.transform_query(
            tickers=["AAPL"], start="2023-01-01", end="2023-12-31"
        )
        panel = provider.transform_data(series.to_frame(), query)
        assert panel.tickers == ["AAPL"]

    def test_all_nan_rows_dropped(self, provider: YFinanceProvider) -> None:
        dates = pd.bdate_range("2023-01-01", periods=5)
        df = pd.DataFrame(
            {"AAPL": [100.0, np.nan, 102.0, np.nan, 104.0]},
            index=dates,
        )
        df.iloc[1] = np.nan
        df.iloc[3] = np.nan
        query = provider.transform_query(
            tickers=["AAPL"], start="2023-01-01", end="2023-12-31"
        )
        panel = provider.transform_data(df, query)
        assert len(panel.prices) < 5
        assert "rows_dropped_all_nan" in panel.metadata

    def test_adj_close_multi_index(self, provider: YFinanceProvider) -> None:
        """Multi-index with 'Adj Close' instead of 'Close'."""
        dates = pd.bdate_range("2023-01-01", periods=10)
        data = {
            ("Adj Close", "AAPL"): 100.0 + np.arange(10, dtype=float),
            ("Adj Close", "MSFT"): 200.0 + np.arange(10, dtype=float),
            ("Volume", "AAPL"): np.ones(10) * 1e6,
            ("Volume", "MSFT"): np.ones(10) * 2e6,
        }
        df = pd.DataFrame(data, index=dates)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        query = provider.transform_query(
            tickers=["AAPL", "MSFT"], start="2023-01-01", end="2023-12-31"
        )
        panel = provider.transform_data(df, query)
        assert panel.tickers == ["AAPL", "MSFT"]

    def test_fallback_multi_index(self, provider: YFinanceProvider) -> None:
        """Multi-index with neither 'Close' nor 'Adj Close'."""
        dates = pd.bdate_range("2023-01-01", periods=10)
        data = {
            ("Price", "AAPL"): 100.0 + np.arange(10, dtype=float),
            ("Price", "MSFT"): 200.0 + np.arange(10, dtype=float),
        }
        df = pd.DataFrame(data, index=dates)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        query = provider.transform_query(
            tickers=["AAPL", "MSFT"], start="2023-01-01", end="2023-12-31"
        )
        panel = provider.transform_data(df, query)
        assert len(panel.tickers) == 2

    def test_series_input(self, provider: YFinanceProvider) -> None:
        """Single-ticker download that returns a Series."""
        dates = pd.bdate_range("2023-01-01", periods=10)
        series = pd.Series(100.0 + np.arange(10, dtype=float), index=dates)
        df = series.to_frame(name="AAPL")
        query = provider.transform_query(
            tickers=["AAPL"], start="2023-01-01", end="2023-12-31"
        )
        panel = provider.transform_data(df, query)
        assert panel.tickers == ["AAPL"]


# ---------------------------------------------------------------------------
# Full TET pipeline (load)
# ---------------------------------------------------------------------------


class TestFullPipeline:
    @patch("quantspt.data.providers.yfinance._require_yfinance")
    def test_load_end_to_end(
        self, mock_yf: MagicMock, provider: YFinanceProvider, mock_prices: pd.DataFrame
    ) -> None:
        mock_mod = MagicMock()
        mock_mod.download.return_value = mock_prices
        mock_yf.return_value = mock_mod

        result = provider.load(
            tickers=["AAPL", "MSFT", "GOOG"],
            start="2023-01-01",
            end="2023-12-31",
        )
        assert result.data is not None
        assert isinstance(result.data, MarketPanel)
        assert result.metadata["provider"] == "YFinanceProvider"
        assert result.computation_time_ms >= 0

    @patch("quantspt.data.providers.yfinance._require_yfinance")
    def test_load_weekly(
        self, mock_yf: MagicMock, provider: YFinanceProvider, mock_prices: pd.DataFrame
    ) -> None:
        mock_mod = MagicMock()
        mock_mod.download.return_value = mock_prices
        mock_yf.return_value = mock_mod

        result = provider.load(
            tickers=["AAPL"],
            start="2023-01-01",
            end="2023-12-31",
            frequency="weekly",
        )
        assert result.data is not None
