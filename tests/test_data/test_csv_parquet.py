"""Tests for data/providers/csv_parquet.py — CSV and Parquet loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from quantspt.data.providers.csv_parquet import CSVProvider, ParquetProvider
from quantspt.data.schemas import MarketPanel
from quantspt.errors import DataProviderError, SPTInvariantError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_csv(tmp_path: Path) -> Path:
    dates = pd.date_range("2020-01-01", periods=20, freq="B")
    df = pd.DataFrame(
        {
            "date": dates,
            "AAPL": np.linspace(100, 120, 20),
            "GOOG": np.linspace(200, 240, 20),
            "MSFT": np.linspace(150, 180, 20),
        }
    )
    path = tmp_path / "prices.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def sample_parquet(tmp_path: Path) -> Path:
    dates = pd.date_range("2020-01-01", periods=20, freq="B")
    df = pd.DataFrame(
        {
            "AAPL": np.linspace(100, 120, 20),
            "GOOG": np.linspace(200, 240, 20),
        },
        index=dates,
    )
    df.index.name = "date"
    path = tmp_path / "prices.parquet"
    df.to_parquet(path)
    return path


# ---------------------------------------------------------------------------
# CSV Provider
# ---------------------------------------------------------------------------


class TestCSVProvider:
    def test_load_all(self, sample_csv: Path) -> None:
        provider = CSVProvider(sample_csv)
        result = provider.load()
        assert isinstance(result.data, MarketPanel)
        assert result.data.n_assets == 3
        assert result.data.n_observations == 20

    def test_load_specific_tickers(self, sample_csv: Path) -> None:
        provider = CSVProvider(sample_csv)
        result = provider.load(tickers=["AAPL", "GOOG"])
        assert result.data.n_assets == 2
        assert result.data.tickers == ["AAPL", "GOOG"]

    def test_date_filtering(self, sample_csv: Path) -> None:
        provider = CSVProvider(sample_csv)
        result = provider.load(start="2020-01-06", end="2020-01-17")
        assert result.data.n_observations <= 20

    def test_round_trip(self, tmp_path: Path) -> None:
        """Write CSV, read back, verify data matches."""
        dates = pd.date_range("2021-06-01", periods=5, freq="B")
        original = pd.DataFrame(
            {"date": dates, "A": [1.0, 2.0, 3.0, 4.0, 5.0], "B": [10, 20, 30, 40, 50]}
        )
        path = tmp_path / "roundtrip.csv"
        original.to_csv(path, index=False)

        provider = CSVProvider(path)
        result = provider.load()
        assert_allclose(result.data.prices["A"].values, [1.0, 2.0, 3.0, 4.0, 5.0])
        assert_allclose(result.data.prices["B"].values, [10, 20, 30, 40, 50])

    def test_missing_ticker(self, sample_csv: Path) -> None:
        provider = CSVProvider(sample_csv)
        with pytest.raises(DataProviderError, match="not found"):
            provider.load(tickers=["TSLA"])

    def test_file_not_found(self) -> None:
        provider = CSVProvider("/nonexistent/file.csv")
        with pytest.raises(SPTInvariantError):
            provider.load()

    def test_auto_date_detection(self, tmp_path: Path) -> None:
        """File with 'Date' column (capital D) is auto-detected."""
        df = pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-01", periods=3, freq="B"),
                "X": [100, 101, 102],
            }
        )
        path = tmp_path / "dated.csv"
        df.to_csv(path, index=False)

        provider = CSVProvider(path)
        result = provider.load()
        assert result.data.n_observations == 3

    def test_metadata(self, sample_csv: Path) -> None:
        provider = CSVProvider(sample_csv)
        result = provider.load()
        assert result.metadata["provider"] == "CSVProvider"
        assert result.computation_time_ms >= 0


# ---------------------------------------------------------------------------
# Parquet Provider
# ---------------------------------------------------------------------------


class TestParquetProvider:
    def test_load_all(self, sample_parquet: Path) -> None:
        provider = ParquetProvider(sample_parquet)
        result = provider.load()
        assert isinstance(result.data, MarketPanel)
        assert result.data.n_assets == 2
        assert result.data.n_observations == 20

    def test_load_specific_tickers(self, sample_parquet: Path) -> None:
        provider = ParquetProvider(sample_parquet)
        result = provider.load(tickers=["GOOG"])
        assert result.data.tickers == ["GOOG"]

    def test_round_trip(self, tmp_path: Path) -> None:
        """Write Parquet, read back, verify."""
        dates = pd.date_range("2022-01-01", periods=5, freq="B")
        original = pd.DataFrame(
            {"X": [10.0, 20.0, 30.0, 40.0, 50.0], "Y": [5, 4, 3, 2, 1]},
            index=dates,
        )
        original.index.name = "date"
        path = tmp_path / "rt.parquet"
        original.to_parquet(path)

        provider = ParquetProvider(path)
        result = provider.load()
        assert_allclose(result.data.prices["X"].values, [10, 20, 30, 40, 50])

    def test_file_not_found(self) -> None:
        provider = ParquetProvider("/nonexistent/file.parquet")
        with pytest.raises(SPTInvariantError):
            provider.load()

    def test_metadata(self, sample_parquet: Path) -> None:
        provider = ParquetProvider(sample_parquet)
        result = provider.load()
        assert result.metadata["provider"] == "ParquetProvider"

    def test_date_filtering(self, sample_parquet: Path) -> None:
        provider = ParquetProvider(sample_parquet)
        result = provider.load(start="2020-01-06", end="2020-01-17")
        assert result.data.n_observations <= 20

    def test_missing_ticker(self, sample_parquet: Path) -> None:
        provider = ParquetProvider(sample_parquet)
        with pytest.raises(DataProviderError, match="not found"):
            provider.load(tickers=["TSLA"])


class TestCSVDateEdgeCases:
    """Cover date-parsing edge cases in csv_parquet.py."""

    def test_timestamp_column_name(self, tmp_path: Path) -> None:
        """A column named 'timestamp' should be auto-detected."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=5, freq="B"),
                "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        path = tmp_path / "ts_col.csv"
        df.to_csv(path, index=False)

        provider = CSVProvider(path)
        result = provider.load()
        assert result.data.n_observations == 5

    def test_datetime64_dtype_detection(self, tmp_path: Path) -> None:
        """A Parquet file with a non-standard datetime column should be
        detected by dtype, not name."""
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        df = pd.DataFrame({"my_dt": dates, "A": [1.0, 2.0, 3.0, 4.0, 5.0]})
        path = tmp_path / "dt_dtype.parquet"
        df.to_parquet(path, index=False)

        provider = ParquetProvider(path)
        result = provider.load()
        assert result.data.n_observations == 5

    def test_index_as_date(self, tmp_path: Path) -> None:
        """CSV written with DatetimeIndex should be parseable."""
        dates = pd.date_range("2020-06-01", periods=4, freq="B")
        df = pd.DataFrame({"X": [10, 20, 30, 40]}, index=dates)
        df.index.name = "date"
        path = tmp_path / "indexed.csv"
        df.to_csv(path)

        provider = CSVProvider(path)
        result = provider.load()
        assert result.data.n_observations == 4

    def test_string_date_params(self, tmp_path: Path) -> None:
        """Verify string start/end params are handled in transform_query."""
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        df = pd.DataFrame(
            {"date": dates, "A": range(10)},
        )
        path = tmp_path / "str_dates.csv"
        df.to_csv(path, index=False)

        provider = CSVProvider(path)
        result = provider.load(start="2020-01-03", end="2020-01-10")
        assert result.data.n_observations <= 10

    def test_parquet_string_date_params(self, tmp_path: Path) -> None:
        """Verify string start/end params work for ParquetProvider."""
        dates = pd.date_range("2021-01-01", periods=10, freq="B")
        df = pd.DataFrame({"A": range(10)}, index=dates)
        df.index.name = "date"
        path = tmp_path / "str_pq.parquet"
        df.to_parquet(path)

        provider = ParquetProvider(path)
        result = provider.load(start="2021-01-05", end="2021-01-12")
        assert result.data.n_observations <= 10
