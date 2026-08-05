"""Tests for data/preprocessing.py — returns, missing data, outliers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from quantspt.data.preprocessing import (
    compute_log_returns,
    compute_simple_returns,
    detect_outliers,
    filter_universe,
    handle_missing,
    winsorise,
)
from quantspt.data.schemas import ReturnsMatrix
from quantspt.errors import SPTInvariantError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def price_df():
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {"A": [100.0, 102.0, 101.0, 105.0, 103.0], "B": [50.0, 51.0, 52.0, 50.0, 53.0]},
        index=dates,
    )


@pytest.fixture()
def price_array():
    return np.array(
        [[100.0, 50.0], [102.0, 51.0], [101.0, 52.0], [105.0, 50.0], [103.0, 53.0]]
    )


# ---------------------------------------------------------------------------
# Simple returns
# ---------------------------------------------------------------------------


class TestSimpleReturns:
    def test_dataframe_output(self, price_df) -> None:
        result = compute_simple_returns(price_df)
        assert isinstance(result, ReturnsMatrix)
        assert result.return_type == "simple"
        assert len(result.tickers) == 2

    def test_values_correct(self, price_df) -> None:
        result = compute_simple_returns(price_df)
        arr = result.to_numpy()
        expected_a0 = (102.0 - 100.0) / 100.0
        assert_allclose(arr[0, 0], expected_a0, atol=1e-10)

    def test_shape(self, price_df) -> None:
        result = compute_simple_returns(price_df)
        assert result.to_numpy().shape == (4, 2)

    def test_no_drop_first(self, price_df) -> None:
        result = compute_simple_returns(price_df, drop_first=False)
        assert result.to_numpy().shape == (5, 2)

    def test_array_input(self, price_array) -> None:
        result = compute_simple_returns(price_array)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 2)

    def test_array_no_drop(self, price_array) -> None:
        result = compute_simple_returns(price_array, drop_first=False)
        assert result.shape == (5, 2)
        assert np.isnan(result[0, 0])


# ---------------------------------------------------------------------------
# Log returns
# ---------------------------------------------------------------------------


class TestLogReturns:
    def test_dataframe_output(self, price_df) -> None:
        result = compute_log_returns(price_df)
        assert isinstance(result, ReturnsMatrix)
        assert result.return_type == "log"

    def test_values_correct(self, price_df) -> None:
        result = compute_log_returns(price_df)
        arr = result.to_numpy()
        expected = np.log(102.0 / 100.0)
        assert_allclose(arr[0, 0], expected, atol=1e-10)

    def test_shape(self, price_df) -> None:
        result = compute_log_returns(price_df)
        assert result.to_numpy().shape == (4, 2)

    def test_array_input(self, price_array) -> None:
        result = compute_log_returns(price_array)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 2)

    def test_log_simple_consistency(self, price_df) -> None:
        """For small returns, log and simple should be close."""
        simple = compute_simple_returns(price_df).to_numpy()
        log = compute_log_returns(price_df).to_numpy()
        assert_allclose(simple, np.exp(log) - 1, atol=1e-10)


# ---------------------------------------------------------------------------
# Missing data
# ---------------------------------------------------------------------------


class TestHandleMissing:
    def test_ffill(self) -> None:
        df = pd.DataFrame({"A": [1.0, np.nan, 3.0], "B": [10.0, 20.0, np.nan]})
        result = handle_missing(df, method="ffill")
        assert_allclose(result["A"].values, [1.0, 1.0, 3.0])
        assert_allclose(result["B"].values, [10.0, 20.0, 20.0])

    def test_drop(self) -> None:
        df = pd.DataFrame({"A": [1.0, np.nan, 3.0], "B": [10.0, 20.0, 30.0]})
        result = handle_missing(df, method="drop")
        assert len(result) == 2

    def test_interpolate(self) -> None:
        df = pd.DataFrame({"A": [1.0, np.nan, 3.0]})
        result = handle_missing(df, method="interpolate")
        assert_allclose(result["A"].values, [1.0, 2.0, 3.0])

    def test_max_gap(self) -> None:
        df = pd.DataFrame({"A": [1.0, np.nan, np.nan, np.nan, 5.0]})
        result = handle_missing(df, method="ffill", max_gap=1)
        assert np.isnan(result["A"].iloc[2])

    def test_invalid_method(self) -> None:
        df = pd.DataFrame({"A": [1.0]})
        with pytest.raises(SPTInvariantError):
            handle_missing(df, method="magic")


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------


class TestOutlierDetection:
    def test_zscore_detects_extreme(self) -> None:
        rng = np.random.default_rng(42)
        normal = rng.standard_normal((100, 2))
        normal[50, 0] = 10.0
        outliers = detect_outliers(normal, method="zscore", threshold=3.0)
        assert outliers[50, 0]
        assert outliers.sum() >= 1

    def test_iqr_method(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 1))
        data[0, 0] = 100.0
        outliers = detect_outliers(data, method="iqr", threshold=1.5)
        assert outliers[0, 0]

    def test_no_outliers(self) -> None:
        data = np.ones((10, 2))
        outliers = detect_outliers(data, method="zscore")
        assert not outliers.any()

    def test_dataframe_input(self) -> None:
        df = pd.DataFrame({"A": np.ones(20), "B": np.ones(20)})
        df.iloc[0, 0] = 100.0
        outliers = detect_outliers(df, method="zscore", threshold=2.0)
        assert outliers[0, 0]

    def test_invalid_method(self) -> None:
        with pytest.raises(SPTInvariantError):
            detect_outliers(np.ones((10, 2)), method="magic")


# ---------------------------------------------------------------------------
# Winsorisation
# ---------------------------------------------------------------------------


class TestWinsorise:
    def test_clips_extremes(self) -> None:
        data = np.array([[0.0, 100.0], [1.0, 2.0], [3.0, 4.0], [5.0, -50.0]])
        result = winsorise(data, lower_pct=10, upper_pct=90)
        assert result.max() < 100.0
        assert result.min() > -50.0

    def test_preserves_shape(self) -> None:
        data = np.random.default_rng(42).standard_normal((50, 3))
        result = winsorise(data)
        assert result.shape == (50, 3)

    def test_dataframe_preserves_index(self) -> None:
        dates = pd.date_range("2020-01-01", periods=5)
        df = pd.DataFrame({"A": [1, 2, 3, 4, 100]}, index=dates)
        result = winsorise(df, lower_pct=10, upper_pct=90)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# Universe filtering
# ---------------------------------------------------------------------------


class TestFilterUniverse:
    def test_drops_sparse_columns(self) -> None:
        df = pd.DataFrame(
            {
                "A": [1.0, 2.0, 3.0, 4.0, 5.0],
                "B": [1.0, np.nan, np.nan, np.nan, np.nan],
            }
        )
        result = filter_universe(df, min_non_nan_fraction=0.5)
        assert "A" in result.columns
        assert "B" not in result.columns

    def test_min_observations(self) -> None:
        df = pd.DataFrame(
            {
                "X": [1.0, 2.0, 3.0],
                "Y": [1.0, np.nan, np.nan],
            }
        )
        result = filter_universe(df, min_observations=2)
        assert "X" in result.columns
        assert "Y" not in result.columns

    def test_all_pass(self) -> None:
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        result = filter_universe(df)
        assert list(result.columns) == ["A", "B"]

    def test_none_pass_raises(self) -> None:
        df = pd.DataFrame({"A": [np.nan, np.nan]})
        with pytest.raises(SPTInvariantError, match="No assets"):
            filter_universe(df, min_non_nan_fraction=0.9)
