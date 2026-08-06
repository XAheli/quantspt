"""Tests for data/corporate_actions.py — splits, dividends, delistings."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from quantspt.data.corporate_actions import (
    adjust_for_dividends,
    adjust_for_splits,
    detect_splits,
    handle_delistings,
)
from quantspt.errors import SPTInvariantError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def dates():
    return pd.date_range("2020-01-01", periods=6, freq="B")


@pytest.fixture()
def split_prices(dates):
    """Prices with a 2:1 split at dates[3]."""
    return pd.DataFrame(
        {"A": [100.0, 102.0, 104.0, 52.0, 53.0, 54.0]},
        index=dates,
    )


@pytest.fixture()
def multi_asset_prices(dates):
    """Two assets, B has a 3:1 split at dates[4]."""
    return pd.DataFrame(
        {
            "X": [200.0, 202.0, 204.0, 206.0, 208.0, 210.0],
            "Y": [90.0, 93.0, 96.0, 99.0, 33.0, 34.0],
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# adjust_for_splits tests
# ---------------------------------------------------------------------------


class TestAdjustForSplits:
    def test_single_split_dict(self, split_prices, dates) -> None:
        splits = {"A": [(dates[3], 2.0)]}
        adjusted = adjust_for_splits(split_prices, splits)
        assert_allclose(adjusted["A"].iloc[0], 50.0)
        assert_allclose(adjusted["A"].iloc[1], 51.0)
        assert_allclose(adjusted["A"].iloc[2], 52.0)
        assert_allclose(adjusted["A"].iloc[3], 52.0)

    def test_split_preserves_post_split_prices(self, split_prices, dates) -> None:
        splits = {"A": [(dates[3], 2.0)]}
        adjusted = adjust_for_splits(split_prices, splits)
        assert_allclose(adjusted["A"].iloc[4], 53.0)
        assert_allclose(adjusted["A"].iloc[5], 54.0)

    def test_multi_asset_split(self, multi_asset_prices, dates) -> None:
        splits = {"Y": [(dates[4], 3.0)]}
        adjusted = adjust_for_splits(multi_asset_prices, splits)
        assert_allclose(adjusted["Y"].iloc[0], 30.0)
        assert_allclose(adjusted["Y"].iloc[3], 33.0)
        assert_allclose(adjusted["X"].iloc[0], 200.0)

    def test_multiple_splits_same_asset(self, dates) -> None:
        prices = pd.DataFrame(
            {"A": [100.0, 50.0, 51.0, 25.5, 26.0, 27.0]},
            index=dates,
        )
        splits = {"A": [(dates[1], 2.0), (dates[3], 2.0)]}
        adjusted = adjust_for_splits(prices, splits)
        assert_allclose(adjusted["A"].iloc[0], 25.0)
        assert_allclose(adjusted["A"].iloc[1], 25.0)
        assert_allclose(adjusted["A"].iloc[2], 25.5)

    def test_dataframe_ratio_format(self, dates) -> None:
        prices = pd.DataFrame(
            {"A": [100.0, 102.0, 51.0, 52.0, 53.0, 54.0]},
            index=dates,
        )
        ratios = pd.DataFrame(
            {"A": [1.0, 1.0, 2.0, 1.0, 1.0, 1.0]},
            index=dates,
        )
        adjusted = adjust_for_splits(prices, ratios)
        assert_allclose(adjusted["A"].iloc[0], 50.0)
        assert_allclose(adjusted["A"].iloc[1], 51.0)

    def test_array_input(self, dates) -> None:
        prices = np.array([[100.0], [102.0], [51.0], [52.0]])
        ratios = pd.DataFrame(
            {"A": [1.0, 1.0, 2.0, 1.0]},
            index=dates[:4],
        )
        adjusted = adjust_for_splits(prices, ratios)
        assert_allclose(adjusted[0, 0], 50.0)
        assert_allclose(adjusted[1, 0], 51.0)

    def test_unknown_ticker_raises(self, split_prices, dates) -> None:
        splits = {"UNKNOWN": [(dates[3], 2.0)]}
        with pytest.raises(SPTInvariantError, match="not found"):
            adjust_for_splits(split_prices, splits)

    def test_negative_ratio_raises(self, split_prices, dates) -> None:
        splits = {"A": [(dates[3], -2.0)]}
        with pytest.raises(SPTInvariantError, match="positive"):
            adjust_for_splits(split_prices, splits)


# ---------------------------------------------------------------------------
# adjust_for_dividends tests
# ---------------------------------------------------------------------------


class TestAdjustForDividends:
    def test_total_return(self, dates) -> None:
        prices = pd.DataFrame(
            {"A": [100.0, 98.0, 99.0, 100.0, 98.0, 99.0]},
            index=dates,
        )
        dividends = pd.DataFrame(
            {"A": [0.0, 2.0, 0.0, 0.0, 2.0, 0.0]},
            index=dates,
        )
        adjusted = adjust_for_dividends(prices, dividends, method="total_return")
        ratio_before = prices["A"].iloc[0] / prices["A"].iloc[1]
        ratio_after = adjusted["A"].iloc[0] / adjusted["A"].iloc[1]
        assert ratio_after < ratio_before

    def test_price_only_no_change(self, dates) -> None:
        prices = pd.DataFrame(
            {"A": [100.0, 98.0, 99.0, 100.0, 98.0, 99.0]}, index=dates
        )
        dividends = pd.DataFrame({"A": [0.0, 5.0, 0.0, 0.0, 5.0, 0.0]}, index=dates)
        adjusted = adjust_for_dividends(prices, dividends, method="price_only")
        assert_allclose(adjusted.values, prices.values)

    def test_proportional_method(self, dates) -> None:
        prices = pd.DataFrame(
            {"A": [100.0, 98.0, 99.0, 100.0, 101.0, 102.0]},
            index=dates,
        )
        dividends = pd.DataFrame(
            {"A": [0.0, 2.0, 0.0, 0.0, 0.0, 0.0]},
            index=dates,
        )
        adjusted = adjust_for_dividends(prices, dividends, method="proportional")
        factor = 1.0 - 2.0 / 100.0
        assert_allclose(adjusted["A"].iloc[0], 100.0 * factor)

    def test_invalid_method_raises(self, dates) -> None:
        prices = pd.DataFrame({"A": [100.0] * 6}, index=dates)
        dividends = pd.DataFrame({"A": [0.0] * 6}, index=dates)
        with pytest.raises(SPTInvariantError, match="Unknown method"):
            adjust_for_dividends(prices, dividends, method="magic")

    def test_shape_mismatch_raises(self, dates) -> None:
        prices = pd.DataFrame({"A": [100.0] * 6, "B": [50.0] * 6}, index=dates)
        dividends = pd.DataFrame({"A": [0.0] * 6}, index=dates)
        with pytest.raises(SPTInvariantError, match="shape"):
            adjust_for_dividends(prices, dividends)

    def test_zero_dividends_no_change(self, dates) -> None:
        prices = pd.DataFrame(
            {"A": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]}, index=dates
        )
        dividends = pd.DataFrame({"A": [0.0] * 6}, index=dates)
        adjusted = adjust_for_dividends(prices, dividends, method="total_return")
        assert_allclose(adjusted.values, prices.values)

    def test_multi_asset(self, dates) -> None:
        prices = pd.DataFrame(
            {
                "A": [100.0, 98.0, 99.0, 100.0, 101.0, 102.0],
                "B": [50.0, 49.0, 50.0, 51.0, 52.0, 53.0],
            },
            index=dates,
        )
        dividends = pd.DataFrame(
            {"A": [0.0, 2.0, 0.0, 0.0, 0.0, 0.0], "B": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]},
            index=dates,
        )
        adjusted = adjust_for_dividends(prices, dividends, method="total_return")
        assert adjusted["A"].iloc[0] < prices["A"].iloc[0]
        assert adjusted["B"].iloc[0] < prices["B"].iloc[0]


# ---------------------------------------------------------------------------
# handle_delistings tests
# ---------------------------------------------------------------------------


class TestHandleDelistings:
    def test_last_price_method(self, dates) -> None:
        prices = pd.DataFrame(
            {"A": [100.0, 101.0, 102.0, np.nan, np.nan, np.nan]},
            index=dates,
        )
        result = handle_delistings(prices, method="last_price")
        assert_allclose(result["A"].iloc[3], 102.0)
        assert_allclose(result["A"].iloc[5], 102.0)

    def test_zero_method(self, dates) -> None:
        prices = pd.DataFrame(
            {"A": [100.0, 101.0, 102.0, np.nan, np.nan, np.nan]},
            index=dates,
        )
        result = handle_delistings(prices, method="zero")
        assert_allclose(result["A"].iloc[3], 0.0)
        assert_allclose(result["A"].iloc[5], 0.0)

    def test_nan_method(self, dates) -> None:
        prices = pd.DataFrame(
            {"A": [100.0, 101.0, 102.0, np.nan, np.nan, np.nan]},
            index=dates,
        )
        result = handle_delistings(prices, method="nan")
        assert np.isnan(result["A"].iloc[3])

    def test_fill_value_method(self, dates) -> None:
        prices = pd.DataFrame(
            {"A": [100.0, 101.0, 102.0, np.nan, np.nan, np.nan]},
            index=dates,
        )
        result = handle_delistings(prices, method="fill_value", fill_value=0.01)
        assert_allclose(result["A"].iloc[4], 0.01)

    def test_explicit_delisting_dict(self, dates) -> None:
        prices = pd.DataFrame(
            {"A": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]},
            index=dates,
        )
        delisted = {"A": dates[3]}
        result = handle_delistings(prices, delisted=delisted, method="zero")
        assert_allclose(result["A"].iloc[3], 103.0)  # delist date itself is kept
        assert_allclose(result["A"].iloc[4], 0.0)
        assert_allclose(result["A"].iloc[5], 0.0)

    def test_no_delisting_no_change(self, dates) -> None:
        prices = pd.DataFrame(
            {"A": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]},
            index=dates,
        )
        result = handle_delistings(prices, method="last_price")
        assert_allclose(result.values, prices.values)

    def test_invalid_method_raises(self, dates) -> None:
        prices = pd.DataFrame({"A": [100.0] * 6}, index=dates)
        with pytest.raises(SPTInvariantError, match="Unknown method"):
            handle_delistings(prices, method="magic")

    def test_multi_asset_delisting(self, dates) -> None:
        prices = pd.DataFrame(
            {
                "A": [100.0, 101.0, np.nan, np.nan, np.nan, np.nan],
                "B": [50.0, 51.0, 52.0, 53.0, np.nan, np.nan],
            },
            index=dates,
        )
        result = handle_delistings(prices, method="last_price")
        assert_allclose(result["A"].iloc[2], 101.0)
        assert_allclose(result["B"].iloc[4], 53.0)


# ---------------------------------------------------------------------------
# detect_splits tests
# ---------------------------------------------------------------------------


class TestDetectSplits:
    def test_detects_2_for_1(self, dates) -> None:
        prices = pd.DataFrame(
            {"A": [100.0, 102.0, 104.0, 52.0, 53.0, 54.0]},
            index=dates,
        )
        detected = detect_splits(prices)
        assert "A" in detected
        assert len(detected["A"]) == 1
        assert detected["A"][0][0] == dates[3]
        assert_allclose(detected["A"][0][1], 2.0)

    def test_detects_3_for_1(self, dates) -> None:
        prices = pd.DataFrame(
            {"A": [90.0, 93.0, 96.0, 32.0, 33.0, 34.0]},
            index=dates,
        )
        detected = detect_splits(prices)
        assert "A" in detected
        assert_allclose(detected["A"][0][1], 3.0)

    def test_detects_reverse_split(self, dates) -> None:
        prices = pd.DataFrame(
            {"A": [50.0, 51.0, 52.0, 104.0, 105.0, 106.0]},
            index=dates,
        )
        detected = detect_splits(prices)
        assert "A" in detected
        assert_allclose(detected["A"][0][1], 0.5)

    def test_no_split_detected(self, dates) -> None:
        prices = pd.DataFrame(
            {"A": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]},
            index=dates,
        )
        detected = detect_splits(prices)
        assert len(detected) == 0

    def test_custom_threshold(self, dates) -> None:
        prices = pd.DataFrame(
            {"A": [100.0, 80.0, 81.0, 82.0, 83.0, 84.0]},
            index=dates,
        )
        detected = detect_splits(prices, threshold=0.5)
        assert len(detected) == 0

    def test_array_input(self) -> None:
        prices = np.array(
            [
                [100.0, 200.0],
                [102.0, 202.0],
                [51.0, 204.0],
                [52.0, 206.0],
            ]
        )
        detected = detect_splits(prices)
        assert len(detected) == 1
        assert detected[0][1] == 0  # asset index 0
        assert_allclose(detected[0][2], 2.0)

    def test_nan_handling(self, dates) -> None:
        prices = pd.DataFrame(
            {"A": [100.0, np.nan, 104.0, 52.0, 53.0, 54.0]},
            index=dates,
        )
        detected = detect_splits(prices)
        assert "A" in detected

    def test_custom_common_ratios(self, dates) -> None:
        prices = pd.DataFrame(
            {"A": [100.0, 102.0, 104.0, 20.8, 21.0, 22.0]},
            index=dates,
        )
        detected = detect_splits(prices, common_ratios=(5.0,))
        assert "A" in detected
        assert_allclose(detected["A"][0][1], 5.0)

    def test_invalid_threshold_raises(self, dates) -> None:
        prices = pd.DataFrame({"A": [100.0] * 6}, index=dates)
        with pytest.raises(SPTInvariantError, match="threshold"):
            detect_splits(prices, threshold=-0.1)
