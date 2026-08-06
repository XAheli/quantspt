"""Tests for post_processing/lot_sizing.py."""

from __future__ import annotations

import numpy as np
import pytest

from quantspt.errors import SPTInvariantError
from quantspt.post_processing.lot_sizing import (
    minimum_trade_filter,
    round_to_lots,
)


class TestRoundToLots:
    def test_rounds_down_to_lot_size(self) -> None:
        shares = np.array([150, 250, 350], dtype=np.intp)
        result = round_to_lots(shares, lot_size=100)
        np.testing.assert_array_equal(result, [100, 200, 300])

    def test_already_multiples(self) -> None:
        shares = np.array([200, 300, 500], dtype=np.intp)
        result = round_to_lots(shares, lot_size=100)
        np.testing.assert_array_equal(result, shares)

    def test_all_multiples_of_lot(self) -> None:
        shares = np.array([123, 456, 789, 1011], dtype=np.intp)
        result = round_to_lots(shares, lot_size=50)
        for s in result:
            assert s % 50 == 0

    def test_lot_size_one(self) -> None:
        shares = np.array([7, 13, 42], dtype=np.intp)
        result = round_to_lots(shares, lot_size=1)
        np.testing.assert_array_equal(result, shares)

    def test_below_lot_size_becomes_zero(self) -> None:
        shares = np.array([50, 99, 10], dtype=np.intp)
        result = round_to_lots(shares, lot_size=100)
        np.testing.assert_array_equal(result, [0, 0, 0])

    def test_rejects_invalid_lot_size(self) -> None:
        with pytest.raises(SPTInvariantError, match=">= 1"):
            round_to_lots(np.array([100], dtype=np.intp), lot_size=0)

    def test_rejects_2d(self) -> None:
        with pytest.raises(SPTInvariantError, match="1-D"):
            round_to_lots(np.ones((2, 2), dtype=np.intp))

    def test_large_lot_size(self) -> None:
        shares = np.array([1000, 500, 1500], dtype=np.intp)
        result = round_to_lots(shares, lot_size=1000)
        np.testing.assert_array_equal(result, [1000, 0, 1000])


class TestMinimumTradeFilter:
    def test_filters_small_trades(self) -> None:
        current = np.array([100, 200, 300], dtype=np.intp)
        target = np.array([101, 250, 300], dtype=np.intp)
        result = minimum_trade_filter(current, target, min_trade=5)
        # trade of 1 filtered, trade of 50 kept, no trade unchanged
        assert result[0] == 100  # filtered (diff=1 < 5)
        assert result[1] == 250  # kept (diff=50 >= 5)
        assert result[2] == 300  # unchanged

    def test_no_filtering_needed(self) -> None:
        current = np.array([100, 200], dtype=np.intp)
        target = np.array([200, 100], dtype=np.intp)
        result = minimum_trade_filter(current, target, min_trade=1)
        np.testing.assert_array_equal(result, target)

    def test_all_filtered(self) -> None:
        current = np.array([100, 200, 300], dtype=np.intp)
        target = np.array([101, 201, 302], dtype=np.intp)
        result = minimum_trade_filter(current, target, min_trade=10)
        np.testing.assert_array_equal(result, current)

    def test_sells_filtered_too(self) -> None:
        current = np.array([100, 200], dtype=np.intp)
        target = np.array([98, 150], dtype=np.intp)
        result = minimum_trade_filter(current, target, min_trade=5)
        assert result[0] == 100  # sell of 2 filtered
        assert result[1] == 150  # sell of 50 kept

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(SPTInvariantError, match="same length"):
            minimum_trade_filter(
                np.array([100], dtype=np.intp),
                np.array([100, 200], dtype=np.intp),
            )

    def test_rejects_invalid_min_trade(self) -> None:
        with pytest.raises(SPTInvariantError, match=">= 1"):
            minimum_trade_filter(
                np.array([100], dtype=np.intp),
                np.array([200], dtype=np.intp),
                min_trade=0,
            )
