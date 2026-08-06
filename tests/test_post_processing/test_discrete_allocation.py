"""Tests for post_processing/discrete_allocation.py."""

from __future__ import annotations

import numpy as np
import pytest

from quantspt.errors import SPTInvariantError
from quantspt.post_processing.discrete_allocation import (
    AllocationResult,
    greedy_allocation,
    lp_allocation,
)


class TestGreedyAllocation:
    def test_basic_allocation(self) -> None:
        weights = np.array([0.6, 0.4])
        prices = np.array([100.0, 50.0])
        total_value = 10000.0

        result = greedy_allocation(weights, prices, total_value)
        assert isinstance(result, AllocationResult)
        assert result.shares.dtype == np.intp
        assert result.leftover_cash >= 0.0

    def test_budget_not_exceeded(self) -> None:
        weights = np.array([0.5, 0.3, 0.2])
        prices = np.array([150.0, 80.0, 25.0])
        total_value = 50000.0

        result = greedy_allocation(weights, prices, total_value)
        spent = (result.shares.astype(np.float64) * prices).sum()
        assert spent <= total_value + 1e-10

    def test_leftover_within_one_share(self) -> None:
        weights = np.array([0.5, 0.3, 0.2])
        prices = np.array([10.0, 20.0, 5.0])
        total_value = 10000.0

        result = greedy_allocation(weights, prices, total_value)
        assert result.leftover_cash < prices.max()

    def test_actual_weights_close_to_target(self) -> None:
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        prices = np.array([50.0, 30.0, 20.0, 10.0])
        total_value = 100000.0

        result = greedy_allocation(weights, prices, total_value)
        np.testing.assert_allclose(result.actual_weights, weights, atol=0.05)

    def test_single_stock(self) -> None:
        weights = np.array([1.0])
        prices = np.array([33.0])
        total_value = 1000.0

        result = greedy_allocation(weights, prices, total_value)
        expected_shares = int(1000.0 / 33.0)
        assert result.shares[0] == expected_shares

    def test_zero_weight_gets_no_shares(self) -> None:
        weights = np.array([0.5, 0.0, 0.5])
        prices = np.array([10.0, 10.0, 10.0])
        total_value = 1000.0

        result = greedy_allocation(weights, prices, total_value)
        assert result.shares[1] == 0

    def test_expensive_stock_partial(self) -> None:
        weights = np.array([0.5, 0.5])
        prices = np.array([7000.0, 10.0])
        total_value = 10000.0

        result = greedy_allocation(weights, prices, total_value)
        assert result.shares[0] <= 1
        assert result.leftover_cash >= 0

    def test_rejects_negative_weights(self) -> None:
        with pytest.raises(SPTInvariantError, match="non-negative"):
            greedy_allocation(
                np.array([-0.1, 1.1]),
                np.array([10.0, 10.0]),
                1000.0,
            )

    def test_rejects_zero_prices(self) -> None:
        with pytest.raises(SPTInvariantError, match="positive"):
            greedy_allocation(
                np.array([0.5, 0.5]),
                np.array([0.0, 10.0]),
                1000.0,
            )

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(SPTInvariantError, match="same length"):
            greedy_allocation(
                np.array([0.5, 0.5]),
                np.array([10.0, 20.0, 30.0]),
                1000.0,
            )


class TestLPAllocation:
    def test_basic_allocation(self) -> None:
        weights = np.array([0.6, 0.4])
        prices = np.array([100.0, 50.0])
        total_value = 10000.0

        result = lp_allocation(weights, prices, total_value)
        assert isinstance(result, AllocationResult)
        assert result.shares.dtype == np.intp
        assert result.leftover_cash >= 0.0

    def test_budget_not_exceeded(self) -> None:
        weights = np.array([0.5, 0.3, 0.2])
        prices = np.array([150.0, 80.0, 25.0])
        total_value = 50000.0

        result = lp_allocation(weights, prices, total_value)
        spent = (result.shares.astype(np.float64) * prices).sum()
        assert spent <= total_value + 1e-10

    def test_tracking_error_reasonable(self) -> None:
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        prices = np.array([50.0, 30.0, 20.0, 10.0])
        total_value = 100000.0

        result = lp_allocation(weights, prices, total_value)
        np.testing.assert_allclose(result.actual_weights, weights, atol=0.05)

    def test_lp_at_least_as_good_as_greedy(self) -> None:
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        prices = np.array([47.0, 31.0, 23.0, 17.0])
        total_value = 10000.0

        lp_result = lp_allocation(weights, prices, total_value)
        greedy_result = greedy_allocation(weights, prices, total_value)

        lp_error = np.sum(np.abs(lp_result.actual_weights - weights))
        greedy_error = np.sum(np.abs(greedy_result.actual_weights - weights))
        assert lp_error <= greedy_error + 0.01

    def test_rejects_negative_total(self) -> None:
        with pytest.raises(SPTInvariantError, match="positive"):
            lp_allocation(
                np.array([0.5, 0.5]),
                np.array([10.0, 10.0]),
                -1000.0,
            )
