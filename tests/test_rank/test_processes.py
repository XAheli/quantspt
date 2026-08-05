"""Tests for rank/processes.py — ranked capitalisation dynamics.

Validates:
- Ranked capitalizations are sorted descending (BFK Eq. 3.1)
- Rank assignment is consistent with sorting
- Ranked weights from path computation
- Drift coefficients for ranked dynamics
"""

from __future__ import annotations

import numpy as np
import pytest

from quantspt.rank.processes import (
    rank_assignment,
    ranked_capitalizations,
    ranked_capitalizations_path,
    ranked_drift_coefficients,
    ranked_weights_from_path,
)


class TestRankedCapitalizations:
    """Z_{(1)} >= Z_{(2)} >= ... >= Z_{(n)} must hold."""

    def test_descending_order(self) -> None:
        log_caps = np.array([3.0, 5.0, 1.0, 4.0, 2.0])
        ranked = ranked_capitalizations(log_caps)
        assert np.all(np.diff(ranked) <= 0)
        np.testing.assert_array_equal(ranked, [5.0, 4.0, 3.0, 2.0, 1.0])

    def test_preserves_values(self) -> None:
        log_caps = np.array([1.5, 3.2, 0.8])
        ranked = ranked_capitalizations(log_caps)
        np.testing.assert_array_equal(np.sort(ranked), np.sort(log_caps))

    def test_equal_values(self) -> None:
        log_caps = np.array([2.0, 2.0, 2.0])
        ranked = ranked_capitalizations(log_caps)
        np.testing.assert_array_equal(ranked, [2.0, 2.0, 2.0])

    def test_two_stocks(self) -> None:
        log_caps = np.array([1.0, 3.0])
        ranked = ranked_capitalizations(log_caps)
        np.testing.assert_array_equal(ranked, [3.0, 1.0])

    def test_rejects_1d_check(self) -> None:
        with pytest.raises(Exception, match="1-D"):
            ranked_capitalizations(np.array([[1.0, 2.0]]))


class TestRankedCapitalizationsPath:
    """Path version must sort each row independently."""

    def test_path_sorting(self) -> None:
        path = np.array(
            [
                [3.0, 1.0, 2.0],
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 1.0],
            ]
        )
        ranked = ranked_capitalizations_path(path)
        for t in range(3):
            assert np.all(np.diff(ranked[t]) <= 0)
            np.testing.assert_array_equal(ranked[t], [3.0, 2.0, 1.0])

    def test_path_shape(self) -> None:
        path = np.random.default_rng(42).standard_normal((100, 5))
        ranked = ranked_capitalizations_path(path)
        assert ranked.shape == (100, 5)


class TestRankAssignment:
    """Rank 0 = largest, rank n-1 = smallest."""

    def test_basic_ranking(self) -> None:
        values = np.array([30.0, 50.0, 10.0, 40.0, 20.0])
        ranks = rank_assignment(values)
        assert ranks[1] == 0  # 50 is largest
        assert ranks[3] == 1  # 40 is second
        assert ranks[0] == 2  # 30 is third
        assert ranks[4] == 3  # 20 is fourth
        assert ranks[2] == 4  # 10 is smallest

    def test_rank_permutation_reconstructs_sorted(self) -> None:
        values = np.array([3.0, 1.0, 4.0, 1.5, 2.0])
        ranks = rank_assignment(values)
        order = np.argsort(ranks)
        sorted_vals = values[order]
        assert np.all(np.diff(sorted_vals) <= 0)

    def test_all_ranks_present(self) -> None:
        n = 10
        values = np.random.default_rng(42).standard_normal(n)
        ranks = rank_assignment(values)
        np.testing.assert_array_equal(np.sort(ranks), np.arange(n))


class TestRankedDriftCoefficients:
    """Drift for ranked dynamics: γ + g_k (BFK Eq. 3.3)."""

    def test_atlas_drift(self) -> None:
        n = 5
        g_param = 0.01
        gamma = 0.05
        g = np.full(n, -g_param)
        g[-1] = (n - 1) * g_param
        drift = ranked_drift_coefficients(g, gamma)
        expected = gamma + g
        np.testing.assert_allclose(drift, expected)

    def test_largest_stock_lowest_drift_atlas(self) -> None:
        g = np.array([-0.02, -0.02, -0.02, 0.06])
        gamma = 0.05
        drift = ranked_drift_coefficients(g, gamma)
        assert drift[-1] > drift[0]


class TestRankedWeightsFromPath:
    """Convert log-cap paths to ranked market weights."""

    def test_weights_sum_to_one(self) -> None:
        rng = np.random.default_rng(42)
        path = rng.standard_normal((50, 5)) + 4.0
        ranked_w = ranked_weights_from_path(path)
        for t in range(50):
            np.testing.assert_allclose(ranked_w[t].sum(), 1.0, atol=1e-12)

    def test_weights_descending(self) -> None:
        rng = np.random.default_rng(42)
        path = rng.standard_normal((50, 5)) + 4.0
        ranked_w = ranked_weights_from_path(path)
        for t in range(50):
            assert np.all(np.diff(ranked_w[t]) <= 1e-14)

    def test_weights_positive(self) -> None:
        rng = np.random.default_rng(42)
        path = rng.standard_normal((50, 5)) + 4.0
        ranked_w = ranked_weights_from_path(path)
        assert np.all(ranked_w > 0)

    def test_monotone_capital_distribution(self) -> None:
        path = np.array([[5.0, 3.0, 1.0], [4.0, 4.0, 2.0]])
        ranked_w = ranked_weights_from_path(path)
        for t in range(2):
            assert ranked_w[t, 0] >= ranked_w[t, 1] >= ranked_w[t, 2]
