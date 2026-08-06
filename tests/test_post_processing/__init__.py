"""Tests for post_processing/clean_weights.py."""

from __future__ import annotations

import numpy as np
import pytest

from quantspt.errors import SPTInvariantError
from quantspt.post_processing.clean_weights import (
    clean_weights,
    enforce_bounds,
    round_weights,
)


class TestCleanWeights:
    def test_zeros_below_cutoff(self) -> None:
        weights = np.array([0.5, 0.4, 0.00005, 0.09995])
        result = clean_weights(weights, cutoff=1e-4)
        assert result[2] == 0.0
        assert np.isclose(result.sum(), 1.0)

    def test_preserves_weights_above_cutoff(self) -> None:
        weights = np.array([0.5, 0.3, 0.2])
        result = clean_weights(weights, cutoff=1e-4)
        np.testing.assert_allclose(result, weights, atol=1e-10)

    def test_renormalises_after_cleaning(self) -> None:
        weights = np.array([0.4, 0.3, 0.2, 0.05, 0.04, 0.01])
        result = clean_weights(weights, cutoff=0.02)
        assert result[-1] == 0.0
        assert np.isclose(result.sum(), 1.0)

    def test_custom_cutoff(self) -> None:
        weights = np.array([0.7, 0.2, 0.08, 0.02])
        result = clean_weights(weights, cutoff=0.05)
        assert result[3] == 0.0
        assert np.isclose(result.sum(), 1.0)

    def test_all_below_cutoff_raises(self) -> None:
        weights = np.array([1e-5, 1e-5, 1e-5])
        with pytest.raises(SPTInvariantError, match="below the cutoff"):
            clean_weights(weights, cutoff=1e-4)

    def test_rejects_2d_input(self) -> None:
        with pytest.raises(SPTInvariantError, match="1-D"):
            clean_weights(np.ones((3, 3)) / 9)

    def test_negative_cutoff_raises(self) -> None:
        with pytest.raises(SPTInvariantError, match="non-negative"):
            clean_weights(np.array([0.5, 0.5]), cutoff=-0.1)

    def test_single_asset(self) -> None:
        result = clean_weights(np.array([1.0]))
        assert np.isclose(result[0], 1.0)

    def test_cutoff_zero_keeps_all(self) -> None:
        weights = np.array([0.5, 0.3, 1e-10, 0.2 - 1e-10])
        result = clean_weights(weights, cutoff=0.0)
        assert np.all(result > 0)
        assert np.isclose(result.sum(), 1.0)


class TestRoundWeights:
    def test_sums_to_one(self) -> None:
        rng = np.random.default_rng(42)
        weights = rng.dirichlet(np.ones(10))
        result = round_weights(weights, decimals=4)
        assert np.isclose(result.sum(), 1.0)

    def test_respects_decimals(self) -> None:
        weights = np.array([0.33333, 0.33333, 0.33334])
        result = round_weights(weights, decimals=2)
        for w in result:
            assert round(float(w), 2) == float(w) or np.isclose(
                float(w), round(float(w), 2), atol=1e-10
            )

    def test_already_rounded(self) -> None:
        weights = np.array([0.5, 0.3, 0.2])
        result = round_weights(weights, decimals=1)
        np.testing.assert_allclose(result, weights, atol=1e-10)

    def test_rejects_2d(self) -> None:
        with pytest.raises(SPTInvariantError, match="1-D"):
            round_weights(np.ones((2, 2)) / 4)

    def test_decimals_zero(self) -> None:
        weights = np.array([0.9, 0.1])
        result = round_weights(weights, decimals=0)
        assert np.isclose(result.sum(), 1.0)


class TestEnforceBounds:
    def test_clips_and_renormalises(self) -> None:
        weights = np.array([0.6, 0.3, 0.1])
        result = enforce_bounds(weights, lower=0.0, upper=0.4)
        assert np.all(result <= 0.4 + 1e-10)
        assert np.isclose(result.sum(), 1.0)

    def test_lower_bound(self) -> None:
        weights = np.array([0.8, 0.15, 0.05])
        result = enforce_bounds(weights, lower=0.1, upper=1.0)
        assert np.all(result >= 0.1 - 1e-10)
        assert np.isclose(result.sum(), 1.0)

    def test_no_clipping_needed(self) -> None:
        weights = np.array([0.4, 0.3, 0.3])
        result = enforce_bounds(weights, lower=0.0, upper=1.0)
        np.testing.assert_allclose(result, weights, atol=1e-10)

    def test_invalid_bounds_raises(self) -> None:
        with pytest.raises(SPTInvariantError, match="<="):
            enforce_bounds(np.array([0.5, 0.5]), lower=0.6, upper=0.4)

    def test_upper_exceeds_one_raises(self) -> None:
        with pytest.raises(SPTInvariantError, match="<= 1.0"):
            enforce_bounds(np.array([0.5, 0.5]), upper=1.5)

    def test_negative_lower_raises(self) -> None:
        with pytest.raises(SPTInvariantError, match="non-negative"):
            enforce_bounds(np.array([0.5, 0.5]), lower=-0.1)

    def test_rejects_2d(self) -> None:
        with pytest.raises(SPTInvariantError, match="1-D"):
            enforce_bounds(np.ones((2, 2)) / 4)
