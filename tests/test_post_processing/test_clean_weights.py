"""Comprehensive tests for post_processing/clean_weights.py.

Covers clean_weights, round_weights, and enforce_bounds with:
- Normal operation and simplex constraint preservation
- Tiny weights zeroed, result sums to 1
- Rounding preserves sum-to-1 within tolerance
- Clipping and renormalization works
- Edge cases: all weights below cutoff, single weight = 1.0, negatives
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt.errors import SPTInvariantError
from quantspt.post_processing.clean_weights import (
    clean_weights,
    enforce_bounds,
    round_weights,
)

# ---------------------------------------------------------------------------
# clean_weights tests
# ---------------------------------------------------------------------------


class TestCleanWeights:
    """Tests for clean_weights: zero negligible weights and renormalise."""

    def test_basic_cleanup(self) -> None:
        """Tiny weights are zeroed and result sums to 1."""
        w = np.array([0.50, 0.30, 0.15, 0.04, 1e-6])
        result = clean_weights(w)
        assert result[4] == 0.0
        assert_allclose(result.sum(), 1.0, atol=1e-10)

    def test_all_weights_above_cutoff_unchanged(self) -> None:
        """If all weights are above cutoff, just renormalise."""
        w = np.array([0.5, 0.3, 0.2])
        result = clean_weights(w, cutoff=0.01)
        assert_allclose(result.sum(), 1.0, atol=1e-10)
        assert np.all(result > 0)

    def test_custom_cutoff(self) -> None:
        """Custom cutoff properly filters."""
        w = np.array([0.50, 0.30, 0.10, 0.09, 0.01])
        result = clean_weights(w, cutoff=0.05)
        assert result[4] == 0.0
        assert result[3] >= 0.05
        assert_allclose(result.sum(), 1.0, atol=1e-10)

    def test_single_weight_one(self) -> None:
        """Single weight = 1.0 passes through."""
        w = np.array([1.0])
        result = clean_weights(w)
        assert_allclose(result, [1.0], atol=1e-10)

    def test_two_weights_one_tiny(self) -> None:
        """Two weights, one tiny — reduced to single weight."""
        w = np.array([0.9999, 0.0001])
        result = clean_weights(w, cutoff=0.001)
        assert result[1] == 0.0
        assert_allclose(result[0], 1.0, atol=1e-10)
        assert_allclose(result.sum(), 1.0, atol=1e-10)

    def test_equal_weights(self) -> None:
        """Equal weights are preserved."""
        w = np.ones(5) / 5.0
        result = clean_weights(w)
        assert_allclose(result, w, atol=1e-10)
        assert_allclose(result.sum(), 1.0, atol=1e-10)

    def test_all_below_cutoff_raises(self) -> None:
        """If all weights fall below cutoff, cannot renormalise."""
        w = np.array([1e-6, 1e-7, 1e-8])
        with pytest.raises(SPTInvariantError, match="below the cutoff"):
            clean_weights(w, cutoff=0.01)

    def test_negative_weights_raises(self) -> None:
        """Negative weights raise SPTInvariantError."""
        w = np.array([0.5, 0.3, -0.2, 0.4])
        with pytest.raises(SPTInvariantError, match="non-negative"):
            clean_weights(w)

    def test_negative_cutoff_raises(self) -> None:
        """Negative cutoff raises SPTInvariantError."""
        w = np.array([0.5, 0.5])
        with pytest.raises(SPTInvariantError, match="non-negative"):
            clean_weights(w, cutoff=-0.01)

    def test_not_1d_raises(self) -> None:
        """2-D input raises SPTInvariantError."""
        w = np.array([[0.5, 0.5]])
        with pytest.raises(SPTInvariantError, match="1-D"):
            clean_weights(w)

    def test_cutoff_zero_preserves_all(self) -> None:
        """cutoff=0 keeps all non-zero weights."""
        w = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
        result = clean_weights(w, cutoff=0.0)
        assert np.all(result > 0)
        assert_allclose(result.sum(), 1.0, atol=1e-10)

    def test_large_portfolio(self) -> None:
        """Works correctly with 100 assets."""
        rng = np.random.default_rng(42)
        w = rng.dirichlet(np.ones(100))
        result = clean_weights(w, cutoff=0.005)
        assert_allclose(result.sum(), 1.0, atol=1e-10)
        assert np.all(result >= 0)

    def test_near_zero_tolerance(self) -> None:
        """Tiny negatives within -1e-10 tolerance pass through."""
        w = np.array([0.5, 0.3, 0.2 - 1e-11])
        result = clean_weights(w)
        assert_allclose(result.sum(), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# round_weights tests
# ---------------------------------------------------------------------------


class TestRoundWeights:
    """Tests for round_weights: round and renormalise."""

    def test_basic_rounding(self) -> None:
        """Rounded weights still sum to 1."""
        w = np.array([0.33333, 0.33333, 0.33334])
        result = round_weights(w, decimals=3)
        assert_allclose(result.sum(), 1.0, atol=1e-10)

    def test_preserves_zero_decimals(self) -> None:
        """decimals=0 rounds to integers (effectively all go to 0 or 1)."""
        w = np.array([0.6, 0.4])
        result = round_weights(w, decimals=0)
        assert_allclose(result.sum(), 1.0, atol=1e-10)

    def test_high_precision(self) -> None:
        """decimals=8 preserves most precision."""
        w = np.array([0.123456789, 0.876543211])
        result = round_weights(w, decimals=8)
        assert_allclose(result, w, atol=1e-7)
        assert_allclose(result.sum(), 1.0, atol=1e-10)

    def test_negative_decimals_raises(self) -> None:
        """Negative decimals raises."""
        w = np.array([0.5, 0.5])
        with pytest.raises(SPTInvariantError, match="non-negative"):
            round_weights(w, decimals=-1)

    def test_not_1d_raises(self) -> None:
        """2-D input raises."""
        w = np.array([[0.5, 0.5]])
        with pytest.raises(SPTInvariantError, match="1-D"):
            round_weights(w)

    def test_equal_weights_roundtrip(self) -> None:
        """Equal weights round to equal values."""
        w = np.ones(4) / 4.0
        result = round_weights(w, decimals=4)
        assert_allclose(result.sum(), 1.0, atol=1e-10)
        assert np.all(np.abs(result - result[0]) < 1e-10)

    def test_all_zero_returns_zero(self) -> None:
        """All-zero input returns all-zero (degenerate but handled)."""
        w = np.array([0.0001, 0.0001, 0.0001])
        result = round_weights(w, decimals=2)
        assert_allclose(result.sum(), 0.0, atol=1e-10)

    def test_single_weight(self) -> None:
        """Single weight rounds and stays at 1."""
        w = np.array([1.0])
        result = round_weights(w, decimals=2)
        assert_allclose(result, [1.0], atol=1e-10)


# ---------------------------------------------------------------------------
# enforce_bounds tests
# ---------------------------------------------------------------------------


class TestEnforceBounds:
    """Tests for enforce_bounds: clip and renormalise."""

    def test_basic_clipping(self) -> None:
        """Upper bound is enforced and weights stay on the simplex.

        A single clip-and-rescale would give ``[0.556, 0.333, 0.111]`` and
        violate the 0.5 cap; the bounded renormalisation pins the first weight
        at the cap and splits the freed budget in the original 3:1 ratio.
        """
        w = np.array([0.6, 0.3, 0.1])
        result = enforce_bounds(w, lower=0.0, upper=0.5)
        assert_allclose(result.sum(), 1.0, atol=1e-10)
        assert np.all(result >= 0)
        assert np.all(result <= 0.5 + 1e-10)
        assert_allclose(result, [0.5, 0.375, 0.125], atol=1e-10)

    def test_lower_bound_clipping(self) -> None:
        """Lower bound is enforced; the freed budget goes to the free weights.

        ``w[2] = 0.05`` sits below the 0.10 floor, so it is pinned there and
        the remaining 0.9 is split between the other two in their original
        16:3 ratio.
        """
        w = np.array([0.80, 0.15, 0.05])
        result = enforce_bounds(w, lower=0.10, upper=1.0)
        assert_allclose(result.sum(), 1.0, atol=1e-10)
        assert np.all(result >= 0.10 - 1e-10)
        assert_allclose(result, [0.9 * 16 / 19, 0.9 * 3 / 19, 0.1], atol=1e-10)

    def test_default_bounds_passthrough(self) -> None:
        """Default bounds [0, 1] don't change valid weights."""
        w = np.array([0.4, 0.35, 0.25])
        result = enforce_bounds(w)
        assert_allclose(result, w, atol=1e-10)
        assert_allclose(result.sum(), 1.0, atol=1e-10)

    def test_tight_bounds(self) -> None:
        """Tight bounds [0.2, 0.4] are both enforced.

        ``w[0] = 0.5`` pins at the 0.4 cap; the remaining 0.6 keeps the 3:2
        ratio of the other two.
        """
        w = np.array([0.5, 0.3, 0.2])
        result = enforce_bounds(w, lower=0.2, upper=0.4)
        assert_allclose(result.sum(), 1.0, atol=1e-10)
        assert np.all(result >= 0.2 - 1e-10)
        assert np.all(result <= 0.4 + 1e-10)
        assert_allclose(result, [0.4, 0.36, 0.24], atol=1e-10)

    def test_invalid_bounds_raises(self) -> None:
        """lower > upper raises."""
        w = np.array([0.5, 0.5])
        with pytest.raises(SPTInvariantError, match="must be <="):
            enforce_bounds(w, lower=0.6, upper=0.4)

    def test_negative_lower_raises(self) -> None:
        """Negative lower bound raises."""
        w = np.array([0.5, 0.5])
        with pytest.raises(SPTInvariantError, match="non-negative"):
            enforce_bounds(w, lower=-0.1)

    def test_upper_above_one_raises(self) -> None:
        """Upper bound > 1 raises."""
        w = np.array([0.5, 0.5])
        with pytest.raises(SPTInvariantError, match=r"<= 1\.0"):
            enforce_bounds(w, upper=1.5)

    def test_all_clipped_to_zero_raises(self) -> None:
        """If all weights clip to zero, cannot renormalise."""
        w = np.array([-0.1, -0.2, -0.3])
        with pytest.raises(SPTInvariantError):
            enforce_bounds(w, lower=0.0, upper=1.0)

    def test_not_1d_raises(self) -> None:
        """2-D input raises."""
        w = np.array([[0.5, 0.5]])
        with pytest.raises(SPTInvariantError, match="1-D"):
            enforce_bounds(w)

    def test_equal_weights_within_bounds(self) -> None:
        """Equal-weighted portfolio within bounds passes through."""
        w = np.ones(5) / 5.0
        result = enforce_bounds(w, lower=0.1, upper=0.3)
        assert_allclose(result, w, atol=1e-10)
        assert_allclose(result.sum(), 1.0, atol=1e-10)

    def test_single_weight_at_upper(self) -> None:
        """Single weight = 1.0 with upper = 1.0 passes."""
        w = np.array([1.0])
        result = enforce_bounds(w, lower=0.0, upper=1.0)
        assert_allclose(result, [1.0], atol=1e-10)

    def test_large_portfolio_bounds(self) -> None:
        """100-asset portfolio: bounds enforced, sum preserved, idempotent.

        The single-step expectation ``clip(w)/sum(clip(w))`` violates the
        0.005 floor for 46 of the 100 weights in this sample, so the guarantee
        (rather than the discarded one-step value) is asserted.
        """
        rng = np.random.default_rng(42)
        w = rng.dirichlet(np.ones(100))
        result = enforce_bounds(w, lower=0.005, upper=0.05)
        assert_allclose(result.sum(), 1.0, atol=1e-10)
        assert np.all(result >= 0.005 - 1e-10)
        assert np.all(result <= 0.05 + 1e-10)
        assert_allclose(
            enforce_bounds(result, lower=0.005, upper=0.05), result, atol=1e-12
        )
