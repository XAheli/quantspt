"""Tests for estimation/diversity -- diversity parameter estimation.

Validates rolling diversity deficit computation, parameter estimation,
and bootstrap confidence intervals against known analytical values.

Mathematical References
-----------------------
- Diversity deficit: FKK Eq. 4.2
- Weak diversity condition: FKK Eq. 4.2
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt.errors import SPTInvariantError
from quantspt.estimation.diversity import (
    bootstrap_diversity_ci,
    estimate_diversity_parameter,
    rolling_diversity_deficit,
)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# =========================================================================
# A. Rolling Diversity Deficit
# =========================================================================


class TestRollingDiversityDeficit:
    """Tests for rolling_diversity_deficit()."""

    def test_equal_weights_known_deficit(self) -> None:
        """For uniform weights 1/n, deficit = n^{1-p} - 1."""
        n = 5
        p = 0.5
        weights = np.ones((10, n)) / n
        deficits = rolling_diversity_deficit(weights, p)
        expected = n ** (1 - p) - 1.0
        assert_allclose(deficits, expected, atol=1e-14)

    def test_concentrated_weights_zero_deficit(self) -> None:
        """For single-stock concentration, deficit = 0."""
        p = 0.5
        weights = np.array([[1.0, 0.0, 0.0]] * 5)
        weights[:, 1:] = 1e-15
        weights[:, 0] = 1.0 - 2e-15
        deficits = rolling_diversity_deficit(weights, p)
        assert_allclose(deficits, 0.0, atol=1e-6)

    def test_output_shape(self) -> None:
        weights = np.ones((20, 3)) / 3
        deficits = rolling_diversity_deficit(weights, 0.5)
        assert deficits.shape == (20,)

    def test_deficit_positive_for_diverse_market(self) -> None:
        """Diverse markets should have positive deficits."""
        weights = np.ones((10, 10)) / 10
        deficits = rolling_diversity_deficit(weights, 0.5)
        assert np.all(deficits > 0)

    def test_rejects_invalid_p(self) -> None:
        weights = np.ones((5, 3)) / 3
        with pytest.raises(SPTInvariantError):
            rolling_diversity_deficit(weights, 0.0)
        with pytest.raises(SPTInvariantError):
            rolling_diversity_deficit(weights, 1.0)

    def test_rejects_1d_input(self) -> None:
        with pytest.raises(SPTInvariantError, match="2-D"):
            rolling_diversity_deficit(np.array([0.5, 0.5]), 0.5)

    def test_monotone_in_p(self) -> None:
        """Deficit should decrease as p increases toward 1."""
        weights = np.ones((10, 5)) / 5
        d_low = rolling_diversity_deficit(weights, 0.3)
        d_high = rolling_diversity_deficit(weights, 0.8)
        assert np.all(d_low > d_high)


# =========================================================================
# B. Estimate Diversity Parameter
# =========================================================================


class TestEstimateDiversityParameter:
    """Tests for estimate_diversity_parameter()."""

    def test_uniform_weights_delta_positive(self) -> None:
        """Uniform weights should yield positive delta."""
        weights = np.ones((100, 10)) / 10
        result = estimate_diversity_parameter(weights, 0.5)
        assert result["delta"] > 0
        assert result["is_weakly_diverse"] == 1.0

    def test_concentrated_weights_not_diverse(self) -> None:
        """Near-concentrated weights should not be diverse."""
        n = 5
        weights = np.zeros((100, n))
        weights[:, 0] = 0.99
        for i in range(1, n):
            weights[:, i] = 0.01 / (n - 1)
        result = estimate_diversity_parameter(weights, 0.5)
        assert result["min_deficit"] > -0.01

    def test_mean_deficit_equals_analytical(self) -> None:
        """For constant weights, mean deficit = n^{1-p} - 1."""
        n = 8
        p = 0.5
        weights = np.ones((50, n)) / n
        result = estimate_diversity_parameter(weights, p)
        expected = n ** (1 - p) - 1.0
        assert_allclose(result["mean_deficit"], expected, atol=1e-14)

    def test_quantile_level(self, rng: np.random.Generator) -> None:
        """Delta at q=0.5 should be higher than at q=0.05."""
        weights = rng.dirichlet(np.ones(5), size=200)
        d_05 = estimate_diversity_parameter(weights, 0.5, quantile=0.05)
        d_50 = estimate_diversity_parameter(weights, 0.5, quantile=0.50)
        assert d_50["delta"] >= d_05["delta"]


# =========================================================================
# C. Bootstrap Confidence Intervals
# =========================================================================


class TestBootstrapDiversityCI:
    """Tests for bootstrap_diversity_ci()."""

    def test_ci_contains_point_estimate(self) -> None:
        """CI should contain the point estimate for stable data."""
        weights = np.ones((200, 5)) / 5
        ci = bootstrap_diversity_ci(
            weights, 0.5, n_bootstrap=500, confidence=0.95, seed=42
        )
        point = estimate_diversity_parameter(weights, 0.5)
        assert ci["ci_lower"] <= point["delta"] <= ci["ci_upper"]

    def test_ci_ordering(self, rng: np.random.Generator) -> None:
        """Lower bound < mean < upper bound."""
        weights = rng.dirichlet(np.ones(5), size=200)
        ci = bootstrap_diversity_ci(weights, 0.5, n_bootstrap=300, seed=42)
        assert ci["ci_lower"] <= ci["delta_mean"] <= ci["ci_upper"]

    def test_wider_ci_for_higher_confidence(self, rng: np.random.Generator) -> None:
        """99% CI should be wider than 90% CI."""
        weights = rng.dirichlet(np.ones(5), size=200)
        ci_90 = bootstrap_diversity_ci(
            weights, 0.5, n_bootstrap=500, confidence=0.90, seed=42
        )
        ci_99 = bootstrap_diversity_ci(
            weights, 0.5, n_bootstrap=500, confidence=0.99, seed=42
        )
        width_90 = ci_90["ci_upper"] - ci_90["ci_lower"]
        width_99 = ci_99["ci_upper"] - ci_99["ci_lower"]
        assert width_99 >= width_90

    def test_reproducible_with_seed(self, rng: np.random.Generator) -> None:
        weights = rng.dirichlet(np.ones(3), size=100)
        ci1 = bootstrap_diversity_ci(weights, 0.5, seed=123)
        ci2 = bootstrap_diversity_ci(weights, 0.5, seed=123)
        assert_allclose(ci1["delta_mean"], ci2["delta_mean"])

    def test_rejects_invalid_p(self) -> None:
        weights = np.ones((50, 3)) / 3
        with pytest.raises(SPTInvariantError):
            bootstrap_diversity_ci(weights, 1.5)

    def test_rejects_invalid_confidence(self) -> None:
        weights = np.ones((50, 3)) / 3
        with pytest.raises(SPTInvariantError):
            bootstrap_diversity_ci(weights, 0.5, confidence=1.5)
