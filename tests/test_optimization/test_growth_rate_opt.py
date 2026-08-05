"""Tests for optimization/growth_rate -- growth rate maximisation.

Validates that the optimiser finds the correct analytical solutions
for known cases, respects constraints, and handles edge cases.

Mathematical References
-----------------------
- Portfolio growth rate: F&K Survey Eq. 1.12-1.13
- Excess growth rate: FKK Eq. 2.8
- Equal-weight optimality for diagonal equal-variance: BFK Eq. 5.14
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt.errors import InfeasibleError
from quantspt.optimization.growth_rate import optimize_growth_rate


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# =========================================================================
# A. Known Analytical Solutions
# =========================================================================


class TestKnownSolutions:
    """Verify optimiser against known analytical results."""

    def test_equal_weight_optimal_for_equal_variance_diagonal(self) -> None:
        """For equal growth rates and diagonal equal-variance covariance,
        equal weights maximise the growth rate (BFK Eq. 5.14).

        When gamma_i = gamma for all i and a = sigma^2 * I, the excess
        growth rate is maximised at pi = (1/n, ..., 1/n).
        """
        n = 5
        gamma = np.zeros(n)
        sigma_sq = 0.04
        cov = sigma_sq * np.eye(n)

        result = optimize_growth_rate(gamma, cov)
        expected = np.ones(n) / n
        assert_allclose(result["weights"], expected, atol=1e-4)

    def test_growth_rate_value_at_optimum(self) -> None:
        """Verify the achieved growth rate matches manual computation."""
        gamma = np.array([0.05, 0.08, 0.03])
        cov = np.diag([0.04, 0.09, 0.01])

        result = optimize_growth_rate(gamma, cov)
        w = result["weights"]

        weighted_g = float(np.dot(w, gamma))
        excess_g = 0.5 * (float(np.dot(w, np.diag(cov))) - float(w @ cov @ w))
        expected = weighted_g + excess_g

        assert_allclose(result["growth_rate"], expected, atol=1e-8)
        assert_allclose(result["excess_growth_rate"], excess_g, atol=1e-8)

    def test_concentrated_optimal_for_dominant_growth(self) -> None:
        """When one stock has much higher growth rate, it gets high weight."""
        gamma = np.array([0.50, 0.01, 0.01])
        cov = np.diag([0.04, 0.04, 0.04])

        result = optimize_growth_rate(gamma, cov)
        assert result["weights"][0] > 0.5

    def test_excess_growth_rate_non_negative(self) -> None:
        """Excess growth rate must be non-negative for long-only."""
        gamma = np.array([0.05, 0.08, 0.03])
        cov = np.diag([0.04, 0.09, 0.01])

        result = optimize_growth_rate(gamma, cov)
        assert result["excess_growth_rate"] >= -1e-8


# =========================================================================
# B. Constraint Satisfaction
# =========================================================================


class TestConstraintSatisfaction:
    """Verify all constraints are respected in the solution."""

    def test_weights_sum_to_one(self) -> None:
        gamma = np.array([0.05, 0.08, 0.03])
        cov = np.diag([0.04, 0.09, 0.01])
        result = optimize_growth_rate(gamma, cov)
        assert_allclose(np.sum(result["weights"]), 1.0, atol=1e-6)

    def test_long_only_default(self) -> None:
        """Default min_weight=0 enforces long-only."""
        gamma = np.array([0.05, -0.10, 0.03])
        cov = np.diag([0.04, 0.09, 0.01])
        result = optimize_growth_rate(gamma, cov)
        assert np.all(result["weights"] >= -1e-8)

    def test_max_weight_respected(self) -> None:
        """max_weight constraint must be respected."""
        gamma = np.array([0.50, 0.01, 0.01])
        cov = np.diag([0.04, 0.04, 0.04])
        result = optimize_growth_rate(gamma, cov, max_weight=0.5)
        assert np.all(result["weights"] <= 0.5 + 1e-6)

    def test_min_weight_respected(self) -> None:
        """min_weight > 0 forces all assets to have positive weight."""
        gamma = np.array([0.10, -0.05, 0.01])
        cov = np.diag([0.04, 0.09, 0.01])
        result = optimize_growth_rate(gamma, cov, min_weight=0.05)
        assert np.all(result["weights"] >= 0.05 - 1e-6)

    def test_turnover_constraint(self) -> None:
        """Turnover constraint limits distance from previous weights."""
        gamma = np.array([0.10, 0.05, 0.03])
        cov = np.diag([0.04, 0.04, 0.04])
        prev = np.array([0.5, 0.3, 0.2])

        result = optimize_growth_rate(gamma, cov, max_turnover=0.1, prev_weights=prev)
        actual_turnover = 0.5 * float(np.sum(np.abs(result["weights"] - prev)))
        assert actual_turnover <= 0.1 + 1e-4

    def test_tracking_error_constraint(self) -> None:
        """Tracking error constraint keeps portfolio close to benchmark."""
        n = 5
        gamma = np.zeros(n)
        cov = 0.04 * np.eye(n)
        benchmark = np.ones(n) / n

        result = optimize_growth_rate(
            gamma, cov, max_tracking_error=0.01, benchmark=benchmark
        )
        diff = result["weights"] - benchmark
        tracking_error = float(np.sqrt(diff @ cov @ diff))
        assert tracking_error <= 0.01 + 1e-4

    def test_tight_bounds_infeasible(self) -> None:
        """Infeasible problem should raise InfeasibleError."""
        gamma = np.array([0.05, 0.05])
        cov = np.diag([0.04, 0.04])
        with pytest.raises(InfeasibleError):
            optimize_growth_rate(gamma, cov, min_weight=0.6, max_weight=0.6)


# =========================================================================
# C. Status and Edge Cases
# =========================================================================


class TestStatusAndEdgeCases:
    """Test solver status reporting and edge cases."""

    def test_status_optimal(self) -> None:
        gamma = np.array([0.05, 0.08])
        cov = np.diag([0.04, 0.09])
        result = optimize_growth_rate(gamma, cov)
        assert result["status"] in ("optimal", "optimal_inaccurate")

    def test_two_asset_case(self) -> None:
        """Minimum viable problem: n=2."""
        gamma = np.array([0.05, 0.08])
        cov = np.diag([0.04, 0.09])
        result = optimize_growth_rate(gamma, cov)
        assert result["weights"].shape == (2,)
        assert_allclose(np.sum(result["weights"]), 1.0, atol=1e-6)

    def test_correlated_assets(self) -> None:
        """Optimiser should handle correlated assets correctly."""
        gamma = np.array([0.05, 0.08, 0.06])
        cov = np.array(
            [
                [0.04, 0.02, 0.01],
                [0.02, 0.09, 0.03],
                [0.01, 0.03, 0.04],
            ]
        )
        result = optimize_growth_rate(gamma, cov)
        assert_allclose(np.sum(result["weights"]), 1.0, atol=1e-6)
        assert np.all(result["weights"] >= -1e-8)

    def test_zero_growth_rates(self) -> None:
        """With zero growth rates, optimiser maximises excess growth rate."""
        n = 4
        gamma = np.zeros(n)
        cov = 0.04 * np.eye(n)
        result = optimize_growth_rate(gamma, cov)
        expected = np.ones(n) / n
        assert_allclose(result["weights"], expected, atol=1e-4)

    def test_higher_growth_rate_than_any_individual(self) -> None:
        """Portfolio growth rate should exceed weighted average of gamma_i
        due to the excess growth rate (diversification return)."""
        gamma = np.array([0.05, 0.08, 0.03])
        cov = np.diag([0.04, 0.09, 0.01])
        result = optimize_growth_rate(gamma, cov)
        w = result["weights"]
        weighted_avg = float(np.dot(w, gamma))
        assert result["growth_rate"] >= weighted_avg - 1e-8

    def test_shape_mismatch_raises(self) -> None:
        """Mismatched gamma and cov shapes should raise."""
        from quantspt.errors import SPTInvariantError

        gamma = np.array([0.05, 0.08])
        cov = np.eye(3) * 0.04
        with pytest.raises(SPTInvariantError, match="incompatible"):
            optimize_growth_rate(gamma, cov)
