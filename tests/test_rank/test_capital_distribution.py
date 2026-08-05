"""Tests for rank/capital_distribution.py.

Validates:
- Capital distribution curve is monotone (descending)
- Log-log fit produces meaningful R² for power-law data
- Pareto exponents are positive for diverse weights
- Stability metric is non-negative
"""

from __future__ import annotations

import numpy as np

from quantspt.rank.capital_distribution import (
    capital_distribution_curve,
    capital_distribution_stability,
    log_log_fit,
    pareto_exponents_empirical,
)


class TestCapitalDistributionCurve:
    """Ranked weights must be monotone descending."""

    def test_descending(self) -> None:
        mu = np.array([0.1, 0.3, 0.15, 0.05, 0.4])
        curve = capital_distribution_curve(mu)
        assert np.all(np.diff(curve) <= 0)

    def test_sum_preserved(self) -> None:
        mu = np.array([0.2, 0.5, 0.3])
        curve = capital_distribution_curve(mu)
        np.testing.assert_allclose(curve.sum(), 1.0, atol=1e-14)

    def test_uniform_weights(self) -> None:
        n = 10
        mu = np.full(n, 1.0 / n)
        curve = capital_distribution_curve(mu)
        np.testing.assert_allclose(curve, 1.0 / n, atol=1e-14)


class TestLogLogFit:
    """Log-log regression on the capital distribution."""

    def test_zipf_law_slope_minus_one(self) -> None:
        n = 100
        ranks = np.arange(1, n + 1, dtype=np.float64)
        weights = 1.0 / ranks
        weights /= weights.sum()
        slope, _, r_sq = log_log_fit(weights)
        np.testing.assert_allclose(slope, -1.0, atol=0.05)
        assert r_sq > 0.99

    def test_uniform_has_near_zero_slope(self) -> None:
        n = 50
        mu = np.full(n, 1.0 / n)
        slope, _, _ = log_log_fit(mu)
        np.testing.assert_allclose(slope, 0.0, atol=1e-10)

    def test_returns_three_values(self) -> None:
        mu = np.array([0.5, 0.3, 0.2])
        result = log_log_fit(mu)
        assert len(result) == 3

    def test_r_squared_bounds(self) -> None:
        rng = np.random.default_rng(42)
        mu = rng.dirichlet(np.ones(20))
        _, _, r_sq = log_log_fit(mu)
        assert 0.0 <= r_sq <= 1.0


class TestParetoExponentsEmpirical:
    """Estimated Pareto exponents from weight ratios."""

    def test_shape(self) -> None:
        mu = np.array([0.4, 0.3, 0.2, 0.1])
        exp = pareto_exponents_empirical(mu)
        assert exp.shape == (3,)

    def test_positive_for_diverse_weights(self) -> None:
        rng = np.random.default_rng(42)
        mu = rng.dirichlet(np.ones(10))
        exp = pareto_exponents_empirical(mu)
        assert np.all(exp > 0)

    def test_inf_for_equal_weights(self) -> None:
        mu = np.array([0.25, 0.25, 0.25, 0.25])
        exp = pareto_exponents_empirical(mu)
        assert np.all(np.isinf(exp))


class TestCapitalDistributionStability:
    """Stability metric for time-varying distributions."""

    def test_nonnegative(self) -> None:
        rng = np.random.default_rng(42)
        paths = rng.dirichlet(np.ones(5), size=50)
        stab = capital_distribution_stability(paths)
        assert stab >= 0.0

    def test_constant_distribution_zero_stability(self) -> None:
        mu = np.array([0.4, 0.3, 0.2, 0.1])
        paths = np.tile(mu, (20, 1))
        stab = capital_distribution_stability(paths)
        np.testing.assert_allclose(stab, 0.0, atol=1e-14)

    def test_variable_has_higher_stability_metric(self) -> None:
        mu = np.array([0.4, 0.3, 0.2, 0.1])
        constant_paths = np.tile(mu, (20, 1))
        rng = np.random.default_rng(42)
        variable_paths = rng.dirichlet(np.ones(4), size=20)
        assert capital_distribution_stability(
            variable_paths
        ) > capital_distribution_stability(constant_paths)
