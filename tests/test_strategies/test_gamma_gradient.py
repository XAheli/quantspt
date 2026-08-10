"""Test suite for GammaGradientStrategy — mathematical correctness.

Tests cover:
1. Gradient correctness (analytical vs finite differences)
2. γ* non-negativity for long-only portfolios
3. Gradient properties at special points
4. Simplex projection correctness
5. Edge cases (single stock, identical stocks, ill-conditioned cov)
6. Input validation
"""

import numpy as np
import pytest

from quantspt.strategies import (
    GammaGradientStrategy,
    project_bounded_simplex,
    project_simplex,
)


class TestGradientCorrectness:
    """Verify the analytical gradient matches finite differences."""

    def test_gradient_matches_finite_differences(self):
        """∂γ*/∂π_i computed analytically must match numerical gradient."""
        np.random.seed(42)
        n = 10
        mu = np.random.dirichlet(np.ones(n))
        A = np.random.randn(n, n) * 0.1
        cov = A @ A.T + 0.01 * np.eye(n)

        strategy = GammaGradientStrategy(lambda_scale=0.1, max_weight=0.5)
        analytical_grad = strategy.compute_gradient(mu, cov)

        # Use directional derivative: perturb π_i without renormalization
        # ∂γ*/∂π_i is the unconstrained partial derivative
        eps = 1e-7
        numerical_grad = np.zeros(n)
        for i in range(n):
            mu_plus = mu.copy()
            mu_plus[i] += eps
            # Don't renormalize — we want the raw partial derivative

            gamma_plus = strategy.gamma_star(mu_plus, cov)
            gamma_base = strategy.gamma_star(mu, cov)
            numerical_grad[i] = (gamma_plus - gamma_base) / eps

        np.testing.assert_allclose(
            analytical_grad, numerical_grad, atol=1e-5, rtol=1e-4
        )

    def test_gradient_formula_explicit(self):
        """Verify ∂γ*/∂π_i = ½(a_{ii} - 2·(aμ)_i) with hand computation."""
        mu = np.array([0.5, 0.3, 0.2])
        cov = np.array(
            [
                [0.04, 0.01, 0.005],
                [0.01, 0.09, 0.02],
                [0.005, 0.02, 0.16],
            ]
        )

        strategy = GammaGradientStrategy()
        grad = strategy.compute_gradient(mu, cov)

        a_mu = cov @ mu  # [0.5*0.04+0.3*0.01+0.2*0.005, ...]
        expected = 0.5 * (np.diag(cov) - 2 * a_mu)
        np.testing.assert_allclose(grad, expected, atol=1e-15)


class TestGammaStarProperties:
    """Verify mathematical properties of the excess growth rate."""

    def test_gamma_star_non_negative_long_only(self):
        """γ* ≥ 0 for any long-only portfolio (Jensen's inequality)."""
        np.random.seed(123)
        for _ in range(100):
            n = np.random.randint(2, 50)
            mu = np.random.dirichlet(np.ones(n))
            A = np.random.randn(n, n) * 0.1
            cov = A @ A.T + 0.01 * np.eye(n)

            strategy = GammaGradientStrategy()
            gamma = strategy.gamma_star(mu, cov)
            assert gamma >= -1e-12, f"gamma* = {gamma} < 0 for long-only portfolio"

    def test_gamma_star_zero_for_single_stock(self):
        """γ* = 0 for a single-stock portfolio (no diversification)."""
        n = 5
        cov = np.eye(n) * 0.04
        for i in range(n):
            pi = np.zeros(n)
            pi[i] = 1.0
            strategy = GammaGradientStrategy()
            gamma = strategy.gamma_star(pi, cov)
            assert abs(gamma) < 1e-14

    def test_gamma_star_maximized_at_equal_weight_diagonal_cov(self):
        """For diagonal cov with equal variances, γ* maximized at equal weight."""
        n = 10
        sigma_sq = 0.04
        cov = sigma_sq * np.eye(n)
        strategy = GammaGradientStrategy()

        gamma_ew = strategy.gamma_star(np.ones(n) / n, cov)

        np.random.seed(99)
        for _ in range(50):
            pi = np.random.dirichlet(np.ones(n))
            gamma_random = strategy.gamma_star(pi, cov)
            assert gamma_ew >= gamma_random - 1e-12

    def test_strategy_improves_gamma_star(self):
        """Strategy weights should have higher γ* than market weights."""
        np.random.seed(7)
        n = 20
        mu = np.random.dirichlet(np.ones(n) * 0.5)  # concentrated market
        A = np.random.randn(n, n) * 0.1
        cov = A @ A.T + 0.02 * np.eye(n)

        strategy = GammaGradientStrategy(lambda_scale=0.1, max_weight=0.15)
        weights = strategy.compute_weights(mu, cov)

        gamma_mkt = strategy.gamma_star(mu, cov)
        gamma_strat = strategy.gamma_star(weights, cov)
        assert gamma_strat > gamma_mkt


class TestWeightConstraints:
    """Verify all weight constraints are satisfied."""

    def test_weights_sum_to_one(self):
        """Output weights must always sum to 1."""
        np.random.seed(42)
        for _ in range(50):
            n = np.random.randint(20, 100)
            mu = np.random.dirichlet(np.ones(n))
            A = np.random.randn(n, n) * 0.1
            cov = A @ A.T + 0.01 * np.eye(n)

            max_w = max(0.1, 2.0 / n)  # ensure feasibility
            strategy = GammaGradientStrategy(lambda_scale=0.2, max_weight=max_w)
            weights = strategy.compute_weights(mu, cov)
            assert abs(weights.sum() - 1.0) < 1e-8

    def test_weights_non_negative(self):
        """All weights must be ≥ 0."""
        np.random.seed(42)
        for _ in range(50):
            n = np.random.randint(20, 100)
            mu = np.random.dirichlet(np.ones(n))
            A = np.random.randn(n, n) * 0.1
            cov = A @ A.T + 0.01 * np.eye(n)

            max_w = max(0.1, 2.0 / n)
            strategy = GammaGradientStrategy(lambda_scale=0.5, max_weight=max_w)
            weights = strategy.compute_weights(mu, cov)
            assert np.all(weights >= -1e-10)

    def test_max_weight_respected(self):
        """No weight exceeds max_weight."""
        np.random.seed(42)
        n = 50
        mu = np.random.dirichlet(np.ones(n))
        A = np.random.randn(n, n) * 0.1
        cov = A @ A.T + 0.01 * np.eye(n)

        for max_w in [0.03, 0.05, 0.10, 0.20, 0.50]:
            strategy = GammaGradientStrategy(lambda_scale=0.3, max_weight=max_w)
            weights = strategy.compute_weights(mu, cov)
            assert weights.max() <= max_w + 1e-8

    def test_lambda_zero_gives_market_weights(self):
        """With λ=0, strategy returns (projected) market weights."""
        n = 10
        mu = np.random.dirichlet(np.ones(n))
        cov = np.eye(n) * 0.04

        strategy = GammaGradientStrategy(lambda_scale=0.0, max_weight=1.0)
        weights = strategy.compute_weights(mu, cov)
        np.testing.assert_allclose(weights, mu, atol=1e-8)


class TestSimplexProjection:
    """Test simplex projection utilities."""

    def test_project_simplex_already_on_simplex(self):
        """Points already on simplex should be unchanged."""
        v = np.array([0.3, 0.5, 0.2])
        result = project_simplex(v)
        np.testing.assert_allclose(result, v, atol=1e-12)

    def test_project_simplex_negative_entries(self):
        """Negative entries should be clipped to zero."""
        v = np.array([0.8, 0.5, -0.3])
        result = project_simplex(v)
        assert np.all(result >= -1e-12)
        assert abs(result.sum() - 1.0) < 1e-10

    def test_project_simplex_large_entries(self):
        """Large entries should be reduced."""
        v = np.array([2.0, 0.5, 0.5])
        result = project_simplex(v)
        assert abs(result.sum() - 1.0) < 1e-10
        assert np.all(result >= -1e-12)

    def test_bounded_simplex_max_weight(self):
        """Max weight constraint must be satisfied."""
        v = np.array([0.8, 0.1, 0.05, 0.05])
        result = project_bounded_simplex(v, max_weight=0.3)
        assert result.max() <= 0.3 + 1e-8
        assert abs(result.sum() - 1.0) < 1e-8
        assert np.all(result >= -1e-10)

    def test_bounded_simplex_preserves_order(self):
        """Projection should approximately preserve relative ordering."""
        v = np.array([0.5, 0.3, 0.15, 0.05])
        result = project_bounded_simplex(v, max_weight=0.35)
        # Largest input should map to one of the largest outputs
        assert result[0] >= result[3]


class TestInputValidation:
    """Test that invalid inputs raise appropriate errors."""

    def test_nan_market_weights(self):
        """NaN in market weights should raise ValueError."""
        mu = np.array([0.5, np.nan, 0.3])
        cov = np.eye(3) * 0.04
        strategy = GammaGradientStrategy()
        with pytest.raises(ValueError, match="NaN"):
            strategy.compute_weights(mu, cov)

    def test_nan_covariance(self):
        """NaN in covariance should raise ValueError."""
        mu = np.array([0.5, 0.3, 0.2])
        cov = np.eye(3) * 0.04
        cov[0, 1] = np.nan
        strategy = GammaGradientStrategy()
        with pytest.raises(ValueError, match="NaN"):
            strategy.compute_weights(mu, cov)

    def test_negative_market_weights(self):
        """Negative market weights should raise ValueError."""
        mu = np.array([0.6, -0.1, 0.5])
        cov = np.eye(3) * 0.04
        strategy = GammaGradientStrategy()
        with pytest.raises(ValueError, match="non-negative"):
            strategy.compute_weights(mu, cov)

    def test_weights_not_summing_to_one(self):
        """Weights not summing to 1 should raise ValueError."""
        mu = np.array([0.5, 0.3, 0.3])  # sums to 1.1
        cov = np.eye(3) * 0.04
        strategy = GammaGradientStrategy()
        with pytest.raises(ValueError, match="sum to 1"):
            strategy.compute_weights(mu, cov)

    def test_non_symmetric_covariance(self):
        """Non-symmetric covariance should raise ValueError."""
        mu = np.array([0.5, 0.3, 0.2])
        cov = np.array([[0.04, 0.01, 0.0], [0.02, 0.04, 0.0], [0.0, 0.0, 0.04]])
        strategy = GammaGradientStrategy()
        with pytest.raises(ValueError, match="symmetric"):
            strategy.compute_weights(mu, cov)

    def test_invalid_lambda_scale(self):
        """Negative lambda should raise ValueError."""
        with pytest.raises(ValueError, match="lambda_scale"):
            GammaGradientStrategy(lambda_scale=-0.1)

    def test_invalid_max_weight(self):
        """max_weight outside (0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="max_weight"):
            GammaGradientStrategy(max_weight=0.0)
        with pytest.raises(ValueError, match="max_weight"):
            GammaGradientStrategy(max_weight=1.5)

    def test_shape_mismatch(self):
        """Mismatched mu/cov dimensions should raise ValueError."""
        mu = np.array([0.5, 0.3, 0.2])
        cov = np.eye(4) * 0.04
        strategy = GammaGradientStrategy()
        with pytest.raises(ValueError, match="shape"):
            strategy.compute_weights(mu, cov)


class TestEdgeCases:
    """Test behavior at mathematical edge cases."""

    def test_two_stocks(self):
        """Strategy works correctly with only 2 stocks."""
        mu = np.array([0.7, 0.3])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        strategy = GammaGradientStrategy(lambda_scale=0.1, max_weight=0.8)
        weights = strategy.compute_weights(mu, cov)
        assert abs(weights.sum() - 1.0) < 1e-8
        assert np.all(weights >= -1e-10)

    def test_identical_stocks(self):
        """Identical stocks should get identical weight tilts."""
        n = 5
        mu = np.ones(n) / n
        cov = 0.04 * np.eye(n)  # all identical, uncorrelated
        strategy = GammaGradientStrategy(lambda_scale=0.1, max_weight=0.5)
        weights = strategy.compute_weights(mu, cov)
        np.testing.assert_allclose(weights, np.ones(n) / n, atol=1e-8)

    def test_large_universe(self):
        """Strategy scales to 500 stocks without errors."""
        np.random.seed(42)
        n = 500
        mu = np.random.dirichlet(np.ones(n) * 0.3)
        A = np.random.randn(n, 50) * 0.05  # factor model for speed
        cov = A @ A.T + 0.02 * np.eye(n)

        strategy = GammaGradientStrategy(lambda_scale=0.1, max_weight=0.02)
        weights = strategy.compute_weights(mu, cov)
        assert abs(weights.sum() - 1.0) < 1e-7
        assert np.all(weights >= -1e-10)
        assert weights.max() <= 0.02 + 1e-8

    def test_near_singular_covariance(self):
        """Strategy handles nearly singular covariance via ridge."""
        n = 10
        mu = np.ones(n) / n
        # Rank-deficient covariance (rank 3)
        A = np.random.randn(n, 3) * 0.1
        cov = A @ A.T  # rank 3, not full rank

        strategy = GammaGradientStrategy(lambda_scale=0.1, max_weight=0.2, ridge=1e-4)
        weights = strategy.compute_weights(mu, cov)
        assert abs(weights.sum() - 1.0) < 1e-7
        assert np.all(weights >= -1e-10)


class TestBacktestEngineIntegration:
    """Test integration with existing BacktestEngine."""

    def test_weight_function_protocol(self):
        """weight_function() returns callable compatible with BacktestEngine."""
        n = 10
        cov = np.eye(n) * 0.04
        strategy = GammaGradientStrategy(lambda_scale=0.1, max_weight=0.2)
        wf = strategy.weight_function(cov)

        mu = np.random.dirichlet(np.ones(n))
        weights = wf(mu)
        assert abs(weights.sum() - 1.0) < 1e-8
        assert np.all(weights >= -1e-10)

    def test_repr(self):
        """String representation is informative."""
        strategy = GammaGradientStrategy(lambda_scale=0.15, max_weight=0.03)
        r = repr(strategy)
        assert "0.15" in r
        assert "0.03" in r
