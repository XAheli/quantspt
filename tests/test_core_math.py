"""Tests for core mathematical modules.

These tests verify correctness against known analytical results from
the SPT papers. Every test references the specific theorem or equation
being validated.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt.core.covariance import (
    non_degeneracy_bounds,
    portfolio_variance,
    relative_covariance,
    tau_bounds,
    tau_diagonal,
    verify_non_degeneracy,
)
from quantspt.core.diversity import (
    arbitrage_horizon_bound,
    concentration_ratio,
    diversity_deficit,
    entropy,
    herfindahl_hirschman_index,
    is_diverse,
    is_weakly_diverse,
    p_diversity,
)
from quantspt.core.growth_rates import (
    atlas_excess_growth_rate_equal_weighted,
    atlas_excess_growth_rate_uncorrelated,
    atlas_market_growth_rate,
    excess_growth_rate,
    excess_growth_rate_bounds,
    excess_growth_rate_from_tau,
    portfolio_growth_rate,
    relative_performance_rate,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_stock_diagonal():
    """Simple 2-stock uncorrelated system for analytical verification."""
    a = np.diag([0.04, 0.04])
    pi_equal = np.array([0.5, 0.5])
    pi_concentrated = np.array([1.0, 0.0])
    mu = np.array([0.6, 0.4])
    return a, pi_equal, pi_concentrated, mu


@pytest.fixture
def three_stock_correlated():
    """3-stock system with known correlation structure."""
    a = np.array(
        [
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.02],
            [0.005, 0.02, 0.0625],
        ]
    )
    mu = np.array([0.5, 0.3, 0.2])
    return a, mu


@pytest.fixture
def atlas_model_params():
    """Parameters for an Atlas model with n=5 stocks."""
    n = 5
    sigma_sq = np.full(n, 0.04)
    gamma = 0.02
    return n, sigma_sq, gamma


# ---------------------------------------------------------------------------
# Tests: core/covariance.py
# ---------------------------------------------------------------------------


class TestRelativeCovariance:
    """Verify τ^π matrix against analytical properties (F&K Lemma 3.1)."""

    def test_null_space_property(self, two_stock_diagonal):
        """τ^π · π = 0 for any π (F&K Lemma 3.1)."""
        a, pi_equal, _, mu = two_stock_diagonal
        for pi in [pi_equal, mu]:
            tau = relative_covariance(a, pi)
            assert_allclose(tau @ pi, 0, atol=1e-14)

    def test_positive_semidefinite(self, three_stock_correlated):
        """τ^π must be PSD (F&K Lemma 3.1)."""
        a, mu = three_stock_correlated
        tau = relative_covariance(a, mu)
        eigenvalues = np.linalg.eigvalsh(tau)
        assert np.all(eigenvalues >= -1e-14)

    def test_symmetry(self, three_stock_correlated):
        """τ^π must be symmetric."""
        a, mu = three_stock_correlated
        tau = relative_covariance(a, mu)
        assert_allclose(tau, tau.T, atol=1e-14)

    def test_two_stock_analytical(self, two_stock_diagonal):
        """For 2 uncorrelated stocks with equal σ², verify analytical τ."""
        a, _, _, mu = two_stock_diagonal
        tau = relative_covariance(a, mu)
        # For diagonal a with equal σ²:
        # τ_{ii} = σ²(1-μ_i)², τ_{ij} = σ²(1-μ_i)(1-μ_j) - no, that's wrong
        # Actually τ_{ij} = a_{ij} - a^π_i - a^π_j + a_{ππ}
        # For diagonal: a^π_i = π_i * a_{ii}
        sigma_sq = 0.04
        a_pi = mu * sigma_sq  # [0.024, 0.016]
        a_pipi = mu @ (a @ mu)  # 0.6²*0.04 + 0.4²*0.04 = 0.0208
        expected_00 = sigma_sq - 2 * a_pi[0] + a_pipi
        expected_11 = sigma_sq - 2 * a_pi[1] + a_pipi
        assert_allclose(tau[0, 0], expected_00, atol=1e-14)
        assert_allclose(tau[1, 1], expected_11, atol=1e-14)

    def test_concentrated_portfolio_zero_tau(self, two_stock_diagonal):
        """Single-stock portfolio has τ^π = 0 except for off-diag."""
        a, _, pi_concentrated, _ = two_stock_diagonal
        tau = relative_covariance(a, pi_concentrated)
        # τ_{11} = a_{11} - 2*a^π_1 + a_{ππ} = σ² - 2σ² + σ² = 0
        assert_allclose(tau[0, 0], 0, atol=1e-14)

    def test_bounds_hold(self, three_stock_correlated):
        """Verify FKK Eq. 5.10 bounds on τ^π_{ii}."""
        a, mu = three_stock_correlated
        tau = relative_covariance(a, mu)
        eps, M = non_degeneracy_bounds(a)
        lower, upper = tau_bounds(mu, eps, M)
        tau_diag = np.diag(tau)
        assert np.all(tau_diag >= lower - 1e-10)
        assert np.all(tau_diag <= upper + 1e-10)


class TestPortfolioVariance:
    def test_equal_weighted_uncorrelated(self, two_stock_diagonal):
        """Equal-weighted, uncorrelated: π'aπ = σ²/n."""
        a, pi_equal, _, _ = two_stock_diagonal
        assert_allclose(portfolio_variance(a, pi_equal), 0.04 * 0.5, atol=1e-14)

    def test_single_stock(self, two_stock_diagonal):
        """Single stock: π'aπ = σ²."""
        a, _, pi_concentrated, _ = two_stock_diagonal
        assert_allclose(portfolio_variance(a, pi_concentrated), 0.04)


class TestNonDegeneracy:
    def test_diagonal_bounds(self, two_stock_diagonal):
        """For diagonal matrix, bounds are min/max diagonal entries."""
        a, _, _, _ = two_stock_diagonal
        eps, M = non_degeneracy_bounds(a)
        assert_allclose(eps, 0.04)
        assert_allclose(M, 0.04)

    def test_correlated_positive_definite(self, three_stock_correlated):
        a, _ = three_stock_correlated
        assert verify_non_degeneracy(a)

    def test_singular_matrix_fails(self):
        a = np.array([[1.0, 1.0], [1.0, 1.0]])
        assert not verify_non_degeneracy(a)


class TestTauDiagonal:
    def test_matches_full_matrix(self, three_stock_correlated):
        """tau_diagonal should match diagonal of full relative_covariance."""
        a, mu = three_stock_correlated
        tau_full = relative_covariance(a, mu)
        tau_d = tau_diagonal(a, mu)
        assert_allclose(tau_d, np.diag(tau_full), atol=1e-14)


# ---------------------------------------------------------------------------
# Tests: core/growth_rates.py
# ---------------------------------------------------------------------------


class TestExcessGrowthRate:
    """Verify γ*_π against known analytical results."""

    def test_equal_weighted_2_stock(self, two_stock_diagonal):
        """γ* = σ²/4 for equal-weighted 2-stock uncorrelated system."""
        a, pi_equal, _, _ = two_stock_diagonal
        result = excess_growth_rate(pi_equal, a)
        assert_allclose(result, 0.01, atol=1e-14)

    def test_concentrated_is_zero(self, two_stock_diagonal):
        """γ* = 0 for single-stock portfolio."""
        a, _, pi_concentrated, _ = two_stock_diagonal
        result = excess_growth_rate(pi_concentrated, a)
        assert_allclose(result, 0.0, atol=1e-14)

    def test_non_negative_long_only(self, three_stock_correlated):
        """γ* ≥ 0 for any long-only portfolio (F&K Lemma 3.3)."""
        a, mu = three_stock_correlated
        # Random long-only portfolios
        rng = np.random.default_rng(42)
        for _ in range(100):
            pi = rng.dirichlet(np.ones(3))
            assert excess_growth_rate(pi, a) >= -1e-14

    def test_equivalent_forms(self, three_stock_correlated):
        """Verify γ* from covariance form equals τ form (FKK Eq. 5.4)."""
        a, mu = three_stock_correlated
        gamma_star_1 = excess_growth_rate(mu, a)
        tau = relative_covariance(a, mu)
        gamma_star_2 = excess_growth_rate_from_tau(mu, tau)
        assert_allclose(gamma_star_1, gamma_star_2, atol=1e-14)

    def test_bounds_hold(self, three_stock_correlated):
        """Verify FKK Eq. 5.12 bounds on γ*_π."""
        a, mu = three_stock_correlated
        eps, M = non_degeneracy_bounds(a)
        rng = np.random.default_rng(123)
        for _ in range(50):
            pi = rng.dirichlet(np.ones(3))
            gamma_star = excess_growth_rate(pi, a)
            lower, upper = excess_growth_rate_bounds(pi, eps, M)
            assert gamma_star >= lower - 1e-10
            assert gamma_star <= upper + 1e-10

    def test_maximum_at_equal_weight_diagonal(self):
        """For diagonal a with equal σ², γ* maximized at equal weights."""
        n = 10
        sigma_sq = 0.04
        a = np.eye(n) * sigma_sq
        # Equal weighted
        pi_equal = np.ones(n) / n
        gamma_star_equal = excess_growth_rate(pi_equal, a)
        # Any other portfolio
        rng = np.random.default_rng(99)
        for _ in range(100):
            pi = rng.dirichlet(np.ones(n))
            assert excess_growth_rate(pi, a) <= gamma_star_equal + 1e-10


class TestPortfolioGrowthRate:
    def test_decomposition(self, three_stock_correlated):
        """γ_π = Σ π_i γ_i + γ*_π (F&K Eq. 1.12)."""
        a, mu = three_stock_correlated
        gamma = np.array([0.03, 0.05, 0.02])
        result = portfolio_growth_rate(mu, gamma, a)
        expected = np.dot(mu, gamma) + excess_growth_rate(mu, a)
        assert_allclose(result, expected, atol=1e-14)


class TestRelativePerformanceRate:
    def test_zero_for_market(self, three_stock_correlated):
        """Relative performance of market vs itself is zero."""
        a, mu = three_stock_correlated
        gamma = np.array([0.03, 0.05, 0.02])
        result = relative_performance_rate(mu, mu, gamma, a)
        assert_allclose(result, 0.0, atol=1e-14)


class TestAtlasSpecialCases:
    def test_equal_weighted_formula(self, atlas_model_params):
        """Verify BFK Eq. 5.14."""
        n, sigma_sq, _ = atlas_model_params
        result = atlas_excess_growth_rate_equal_weighted(n, sigma_sq)
        expected = (n - 1) / (2 * n**2) * np.sum(sigma_sq)
        assert_allclose(result, expected)

    def test_uncorrelated_vs_general(self, atlas_model_params):
        """Atlas uncorrelated formula should match general formula for diag a."""
        n, sigma_sq, _ = atlas_model_params
        a = np.diag(sigma_sq)
        pi = np.ones(n) / n
        from_general = excess_growth_rate(pi, a)
        from_atlas = atlas_excess_growth_rate_uncorrelated(pi, sigma_sq)
        assert_allclose(from_general, from_atlas, atol=1e-14)

    def test_market_growth_rate(self, atlas_model_params):
        """BFK Eq. 5.10: G^μ(n) = γ."""
        _, _, gamma = atlas_model_params
        assert atlas_market_growth_rate(gamma) == gamma

    def test_large_n_limit(self):
        """As n → ∞, γ*_η → σ²/2 for constant volatility."""
        sigma_sq_val = 0.04
        for n in [50, 100, 500, 1000]:
            sigma_sq = np.full(n, sigma_sq_val)
            gamma_star = atlas_excess_growth_rate_equal_weighted(n, sigma_sq)
            assert_allclose(gamma_star, sigma_sq_val / 2, atol=1.0 / n)


# ---------------------------------------------------------------------------
# Tests: core/diversity.py
# ---------------------------------------------------------------------------


class TestDiversityMeasures:
    def test_entropy_uniform(self):
        """Entropy of uniform weights = log(n)."""
        n = 10
        mu = np.ones(n) / n
        assert_allclose(entropy(mu), np.log(n), atol=1e-14)

    def test_entropy_concentrated(self):
        """Entropy approaches 0 as weights concentrate."""
        mu = np.array([0.99, 0.005, 0.005])
        assert entropy(mu) < 0.1

    def test_p_diversity_uniform(self):
        """p-diversity of uniform weights.

        D_p(1/n, ..., 1/n) = (n · (1/n)^p)^{1/(1-p)} = n^{(1-p)/(1-p)} = n
        """
        n = 10
        mu = np.ones(n) / n
        p = 0.5
        # (n * (1/n)^p)^{1/(1-p)} = (n^{1-p})^{1/(1-p)} = n
        expected = float(n)
        assert_allclose(p_diversity(mu, p), expected, atol=1e-10)

    def test_hhi_uniform(self):
        """HHI of uniform weights = 1/n."""
        n = 20
        mu = np.ones(n) / n
        assert_allclose(herfindahl_hirschman_index(mu), 1.0 / n, atol=1e-14)

    def test_hhi_concentrated(self):
        """HHI of single stock = 1."""
        mu = np.array([1.0 - 1e-10, 1e-10 / 2, 1e-10 / 2])
        assert_allclose(herfindahl_hirschman_index(mu), 1.0, atol=1e-5)

    def test_concentration_ratio(self):
        mu = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
        assert_allclose(concentration_ratio(mu, 2), 0.7)
        assert_allclose(concentration_ratio(mu, 5), 1.0)


class TestDiversityConditions:
    def test_strict_diversity(self):
        """FKK Eq. 4.1: max μ_i ≤ 1 - δ."""
        mu_diverse = np.array([0.3, 0.25, 0.25, 0.2])
        mu_concentrated = np.array([0.8, 0.1, 0.05, 0.05])
        assert is_diverse(mu_diverse, delta=0.5)
        assert not is_diverse(mu_concentrated, delta=0.5)

    def test_weak_diversity(self):
        """FKK Eq. 4.2: Σ μ_i^p ≥ 1 + δ."""
        n = 10
        mu = np.ones(n) / n
        # For uniform weights: Σ (1/n)^p = n^{1-p} which is > 1 for n > 1
        assert is_weakly_diverse(mu, delta=0.5, p=0.5)

    def test_diversity_deficit_positive_for_diverse(self):
        """Diverse market has positive diversity deficit."""
        mu = np.ones(10) / 10
        deficit = diversity_deficit(mu, p=0.5)
        assert deficit > 0


class TestArbitrageHorizon:
    def test_formula(self):
        """Verify FKK Eq. 4.5: T ≥ 2 log(n) / (p ε δ)."""
        T = arbitrage_horizon_bound(n=100, p=0.5, eps=0.01, delta=0.1)
        expected = 2 * np.log(100) / (0.5 * 0.01 * 0.1)
        assert_allclose(T, expected)

    def test_more_stocks_longer_horizon(self):
        """More stocks → longer minimum horizon (harder to arbitrage)."""
        T_10 = arbitrage_horizon_bound(n=10, p=0.5, eps=0.01, delta=0.1)
        T_100 = arbitrage_horizon_bound(n=100, p=0.5, eps=0.01, delta=0.1)
        assert T_100 > T_10

    def test_stronger_diversity_shorter_horizon(self):
        """Larger δ → shorter horizon (easier to arbitrage)."""
        T_weak = arbitrage_horizon_bound(n=50, p=0.5, eps=0.01, delta=0.05)
        T_strong = arbitrage_horizon_bound(n=50, p=0.5, eps=0.01, delta=0.2)
        assert T_strong < T_weak
