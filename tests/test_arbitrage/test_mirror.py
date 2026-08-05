"""Exhaustive tests for arbitrage/mirror.py.

Validates mirror portfolio construction, long-only checks, relative
covariance rate computation, and the performance identity residual.

Mathematical References
-----------------------
- Mirror definition: FKK Eq. 8.1, π̂ = 2μ − π
- Mirror long-only condition: FKK §8
- Relative covariance rate: FKK Eq. 8.3–8.4
- Performance identity: FKK Eq. 8.7
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt.arbitrage.mirror import (
    mirror_covariance_rate,
    mirror_is_long_only,
    mirror_performance_residual,
    mirror_portfolio,
)
from quantspt.errors import SPTInvariantError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mu_3() -> np.ndarray:
    return np.array([0.5, 0.3, 0.2])


@pytest.fixture()
def mu_5() -> np.ndarray:
    return np.ones(5) / 5.0


@pytest.fixture()
def cov_3() -> np.ndarray:
    return np.array(
        [
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.02],
            [0.005, 0.02, 0.16],
        ]
    )


# =========================================================================
# A. mirror_portfolio  (FKK Eq. 8.1)
# =========================================================================


class TestMirrorPortfolio:
    """π̂ = 2μ − π.

    References: FKK Eq. 8.1
    """

    def test_formula_exact(self, mu_3: np.ndarray) -> None:
        """Verify π̂ = 2μ − π for known values."""
        pi = np.array([0.6, 0.25, 0.15])
        pi_hat = mirror_portfolio(mu_3, pi)
        expected = 2.0 * mu_3 - pi
        assert_allclose(pi_hat, expected, atol=1e-14)

    def test_mirror_of_market_is_market(self, mu_3: np.ndarray) -> None:
        """Mirror of the market portfolio is the market itself: 2μ − μ = μ."""
        pi_hat = mirror_portfolio(mu_3, mu_3)
        assert_allclose(pi_hat, mu_3, atol=1e-14)

    def test_mirror_of_mirror_is_original(self, mu_3: np.ndarray) -> None:
        """Double mirror: π̂̂ = 2μ − (2μ − π) = π."""
        pi = np.array([0.55, 0.30, 0.15])
        pi_hat = mirror_portfolio(mu_3, pi)
        pi_hat_hat = mirror_portfolio(mu_3, pi_hat)
        assert_allclose(pi_hat_hat, pi, atol=1e-14)

    def test_weights_sum_to_one(self, mu_3: np.ndarray) -> None:
        """If Σπ_i = 1 and Σμ_i = 1, then Σπ̂_i = 2·1 − 1 = 1."""
        pi = np.array([0.6, 0.25, 0.15])
        pi_hat = mirror_portfolio(mu_3, pi)
        assert_allclose(np.sum(pi_hat), 1.0, atol=1e-14)

    def test_overweight_becomes_underweight(self, mu_3: np.ndarray) -> None:
        """If π_i > μ_i, then π̂_i = 2μ_i − π_i < μ_i."""
        pi = np.array([0.7, 0.2, 0.1])
        pi_hat = mirror_portfolio(mu_3, pi)
        assert pi_hat[0] < mu_3[0]

    def test_underweight_becomes_overweight(self, mu_3: np.ndarray) -> None:
        """If π_i < μ_i, then π̂_i > μ_i."""
        pi = np.array([0.3, 0.4, 0.3])
        pi_hat = mirror_portfolio(mu_3, pi)
        assert pi_hat[0] > mu_3[0]

    def test_symmetric_deviation(self, mu_3: np.ndarray) -> None:
        """π − μ and π̂ − μ are negatives of each other."""
        pi = np.array([0.6, 0.25, 0.15])
        pi_hat = mirror_portfolio(mu_3, pi)
        assert_allclose(pi - mu_3, -(pi_hat - mu_3), atol=1e-14)

    def test_2_stock(self) -> None:
        mu = np.array([0.6, 0.4])
        pi = np.array([0.8, 0.2])
        pi_hat = mirror_portfolio(mu, pi)
        assert_allclose(pi_hat, np.array([0.4, 0.6]), atol=1e-14)

    def test_5_stock_uniform(self, mu_5: np.ndarray) -> None:
        pi = np.array([0.3, 0.25, 0.2, 0.15, 0.10])
        pi_hat = mirror_portfolio(mu_5, pi)
        assert_allclose(np.sum(pi_hat), 1.0, atol=1e-14)
        assert_allclose(pi_hat, 2.0 * mu_5 - pi, atol=1e-14)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(SPTInvariantError, match="length"):
            mirror_portfolio(np.array([0.5, 0.5]), np.array([0.5, 0.3, 0.2]))


# =========================================================================
# B. mirror_is_long_only  (FKK §8)
# =========================================================================


class TestMirrorIsLongOnly:
    """π̂_i ≥ 0 ⟺ π_i ≤ 2μ_i.

    References: FKK §8
    """

    def test_close_to_market_long_only(self, mu_3: np.ndarray) -> None:
        """π ≈ μ → mirror ≈ μ → long-only."""
        pi = mu_3 + np.array([0.05, -0.03, -0.02])
        assert mirror_is_long_only(mu_3, pi) is True

    def test_market_itself_long_only(self, mu_3: np.ndarray) -> None:
        """π = μ → π̂ = μ (always long-only)."""
        assert mirror_is_long_only(mu_3, mu_3) is True

    def test_extreme_deviation_not_long_only(self) -> None:
        """π_i > 2μ_i for some i → π̂_i = 2μ_i - π_i < 0."""
        mu = np.array([0.2, 0.3, 0.5])
        pi = np.array([0.5, 0.3, 0.2])
        assert mirror_is_long_only(mu, pi) is False

    def test_boundary_case_pi_equals_2mu(self) -> None:
        """π_i = 2μ_i → π̂_i = 0 (boundary, still long-only)."""
        mu = np.array([0.5, 0.5])
        pi = np.array([1.0, 0.0])
        assert mirror_is_long_only(mu, pi) is True

    def test_just_beyond_boundary(self) -> None:
        """π_i slightly above 2μ_i → π̂_i < 0."""
        mu = np.array([0.3, 0.7])
        pi = np.array([0.61, 0.39])
        assert mirror_is_long_only(mu, pi) is False

    def test_diversity_portfolio_long_only(self, mu_3: np.ndarray) -> None:
        """Diversity-weighted portfolio is close to μ → mirror is long-only."""
        from quantspt.arbitrage.construction import diversity_arbitrage_portfolio

        pi = diversity_arbitrage_portfolio(mu_3, p=0.5)
        assert mirror_is_long_only(mu_3, pi) is True


# =========================================================================
# C. mirror_covariance_rate  (FKK Eq. 8.3–8.4)
# =========================================================================


class TestMirrorCovarianceRate:
    r"""τ^μ_{ππ} = (π − μ)' a (π − μ).

    References: FKK Eq. 8.3–8.4
    """

    def test_zero_when_pi_equals_mu(self, mu_3: np.ndarray, cov_3: np.ndarray) -> None:
        """π = μ → deviation is zero → τ = 0."""
        tau = mirror_covariance_rate(mu_3, mu_3, cov_3)
        assert_allclose(tau, 0.0, atol=1e-14)

    def test_positive_when_different(self, mu_3: np.ndarray, cov_3: np.ndarray) -> None:
        """π ≠ μ with PSD a → τ > 0."""
        pi = np.array([0.6, 0.25, 0.15])
        tau = mirror_covariance_rate(pi, mu_3, cov_3)
        assert tau > 0.0

    def test_formula_verification(self, mu_3: np.ndarray, cov_3: np.ndarray) -> None:
        """Verify τ = (π−μ)' a (π−μ) directly."""
        pi = np.array([0.6, 0.25, 0.15])
        tau = mirror_covariance_rate(pi, mu_3, cov_3)
        diff = pi - mu_3
        expected = float(diff @ cov_3 @ diff)
        assert_allclose(tau, expected, atol=1e-14)

    def test_non_negative_psd(self, mu_3: np.ndarray, cov_3: np.ndarray) -> None:
        """For PSD a, the quadratic form is always ≥ 0."""
        rng = np.random.default_rng(77)
        for _ in range(20):
            pi = rng.dirichlet(np.ones(3))
            tau = mirror_covariance_rate(pi, mu_3, cov_3)
            assert tau >= -1e-15

    def test_symmetric_for_mirror(self, mu_3: np.ndarray, cov_3: np.ndarray) -> None:
        """τ^μ_{ππ} = τ^μ_{π̂π̂} because deviations are negatives."""
        pi = np.array([0.6, 0.25, 0.15])
        pi_hat = mirror_portfolio(mu_3, pi)
        tau_pi = mirror_covariance_rate(pi, mu_3, cov_3)
        tau_hat = mirror_covariance_rate(pi_hat, mu_3, cov_3)
        assert_allclose(tau_pi, tau_hat, atol=1e-14)

    def test_scales_with_deviation(self, mu_3: np.ndarray, cov_3: np.ndarray) -> None:
        """Doubling the deviation quadruples the covariance rate."""
        small_dev = np.array([0.02, -0.01, -0.01])
        pi_small = mu_3 + small_dev
        pi_large = mu_3 + 2.0 * small_dev
        tau_small = mirror_covariance_rate(pi_small, mu_3, cov_3)
        tau_large = mirror_covariance_rate(pi_large, mu_3, cov_3)
        assert_allclose(tau_large / tau_small, 4.0, rtol=1e-10)

    def test_identity_covariance(self, mu_3: np.ndarray) -> None:
        """With a = I, τ = ||π − μ||²."""
        pi = np.array([0.6, 0.25, 0.15])
        a = np.eye(3)
        tau = mirror_covariance_rate(pi, mu_3, a)
        expected = float(np.sum((pi - mu_3) ** 2))
        assert_allclose(tau, expected, atol=1e-14)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(SPTInvariantError, match="shape"):
            mirror_covariance_rate(
                np.array([0.5, 0.5]),
                np.array([0.5, 0.5]),
                np.eye(3),
            )

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(SPTInvariantError, match="length"):
            mirror_covariance_rate(
                np.array([0.5, 0.5]),
                np.array([0.5, 0.3, 0.2]),
                np.eye(2),
            )


# =========================================================================
# D. mirror_performance_residual  (FKK Eq. 8.7)
# =========================================================================


class TestMirrorPerformanceResidual:
    r"""Residual: log V^π + log V^π̂ − 2 log V^μ ≥ 0.

    The mirror identity (FKK Eq. 8.7):
        log V^π(t) + log V^π̂(t) = 2 log V^μ(t) + ∫₀ᵗ τ^μ_{ππ}(s) ds

    Since the integral is non-negative, the residual is ≥ 0.

    References: FKK Eq. 8.7
    """

    def test_formula_verification(self) -> None:
        """Directly verify residual = log V^π + log V^π̂ − 2 log V^μ."""
        V_pi = np.array([1.0, 1.05, 1.10, 1.08])
        V_hat = np.array([1.0, 0.98, 1.02, 1.05])
        V_mu = np.array([1.0, 1.01, 1.05, 1.06])
        residual = mirror_performance_residual(V_pi, V_hat, V_mu)
        expected = np.log(V_pi) + np.log(V_hat) - 2.0 * np.log(V_mu)
        assert_allclose(residual, expected, atol=1e-14)

    def test_residual_at_time_zero(self) -> None:
        """At t=0, all values are 1.0, so residual = 0."""
        T = 10
        V_pi = np.ones(T)
        V_hat = np.ones(T)
        V_mu = np.ones(T)
        residual = mirror_performance_residual(V_pi, V_hat, V_mu)
        assert_allclose(residual[0], 0.0, atol=1e-14)

    def test_identical_portfolios_residual_zero(self) -> None:
        """If π = μ, then π̂ = μ, so V^π = V^π̂ = V^μ → residual = 0."""
        V = np.array([1.0, 1.02, 1.05, 1.03, 1.07])
        residual = mirror_performance_residual(V, V, V)
        assert_allclose(residual, 0.0, atol=1e-14)

    def test_output_shape(self) -> None:
        T = 20
        V = np.linspace(1.0, 1.5, T)
        residual = mirror_performance_residual(V, V, V)
        assert residual.shape == (T,)

    def test_simulated_non_negative_residual(self) -> None:
        """On simulated data, verify residual ≥ 0 (approximately).

        Construct V^π and V^π̂ from a simple model where the identity
        log V^π + log V^π̂ ≈ 2 log V^μ + accumulated τ holds.
        """
        rng = np.random.default_rng(42)
        T = 100
        tau_integral = np.cumsum(rng.exponential(0.001, size=T))
        tau_integral = np.insert(tau_integral, 0, 0.0)

        log_V_mu = np.cumsum(np.insert(rng.normal(0.001, 0.01, size=T), 0, 0.0))
        alpha = rng.normal(0, 0.005, size=T)
        alpha = np.insert(alpha, 0, 0.0)

        log_V_pi = log_V_mu + alpha + tau_integral / 2.0
        log_V_hat = log_V_mu - alpha + tau_integral / 2.0

        V_pi = np.exp(log_V_pi)
        V_hat = np.exp(log_V_hat)
        V_mu = np.exp(log_V_mu)

        residual = mirror_performance_residual(V_pi, V_hat, V_mu)
        assert np.all(residual >= -1e-10)

    def test_length_mismatch_V_pi(self) -> None:
        with pytest.raises(SPTInvariantError, match="equal length"):
            mirror_performance_residual(
                np.array([1.0, 1.1]),
                np.array([1.0, 1.1, 1.2]),
                np.array([1.0, 1.1, 1.2]),
            )

    def test_length_mismatch_V_mu(self) -> None:
        with pytest.raises(SPTInvariantError, match="equal length"):
            mirror_performance_residual(
                np.array([1.0, 1.1, 1.2]),
                np.array([1.0, 1.1, 1.2]),
                np.array([1.0, 1.1]),
            )

    def test_zero_value_raises(self) -> None:
        with pytest.raises(SPTInvariantError, match="positive"):
            mirror_performance_residual(
                np.array([1.0, 0.0]),
                np.array([1.0, 1.1]),
                np.array([1.0, 1.1]),
            )

    def test_negative_value_raises(self) -> None:
        with pytest.raises(SPTInvariantError, match="positive"):
            mirror_performance_residual(
                np.array([1.0, -0.5]),
                np.array([1.0, 1.1]),
                np.array([1.0, 1.1]),
            )
