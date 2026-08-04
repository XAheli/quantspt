"""Tests for core/market.py — market weight dynamics and coherence.

These tests verify mathematical properties from the SPT literature,
not just that "code runs".
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt.core.market import (
    capital_distribution_curve,
    coherence_residual,
    log_log_capital_curve,
    market_excess_growth_rate,
    market_weight_diffusion,
    market_weight_drift,
    rank_permutation,
    ranked_weights,
    validate_weights,
    verify_coherence,
)
from quantspt.errors import SPTInvariantError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def uniform_3():
    """Uniform 3-stock market."""
    return np.ones(3) / 3


@pytest.fixture()
def skewed_4():
    """Skewed 4-stock market."""
    return np.array([0.5, 0.25, 0.15, 0.1])


@pytest.fixture()
def diagonal_cov_3():
    """3×3 diagonal covariance matrix."""
    return np.diag([0.04, 0.09, 0.0625])


@pytest.fixture()
def corr_cov_3():
    """3×3 correlated covariance matrix."""
    return np.array(
        [
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.02],
            [0.005, 0.02, 0.0625],
        ]
    )


# ---------------------------------------------------------------------------
# Tests: validate_weights
# ---------------------------------------------------------------------------


class TestValidateWeights:
    def test_valid_weights_pass(self, uniform_3: np.ndarray) -> None:
        validate_weights(uniform_3)

    def test_negative_weight_raises(self) -> None:
        with pytest.raises(SPTInvariantError, match="must be > 0"):
            validate_weights(np.array([0.5, -0.1, 0.6]))

    def test_wrong_sum_raises(self) -> None:
        with pytest.raises(SPTInvariantError, match="must sum to 1"):
            validate_weights(np.array([0.3, 0.3, 0.3]))

    def test_single_asset_raises(self) -> None:
        with pytest.raises(SPTInvariantError, match="≥ 2"):
            validate_weights(np.array([1.0]))

    def test_2d_array_raises(self) -> None:
        with pytest.raises(SPTInvariantError, match="1-D"):
            validate_weights(np.ones((2, 2)) / 4)


# ---------------------------------------------------------------------------
# Tests: market weight dynamics (F&K Survey Eq. 2.4)
# ---------------------------------------------------------------------------


class TestMarketWeightDrift:
    def test_drift_sums_to_zero(
        self, skewed_4: np.ndarray, diagonal_cov_3: np.ndarray
    ) -> None:
        """Market weight drifts must sum to zero (simplex constraint)."""
        mu = np.array([0.5, 0.3, 0.2])
        gamma = np.array([0.03, 0.05, 0.02])
        a = diagonal_cov_3
        drift = market_weight_drift(mu, gamma, a)
        assert_allclose(np.sum(drift), 0.0, atol=1e-14)

    def test_drift_sums_to_zero_correlated(self, corr_cov_3: np.ndarray) -> None:
        """Drift sums to zero even with correlations."""
        mu = np.array([0.5, 0.3, 0.2])
        gamma = np.array([0.03, 0.05, 0.02])
        drift = market_weight_drift(mu, gamma, corr_cov_3)
        assert_allclose(np.sum(drift), 0.0, atol=1e-13)

    def test_equal_rates_zero_drift_diagonal(self) -> None:
        """When all b_i are equal, drift is zero (equal-volatility case)."""
        n = 4
        sigma_sq = 0.04
        a = np.eye(n) * sigma_sq
        gamma = np.full(n, 0.05)
        mu = np.array([0.4, 0.3, 0.2, 0.1])
        drift = market_weight_drift(mu, gamma, a)
        assert_allclose(drift, 0.0, atol=1e-14)


class TestMarketWeightDiffusion:
    def test_diffusion_rows_sum_to_zero(self) -> None:
        """Σ_i (diffusion row)_ν = 0 for each factor ν."""
        mu = np.array([0.5, 0.3, 0.2])
        sigma = np.array([[0.2, 0.0], [0.0, 0.3], [0.1, 0.1]])
        diff = market_weight_diffusion(mu, sigma)
        assert_allclose(np.sum(diff, axis=0), 0.0, atol=1e-14)

    def test_shape(self) -> None:
        mu = np.array([0.6, 0.4])
        sigma = np.array([[0.2, 0.1, 0.0], [0.1, 0.2, 0.1]])
        diff = market_weight_diffusion(mu, sigma)
        assert diff.shape == (2, 3)


# ---------------------------------------------------------------------------
# Tests: ranked weights (F&K Survey Eq. 1.18)
# ---------------------------------------------------------------------------


class TestRankedWeights:
    def test_descending_order(self, skewed_4: np.ndarray) -> None:
        rw = ranked_weights(skewed_4)
        assert np.all(rw[:-1] >= rw[1:])

    def test_sum_preserved(self, skewed_4: np.ndarray) -> None:
        """Ranking must not change the total weight."""
        rw = ranked_weights(skewed_4)
        assert_allclose(np.sum(rw), np.sum(skewed_4), atol=1e-14)

    def test_permutation_recovers_ranked(self, skewed_4: np.ndarray) -> None:
        p = rank_permutation(skewed_4)
        assert_allclose(skewed_4[p], ranked_weights(skewed_4))


# ---------------------------------------------------------------------------
# Tests: market excess growth rate
# ---------------------------------------------------------------------------


class TestMarketExcessGrowthRate:
    def test_matches_general_formula(self, corr_cov_3: np.ndarray) -> None:
        from quantspt.core.growth_rates import excess_growth_rate

        mu = np.array([0.5, 0.3, 0.2])
        expected = excess_growth_rate(mu, corr_cov_3)
        result = market_excess_growth_rate(mu, corr_cov_3)
        assert_allclose(result, expected, atol=1e-14)


# ---------------------------------------------------------------------------
# Tests: coherence condition (F&K Survey Eq. 2.9)
# ---------------------------------------------------------------------------


class TestCoherence:
    def test_coherence_holds_by_definition(self) -> None:
        """Coherence Σ γ_i μ_i + γ*_μ = γ_μ holds when γ_μ is derived."""
        mu = np.array([0.5, 0.3, 0.2])
        gamma = np.array([0.03, 0.05, 0.02])
        a = np.diag([0.04, 0.09, 0.0625])
        assert verify_coherence(gamma, mu, a)

    def test_coherence_residual_zero(self) -> None:
        """Residual is zero when γ_μ is computed from the same γ_i, μ_i."""
        mu = np.array([0.6, 0.4])
        gamma = np.array([0.02, 0.04])
        a = np.diag([0.04, 0.04])
        residual = coherence_residual(gamma, mu, a)
        assert_allclose(residual, 0.0, atol=1e-14)

    def test_coherence_fails_with_wrong_gamma_mu(self) -> None:
        """Supplying an inconsistent γ_μ violates coherence."""
        mu = np.array([0.5, 0.3, 0.2])
        gamma = np.array([0.03, 0.05, 0.02])
        a = np.diag([0.04, 0.09, 0.0625])
        assert not verify_coherence(gamma, mu, a, gamma_mu=999.0)

    def test_coherence_random_portfolios(self) -> None:
        """Coherence holds for random parameter sets (tautological test)."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            n = rng.integers(2, 8)
            mu = rng.dirichlet(np.ones(n))
            gamma = rng.standard_normal(n) * 0.1
            L = rng.standard_normal((n, n))
            a = L @ L.T + np.eye(n) * 0.01
            assert verify_coherence(gamma, mu, a)


# ---------------------------------------------------------------------------
# Tests: capital distribution curve
# ---------------------------------------------------------------------------


class TestCapitalDistribution:
    def test_curve_is_descending(self, skewed_4: np.ndarray) -> None:
        curve = capital_distribution_curve(skewed_4)
        assert np.all(curve[:-1] >= curve[1:])

    def test_log_log_shapes(self, skewed_4: np.ndarray) -> None:
        log_r, log_w = log_log_capital_curve(skewed_4)
        assert log_r.shape == (4,)
        assert log_w.shape == (4,)

    def test_log_rank_starts_at_zero(self, skewed_4: np.ndarray) -> None:
        """log(rank=1) = 0."""
        log_r, _ = log_log_capital_curve(skewed_4)
        assert_allclose(log_r[0], 0.0)
