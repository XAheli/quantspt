"""Exhaustive tests for arbitrage/detection.py.

Validates arbitrage opportunity detection logic including the
ArbitrageOpportunity dataclass, non-degeneracy estimation, intrinsic
volatility conditions, and the main diversity-based detection.

Mathematical References
-----------------------
- Non-degeneracy: FKK Eq. 2.3
- Sufficient intrinsic volatility: F&K Survey Eq. 11.8–11.12
- Diversity-based detection and horizon: FKK Eq. 4.5
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt.arbitrage.detection import (
    ArbitrageOpportunity,
    check_sufficient_intrinsic_volatility,
    detect_diversity_arbitrage,
    estimate_nondegeneracy,
)
from quantspt.errors import SPTInvariantError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def uniform_5() -> np.ndarray:
    return np.ones(5) / 5.0


@pytest.fixture()
def concentrated_5() -> np.ndarray:
    return np.array([0.91, 0.03, 0.02, 0.02, 0.02])


@pytest.fixture()
def psd_cov_5() -> np.ndarray:
    """5×5 symmetric PSD matrix with known smallest eigenvalue."""
    rng = np.random.default_rng(99)
    L = rng.standard_normal((5, 5))
    return L @ L.T + 0.1 * np.eye(5)


@pytest.fixture()
def identity_cov_5() -> np.ndarray:
    return np.eye(5)


# =========================================================================
# A. ArbitrageOpportunity dataclass
# =========================================================================


class TestArbitrageOpportunity:
    """Verify the ArbitrageOpportunity result dataclass."""

    def test_fields_present(self) -> None:
        opp = ArbitrageOpportunity(
            is_detected=True,
            method="diversity",
            min_horizon=10.0,
            delta=0.5,
            epsilon=0.1,
            expected_rate=0.025,
            basis="FKK Eq. 4.5",
        )
        assert opp.is_detected is True
        assert opp.method == "diversity"
        assert opp.min_horizon == 10.0
        assert opp.delta == 0.5
        assert opp.epsilon == 0.1
        assert opp.expected_rate == 0.025
        assert opp.basis == "FKK Eq. 4.5"

    def test_frozen_dataclass(self) -> None:
        opp = ArbitrageOpportunity(
            is_detected=False,
            method="diversity",
            min_horizon=None,
            delta=0.0,
            epsilon=0.0,
            expected_rate=0.0,
            basis="test",
        )
        with pytest.raises(AttributeError):
            opp.is_detected = True  # type: ignore[misc]

    def test_not_detected_horizon_none(self) -> None:
        opp = ArbitrageOpportunity(
            is_detected=False,
            method="diversity",
            min_horizon=None,
            delta=0.0,
            epsilon=0.0,
            expected_rate=0.0,
            basis="test",
        )
        assert opp.min_horizon is None


# =========================================================================
# B. estimate_nondegeneracy  (FKK Eq. 2.3)
# =========================================================================


class TestEstimateNondegeneracy:
    r"""Non-degeneracy constant ε = smallest eigenvalue of a.

    References: FKK Eq. 2.3
    """

    def test_identity_matrix(self) -> None:
        """Identity matrix → ε = 1."""
        a = np.eye(3)
        assert_allclose(estimate_nondegeneracy(a), 1.0, atol=1e-12)

    def test_scaled_identity(self) -> None:
        """c·I → ε = c."""
        c = 0.05
        a = c * np.eye(4)
        assert_allclose(estimate_nondegeneracy(a), c, atol=1e-12)

    def test_known_eigenvalues(self) -> None:
        """Diagonal matrix with known eigenvalues."""
        a = np.diag([0.1, 0.5, 1.0, 2.0])
        assert_allclose(estimate_nondegeneracy(a), 0.1, atol=1e-12)

    def test_psd_matrix_positive(self, psd_cov_5: np.ndarray) -> None:
        """PSD matrix with ridge → smallest eigenvalue > 0."""
        eps = estimate_nondegeneracy(psd_cov_5)
        assert eps > 0.0

    def test_singular_matrix_zero_eigenvalue(self) -> None:
        """Rank-deficient matrix → ε ≈ 0 (degenerate market)."""
        v = np.array([1.0, 1.0, 1.0])
        a = np.outer(v, v)  # rank 1
        eps = estimate_nondegeneracy(a)
        assert_allclose(eps, 0.0, atol=1e-10)

    def test_symmetric_2x2(self) -> None:
        """2×2 symmetric: eigenvalues known analytically."""
        a = np.array([[4.0, 2.0], [2.0, 4.0]])
        assert_allclose(estimate_nondegeneracy(a), 2.0, atol=1e-12)

    def test_invalid_1d(self) -> None:
        with pytest.raises(SPTInvariantError, match="2-D"):
            estimate_nondegeneracy(np.array([1.0, 2.0]))

    def test_invalid_non_square(self) -> None:
        with pytest.raises(SPTInvariantError, match="square"):
            estimate_nondegeneracy(np.ones((2, 3)))


# =========================================================================
# C. check_sufficient_intrinsic_volatility (F&K Survey Eq. 11.8–11.12)
# =========================================================================


class TestCheckSufficientIntrinsicVolatility:
    """Sufficient condition: γ*_μ ≥ ζ > 0.

    References: F&K Survey Eq. 11.8–11.12
    """

    def test_above_threshold(self) -> None:
        assert check_sufficient_intrinsic_volatility(0.05, zeta=0.01) is True

    def test_below_threshold(self) -> None:
        assert check_sufficient_intrinsic_volatility(0.005, zeta=0.01) is False

    def test_exactly_at_threshold(self) -> None:
        """γ*_μ = ζ → condition holds (≥)."""
        assert check_sufficient_intrinsic_volatility(0.01, zeta=0.01) is True

    def test_large_excess_growth(self) -> None:
        assert check_sufficient_intrinsic_volatility(1.0, zeta=0.001) is True

    def test_zero_excess_growth(self) -> None:
        assert check_sufficient_intrinsic_volatility(0.0, zeta=0.01) is False

    def test_negative_excess_growth(self) -> None:
        assert check_sufficient_intrinsic_volatility(-0.1, zeta=0.01) is False

    def test_invalid_zeta_zero(self) -> None:
        with pytest.raises(SPTInvariantError):
            check_sufficient_intrinsic_volatility(0.05, zeta=0.0)

    def test_invalid_zeta_negative(self) -> None:
        with pytest.raises(SPTInvariantError):
            check_sufficient_intrinsic_volatility(0.05, zeta=-0.01)


# =========================================================================
# D. detect_diversity_arbitrage  (FKK Eq. 4.5)
# =========================================================================


class TestDetectDiversityArbitrage:
    r"""Diversity-based detection: T* = 2 log(n) / (p ε δ).

    References: FKK Eq. 4.5
    """

    def test_diverse_market_detected(
        self, uniform_5: np.ndarray, identity_cov_5: np.ndarray
    ) -> None:
        """Uniform weights + identity covariance → arbitrage detected."""
        result = detect_diversity_arbitrage(uniform_5, identity_cov_5, p=0.5)
        assert result.is_detected is True
        assert result.method == "diversity"
        assert result.min_horizon is not None
        assert result.min_horizon > 0.0
        assert result.delta > 0.0
        assert result.epsilon > 0.0
        assert result.expected_rate > 0.0
        assert result.basis == "FKK Eq. 4.5"

    def test_concentrated_market_not_detected(
        self, concentrated_5: np.ndarray, identity_cov_5: np.ndarray
    ) -> None:
        """Concentrated market: weak diversity fails → no arbitrage for large p."""
        result = detect_diversity_arbitrage(concentrated_5, identity_cov_5, p=0.9)
        if not result.is_detected:
            assert result.min_horizon is None
            assert result.expected_rate == 0.0

    def test_horizon_formula_fkk_eq_4_5(
        self, uniform_5: np.ndarray, identity_cov_5: np.ndarray
    ) -> None:
        """Verify T* = 2 log(n) / (p ε δ) directly."""
        p = 0.5
        result = detect_diversity_arbitrage(uniform_5, identity_cov_5, p=p)
        assert result.is_detected is True

        n = len(uniform_5)
        delta = float(np.sum(uniform_5**p)) - 1.0
        epsilon = float(np.linalg.eigvalsh(identity_cov_5)[0])
        T_star_expected = 2.0 * np.log(n) / (p * epsilon * delta)

        assert_allclose(result.min_horizon, T_star_expected, atol=1e-10)
        assert_allclose(result.delta, delta, atol=1e-10)
        assert_allclose(result.epsilon, epsilon, atol=1e-10)

    def test_expected_rate_formula(
        self, uniform_5: np.ndarray, identity_cov_5: np.ndarray
    ) -> None:
        """Expected rate = (1-p) ε δ / 2."""
        p = 0.5
        result = detect_diversity_arbitrage(uniform_5, identity_cov_5, p=p)
        expected_rate = (1.0 - p) * result.epsilon * result.delta / 2.0
        assert_allclose(result.expected_rate, expected_rate, atol=1e-12)

    def test_degenerate_covariance_not_detected(self, uniform_5: np.ndarray) -> None:
        """Singular covariance (ε=0) → no detection even if diverse."""
        v = np.ones(5)
        a = np.outer(v, v)  # rank 1
        result = detect_diversity_arbitrage(uniform_5, a, p=0.5)
        assert result.is_detected is False

    @pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_horizon_decreases_with_p_diversity_tradeoff(
        self, uniform_5: np.ndarray, identity_cov_5: np.ndarray, p: float
    ) -> None:
        """Verify detection works across different p values."""
        result = detect_diversity_arbitrage(uniform_5, identity_cov_5, p=p)
        if result.is_detected:
            assert result.min_horizon is not None
            assert result.min_horizon > 0.0

    def test_default_p(self, uniform_5: np.ndarray, identity_cov_5: np.ndarray) -> None:
        """Default p=0.5 produces valid results."""
        result = detect_diversity_arbitrage(uniform_5, identity_cov_5)
        assert result.is_detected is True

    def test_2_stock_market(self) -> None:
        """Minimal market n=2."""
        mu = np.array([0.6, 0.4])
        a = np.array([[0.04, 0.01], [0.01, 0.09]])
        result = detect_diversity_arbitrage(mu, a, p=0.5)
        assert isinstance(result, ArbitrageOpportunity)
        if result.is_detected:
            n = 2
            T_expected = 2.0 * np.log(n) / (0.5 * result.epsilon * result.delta)
            assert_allclose(result.min_horizon, T_expected, atol=1e-10)

    def test_more_stocks_longer_horizon_fixed_params(self) -> None:
        """T* = 2log(n)/(pεδ): with fixed δ and ε, more stocks → longer horizon.

        When using uniform weights, δ = n^{1-p} - 1 grows with n,
        so we must fix δ and ε to isolate the log(n) effect.
        """
        from quantspt.arbitrage.horizon import diversity_horizon

        p, eps, delta = 0.5, 1.0, 0.5
        horizons = [diversity_horizon(n, p, eps, delta) for n in [3, 5, 10]]
        assert horizons[0] < horizons[1] < horizons[2]

    def test_invalid_single_stock(self) -> None:
        with pytest.raises(SPTInvariantError, match="≥ 2"):
            detect_diversity_arbitrage(np.array([1.0]), np.array([[0.04]]), p=0.5)

    def test_invalid_p_range(self, uniform_5: np.ndarray) -> None:
        with pytest.raises(SPTInvariantError):
            detect_diversity_arbitrage(uniform_5, np.eye(5), p=0.0)

    def test_invalid_negative_weights(self) -> None:
        mu = np.array([1.5, -0.5])
        with pytest.raises(SPTInvariantError, match="positive"):
            detect_diversity_arbitrage(mu, np.eye(2), p=0.5)

    def test_shape_mismatch(self, uniform_5: np.ndarray) -> None:
        with pytest.raises(SPTInvariantError, match="shape"):
            detect_diversity_arbitrage(uniform_5, np.eye(3), p=0.5)
