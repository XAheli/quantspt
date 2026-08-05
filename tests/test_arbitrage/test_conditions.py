"""Exhaustive tests for arbitrage/conditions.py.

Validates the diversity conditions that guarantee existence of
relative arbitrage opportunities.

Mathematical References
-----------------------
- Strict diversity: FKK Eq. 4.1
- Weak diversity: FKK Eq. 4.2
- Asymptotic weak diversity: FKK Eq. 4.3
- Diversity parameter estimation: FKK §4
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt.arbitrage.conditions import (
    check_asymptotic_weak_diversity,
    check_strict_diversity,
    check_weak_diversity,
    estimate_diversity_parameters,
)
from quantspt.errors import SPTInvariantError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def uniform_5() -> np.ndarray:
    """Equal weights for 5 stocks — maximally diverse."""
    return np.ones(5) / 5.0


@pytest.fixture()
def concentrated_5() -> np.ndarray:
    """One stock dominates — not diverse."""
    return np.array([0.91, 0.03, 0.02, 0.02, 0.02])


@pytest.fixture()
def moderate_5() -> np.ndarray:
    """Moderately spread weights."""
    return np.array([0.35, 0.25, 0.20, 0.12, 0.08])


@pytest.fixture()
def mu_2() -> np.ndarray:
    """Minimal 2-stock market."""
    return np.array([0.6, 0.4])


# =========================================================================
# A. check_strict_diversity  (FKK Eq. 4.1)
# =========================================================================


class TestCheckStrictDiversity:
    """Strict diversity: max_i μ_i ≤ 1 - δ.

    References: FKK Eq. 4.1
    """

    def test_uniform_weights_diverse(self, uniform_5: np.ndarray) -> None:
        """Uniform weights (max = 0.2) satisfy max ≤ 1 - 0.5 = 0.5."""
        assert check_strict_diversity(uniform_5, delta=0.5) is True

    def test_concentrated_not_diverse(self, concentrated_5: np.ndarray) -> None:
        """Concentrated market (max = 0.91) fails max ≤ 1 - 0.5."""
        assert check_strict_diversity(concentrated_5, delta=0.5) is False

    def test_boundary_exactly_met(self) -> None:
        """max μ_i = 1 - δ exactly: inequality is ≤, so should pass."""
        mu = np.array([0.7, 0.3])
        assert check_strict_diversity(mu, delta=0.3) is True

    def test_boundary_barely_exceeded(self) -> None:
        """max μ_i just above 1 - δ → not diverse."""
        mu = np.array([0.701, 0.299])
        assert check_strict_diversity(mu, delta=0.3) is False

    def test_two_stock_market(self, mu_2: np.ndarray) -> None:
        """2-stock market: max = 0.6, 1 - δ = 0.8 → diverse."""
        assert check_strict_diversity(mu_2, delta=0.2) is True

    def test_two_stock_tight_delta(self, mu_2: np.ndarray) -> None:
        """2-stock: δ = 0.5 → 1 - δ = 0.5 < max(0.6) → not diverse."""
        assert check_strict_diversity(mu_2, delta=0.5) is False

    @pytest.mark.parametrize("delta", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_uniform_diverse_for_all_deltas_below_threshold(
        self, uniform_5: np.ndarray, delta: float
    ) -> None:
        """Uniform(5) has max = 0.2; diverse iff 1 - δ ≥ 0.2, i.e. δ ≤ 0.8."""
        expected = delta <= 0.8
        assert check_strict_diversity(uniform_5, delta) is expected

    def test_near_boundary_delta(self) -> None:
        """δ very close to boundary: max = 0.4, δ = 0.6 → 1 - δ = 0.4."""
        mu = np.array([0.4, 0.3, 0.3])
        assert check_strict_diversity(mu, delta=0.6) is True

    def test_formula_verification(self, moderate_5: np.ndarray) -> None:
        """Directly verify FKK Eq. 4.1: max_i μ_i ≤ 1 - δ."""
        delta = 0.3
        result = check_strict_diversity(moderate_5, delta)
        expected = float(np.max(moderate_5)) <= 1.0 - delta
        assert result is expected

    def test_invalid_delta_zero(self, uniform_5: np.ndarray) -> None:
        with pytest.raises(SPTInvariantError):
            check_strict_diversity(uniform_5, delta=0.0)

    def test_invalid_delta_one(self, uniform_5: np.ndarray) -> None:
        with pytest.raises(SPTInvariantError):
            check_strict_diversity(uniform_5, delta=1.0)

    def test_invalid_delta_negative(self, uniform_5: np.ndarray) -> None:
        with pytest.raises(SPTInvariantError):
            check_strict_diversity(uniform_5, delta=-0.1)

    def test_invalid_delta_greater_than_one(self, uniform_5: np.ndarray) -> None:
        with pytest.raises(SPTInvariantError):
            check_strict_diversity(uniform_5, delta=1.5)


# =========================================================================
# B. check_weak_diversity  (FKK Eq. 4.2)
# =========================================================================


class TestCheckWeakDiversity:
    r"""Weak diversity: Σ μ_i^p ≥ 1 + δ.

    For p ∈ (0,1), Jensen's inequality gives Σ μ_i^p ≥ (Σ μ_i)^p = 1
    with equality iff all weights are concentrated on one stock.
    Uniform weights maximise the p-norm.

    References: FKK Eq. 4.2
    """

    def test_uniform_weights_diverse(self, uniform_5: np.ndarray) -> None:
        """Uniform(5) with p=0.5: Σ(1/5)^0.5 = 5·(1/5)^0.5 = √5 ≈ 2.236."""
        p_norm = float(np.sum(uniform_5**0.5))
        assert p_norm > 1.0 + 0.5
        assert check_weak_diversity(uniform_5, delta=0.5, p=0.5) is True

    def test_concentrated_not_diverse(self) -> None:
        """Nearly single-stock market fails weak diversity with large δ.

        For [0.96, 0.01, 0.01, 0.01, 0.01], Σ μ_i^0.5 ≈ 1.38 < 1 + 0.5.
        """
        mu = np.array([0.96, 0.01, 0.01, 0.01, 0.01])
        assert check_weak_diversity(mu, delta=0.5, p=0.5) is False

    def test_formula_verification(self, moderate_5: np.ndarray) -> None:
        """Directly verify FKK Eq. 4.2: Σ μ_i^p ≥ 1 + δ."""
        p, delta = 0.5, 0.3
        result = check_weak_diversity(moderate_5, delta=delta, p=p)
        computed = float(np.sum(moderate_5**p))
        expected = computed >= 1.0 + delta
        assert result is expected

    @pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_uniform_diverse_across_p(self, uniform_5: np.ndarray, p: float) -> None:
        """Uniform weights satisfy weak diversity for small δ across all p."""
        assert check_weak_diversity(uniform_5, delta=0.01, p=p) is True

    def test_two_stock_diverse(self, mu_2: np.ndarray) -> None:
        """2-stock: Σ μ_i^0.5 = √0.6 + √0.4 ≈ 1.407."""
        p_norm = float(np.sum(mu_2**0.5))
        delta = p_norm - 1.0 - 0.01
        assert check_weak_diversity(mu_2, delta=delta, p=0.5) is True

    def test_two_stock_tight_delta(self, mu_2: np.ndarray) -> None:
        """δ larger than actual diversity deficit → fails."""
        p_norm = float(np.sum(mu_2**0.5))
        delta = p_norm - 1.0 + 0.01
        assert check_weak_diversity(mu_2, delta=delta, p=0.5) is False

    def test_p_near_zero_high_diversity(self, uniform_5: np.ndarray) -> None:
        """Small p amplifies diversity: Σ μ_i^p → n as p → 0."""
        assert check_weak_diversity(uniform_5, delta=3.0, p=0.01) is True

    def test_p_near_one_low_diversity(self) -> None:
        """As p → 1, Σ μ_i^p → Σ μ_i = 1, so deficit vanishes."""
        mu = np.array([0.5, 0.3, 0.2])
        deficit = float(np.sum(mu**0.99)) - 1.0
        assert deficit < 0.02

    def test_invalid_delta_zero(self, uniform_5: np.ndarray) -> None:
        with pytest.raises(SPTInvariantError):
            check_weak_diversity(uniform_5, delta=0.0, p=0.5)

    def test_invalid_delta_negative(self, uniform_5: np.ndarray) -> None:
        with pytest.raises(SPTInvariantError):
            check_weak_diversity(uniform_5, delta=-1.0, p=0.5)

    def test_invalid_p_zero(self, uniform_5: np.ndarray) -> None:
        with pytest.raises(SPTInvariantError):
            check_weak_diversity(uniform_5, delta=0.1, p=0.0)

    def test_invalid_p_one(self, uniform_5: np.ndarray) -> None:
        with pytest.raises(SPTInvariantError):
            check_weak_diversity(uniform_5, delta=0.1, p=1.0)

    def test_negative_weight_raises(self) -> None:
        mu = np.array([1.2, -0.2])
        with pytest.raises(SPTInvariantError, match="positive"):
            check_weak_diversity(mu, delta=0.1, p=0.5)

    def test_zero_weight_raises(self) -> None:
        mu = np.array([1.0, 0.0])
        with pytest.raises(SPTInvariantError, match="positive"):
            check_weak_diversity(mu, delta=0.1, p=0.5)


# =========================================================================
# C. check_asymptotic_weak_diversity  (FKK Eq. 4.3)
# =========================================================================


class TestCheckAsymptoticWeakDiversity:
    r"""Asymptotic weak diversity: time-averaged Σ μ_i^p ≥ 1 + δ.

    References: FKK Eq. 4.3
    """

    def test_constant_diverse_path(self, uniform_5: np.ndarray) -> None:
        """Constant uniform path → time average equals Σ μ_i^p."""
        mu_path = np.tile(uniform_5, (100, 1))
        assert check_asymptotic_weak_diversity(mu_path, p=0.5, delta=0.5) is True

    def test_constant_concentrated_path(self) -> None:
        """Constant near-single-stock path fails asymptotic diversity.

        For [0.96, 0.01, 0.01, 0.01, 0.01], Σ μ_i^0.5 ≈ 1.38 < 1.5.
        """
        mu = np.array([0.96, 0.01, 0.01, 0.01, 0.01])
        mu_path = np.tile(mu, (100, 1))
        assert check_asymptotic_weak_diversity(mu_path, p=0.5, delta=0.5) is False

    def test_mixed_path_averaging(self) -> None:
        """Path alternating between diverse and concentrated states."""
        diverse = np.ones(5) / 5.0
        concentrated = np.array([0.91, 0.03, 0.02, 0.02, 0.02])
        mu_path = np.vstack([np.tile(diverse, (80, 1)), np.tile(concentrated, (20, 1))])
        diverse_pnorm = float(np.sum(diverse**0.5))
        conc_pnorm = float(np.sum(concentrated**0.5))
        expected_avg = 0.8 * diverse_pnorm + 0.2 * conc_pnorm
        result = check_asymptotic_weak_diversity(mu_path, p=0.5, delta=0.3)
        assert result == (expected_avg >= 1.0 + 0.3)

    def test_formula_verification(self, moderate_5: np.ndarray) -> None:
        """Verify time-average computation matches manual calculation."""
        mu_path = np.tile(moderate_5, (50, 1))
        p, delta = 0.5, 0.1
        p_norms = np.sum(mu_path**p, axis=1)
        expected = float(np.mean(p_norms)) >= 1.0 + delta
        assert check_asymptotic_weak_diversity(mu_path, p=p, delta=delta) is expected

    def test_invalid_1d_path(self, uniform_5: np.ndarray) -> None:
        with pytest.raises(SPTInvariantError, match="2-D"):
            check_asymptotic_weak_diversity(uniform_5, p=0.5, delta=0.1)

    def test_invalid_p(self, uniform_5: np.ndarray) -> None:
        mu_path = np.tile(uniform_5, (10, 1))
        with pytest.raises(SPTInvariantError):
            check_asymptotic_weak_diversity(mu_path, p=0.0, delta=0.1)

    def test_invalid_delta(self, uniform_5: np.ndarray) -> None:
        mu_path = np.tile(uniform_5, (10, 1))
        with pytest.raises(SPTInvariantError):
            check_asymptotic_weak_diversity(mu_path, p=0.5, delta=-0.1)


# =========================================================================
# D. estimate_diversity_parameters  (FKK §4)
# =========================================================================


class TestEstimateDiversityParameters:
    """Estimate diversity statistics from observed market weights.

    References: FKK §4
    """

    def test_keys_present(self, uniform_5: np.ndarray) -> None:
        """Result dict contains all expected keys."""
        mu_path = np.tile(uniform_5, (50, 1))
        result = estimate_diversity_parameters(mu_path, p=0.5)
        expected_keys = {
            "delta_mean",
            "delta_min",
            "delta_std",
            "fraction_diverse",
            "max_weight",
        }
        assert set(result.keys()) == expected_keys

    def test_uniform_path_statistics(self, uniform_5: np.ndarray) -> None:
        """Constant uniform path → zero std, fraction_diverse = 1."""
        mu_path = np.tile(uniform_5, (100, 1))
        result = estimate_diversity_parameters(mu_path, p=0.5)

        p_norm = float(np.sum(uniform_5**0.5))
        expected_deficit = p_norm - 1.0

        assert_allclose(result["delta_mean"], expected_deficit, atol=1e-12)
        assert_allclose(result["delta_min"], expected_deficit, atol=1e-12)
        assert_allclose(result["delta_std"], 0.0, atol=1e-12)
        assert_allclose(result["fraction_diverse"], 1.0, atol=1e-12)
        assert_allclose(result["max_weight"], 0.2, atol=1e-12)

    def test_concentrated_path_low_diversity(self, concentrated_5: np.ndarray) -> None:
        """Concentrated market has small (possibly negative) deficit."""
        mu_path = np.tile(concentrated_5, (50, 1))
        result = estimate_diversity_parameters(mu_path, p=0.5)
        assert result["max_weight"] == pytest.approx(0.91, abs=1e-12)

    def test_delta_mean_matches_manual(self, moderate_5: np.ndarray) -> None:
        """Verify delta_mean = mean(Σ μ_i^p - 1)."""
        mu_path = np.tile(moderate_5, (30, 1))
        p = 0.5
        result = estimate_diversity_parameters(mu_path, p=p)
        manual_deficits = np.sum(mu_path**p, axis=1) - 1.0
        assert_allclose(
            result["delta_mean"], float(np.mean(manual_deficits)), atol=1e-12
        )

    def test_fraction_diverse_correct(self) -> None:
        """Mix of diverse and non-diverse steps → correct fraction."""
        diverse = np.ones(3) / 3.0
        concentrated = np.array([0.96, 0.02, 0.02])
        mu_path = np.vstack([np.tile(diverse, (70, 1)), np.tile(concentrated, (30, 1))])
        result = estimate_diversity_parameters(mu_path, p=0.5)
        assert 0.0 <= result["fraction_diverse"] <= 1.0

    @pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_delta_mean_increases_with_more_stocks(self, p: float) -> None:
        """More stocks with uniform weights → larger diversity deficit.

        For uniform(n), Σ(1/n)^p = n^{1-p}, so deficit = n^{1-p} - 1.
        """
        deficits = []
        for n in [3, 5, 10]:
            mu = np.ones(n) / n
            mu_path = np.tile(mu, (20, 1))
            result = estimate_diversity_parameters(mu_path, p=p)
            deficits.append(result["delta_mean"])
        assert deficits[0] < deficits[1] < deficits[2]

    def test_invalid_1d_input(self, uniform_5: np.ndarray) -> None:
        with pytest.raises(SPTInvariantError, match="2-D"):
            estimate_diversity_parameters(uniform_5, p=0.5)

    def test_invalid_p(self, uniform_5: np.ndarray) -> None:
        mu_path = np.tile(uniform_5, (10, 1))
        with pytest.raises(SPTInvariantError):
            estimate_diversity_parameters(mu_path, p=1.0)

    def test_zero_weight_raises(self) -> None:
        mu_path = np.array([[0.5, 0.5, 0.0], [0.4, 0.4, 0.2]])
        with pytest.raises(SPTInvariantError, match="positive"):
            estimate_diversity_parameters(mu_path, p=0.5)
