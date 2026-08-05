"""Exhaustive tests for arbitrage/horizon.py.

Validates minimum horizon computations and parameter sensitivity
analysis for diversity-based and entropy-based relative arbitrage.

Mathematical References
-----------------------
- Diversity horizon: FKK Eq. 4.5, T* = 2 log(n) / (p ε δ)
- Entropy horizon: F&K Survey Eq. 11.8–11.12, T > H(μ(0)) / ζ
- Sensitivity analysis: FKK §4
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt.arbitrage.horizon import (
    diversity_horizon,
    entropy_horizon,
    horizon_sensitivity,
)
from quantspt.errors import SPTInvariantError

# =========================================================================
# A. diversity_horizon  (FKK Eq. 4.5)
# =========================================================================


class TestDiversityHorizon:
    r"""T* = 2 log(n) / (p ε δ).

    References: FKK Eq. 4.5
    """

    def test_formula_exact(self) -> None:
        """Verify T* = 2 log(n) / (p ε δ) for known parameters."""
        n, p, eps, delta = 5, 0.5, 0.1, 0.3
        expected = 2.0 * np.log(n) / (p * eps * delta)
        assert_allclose(diversity_horizon(n, p, eps, delta), expected, atol=1e-12)

    def test_two_stocks(self) -> None:
        """Minimal market n=2."""
        T = diversity_horizon(n=2, p=0.5, eps=1.0, delta=0.5)
        expected = 2.0 * np.log(2) / (0.5 * 1.0 * 0.5)
        assert_allclose(T, expected, atol=1e-12)

    @pytest.mark.parametrize("n", [2, 5, 10, 50, 100])
    def test_horizon_increases_with_n(self, n: int) -> None:
        """T* ∝ log(n): more stocks → longer horizon."""
        T = diversity_horizon(n, p=0.5, eps=0.1, delta=0.3)
        expected = 2.0 * np.log(n) / (0.5 * 0.1 * 0.3)
        assert_allclose(T, expected, atol=1e-10)

    def test_monotone_in_n(self) -> None:
        """T*(n) is strictly increasing in n."""
        horizons = [diversity_horizon(n, 0.5, 0.1, 0.3) for n in [2, 5, 10, 50]]
        for i in range(len(horizons) - 1):
            assert horizons[i] < horizons[i + 1]

    def test_horizon_decreases_with_eps(self) -> None:
        """Larger ε (stronger non-degeneracy) → shorter horizon."""
        T1 = diversity_horizon(5, 0.5, eps=0.1, delta=0.3)
        T2 = diversity_horizon(5, 0.5, eps=0.5, delta=0.3)
        assert T2 < T1

    def test_horizon_decreases_with_delta(self) -> None:
        """Larger δ (more diverse) → shorter horizon."""
        T1 = diversity_horizon(5, 0.5, 0.1, delta=0.1)
        T2 = diversity_horizon(5, 0.5, 0.1, delta=0.5)
        assert T2 < T1

    def test_horizon_inversely_proportional_to_p(self) -> None:
        """T* ∝ 1/p: verify by doubling p halves T*."""
        T1 = diversity_horizon(5, p=0.25, eps=0.1, delta=0.3)
        T2 = diversity_horizon(5, p=0.5, eps=0.1, delta=0.3)
        assert_allclose(T1 / T2, 2.0, atol=1e-12)

    def test_large_market(self) -> None:
        """n=1000 stocks: finite, positive horizon."""
        T = diversity_horizon(1000, 0.5, 0.01, 0.1)
        assert T > 0.0
        assert np.isfinite(T)

    def test_invalid_n_one(self) -> None:
        with pytest.raises(SPTInvariantError, match="≥ 2"):
            diversity_horizon(1, 0.5, 0.1, 0.3)

    def test_invalid_p_zero(self) -> None:
        with pytest.raises(SPTInvariantError):
            diversity_horizon(5, 0.0, 0.1, 0.3)

    def test_invalid_p_one(self) -> None:
        with pytest.raises(SPTInvariantError):
            diversity_horizon(5, 1.0, 0.1, 0.3)

    def test_invalid_eps_zero(self) -> None:
        with pytest.raises(SPTInvariantError):
            diversity_horizon(5, 0.5, 0.0, 0.3)

    def test_invalid_eps_negative(self) -> None:
        with pytest.raises(SPTInvariantError):
            diversity_horizon(5, 0.5, -0.1, 0.3)

    def test_invalid_delta_zero(self) -> None:
        with pytest.raises(SPTInvariantError):
            diversity_horizon(5, 0.5, 0.1, 0.0)

    def test_invalid_delta_negative(self) -> None:
        with pytest.raises(SPTInvariantError):
            diversity_horizon(5, 0.5, 0.1, -0.3)


# =========================================================================
# B. entropy_horizon  (F&K Survey Eq. 11.8–11.12)
# =========================================================================


class TestEntropyHorizon:
    r"""T > H(μ(0)) / ζ.

    References: F&K Survey Eq. 11.8–11.12
    """

    def test_formula_exact(self) -> None:
        """T = H / ζ for known values."""
        H, zeta = 1.5, 0.05
        expected = H / zeta
        assert_allclose(entropy_horizon(H, zeta), expected, atol=1e-12)

    def test_uniform_5_entropy(self) -> None:
        """Uniform(5) entropy: H = log(5) ≈ 1.609."""
        mu = np.ones(5) / 5.0
        H = -float(np.sum(mu * np.log(mu)))
        zeta = 0.01
        T = entropy_horizon(H, zeta)
        assert_allclose(T, H / zeta, atol=1e-12)

    def test_concentrated_short_horizon(self) -> None:
        """Low entropy → short horizon."""
        H_low = 0.1
        T = entropy_horizon(H_low, zeta=0.05)
        assert pytest.approx(2.0) == T

    def test_high_entropy_long_horizon(self) -> None:
        """High entropy → long horizon."""
        H_high = 5.0
        T = entropy_horizon(H_high, zeta=0.05)
        assert pytest.approx(100.0) == T

    def test_horizon_proportional_to_entropy(self) -> None:
        """T ∝ H: doubling entropy doubles horizon."""
        T1 = entropy_horizon(1.0, 0.05)
        T2 = entropy_horizon(2.0, 0.05)
        assert_allclose(T2 / T1, 2.0, atol=1e-12)

    def test_horizon_inversely_proportional_to_zeta(self) -> None:
        """T ∝ 1/ζ: doubling ζ halves horizon."""
        T1 = entropy_horizon(1.5, 0.05)
        T2 = entropy_horizon(1.5, 0.10)
        assert_allclose(T1 / T2, 2.0, atol=1e-12)

    def test_zero_entropy(self) -> None:
        """H = 0 (single stock) → T = 0."""
        T = entropy_horizon(0.0, zeta=0.05)
        assert_allclose(T, 0.0, atol=1e-12)

    def test_invalid_negative_entropy(self) -> None:
        with pytest.raises(SPTInvariantError, match="non-negative"):
            entropy_horizon(-0.1, zeta=0.05)

    def test_invalid_zeta_zero(self) -> None:
        with pytest.raises(SPTInvariantError):
            entropy_horizon(1.0, zeta=0.0)

    def test_invalid_zeta_negative(self) -> None:
        with pytest.raises(SPTInvariantError):
            entropy_horizon(1.0, zeta=-0.01)


# =========================================================================
# C. horizon_sensitivity  (FKK §4)
# =========================================================================


class TestHorizonSensitivity:
    """Semi-elasticity of T* w.r.t. each parameter.

    T* = 2 log(n) / (p ε δ)

    Analytical semi-elasticities:
    - ∂T*/∂ε · 1/T* = -1/ε  (inverse proportional)
    - ∂T*/∂δ · 1/T* = -1/δ  (inverse proportional)
    - ∂T*/∂p · 1/T* = -1/p  (inverse proportional)

    References: FKK §4
    """

    def test_keys_present(self) -> None:
        result = horizon_sensitivity(5, 0.5, 0.1, 0.3)
        assert set(result.keys()) == {"n", "p", "eps", "delta"}

    def test_eps_sensitivity_negative(self) -> None:
        """Increasing ε reduces T* → negative semi-elasticity."""
        result = horizon_sensitivity(5, 0.5, 0.1, 0.3)
        assert result["eps"] < 0

    def test_delta_sensitivity_negative(self) -> None:
        """Increasing δ reduces T* → negative semi-elasticity."""
        result = horizon_sensitivity(5, 0.5, 0.1, 0.3)
        assert result["delta"] < 0

    def test_p_sensitivity_negative(self) -> None:
        """Increasing p reduces T* → negative semi-elasticity."""
        result = horizon_sensitivity(5, 0.5, 0.1, 0.3)
        assert result["p"] < 0

    def test_n_sensitivity_positive(self) -> None:
        """Increasing n increases T* → positive sensitivity."""
        result = horizon_sensitivity(5, 0.5, 0.1, 0.3)
        assert result["n"] > 0

    def test_eps_sensitivity_analytical(self) -> None:
        r"""For T* = C/ε, ∂T*/∂ε / T* = -1/ε (analytically).

        The numerical derivative should approximate this.
        """
        eps = 0.1
        result = horizon_sensitivity(5, 0.5, eps, 0.3, perturbation=0.001)
        assert_allclose(result["eps"], -1.0 / eps, rtol=0.01)

    def test_delta_sensitivity_analytical(self) -> None:
        r"""For T* = C/δ, ∂T*/∂δ / T* = -1/δ."""
        delta = 0.3
        result = horizon_sensitivity(5, 0.5, 0.1, delta, perturbation=0.001)
        assert_allclose(result["delta"], -1.0 / delta, rtol=0.01)

    def test_p_sensitivity_analytical(self) -> None:
        r"""For T* = C/p, ∂T*/∂p / T* = -1/p."""
        p = 0.5
        result = horizon_sensitivity(5, p, 0.1, 0.3, perturbation=0.001)
        assert_allclose(result["p"], -1.0 / p, rtol=0.01)

    def test_different_perturbation_sizes(self) -> None:
        """Smaller perturbation → more accurate sensitivity."""
        r1 = horizon_sensitivity(5, 0.5, 0.1, 0.3, perturbation=0.1)
        r2 = horizon_sensitivity(5, 0.5, 0.1, 0.3, perturbation=0.001)
        assert abs(r2["eps"] - (-10.0)) < abs(r1["eps"] - (-10.0))

    def test_invalid_perturbation(self) -> None:
        with pytest.raises(SPTInvariantError):
            horizon_sensitivity(5, 0.5, 0.1, 0.3, perturbation=-0.01)

    def test_invalid_n(self) -> None:
        with pytest.raises(SPTInvariantError, match="≥ 2"):
            horizon_sensitivity(1, 0.5, 0.1, 0.3)

    def test_invalid_p(self) -> None:
        with pytest.raises(SPTInvariantError):
            horizon_sensitivity(5, 0.0, 0.1, 0.3)

    def test_invalid_eps(self) -> None:
        with pytest.raises(SPTInvariantError):
            horizon_sensitivity(5, 0.5, 0.0, 0.3)

    def test_invalid_delta(self) -> None:
        with pytest.raises(SPTInvariantError):
            horizon_sensitivity(5, 0.5, 0.1, 0.0)
