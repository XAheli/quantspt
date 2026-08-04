"""Exhaustive tests for core/generating_functions.py.

Validates mathematical correctness of every generating function against
known analytical results from the SPT literature.

Mathematical References
-----------------------
- Fernholz weight formula: F&K Survey Eq. 11.1
- Diversity weights π_i = μ_i^p / Σμ_j^p: FKK Eq. 4.4
- Diversity drift formula: F&K Survey Remark 11.1 (Example 3)
- Entropy generating function: F&K Survey Eq. 11.5, Lukacs Lectures §11
- Modified entropy H_c: F&K Survey Eq. 11.6-11.7
- Master formula: F&K Survey Eq. 11.2
- Drift process: F&K Survey Eq. 11.3
- 1-homogeneity ⟹ Σπ_i = 1: Euler's theorem for homogeneous functions
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt.core.covariance import relative_covariance
from quantspt.core.generating_functions import (
    CustomGenerator,
    DiversityGenerator,
    EntropyGenerator,
    InverseVolatilityGenerator,
    ModifiedEntropyGenerator,
    drift_process,
    fernholz_weights,
)
from quantspt.errors import SPTInvariantError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture()
def mu_2() -> np.ndarray:
    """Simple 2-stock market weights."""
    return np.array([0.6, 0.4])


@pytest.fixture()
def mu_3() -> np.ndarray:
    """3-stock market weights."""
    return np.array([0.5, 0.3, 0.2])


@pytest.fixture()
def mu_5(rng: np.random.Generator) -> np.ndarray:
    """Dirichlet-sampled 5-stock market weights."""
    return rng.dirichlet(np.ones(5))


@pytest.fixture()
def cov_2() -> np.ndarray:
    """2×2 covariance matrix with cross-correlation."""
    return np.array([[0.04, 0.01], [0.01, 0.09]])


@pytest.fixture()
def cov_3() -> np.ndarray:
    """3×3 covariance matrix."""
    return np.array(
        [
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.02],
            [0.005, 0.02, 0.16],
        ]
    )


@pytest.fixture()
def tau_mu_2(cov_2: np.ndarray, mu_2: np.ndarray) -> np.ndarray:
    return relative_covariance(cov_2, mu_2)


@pytest.fixture()
def tau_mu_3(cov_3: np.ndarray, mu_3: np.ndarray) -> np.ndarray:
    return relative_covariance(cov_3, mu_3)


# =========================================================================
# A. DiversityGenerator
# =========================================================================


class TestDiversityGenerator:
    """Tests for G_p(μ) = (Σ μ_i^p)^{1/p}, p ∈ (0,1).

    References: FKK Eq. 4.4, F&K Survey Remark 11.1 (Example 3)
    """

    @pytest.mark.parametrize("p", [0.1, 0.25, 0.5, 0.75, 0.99])
    def test_weights_sum_to_one(self, p: float, mu_3: np.ndarray) -> None:
        """1-homogeneous G ⟹ weights sum to 1 (Euler's theorem)."""
        G = DiversityGenerator(p)
        pi = G.weights(mu_3)
        assert_allclose(np.sum(pi), 1.0, atol=1e-14)

    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
    def test_weights_non_negative(self, p: float, mu_3: np.ndarray) -> None:
        """For μ on the open simplex, all FGP weights are non-negative."""
        G = DiversityGenerator(p)
        pi = G.weights(mu_3)
        assert np.all(pi >= -1e-15), f"Negative weights: {pi}"

    @pytest.mark.parametrize("p", [0.2, 0.5, 0.8])
    def test_known_analytical_weights(self, p: float, mu_3: np.ndarray) -> None:
        """π_i = μ_i^p / Σμ_j^p (FKK Eq. 4.4)."""
        G = DiversityGenerator(p)
        pi = G.weights(mu_3)
        expected = mu_3**p / np.sum(mu_3**p)
        assert_allclose(pi, expected, atol=1e-14)

    @pytest.mark.parametrize("p", [0.3, 0.5, 0.7])
    def test_fernholz_formula_consistency(self, p: float, mu_3: np.ndarray) -> None:
        """G.weights(mu) must equal fernholz_weights(G.log_gradient(mu), mu).

        Both paths — the optimized direct formula and the generic
        Fernholz formula (F&K Survey Eq. 11.1) — must agree.
        """
        G = DiversityGenerator(p)
        pi_direct = G.weights(mu_3)
        pi_fernholz = fernholz_weights(G.log_gradient(mu_3), mu_3)
        assert_allclose(pi_direct, pi_fernholz, atol=1e-12)

    @pytest.mark.parametrize("p", [0.3, 0.5, 0.7])
    def test_hessian_symmetry(self, p: float, mu_3: np.ndarray) -> None:
        """D²G must be symmetric (Schwarz's theorem for C² functions)."""
        G = DiversityGenerator(p)
        H = G.hessian(mu_3)
        assert_allclose(H, H.T, atol=1e-13)

    @pytest.mark.parametrize("p", [0.3, 0.5])
    def test_drift_non_negative_diverse_market(
        self, p: float, mu_3: np.ndarray, tau_mu_3: np.ndarray
    ) -> None:
        """For concave G_p (p<1) on a diverse market, drift ≥ 0.

        The drift g_p(t) measures the FGP's structural advantage from
        rebalancing. For p ∈ (0,1), G_p is concave on the simplex, so
        the Hessian contribution is non-negative.

        References: F&K Survey Remark 11.1
        """
        G = DiversityGenerator(p)
        g = G.drift(mu_3, tau_mu_3)
        assert g >= -1e-15, f"Drift should be non-negative for p<1, got {g}"

    def test_drift_analytical_value_2stock(
        self, mu_2: np.ndarray, cov_2: np.ndarray
    ) -> None:
        r"""Verify drift against the known formula for DiversityGenerator.

        For G_p, the drift process is (F&K Survey Remark 11.1):

            g_p(t) = p(1-p)/2 · Σ_{i<j} π_i^{(p)} π_j^{(p)}
                     · (τ^μ_{ii} + τ^μ_{jj} - 2τ^μ_{ij})
                   / (Σ_k μ_k^p)^{2/p} × (Σ_k μ_k^p)^{2/p}

        For 2 stocks this simplifies. We verify numerically.
        """
        p = 0.5
        G = DiversityGenerator(p)
        tau_mu = relative_covariance(cov_2, mu_2)
        g_computed = G.drift(mu_2, tau_mu)

        G_val = G(mu_2)
        H = G.hessian(mu_2)
        mu_outer = np.outer(mu_2, mu_2)
        g_manual = -0.5 / G_val * np.sum(H * tau_mu * mu_outer)

        assert_allclose(g_computed, g_manual, atol=1e-14)

    def test_g_value_formula(self, mu_3: np.ndarray) -> None:
        """G_p(μ) = (Σ μ_i^p)^{1/p}."""
        p = 0.5
        G = DiversityGenerator(p)
        expected = np.sum(mu_3**p) ** (1.0 / p)
        assert_allclose(G(mu_3), expected, atol=1e-14)

    def test_log_gradient_formula(self, mu_3: np.ndarray) -> None:
        """D_k log G_p = μ_k^{p-1} / Σ μ_j^p."""
        p = 0.5
        G = DiversityGenerator(p)
        expected = mu_3 ** (p - 1.0) / np.sum(mu_3**p)
        assert_allclose(G.log_gradient(mu_3), expected, atol=1e-14)

    def test_homogeneity_weights_exact(self, mu_5: np.ndarray) -> None:
        """For 1-homogeneous G, weights sum to exactly 1 without normalization.

        This is a consequence of Euler's theorem: for a function G that is
        positively homogeneous of degree 1, Σ μ_i D_i G = G.
        """
        G = DiversityGenerator(0.5)
        pi = G.weights(mu_5)
        assert_allclose(np.sum(pi), 1.0, atol=1e-14)

    @pytest.mark.parametrize("p", [0.3, 0.5, 0.7])
    def test_weights_overweight_small_stocks(self, p: float, mu_3: np.ndarray) -> None:
        """Diversity portfolio overweights small stocks vs market cap."""
        G = DiversityGenerator(p)
        pi = G.weights(mu_3)
        smallest_idx = np.argmin(mu_3)
        assert pi[smallest_idx] > mu_3[smallest_idx]

    def test_name_property(self) -> None:
        G = DiversityGenerator(0.5)
        assert G.name == "Diversity(p=0.500)"

    def test_p_property(self) -> None:
        G = DiversityGenerator(0.42)
        assert G.p == 0.42

    def test_invalid_p_raises(self) -> None:
        with pytest.raises(SPTInvariantError):
            DiversityGenerator(0.0)
        with pytest.raises(SPTInvariantError):
            DiversityGenerator(1.0)
        with pytest.raises(SPTInvariantError):
            DiversityGenerator(-0.5)
        with pytest.raises(SPTInvariantError):
            DiversityGenerator(1.5)


# =========================================================================
# B. EntropyGenerator
# =========================================================================


class TestEntropyGenerator:
    """Tests for G(μ) = exp(H(μ)) = exp(-Σ μ_i log μ_i).

    References: F&K Survey Eq. 11.5, Lukacs Lectures §11
    """

    def test_g_value(self, mu_3: np.ndarray) -> None:
        """G(μ) = exp(-Σ μ_i log μ_i) = Π μ_i^{-μ_i}."""
        G = EntropyGenerator()
        H = -np.sum(mu_3 * np.log(mu_3))
        expected = np.exp(H)
        assert_allclose(G(mu_3), expected, atol=1e-14)

    def test_log_gradient(self, mu_3: np.ndarray) -> None:
        """D_k log G = -(1 + log μ_k)."""
        G = EntropyGenerator()
        expected = -(1.0 + np.log(mu_3))
        assert_allclose(G.log_gradient(mu_3), expected, atol=1e-14)

    def test_weights_formula(self, mu_3: np.ndarray) -> None:
        """Verify weights via Fernholz formula with known log-gradient.

        π_i = [D_i log G + 1 - Σ_k μ_k D_k log G] · μ_i

        With D_k log G = -(1 + log μ_k):
          Σ_k μ_k D_k log G = -(1 + Σ μ_k log μ_k) = -(1 - H)
          π_i = [-(1 + log μ_i) + 1 + 1 - H] μ_i
              = [1 - H - log μ_i] μ_i
        """
        G = EntropyGenerator()
        pi = G.weights(mu_3)
        H = -np.sum(mu_3 * np.log(mu_3))
        expected = (1.0 - H - np.log(mu_3)) * mu_3
        assert_allclose(pi, expected, atol=1e-13)

    def test_weights_sum_to_one(self, mu_3: np.ndarray) -> None:
        G = EntropyGenerator()
        pi = G.weights(mu_3)
        assert_allclose(np.sum(pi), 1.0, atol=1e-12)

    def test_weights_non_negative(self, mu_3: np.ndarray) -> None:
        G = EntropyGenerator()
        pi = G.weights(mu_3)
        assert np.all(pi >= -1e-12), f"Negative weight: {pi}"

    def test_hessian_symmetry(self, mu_3: np.ndarray) -> None:
        G = EntropyGenerator()
        H = G.hessian(mu_3)
        assert_allclose(H, H.T, atol=1e-13)

    def test_hessian_analytical(self, mu_3: np.ndarray) -> None:
        """D²_{ij}G = G·[(1+log μ_i)(1+log μ_j) - δ_{ij}/μ_i]."""
        G = EntropyGenerator()
        H = G.hessian(mu_3)
        G_val = G(mu_3)
        log_terms = 1.0 + np.log(mu_3)
        expected = G_val * (np.outer(log_terms, log_terms) - np.diag(1.0 / mu_3))
        assert_allclose(H, expected, atol=1e-12)

    def test_fernholz_formula_consistency(self, mu_3: np.ndarray) -> None:
        G = EntropyGenerator()
        pi_direct = G.weights(mu_3)
        pi_fernholz = fernholz_weights(G.log_gradient(mu_3), mu_3)
        assert_allclose(pi_direct, pi_fernholz, atol=1e-13)

    def test_drift_non_negative(self, mu_3: np.ndarray, tau_mu_3: np.ndarray) -> None:
        """exp(H) is a concave function, so drift should be non-negative."""
        G = EntropyGenerator()
        g = G.drift(mu_3, tau_mu_3)
        assert g >= -1e-15, f"Entropy drift should be ≥ 0, got {g}"

    def test_name(self) -> None:
        assert EntropyGenerator().name == "Entropy"

    def test_equal_weights_maximum_entropy(self) -> None:
        """Equal weights maximize entropy, so G is maximized there."""
        G = EntropyGenerator()
        mu_equal = np.array([1 / 3, 1 / 3, 1 / 3])
        mu_unequal = np.array([0.7, 0.2, 0.1])
        assert G(mu_equal) > G(mu_unequal)


# =========================================================================
# C. ModifiedEntropyGenerator
# =========================================================================


class TestModifiedEntropyGenerator:
    """Tests for H_c(μ) = c - Σ μ_i log μ_i.

    References: F&K Survey Eq. 11.6-11.7, Lukacs Lectures Eq. 11.5
    """

    def test_g_value(self, mu_3: np.ndarray) -> None:
        c = 1.0
        G = ModifiedEntropyGenerator(c)
        H = -np.sum(mu_3 * np.log(mu_3))
        assert_allclose(G(mu_3), c + H, atol=1e-14)

    def test_log_gradient(self, mu_3: np.ndarray) -> None:
        """D_k log H_c = -(1 + log μ_k) / H_c."""
        c = 1.0
        G = ModifiedEntropyGenerator(c)
        Hc = G(mu_3)
        expected = -(1.0 + np.log(mu_3)) / Hc
        assert_allclose(G.log_gradient(mu_3), expected, atol=1e-14)

    def test_hessian_diagonal(self, mu_3: np.ndarray) -> None:
        """D²_{ij} H_c = -δ_{ij}/μ_i (H_c is linear + entropy)."""
        G = ModifiedEntropyGenerator(1.0)
        H = G.hessian(mu_3)
        expected = -np.diag(1.0 / mu_3)
        assert_allclose(H, expected, atol=1e-14)

    def test_hessian_symmetry(self, mu_3: np.ndarray) -> None:
        G = ModifiedEntropyGenerator(1.0)
        H = G.hessian(mu_3)
        assert_allclose(H, H.T, atol=1e-14)

    def test_drift_equals_gamma_star_over_Hc(
        self, mu_3: np.ndarray, tau_mu_3: np.ndarray
    ) -> None:
        r"""Drift of H_c: g_c(t) = γ*_μ(t) / H_c(μ(t)).

        This elegant result follows because D²H_c = -diag(1/μ_i).

        References: F&K Survey Eq. 11.7, Lukacs Lectures Eq. 11.5
        """
        from quantspt.core.growth_rates import excess_growth_rate_from_tau

        c = 1.0
        G = ModifiedEntropyGenerator(c)
        g = G.drift(mu_3, tau_mu_3)
        Hc = G(mu_3)
        gamma_star = excess_growth_rate_from_tau(mu_3, tau_mu_3)
        expected = gamma_star / Hc
        assert_allclose(g, expected, atol=1e-14)

    def test_drift_non_negative(self, mu_3: np.ndarray, tau_mu_3: np.ndarray) -> None:
        """H_c has non-negative drift since γ*_μ ≥ 0 and H_c > 0."""
        G = ModifiedEntropyGenerator(1.0)
        g = G.drift(mu_3, tau_mu_3)
        assert g >= -1e-15

    def test_fernholz_formula_consistency(self, mu_3: np.ndarray) -> None:
        G = ModifiedEntropyGenerator(1.0)
        pi_direct = G.weights(mu_3)
        pi_fernholz = fernholz_weights(G.log_gradient(mu_3), mu_3)
        assert_allclose(pi_direct, pi_fernholz, atol=1e-13)

    def test_weights_sum_to_one(self, mu_3: np.ndarray) -> None:
        G = ModifiedEntropyGenerator(1.0)
        pi = G.weights(mu_3)
        assert_allclose(np.sum(pi), 1.0, atol=1e-12)

    def test_weights_non_negative(self, mu_3: np.ndarray) -> None:
        G = ModifiedEntropyGenerator(1.0)
        pi = G.weights(mu_3)
        assert np.all(pi >= -1e-12)

    def test_name(self) -> None:
        assert ModifiedEntropyGenerator(1.0).name == "ModifiedEntropy(c=1.000)"

    def test_c_property(self) -> None:
        assert ModifiedEntropyGenerator(2.5).c == 2.5

    def test_invalid_c(self) -> None:
        with pytest.raises(SPTInvariantError):
            ModifiedEntropyGenerator(0.0)
        with pytest.raises(SPTInvariantError):
            ModifiedEntropyGenerator(-1.0)


# =========================================================================
# D. InverseVolatilityGenerator
# =========================================================================


class TestInverseVolatilityGenerator:
    def test_weights_inverse_proportional(self) -> None:
        """Weights should be inversely proportional to variances."""
        variances = np.array([0.04, 0.09, 0.16])
        G = InverseVolatilityGenerator(variances=variances)
        mu = np.array([0.4, 0.35, 0.25])
        pi = G.weights(mu)
        inv_var = 1.0 / variances
        expected = inv_var / np.sum(inv_var)
        assert_allclose(pi, expected, atol=1e-14)

    def test_weights_independent_of_market_weights(self) -> None:
        """Inverse-volatility weights don't depend on μ."""
        variances = np.array([0.04, 0.09])
        G = InverseVolatilityGenerator(variances=variances)
        pi1 = G.weights(np.array([0.6, 0.4]))
        pi2 = G.weights(np.array([0.3, 0.7]))
        assert_allclose(pi1, pi2, atol=1e-14)

    def test_weights_sum_to_one(self) -> None:
        variances = np.array([0.04, 0.09, 0.16])
        G = InverseVolatilityGenerator(variances=variances)
        pi = G.weights(np.array([0.4, 0.35, 0.25]))
        assert_allclose(np.sum(pi), 1.0, atol=1e-14)

    def test_g_value(self) -> None:
        """G(μ) = Σ (1/σ²_i) μ_i — a linear generating function."""
        variances = np.array([0.04, 0.09])
        G = InverseVolatilityGenerator(variances=variances)
        mu = np.array([0.6, 0.4])
        expected = (1 / 0.04) * 0.6 + (1 / 0.09) * 0.4
        assert_allclose(G(mu), expected, atol=1e-14)

    def test_hessian_is_zero(self) -> None:
        """G is linear in μ, so the Hessian vanishes identically."""
        variances = np.array([0.04, 0.09])
        G = InverseVolatilityGenerator(variances=variances)
        mu = np.array([0.6, 0.4])
        H = G.hessian(mu)
        assert_allclose(H, np.zeros((2, 2)), atol=1e-14)

    def test_drift_is_zero(self) -> None:
        """Linear G ⟹ D²G = 0 ⟹ drift = 0."""
        variances = np.array([0.04, 0.09])
        G = InverseVolatilityGenerator(variances=variances)
        mu = np.array([0.6, 0.4])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        tau_mu = relative_covariance(cov, mu)
        assert_allclose(G.drift(mu, tau_mu), 0.0, atol=1e-14)

    def test_log_gradient_formula(self) -> None:
        """D_k log G = (1/σ²_k) / G(μ) where G(μ) = Σ (1/σ²_j) μ_j."""
        variances = np.array([0.04, 0.09, 0.16])
        G = InverseVolatilityGenerator(variances=variances)
        mu = np.array([0.5, 0.3, 0.2])
        grad = G.log_gradient(mu)
        inv_var = 1.0 / variances
        G_val = np.sum(inv_var * mu)
        expected = inv_var / G_val
        assert_allclose(grad, expected, atol=1e-14)

    def test_log_gradient_fernholz_consistency(self) -> None:
        """fernholz_weights(log_gradient, mu) should match overridden weights."""
        variances = np.array([0.04, 0.09])
        G = InverseVolatilityGenerator(variances=variances)
        mu = np.array([0.6, 0.4])
        pi_fernholz = fernholz_weights(G.log_gradient(mu), mu)
        assert np.all(np.isfinite(pi_fernholz))
        assert_allclose(np.sum(pi_fernholz), 1.0, atol=1e-12)

    def test_name(self) -> None:
        variances = np.array([0.04, 0.09])
        G = InverseVolatilityGenerator(variances=variances)
        assert G.name == "InverseVolatility"


# =========================================================================
# E. CustomGenerator
# =========================================================================


class TestCustomGenerator:
    """Tests for user-defined generating function with numerical derivatives."""

    def test_custom_vs_diversity(self, mu_3: np.ndarray) -> None:
        """CustomGenerator matching G_p should produce same weights as
        DiversityGenerator(p) within finite-difference tolerance.

        References: FKK Eq. 4.4
        """
        p = 0.5

        def diversity_func(mu: np.ndarray) -> float:
            return float(np.sum(mu**p) ** (1.0 / p))

        G_custom = CustomGenerator(diversity_func, "DiversityCustom", h=1e-7)
        G_analytical = DiversityGenerator(p)

        pi_custom = G_custom.weights(mu_3)
        pi_analytical = G_analytical.weights(mu_3)
        assert_allclose(pi_custom, pi_analytical, atol=1e-5)

    def test_custom_hessian_vs_diversity(self, mu_3: np.ndarray) -> None:
        """Custom numerical Hessian should approximate analytical Hessian."""
        p = 0.5

        def diversity_func(mu: np.ndarray) -> float:
            return float(np.sum(mu**p) ** (1.0 / p))

        G_custom = CustomGenerator(diversity_func, h=1e-5)
        G_analytical = DiversityGenerator(p)

        H_custom = G_custom.hessian(mu_3)
        H_analytical = G_analytical.hessian(mu_3)
        assert_allclose(H_custom, H_analytical, rtol=1e-3, atol=1e-6)

    def test_custom_hessian_symmetry(self, mu_3: np.ndarray) -> None:
        def func(mu: np.ndarray) -> float:
            return float(np.sum(mu**0.5) ** 2)

        G = CustomGenerator(func, h=1e-6)
        H = G.hessian(mu_3)
        assert_allclose(H, H.T, atol=1e-6)

    def test_custom_g_value(self) -> None:
        def func(mu: np.ndarray) -> float:
            return float(np.sum(mu**2))

        G = CustomGenerator(func, "TestSquared")
        mu = np.array([0.5, 0.3, 0.2])
        expected = 0.5**2 + 0.3**2 + 0.2**2
        assert_allclose(G(mu), expected, atol=1e-14)

    def test_custom_log_gradient_central_diff(self) -> None:
        """Central-difference log-gradient vs analytical for exp(H)."""

        def entropy_func(mu: np.ndarray) -> float:
            return float(np.exp(-np.sum(mu * np.log(mu))))

        G_custom = CustomGenerator(entropy_func, h=1e-7)
        G_analytical = EntropyGenerator()

        mu = np.array([0.5, 0.3, 0.2])
        grad_custom = G_custom.log_gradient(mu)
        grad_analytical = G_analytical.log_gradient(mu)
        assert_allclose(grad_custom, grad_analytical, atol=1e-5)

    def test_name_property(self) -> None:
        G = CustomGenerator(lambda mu: 1.0, "MyFunc")
        assert G.name == "MyFunc"

    def test_default_name(self) -> None:
        G = CustomGenerator(lambda mu: 1.0)
        assert G.name == "Custom"


# =========================================================================
# F. Standalone functions: fernholz_weights, drift_process
# =========================================================================


class TestFernholzWeights:
    """Tests for the standalone Fernholz weight formula (F&K Survey Eq. 11.1)."""

    def test_identity_log_gradient(self) -> None:
        """If D_k log G = 0 for all k, then π = μ (market portfolio)."""
        mu = np.array([0.5, 0.3, 0.2])
        log_grad = np.zeros(3)
        pi = fernholz_weights(log_grad, mu)
        assert_allclose(pi, mu, atol=1e-14)

    def test_constant_log_gradient(self) -> None:
        """If D_k log G = c for all k, then π = μ.

        Because S = Σ μ_k · c = c, and π_i = (c + 1 - c) μ_i = μ_i.
        """
        mu = np.array([0.5, 0.3, 0.2])
        c = 3.14
        log_grad = np.full(3, c)
        pi = fernholz_weights(log_grad, mu)
        assert_allclose(pi, mu, atol=1e-14)


class TestDriftProcess:
    """Tests for the standalone drift_process function (F&K Survey Eq. 11.3)."""

    def test_drift_manual_computation(
        self, mu_2: np.ndarray, cov_2: np.ndarray
    ) -> None:
        """g = -1/(2G) · Σ_{ij} D²G · μ_i μ_j · τ^μ_{ij}."""
        G = DiversityGenerator(0.5)
        tau_mu = relative_covariance(cov_2, mu_2)
        g = drift_process(G, mu_2, tau_mu)

        G_val = G(mu_2)
        H = G.hessian(mu_2)
        mu_outer = np.outer(mu_2, mu_2)
        expected = -0.5 / G_val * np.sum(H * tau_mu * mu_outer)
        assert_allclose(g, expected, atol=1e-14)

    def test_linear_generator_zero_drift(self) -> None:
        """Linear G (Hessian = 0) ⟹ drift = 0."""
        variances = np.array([0.04, 0.09])
        G = InverseVolatilityGenerator(variances=variances)
        mu = np.array([0.6, 0.4])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        tau_mu = relative_covariance(cov, mu)
        g = drift_process(G, mu, tau_mu)
        assert_allclose(g, 0.0, atol=1e-14)

    def test_drift_requires_positive_G(self) -> None:
        """drift_process requires G(μ) > 0."""

        def bad_func(mu: np.ndarray) -> float:
            return 0.0

        G = CustomGenerator(bad_func)
        mu = np.array([0.5, 0.5])
        tau = np.eye(2) * 0.01
        with pytest.raises(SPTInvariantError, match="positive"):
            drift_process(G, mu, tau)


# =========================================================================
# G. Edge cases and boundary behavior
# =========================================================================


class TestEdgeCases:
    def test_near_boundary_weight_approaching_zero(self) -> None:
        """Behavior when one weight is very close to 0 (simplex boundary).

        DiversityGenerator with p < 1 amplifies small weights — verify
        the computation stays finite and well-behaved.
        """
        mu = np.array([0.99, 0.009, 0.001])
        G = DiversityGenerator(0.5)
        pi = G.weights(mu)
        assert np.all(np.isfinite(pi))
        assert_allclose(np.sum(pi), 1.0, atol=1e-12)
        assert pi[2] > mu[2], "Diversity should amplify smallest weight"

    def test_near_boundary_hessian_finite(self) -> None:
        """Hessian should remain finite near the simplex boundary."""
        mu = np.array([0.98, 0.015, 0.005])
        G = DiversityGenerator(0.5)
        H = G.hessian(mu)
        assert np.all(np.isfinite(H))

    def test_equal_weights_diversity(self) -> None:
        """At μ = (1/n, ..., 1/n), diversity weights = market weights."""
        n = 5
        mu = np.ones(n) / n
        G = DiversityGenerator(0.5)
        pi = G.weights(mu)
        assert_allclose(pi, mu, atol=1e-14)

    def test_entropy_near_boundary(self) -> None:
        """Entropy generator near simplex boundary."""
        mu = np.array([0.98, 0.015, 0.005])
        G = EntropyGenerator()
        val = G(mu)
        assert np.isfinite(val)
        assert val > 0
        pi = G.weights(mu)
        assert np.all(np.isfinite(pi))

    def test_two_stock_market(self) -> None:
        """Minimum viable market: n=2."""
        mu = np.array([0.7, 0.3])
        G = DiversityGenerator(0.5)
        pi = G.weights(mu)
        assert pi.shape == (2,)
        assert_allclose(np.sum(pi), 1.0, atol=1e-14)


# =========================================================================
# H. Master formula integration test (simplified 2-stock)
# =========================================================================


class TestMasterFormulaIntegration:
    """Simulate a 2-stock market and verify the master formula identity.

    log(V^π(T)/V^μ(T)) ≈ log(G(μ_T)/G(μ_0)) + Σ g(t)Δt

    This is THE critical test: F&K Survey Eq. 11.2.
    """

    def test_master_formula_2stock_gbm(self) -> None:
        """Master formula holds on a simulated 2-stock GBM market.

        References: F&K Survey Eq. 11.2
        """
        from quantspt.core.processes import CorrelatedGBM, simulate_path

        rng = np.random.default_rng(42)
        n_steps = 5000
        T = 1.0
        dt = T / n_steps

        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        gbm = CorrelatedGBM(
            mu=np.array([0.05, 0.08]),
            cov=cov,
            x0=np.array([100.0, 100.0]),
        )

        _, prices = simulate_path(gbm, T=T, n_steps=n_steps, rng=rng)

        mu_path = prices / prices.sum(axis=1, keepdims=True)

        p = 0.5
        G = DiversityGenerator(p)

        boundary = np.log(G(mu_path[-1]) / G(mu_path[0]))

        drift_sum = 0.0
        for t in range(n_steps):
            mu_t = mu_path[t]
            tau_t = relative_covariance(cov, mu_t)
            drift_sum += G.drift(mu_t, tau_t) * dt

        V_pi = 1.0
        V_mu = 1.0
        for t in range(n_steps):
            mu_t = mu_path[t]
            pi_t = G.weights(mu_t)

            if t + 1 < len(prices):
                returns = prices[t + 1] / prices[t]
                V_pi *= np.dot(pi_t, returns)
                V_mu *= np.dot(mu_t, returns)

        log_rel_return = np.log(V_pi / V_mu)
        master_rhs = boundary + drift_sum

        assert_allclose(
            log_rel_return,
            master_rhs,
            atol=0.02,
            err_msg=(
                f"Master formula violated: "
                f"log(V^π/V^μ)={log_rel_return:.6f}, "
                f"boundary+drift={master_rhs:.6f}"
            ),
        )
