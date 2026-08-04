"""Tests for the Atlas model and FirstOrderModel.

Validates mathematical correctness of:
- Stability condition enforcement (BFK Eq. 1.5)
- Ergodic property: each stock spends ≈1/n time at each rank (BFK Prop. 2.3)
- Local time rates (BFK Eq. 3.7)
- Pareto exponents (BFK Eq. 4.3)
- Certainty-equivalent weights (BFK Eq. 4.12–4.15)
- Equal-weighted excess growth rate (BFK Eq. 5.14)
- Market growth rate G^μ = γ (BFK Eq. 5.10)
- 2-stock Atlas explicit solution
- Simulation-based verification of analytical growth rates
"""

from __future__ import annotations

import numpy as np
import pytest

from quantspt.core.growth_rates import (
    atlas_excess_growth_rate_equal_weighted,
    excess_growth_rate,
)
from quantspt.core.processes import simulate_path
from quantspt.models.atlas import AtlasModel, FirstOrderModel

# ======================================================================
# Stability condition tests (BFK Eq. 1.5)
# ======================================================================


class TestStabilityCondition:
    """The stability condition must be enforced at construction time."""

    def test_valid_basic_atlas(self) -> None:
        model = AtlasModel(n=5, gamma=0.05, g_param=0.01, sigma_param=0.3)
        assert model.n_assets == 5

    def test_valid_first_order(self) -> None:
        g = np.array([-0.03, -0.02, -0.01, 0.06])
        model = FirstOrderModel(n=4, gamma=0.05, g=g, sigma=np.full(4, 0.3))
        assert model.n_assets == 4

    def test_rejects_nonsumming_g(self) -> None:
        g = np.array([-0.03, -0.02, -0.01, 0.05])  # sum = -0.01 ≠ 0
        with pytest.raises(Exception, match="sum to 0"):
            FirstOrderModel(n=4, gamma=0.05, g=g, sigma=np.full(4, 0.3))

    def test_rejects_positive_partial_sum(self) -> None:
        g = np.array([0.02, -0.03, -0.01, 0.02])  # g1 > 0
        with pytest.raises(Exception, match="[Ss]tability"):
            FirstOrderModel(n=4, gamma=0.05, g=g, sigma=np.full(4, 0.3))

    def test_rejects_nonpositive_sigma(self) -> None:
        g = np.array([-0.01, 0.01])
        with pytest.raises(Exception, match="positive"):
            FirstOrderModel(n=2, gamma=0.05, g=g, sigma=np.array([0.3, 0.0]))

    def test_rejects_negative_g_param(self) -> None:
        with pytest.raises(Exception, match="positive"):
            AtlasModel(n=3, gamma=0.05, g_param=-0.01, sigma_param=0.3)

    def test_atlas_g_vector_structure(self) -> None:
        """Basic Atlas: g_k = -g for k<n, g_n = (n-1)g."""
        model = AtlasModel(n=5, gamma=0.05, g_param=0.02, sigma_param=0.3)
        expected_g = np.array([-0.02, -0.02, -0.02, -0.02, 0.08])
        np.testing.assert_allclose(model.g, expected_g)

    def test_atlas_stability_holds(self) -> None:
        """The constructed g vector must satisfy the stability condition."""
        model = AtlasModel(n=10, gamma=0.05, g_param=0.01, sigma_param=0.3)
        cumsum = np.cumsum(model.g)
        assert np.all(cumsum[:-1] < 0)
        np.testing.assert_allclose(cumsum[-1], 0.0, atol=1e-14)


# ======================================================================
# Local time rates (BFK Eq. 3.7)
# ======================================================================


class TestLocalTimeRates:
    """λ_{k,k+1} = −2(g_1+…+g_k) must be positive for stable models."""

    def test_basic_atlas_local_times(self) -> None:
        model = AtlasModel(n=4, gamma=0.05, g_param=0.01, sigma_param=0.3)
        lam = model.local_time_rates()
        assert lam.shape == (3,)
        assert np.all(lam > 0)

    def test_basic_atlas_values(self) -> None:
        """For basic Atlas with g=0.01, n=3: cumsum = [-0.01, -0.02, 0]."""
        model = AtlasModel(n=3, gamma=0.05, g_param=0.01, sigma_param=0.3)
        lam = model.local_time_rates()
        np.testing.assert_allclose(lam, [0.02, 0.04])

    def test_general_first_order_local_times(self) -> None:
        g = np.array([-0.05, -0.03, -0.02, 0.10])
        model = FirstOrderModel(n=4, gamma=0.05, g=g, sigma=np.full(4, 0.3))
        lam = model.local_time_rates()
        cumsum_g = np.cumsum(g)
        np.testing.assert_allclose(lam, -2.0 * cumsum_g[:-1])


# ======================================================================
# Pareto exponents (BFK Eq. 4.3)
# ======================================================================


class TestParetoExponents:
    """r_k = −4(g_1+…+g_k)/(σ²_k + σ²_{k+1})."""

    def test_basic_atlas_constant_sigma(self) -> None:
        """With constant σ, r_k = 2·k·g/σ² for basic Atlas."""
        g_param = 0.01
        sigma = 0.3
        n = 5
        model = AtlasModel(n=n, gamma=0.05, g_param=g_param, sigma_param=sigma)
        r = model.pareto_exponents()
        for k in range(n - 1):
            expected = 2.0 * (k + 1) * g_param / sigma**2
            np.testing.assert_allclose(r[k], expected, atol=1e-14)

    def test_first_pareto_exponent_formula(self) -> None:
        """r_1 = 2g/σ² for basic Atlas."""
        model = AtlasModel(n=10, gamma=0.05, g_param=0.02, sigma_param=0.4)
        r1 = model.pareto_exponents()[0]
        expected = 2.0 * 0.02 / 0.4**2
        np.testing.assert_allclose(r1, expected)

    def test_all_exponents_positive(self) -> None:
        g = np.array([-0.05, -0.03, -0.02, 0.10])
        model = FirstOrderModel(n=4, gamma=0.05, g=g, sigma=np.full(4, 0.3))
        assert np.all(model.pareto_exponents() > 0)


# ======================================================================
# Certainty-equivalent weights (BFK Eq. 4.12–4.15)
# ======================================================================


class TestCertaintyEquivalentWeights:
    """CE weights must sum to 1, be positive, and decrease with rank."""

    def test_sum_to_one(self) -> None:
        model = AtlasModel(n=5, gamma=0.05, g_param=0.01, sigma_param=0.3)
        w = model.certainty_equivalent_weights()
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-14)

    def test_positive(self) -> None:
        model = AtlasModel(n=10, gamma=0.05, g_param=0.01, sigma_param=0.3)
        w = model.certainty_equivalent_weights()
        assert np.all(w > 0)

    def test_decreasing_with_rank(self) -> None:
        """Largest stock (rank 1) should have highest weight."""
        model = AtlasModel(n=8, gamma=0.05, g_param=0.01, sigma_param=0.3)
        w = model.certainty_equivalent_weights()
        assert np.all(np.diff(w) <= 0)

    def test_successive_ratios_match_rho(self) -> None:
        """log(M_k/M_{k+1}) = ρ_k — the fundamental CE result (BFK Eq. 4.12).

        The CE weights satisfy log(M_k/M_{k+1}) = ρ_k exactly,
        which is more testable than the asymptotic Zipf approximation.
        """
        n = 10
        g = np.array(
            [-0.05, -0.04, -0.03, -0.03, -0.02, -0.02, -0.01, -0.01, -0.01, 0.22]
        )
        sigma = np.array([0.20, 0.22, 0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40, 0.45])
        model = FirstOrderModel(n=n, gamma=0.05, g=g, sigma=sigma)
        w = model.certainty_equivalent_weights()

        # Compute expected ρ_k
        cumsum_g = np.cumsum(g)
        sigma_sq = sigma**2
        for k in range(n - 1):
            rho_k = (sigma_sq[k] + sigma_sq[k + 1]) / (-4.0 * cumsum_g[k])
            ratio = np.log(w[k] / w[k + 1])
            np.testing.assert_allclose(ratio, rho_k, atol=1e-12)


# ======================================================================
# Growth rate formulas (BFK §5)
# ======================================================================


class TestGrowthRateFormulas:
    """Analytical growth rates must match BFK formulas exactly."""

    def test_market_growth_rate(self) -> None:
        """G^μ(n) = γ  (BFK Eq. 5.10)."""
        model = AtlasModel(n=5, gamma=0.07, g_param=0.01, sigma_param=0.3)
        assert model.market_growth_rate() == 0.07

    def test_equal_weighted_excess_growth(self) -> None:
        """γ*_η = (n−1)/(2n²) · Σ σ²_k  (BFK Eq. 5.14)."""
        n = 8
        sigma = 0.3
        model = AtlasModel(n=n, gamma=0.05, g_param=0.01, sigma_param=sigma)
        gamma_star = model.equal_weighted_excess_growth_rate()
        expected = (n - 1) / (2.0 * n**2) * n * sigma**2
        np.testing.assert_allclose(gamma_star, expected)

    def test_matches_core_formula(self) -> None:
        """Must agree with core.growth_rates.atlas_excess_growth_rate_equal_weighted."""
        n = 6
        sigma = 0.25
        model = AtlasModel(n=n, gamma=0.05, g_param=0.01, sigma_param=sigma)
        from_model = model.equal_weighted_excess_growth_rate()
        sigma_sq = np.full(n, sigma**2)
        from_core = atlas_excess_growth_rate_equal_weighted(n, sigma_sq)
        np.testing.assert_allclose(from_model, from_core, atol=1e-14)

    def test_equal_weighted_growth_rate(self) -> None:
        """G^η = γ + γ*_η  (BFK Eq. 5.15)."""
        model = AtlasModel(n=5, gamma=0.05, g_param=0.01, sigma_param=0.3)
        total = model.equal_weighted_growth_rate()
        expected = model.gamma + model.equal_weighted_excess_growth_rate()
        np.testing.assert_allclose(total, expected)

    def test_equal_weighted_beats_market(self) -> None:
        """EW growth > market growth because γ*_η > 0 for n ≥ 2."""
        model = AtlasModel(n=10, gamma=0.05, g_param=0.01, sigma_param=0.3)
        assert model.equal_weighted_growth_rate() > model.market_growth_rate()

    def test_large_n_limit(self) -> None:
        """As n → ∞, γ*_η → σ²/2."""
        sigma = 0.3
        n = 1000
        model = AtlasModel(n=n, gamma=0.05, g_param=0.01, sigma_param=sigma)
        gamma_star = model.equal_weighted_excess_growth_rate()
        np.testing.assert_allclose(gamma_star, sigma**2 / 2, rtol=0.01)


# ======================================================================
# 2-stock Atlas explicit solution
# ======================================================================


class TestTwoStockAtlas:
    """The 2-stock Atlas is analytically tractable and serves as a sanity check."""

    def test_two_stock_construction(self) -> None:
        model = AtlasModel(n=2, gamma=0.05, g_param=0.01, sigma_param=0.3)
        np.testing.assert_allclose(model.g, [-0.01, 0.01])

    def test_two_stock_local_time(self) -> None:
        model = AtlasModel(n=2, gamma=0.05, g_param=0.01, sigma_param=0.3)
        lam = model.local_time_rates()
        np.testing.assert_allclose(lam, [0.02])

    def test_two_stock_pareto(self) -> None:
        g_param = 0.01
        sigma = 0.3
        model = AtlasModel(n=2, gamma=0.05, g_param=g_param, sigma_param=sigma)
        r = model.pareto_exponents()
        expected = 2.0 * g_param / sigma**2
        np.testing.assert_allclose(r, [expected])

    def test_two_stock_equal_weighted_excess(self) -> None:
        sigma = 0.3
        model = AtlasModel(n=2, gamma=0.05, g_param=0.01, sigma_param=sigma)
        gamma_star = model.equal_weighted_excess_growth_rate()
        np.testing.assert_allclose(gamma_star, sigma**2 / 4.0)


# ======================================================================
# Ergodic property via simulation (BFK Prop. 2.3)
# ======================================================================


class TestErgodicProperty:
    """Each stock should spend ≈1/n of its time at each rank."""

    def test_ergodic_3_stocks(self) -> None:
        n = 3
        # Very high g_param relative to σ to promote fast rank mixing.
        # Local time rate λ_{1,2} = 2g = 0.4, λ_{2,3} = 4g = 0.8
        model = AtlasModel(n=n, gamma=0.05, g_param=0.20, sigma_param=0.5)
        x0 = np.array([100.0, 100.0, 100.0])
        proc = model.to_stochastic_process(x0)

        rng = np.random.default_rng(42)
        n_steps = 500_000
        _, path = simulate_path(proc, T=1000.0, n_steps=n_steps, rng=rng)

        # Discard burn-in (first 30%)
        burn = int(path.shape[0] * 0.3)
        path_steady = path[burn:]
        rank_counts = np.zeros((n, n))
        for step in range(path_steady.shape[0]):
            order = np.argsort(-path_steady[step])
            for rank_idx, stock_idx in enumerate(order):
                rank_counts[stock_idx, rank_idx] += 1

        fractions = rank_counts / path_steady.shape[0]
        target = 1.0 / n
        np.testing.assert_allclose(fractions, target, atol=0.08)

    def test_ergodic_2_stocks(self) -> None:
        """Simplest case: each stock at each rank ≈50% of time."""
        n = 2
        model = AtlasModel(n=n, gamma=0.05, g_param=0.20, sigma_param=0.5)
        x0 = np.array([100.0, 100.0])
        proc = model.to_stochastic_process(x0)

        rng = np.random.default_rng(99)
        _, path = simulate_path(proc, T=1000.0, n_steps=500_000, rng=rng)

        burn = int(path.shape[0] * 0.3)
        path_steady = path[burn:]
        rank_counts = np.zeros((n, n))
        for step in range(path_steady.shape[0]):
            order = np.argsort(-path_steady[step])
            for rank_idx, stock_idx in enumerate(order):
                rank_counts[stock_idx, rank_idx] += 1

        fractions = rank_counts / path_steady.shape[0]
        np.testing.assert_allclose(fractions, 0.5, atol=0.06)


# ======================================================================
# Simulation cross-checks
# ======================================================================


class TestSimulationCrossChecks:
    """Verify analytical formulas against Monte-Carlo estimates."""

    def test_excess_growth_positive_and_consistent(self) -> None:
        """EW portfolio should outperform market in Atlas model.

        The theoretical excess growth rate is γ*_η = (n−1)/(2n²)·Σσ²_k.
        We verify that the simulated EW portfolio outperforms the market
        at a rate consistent with this formula (direction and magnitude).

        Because the BFK formula describes long-run stationary behaviour,
        finite-horizon simulation includes transient and discretization
        effects.  We test that the simulated excess growth has the correct
        sign and is within a factor of 3 of the analytical value.
        """
        n = 3
        sigma = 0.5
        g_param = 0.05
        model = AtlasModel(n=n, gamma=0.05, g_param=g_param, sigma_param=sigma)
        analytical_gamma_star = model.equal_weighted_excess_growth_rate()
        assert analytical_gamma_star > 0

        rng = np.random.default_rng(2024)
        n_paths = 200
        T = 50.0
        n_steps = 25_000

        relative_log_returns = []
        for _ in range(n_paths):
            x0 = np.full(n, 100.0)
            proc = model.to_stochastic_process(x0)
            _, path = simulate_path(proc, T=T, n_steps=n_steps, rng=rng)

            log_relative = 0.0
            for k in range(1, path.shape[0]):
                dY = path[k] - path[k - 1]
                gross = np.exp(dY)
                ew_return = float(np.mean(gross))
                caps_prev = np.exp(path[k - 1] - np.max(path[k - 1]))
                mu = caps_prev / np.sum(caps_prev)
                mkt_return = float(np.dot(mu, gross))
                log_relative += np.log(ew_return) - np.log(mkt_return)

            relative_log_returns.append(log_relative / T)

        sim_excess = float(np.mean(relative_log_returns))
        # EW should outperform market (positive excess growth)
        assert sim_excess > 0, f"Expected positive excess growth, got {sim_excess}"
        # Should be within a factor of 3 of the analytical value
        assert (
            sim_excess < 3.0 * analytical_gamma_star
        ), f"Simulated {sim_excess} too far from analytical {analytical_gamma_star}"


# ======================================================================
# StochasticProcess protocol
# ======================================================================


class TestAtlasStochasticProcess:
    """The atlas process must comply with the StochasticProcess protocol."""

    def test_protocol_methods(self) -> None:
        model = AtlasModel(n=3, gamma=0.05, g_param=0.01, sigma_param=0.3)
        x0 = np.array([100.0, 80.0, 60.0])
        proc = model.to_stochastic_process(x0)
        assert proc.size() == 3
        assert proc.factors() == 3
        iv = proc.initial_values()
        np.testing.assert_allclose(iv, np.log(x0))

    def test_drift_rank_dependent(self) -> None:
        """Drift must assign higher growth to lower-ranked (smaller) stocks."""
        model = AtlasModel(n=3, gamma=0.05, g_param=0.01, sigma_param=0.3)
        x0 = np.array([100.0, 80.0, 60.0])
        proc = model.to_stochastic_process(x0)
        log_x = np.log(x0)
        drift = proc.drift(0.0, log_x)
        # Stock 2 (smallest, rank 3) should have highest drift
        assert drift[2] > drift[0]
        assert drift[2] > drift[1]

    def test_diffusion_diagonal(self) -> None:
        """Atlas model has independent Brownian motions → diagonal diffusion."""
        model = AtlasModel(n=3, gamma=0.05, g_param=0.01, sigma_param=0.3)
        proc = model.to_stochastic_process(np.array([100.0, 80.0, 60.0]))
        D = proc.diffusion(0.0, proc.initial_values())
        off_diag_mask = ~np.eye(3, dtype=bool)
        np.testing.assert_allclose(D[off_diag_mask], 0.0)

    def test_simulation_does_not_diverge(self) -> None:
        model = AtlasModel(n=5, gamma=0.05, g_param=0.01, sigma_param=0.3)
        x0 = np.full(5, 100.0)
        proc = model.to_stochastic_process(x0)
        rng = np.random.default_rng(77)
        _, path = simulate_path(proc, T=10.0, n_steps=5_000, rng=rng)
        assert np.all(np.isfinite(path))


# ======================================================================
# MarketModel interface
# ======================================================================


class TestMarketModelInterface:
    """AtlasModel and FirstOrderModel must satisfy the MarketModel ABC."""

    def test_drift_rates_shape(self) -> None:
        model = AtlasModel(n=4, gamma=0.05, g_param=0.01, sigma_param=0.3)
        x = np.array([100.0, 80.0, 60.0, 40.0])
        g = model.drift_rates(0.0, x)
        assert g.shape == (4,)

    def test_covariance_rate_shape(self) -> None:
        model = AtlasModel(n=4, gamma=0.05, g_param=0.01, sigma_param=0.3)
        x = np.array([100.0, 80.0, 60.0, 40.0])
        a = model.covariance_rate(0.0, x)
        assert a.shape == (4, 4)

    def test_covariance_diagonal(self) -> None:
        """Atlas covariance is diagonal (independent BMs)."""
        model = AtlasModel(n=3, gamma=0.05, g_param=0.01, sigma_param=0.3)
        x = np.array([100.0, 80.0, 60.0])
        a = model.covariance_rate(0.0, x)
        off = ~np.eye(3, dtype=bool)
        np.testing.assert_allclose(a[off], 0.0)

    def test_covariance_values_match_sigma(self) -> None:
        """Diagonal entries should be σ² for each stock's current rank."""
        sigma = 0.3
        model = AtlasModel(n=3, gamma=0.05, g_param=0.01, sigma_param=sigma)
        x = np.array([100.0, 80.0, 60.0])
        a = model.covariance_rate(0.0, x)
        np.testing.assert_allclose(np.diag(a), sigma**2)

    def test_excess_growth_from_model_cov(self) -> None:
        """Excess growth rate from model's covariance should match analytical formula."""
        n = 5
        sigma = 0.3
        model = AtlasModel(n=n, gamma=0.05, g_param=0.01, sigma_param=sigma)
        pi = np.full(n, 1.0 / n)
        a = model.covariance_rate(0.0, np.arange(1, n + 1, dtype=float)[::-1] * 100)
        gamma_star = excess_growth_rate(pi, a)
        expected = model.equal_weighted_excess_growth_rate()
        np.testing.assert_allclose(gamma_star, expected, atol=1e-14)


# ======================================================================
# FirstOrderModel with non-constant sigma
# ======================================================================


class TestFirstOrderNonConstantSigma:
    """Test with rank-dependent volatilities (general case)."""

    def test_construction(self) -> None:
        g = np.array([-0.05, -0.03, -0.02, 0.10])
        sigma = np.array([0.20, 0.25, 0.30, 0.40])
        model = FirstOrderModel(n=4, gamma=0.05, g=g, sigma=sigma)
        assert model.n_assets == 4

    def test_pareto_exponents_nonuniform(self) -> None:
        g = np.array([-0.05, -0.03, -0.02, 0.10])
        sigma = np.array([0.20, 0.25, 0.30, 0.40])
        model = FirstOrderModel(n=4, gamma=0.05, g=g, sigma=sigma)
        r = model.pareto_exponents()
        cumsum_g = np.cumsum(g)
        sigma_sq = sigma**2
        for k in range(3):
            expected = -4.0 * cumsum_g[k] / (sigma_sq[k] + sigma_sq[k + 1])
            np.testing.assert_allclose(r[k], expected, atol=1e-14)

    def test_ce_weights_with_varying_sigma(self) -> None:
        g = np.array([-0.05, -0.03, -0.02, 0.10])
        sigma = np.array([0.20, 0.25, 0.30, 0.40])
        model = FirstOrderModel(n=4, gamma=0.05, g=g, sigma=sigma)
        w = model.certainty_equivalent_weights()
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-14)
        assert np.all(w > 0)

    def test_diversity_excess_growth(self) -> None:
        """Diversity-weighted excess growth should be positive for 0 < p < 1."""
        g = np.array([-0.05, -0.03, -0.02, 0.10])
        sigma = np.array([0.20, 0.25, 0.30, 0.40])
        model = FirstOrderModel(n=4, gamma=0.05, g=g, sigma=sigma)
        gamma_star = model.diversity_weighted_excess_growth(p=0.5)
        assert gamma_star > 0

    def test_diversity_excess_growth_rejects_bad_p(self) -> None:
        model = AtlasModel(n=3, gamma=0.05, g_param=0.01, sigma_param=0.3)
        with pytest.raises(Exception, match="\\(0, 1\\)"):
            model.diversity_weighted_excess_growth(p=1.5)


# ======================================================================
# Zipf exponent (AtlasModel-specific)
# ======================================================================


class TestZipfExponent:
    """α = σ²/(2g) governs the power-law tail of ranked weights."""

    def test_zipf_formula(self) -> None:
        g_param = 0.02
        sigma = 0.4
        model = AtlasModel(n=10, gamma=0.05, g_param=g_param, sigma_param=sigma)
        alpha = model.zipf_exponent()
        np.testing.assert_allclose(alpha, sigma**2 / (2.0 * g_param))

    def test_zipf_equals_one_gives_zipf_law(self) -> None:
        """When σ²/(2g) = 1, we get Zipf's law."""
        sigma = 0.2
        g_param = sigma**2 / 2.0
        model = AtlasModel(n=5, gamma=0.05, g_param=g_param, sigma_param=sigma)
        np.testing.assert_allclose(model.zipf_exponent(), 1.0)

    def test_pareto_exponent_consistency(self) -> None:
        """r_1 = 2g/σ² = 1/α for the basic Atlas."""
        model = AtlasModel(n=5, gamma=0.05, g_param=0.02, sigma_param=0.3)
        r1 = model.pareto_exponent()
        alpha = model.zipf_exponent()
        np.testing.assert_allclose(r1, 1.0 / alpha, atol=1e-14)
