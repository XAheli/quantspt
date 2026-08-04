"""Tests for the volatility-stabilised market model.

Validates:
- σ_i = σ/√(nμ_i) produces correct covariance
- Volatility explosion for small market weights
- Market excess growth rate γ*_μ = σ²(n−1)/(2n)
- Market weights sum to 1 under simulation
- StochasticProcess protocol compliance
"""

from __future__ import annotations

import numpy as np
import pytest

from quantspt.core.growth_rates import excess_growth_rate
from quantspt.core.processes import simulate_path
from quantspt.errors import SPTInvariantError
from quantspt.models.volatility_stabilized import VolatilityStabilizedMarket

# ======================================================================
# Construction
# ======================================================================


class TestConstruction:
    """Parameter validation."""

    def test_basic(self) -> None:
        model = VolatilityStabilizedMarket(n=5, sigma=0.3)
        assert model.n_assets == 5

    def test_rejects_n_less_than_2(self) -> None:
        with pytest.raises(Exception, match="at least 2"):
            VolatilityStabilizedMarket(n=1, sigma=0.3)

    def test_rejects_nonpositive_sigma(self) -> None:
        with pytest.raises(Exception, match="positive"):
            VolatilityStabilizedMarket(n=3, sigma=0.0)

    def test_rejects_negative_sigma(self) -> None:
        with pytest.raises(Exception, match="positive"):
            VolatilityStabilizedMarket(n=3, sigma=-0.1)


# ======================================================================
# Covariance formula: a_{ii} = σ²/(nμ_i)
# ======================================================================


class TestCovarianceFormula:
    """Covariance rate must satisfy σ²/(n·μ_i) for each stock."""

    def test_stock_variance_formula(self) -> None:
        model = VolatilityStabilizedMarket(n=5, sigma=0.3)
        mu_i = 0.2
        expected = 0.3**2 / (5 * 0.2)
        np.testing.assert_allclose(model.stock_variance(mu_i), expected)

    def test_stock_variance_multiple_weights(self) -> None:
        model = VolatilityStabilizedMarket(n=4, sigma=0.5)
        for mu_i in [0.05, 0.1, 0.25, 0.5]:
            expected = 0.5**2 / (4 * mu_i)
            np.testing.assert_allclose(model.stock_variance(mu_i), expected)

    def test_covariance_rate_matrix(self) -> None:
        """Full covariance matrix must be diagonal with correct entries."""
        n = 4
        sigma = 0.3
        model = VolatilityStabilizedMarket(n=n, sigma=sigma)
        x = np.array([100.0, 200.0, 300.0, 400.0])
        mu = x / x.sum()
        a = model.covariance_rate(0.0, x)

        assert a.shape == (n, n)
        off_diag = ~np.eye(n, dtype=bool)
        np.testing.assert_allclose(a[off_diag], 0.0)

        for i in range(n):
            expected = sigma**2 / (n * mu[i])
            np.testing.assert_allclose(a[i, i], expected, atol=1e-14)

    def test_covariance_is_psd(self) -> None:
        model = VolatilityStabilizedMarket(n=5, sigma=0.3)
        x = np.array([50.0, 100.0, 150.0, 200.0, 250.0])
        a = model.covariance_rate(0.0, x)
        eigvals = np.linalg.eigvalsh(a)
        assert np.all(eigvals >= -1e-14)

    def test_stock_variance_rejects_zero_weight(self) -> None:
        model = VolatilityStabilizedMarket(n=3, sigma=0.3)
        with pytest.raises(Exception, match="positive"):
            model.stock_variance(0.0)


# ======================================================================
# Volatility explosion near boundary
# ======================================================================


class TestVolatilityExplosion:
    """As μ_i → 0, σ_i → ∞. This is the key feature of the model."""

    def test_small_weight_large_variance(self) -> None:
        model = VolatilityStabilizedMarket(n=5, sigma=0.3)
        var_small = model.stock_variance(0.001)
        var_large = model.stock_variance(0.5)
        assert var_small > var_large
        assert var_small > 100 * var_large

    def test_monotone_decreasing_variance(self) -> None:
        """Variance should strictly decrease as weight increases."""
        model = VolatilityStabilizedMarket(n=5, sigma=0.3)
        weights = [0.01, 0.05, 0.10, 0.20, 0.50]
        variances = [model.stock_variance(w) for w in weights]
        for i in range(len(variances) - 1):
            assert variances[i] > variances[i + 1]

    def test_inverse_proportionality(self) -> None:
        """a_{ii} ∝ 1/μ_i: doubling weight halves variance."""
        model = VolatilityStabilizedMarket(n=5, sigma=0.3)
        v1 = model.stock_variance(0.10)
        v2 = model.stock_variance(0.20)
        np.testing.assert_allclose(v1 / v2, 2.0, atol=1e-14)


# ======================================================================
# Market excess growth rate
# ======================================================================


class TestMarketExcessGrowthRate:
    """γ*_μ = σ²(n−1)/(2n) must hold exactly."""

    def test_formula(self) -> None:
        sigma = 0.3
        n = 5
        model = VolatilityStabilizedMarket(n=n, sigma=sigma)
        expected = sigma**2 * (n - 1) / (2.0 * n)
        np.testing.assert_allclose(
            model.market_excess_growth_rate(), expected, atol=1e-14
        )

    def test_various_n(self) -> None:
        sigma = 0.4
        for n in [2, 5, 10, 50, 100]:
            model = VolatilityStabilizedMarket(n=n, sigma=sigma)
            expected = sigma**2 * (n - 1) / (2.0 * n)
            np.testing.assert_allclose(model.market_excess_growth_rate(), expected)

    def test_large_n_limit(self) -> None:
        """As n → ∞, γ*_μ → σ²/2."""
        sigma = 0.3
        model = VolatilityStabilizedMarket(n=1000, sigma=sigma)
        np.testing.assert_allclose(
            model.market_excess_growth_rate(), sigma**2 / 2, rtol=0.01
        )

    def test_matches_core_excess_growth(self) -> None:
        """γ*_μ from the analytical formula must match the core computation."""
        n = 5
        sigma = 0.3
        model = VolatilityStabilizedMarket(n=n, sigma=sigma)
        x = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        mu = x / x.sum()
        a = model.covariance_rate(0.0, x)
        core_gamma_star = excess_growth_rate(mu, a)
        expected = model.market_excess_growth_rate()
        np.testing.assert_allclose(core_gamma_star, expected, atol=1e-12)

    def test_n2_case(self) -> None:
        """n=2: γ*_μ = σ²/4."""
        sigma = 0.4
        model = VolatilityStabilizedMarket(n=2, sigma=sigma)
        np.testing.assert_allclose(model.market_excess_growth_rate(), sigma**2 / 4.0)


# ======================================================================
# Drift rates
# ======================================================================


class TestDriftRates:
    """Drift rates γ_i depend on market weight."""

    def test_shape(self) -> None:
        model = VolatilityStabilizedMarket(n=4, sigma=0.3)
        x = np.array([100.0, 200.0, 300.0, 400.0])
        g = model.drift_rates(0.0, x)
        assert g.shape == (4,)

    def test_smaller_stock_lower_growth(self) -> None:
        """Small stocks have higher variance → lower growth rate."""
        model = VolatilityStabilizedMarket(n=3, sigma=0.3)
        x = np.array([10.0, 100.0, 1000.0])
        g = model.drift_rates(0.0, x)
        # Stock with largest cap should have highest growth rate
        assert g[2] > g[1] > g[0]


# ======================================================================
# Simulation
# ======================================================================


class TestSimulation:
    """Process should simulate without divergence; weights should stay valid."""

    def test_process_protocol(self) -> None:
        model = VolatilityStabilizedMarket(n=3, sigma=0.3)
        x0 = np.array([100.0, 100.0, 100.0])
        proc = model.to_stochastic_process(x0)
        assert proc.size() == 3
        assert proc.factors() == 3
        np.testing.assert_allclose(proc.initial_values(), np.log(x0))

    def test_simulation_finite(self) -> None:
        model = VolatilityStabilizedMarket(n=4, sigma=0.3)
        x0 = np.array([100.0, 100.0, 100.0, 100.0])
        proc = model.to_stochastic_process(x0)
        rng = np.random.default_rng(42)
        _, path = simulate_path(proc, T=5.0, n_steps=5_000, rng=rng)
        assert np.all(np.isfinite(path))

    def test_weights_sum_to_one_in_simulation(self) -> None:
        """Market weights from simulated caps must always sum to 1."""
        model = VolatilityStabilizedMarket(n=3, sigma=0.3)
        x0 = np.array([100.0, 100.0, 100.0])
        proc = model.to_stochastic_process(x0)
        rng = np.random.default_rng(123)
        _, path = simulate_path(proc, T=2.0, n_steps=2_000, rng=rng)

        caps = np.exp(path)
        for step in range(caps.shape[0]):
            weights = model.market_weights(caps[step])
            np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-12)

    def test_rejects_wrong_x0(self) -> None:
        model = VolatilityStabilizedMarket(n=3, sigma=0.3)
        with pytest.raises(SPTInvariantError, match="mismatch"):
            model.to_stochastic_process(np.array([100.0, 100.0]))

    def test_rejects_nonpositive_x0(self) -> None:
        model = VolatilityStabilizedMarket(n=3, sigma=0.3)
        with pytest.raises(SPTInvariantError, match="positive"):
            model.to_stochastic_process(np.array([100.0, -1.0, 100.0]))


# ======================================================================
# Cross-check: excess growth rate is weight-independent
# ======================================================================


class TestExcessGrowthWeightIndependent:
    """The analytical γ*_μ = σ²(n−1)/(2n) does not depend on x."""

    def test_same_for_different_x(self) -> None:
        model = VolatilityStabilizedMarket(n=4, sigma=0.3)
        x1 = np.array([100.0, 200.0, 300.0, 400.0])
        x2 = np.array([10.0, 10.0, 10.0, 10.0])
        x3 = np.array([1.0, 100.0, 10000.0, 1000000.0])

        for x in [x1, x2, x3]:
            mu = x / x.sum()
            a = model.covariance_rate(0.0, x)
            gamma_star = excess_growth_rate(mu, a)
            np.testing.assert_allclose(
                gamma_star, model.market_excess_growth_rate(), atol=1e-12
            )
