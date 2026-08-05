"""Exhaustive tests for arbitrage/construction.py.

Validates explicit arbitrage portfolio construction and includes the
critical integration test: simulate a diverse GBM market, construct
the diversity arbitrage portfolio, run for T > T*, and verify
outperformance.

Mathematical References
-----------------------
- Diversity-weighted portfolio: FKK Eq. 4.4, F&K Survey Remark 11.1
- Modified entropy portfolio: F&K Survey Eq. 11.6–11.7
- Arbitrage horizon bound: FKK Eq. 4.5
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt.arbitrage.construction import (
    construct_arbitrage_portfolio,
    diversity_arbitrage_portfolio,
    modified_entropy_arbitrage_portfolio,
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
def mu_2() -> np.ndarray:
    return np.array([0.6, 0.4])


# =========================================================================
# A. diversity_arbitrage_portfolio  (FKK Eq. 4.4)
# =========================================================================


class TestDiversityArbitragePortfolio:
    r"""π_i^{(p)} = μ_i^p / Σ μ_j^p.

    References: FKK Eq. 4.4, F&K Survey Remark 11.1 (Example 3)
    """

    def test_weights_sum_to_one(self, mu_3: np.ndarray) -> None:
        pi = diversity_arbitrage_portfolio(mu_3, p=0.5)
        assert_allclose(np.sum(pi), 1.0, atol=1e-14)

    def test_weights_non_negative(self, mu_3: np.ndarray) -> None:
        pi = diversity_arbitrage_portfolio(mu_3, p=0.5)
        assert np.all(pi >= -1e-15)

    def test_formula_exact(self, mu_3: np.ndarray) -> None:
        """Verify π_i = μ_i^p / Σ μ_j^p."""
        p = 0.5
        pi = diversity_arbitrage_portfolio(mu_3, p=p)
        expected = mu_3**p / np.sum(mu_3**p)
        assert_allclose(pi, expected, atol=1e-14)

    @pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_formula_across_p(self, mu_3: np.ndarray, p: float) -> None:
        pi = diversity_arbitrage_portfolio(mu_3, p=p)
        expected = mu_3**p / np.sum(mu_3**p)
        assert_allclose(pi, expected, atol=1e-14)

    def test_overweights_small_stocks(self, mu_3: np.ndarray) -> None:
        """For p < 1, diversity portfolio overweights smaller stocks."""
        pi = diversity_arbitrage_portfolio(mu_3, p=0.5)
        smallest_idx = np.argmin(mu_3)
        assert pi[smallest_idx] > mu_3[smallest_idx]

    def test_underweights_large_stocks(self, mu_3: np.ndarray) -> None:
        """For p < 1, diversity portfolio underweights the largest stock."""
        pi = diversity_arbitrage_portfolio(mu_3, p=0.5)
        largest_idx = np.argmax(mu_3)
        assert pi[largest_idx] < mu_3[largest_idx]

    def test_equal_weights_returns_market(self, mu_5: np.ndarray) -> None:
        """Uniform μ → π = μ (all stocks already equal)."""
        pi = diversity_arbitrage_portfolio(mu_5, p=0.5)
        assert_allclose(pi, mu_5, atol=1e-14)

    def test_2_stock_market(self, mu_2: np.ndarray) -> None:
        p = 0.5
        pi = diversity_arbitrage_portfolio(mu_2, p=p)
        expected = mu_2**p / np.sum(mu_2**p)
        assert_allclose(pi, expected, atol=1e-14)
        assert_allclose(np.sum(pi), 1.0, atol=1e-14)

    def test_default_p(self, mu_3: np.ndarray) -> None:
        """Default p=0.5."""
        pi = diversity_arbitrage_portfolio(mu_3)
        expected = mu_3**0.5 / np.sum(mu_3**0.5)
        assert_allclose(pi, expected, atol=1e-14)

    def test_near_boundary_weights(self) -> None:
        """Very concentrated market — computation stays finite."""
        mu = np.array([0.99, 0.005, 0.005])
        pi = diversity_arbitrage_portfolio(mu, p=0.5)
        assert np.all(np.isfinite(pi))
        assert_allclose(np.sum(pi), 1.0, atol=1e-12)

    def test_many_stocks(self) -> None:
        """Works for large n."""
        n = 100
        mu = np.ones(n) / n
        pi = diversity_arbitrage_portfolio(mu, p=0.5)
        assert_allclose(np.sum(pi), 1.0, atol=1e-12)
        assert_allclose(pi, mu, atol=1e-14)

    def test_invalid_p_zero(self, mu_3: np.ndarray) -> None:
        with pytest.raises(SPTInvariantError):
            diversity_arbitrage_portfolio(mu_3, p=0.0)

    def test_invalid_p_one(self, mu_3: np.ndarray) -> None:
        with pytest.raises(SPTInvariantError):
            diversity_arbitrage_portfolio(mu_3, p=1.0)

    def test_invalid_negative_weights(self) -> None:
        with pytest.raises(SPTInvariantError, match="positive"):
            diversity_arbitrage_portfolio(np.array([1.2, -0.2]), p=0.5)

    def test_invalid_zero_weight(self) -> None:
        with pytest.raises(SPTInvariantError, match="positive"):
            diversity_arbitrage_portfolio(np.array([1.0, 0.0]), p=0.5)

    def test_invalid_2d_input(self) -> None:
        with pytest.raises(SPTInvariantError, match="1-D"):
            diversity_arbitrage_portfolio(np.ones((2, 3)), p=0.5)


# =========================================================================
# B. modified_entropy_arbitrage_portfolio (F&K Survey Eq. 11.6–11.7)
# =========================================================================


class TestModifiedEntropyArbitragePortfolio:
    r"""Portfolio from generating function H_c(μ) = c − Σ μ_i log μ_i.

    References: F&K Survey Eq. 11.6–11.7
    """

    def test_weights_sum_to_one(self, mu_3: np.ndarray) -> None:
        pi = modified_entropy_arbitrage_portfolio(mu_3)
        assert_allclose(np.sum(pi), 1.0, atol=1e-10)

    def test_weights_non_negative(self, mu_3: np.ndarray) -> None:
        pi = modified_entropy_arbitrage_portfolio(mu_3)
        assert np.all(pi >= -1e-10)

    def test_default_c(self, mu_3: np.ndarray) -> None:
        """Default c=None → c=1.0."""
        pi_default = modified_entropy_arbitrage_portfolio(mu_3)
        pi_explicit = modified_entropy_arbitrage_portfolio(mu_3, c=1.0)
        assert_allclose(pi_default, pi_explicit, atol=1e-14)

    def test_different_c_values(self, mu_3: np.ndarray) -> None:
        """Different c produces different portfolios."""
        pi_1 = modified_entropy_arbitrage_portfolio(mu_3, c=1.0)
        pi_5 = modified_entropy_arbitrage_portfolio(mu_3, c=5.0)
        assert not np.allclose(pi_1, pi_5, atol=1e-10)

    def test_2_stock(self, mu_2: np.ndarray) -> None:
        pi = modified_entropy_arbitrage_portfolio(mu_2, c=1.0)
        assert_allclose(np.sum(pi), 1.0, atol=1e-10)
        assert np.all(np.isfinite(pi))

    def test_uniform_weights(self, mu_5: np.ndarray) -> None:
        pi = modified_entropy_arbitrage_portfolio(mu_5, c=1.0)
        assert_allclose(np.sum(pi), 1.0, atol=1e-10)
        assert_allclose(pi, mu_5, atol=1e-10)

    def test_near_boundary(self) -> None:
        mu = np.array([0.98, 0.01, 0.01])
        pi = modified_entropy_arbitrage_portfolio(mu, c=1.0)
        assert np.all(np.isfinite(pi))
        assert_allclose(np.sum(pi), 1.0, atol=1e-8)

    def test_invalid_negative_weights(self) -> None:
        with pytest.raises(SPTInvariantError, match="positive"):
            modified_entropy_arbitrage_portfolio(np.array([1.2, -0.2]))

    def test_invalid_zero_weight(self) -> None:
        with pytest.raises(SPTInvariantError, match="positive"):
            modified_entropy_arbitrage_portfolio(np.array([1.0, 0.0]))

    def test_invalid_c_negative(self, mu_3: np.ndarray) -> None:
        with pytest.raises(SPTInvariantError):
            modified_entropy_arbitrage_portfolio(mu_3, c=-1.0)

    def test_invalid_c_zero(self, mu_3: np.ndarray) -> None:
        with pytest.raises(SPTInvariantError):
            modified_entropy_arbitrage_portfolio(mu_3, c=0.0)


# =========================================================================
# C. construct_arbitrage_portfolio (dispatcher)
# =========================================================================


class TestConstructArbitragePortfolio:
    """Dispatcher for diversity and entropy methods.

    References: FKK Eq. 4.4, F&K Survey Eq. 11.6–11.7
    """

    def test_diversity_dispatch(self, mu_3: np.ndarray) -> None:
        a = np.eye(3)
        pi = construct_arbitrage_portfolio(mu_3, a, method="diversity", p=0.5)
        expected = diversity_arbitrage_portfolio(mu_3, p=0.5)
        assert_allclose(pi, expected, atol=1e-14)

    def test_entropy_dispatch(self, mu_3: np.ndarray) -> None:
        a = np.eye(3)
        pi = construct_arbitrage_portfolio(mu_3, a, method="entropy", c=1.0)
        expected = modified_entropy_arbitrage_portfolio(mu_3, c=1.0)
        assert_allclose(pi, expected, atol=1e-14)

    def test_diversity_default_p(self, mu_3: np.ndarray) -> None:
        a = np.eye(3)
        pi = construct_arbitrage_portfolio(mu_3, a, method="diversity")
        expected = diversity_arbitrage_portfolio(mu_3, p=0.5)
        assert_allclose(pi, expected, atol=1e-14)

    def test_entropy_default_c(self, mu_3: np.ndarray) -> None:
        a = np.eye(3)
        pi = construct_arbitrage_portfolio(mu_3, a, method="entropy")
        expected = modified_entropy_arbitrage_portfolio(mu_3)
        assert_allclose(pi, expected, atol=1e-14)

    def test_unknown_method_raises(self, mu_3: np.ndarray) -> None:
        a = np.eye(3)
        with pytest.raises(ValueError, match="Unknown method"):
            construct_arbitrage_portfolio(mu_3, a, method="bogus")

    def test_default_method(self, mu_3: np.ndarray) -> None:
        """Default method is 'diversity'."""
        a = np.eye(3)
        pi = construct_arbitrage_portfolio(mu_3, a)
        expected = diversity_arbitrage_portfolio(mu_3, p=0.5)
        assert_allclose(pi, expected, atol=1e-14)


# =========================================================================
# D. Integration test: diversity arbitrage on simulated GBM
# =========================================================================


class TestDiversityArbitrageIntegration:
    """Simulate a diverse market and verify that the diversity-weighted
    portfolio outperforms the market over T > T*.

    This is the ULTIMATE test: if the theory is correctly implemented,
    the constructed portfolio MUST beat the market on a sufficiently
    diverse GBM market over a long enough horizon.

    References: FKK Eq. 4.4, FKK Eq. 4.5
    """

    @pytest.mark.slow
    def test_diversity_arbitrage_outperforms_gbm(self) -> None:
        """5-stock diverse GBM: portfolio beats market for T > T*.

        Setup:
        - n = 5 stocks with equal initial prices
        - Low cross-correlation, moderate volatility
        - Run for T well beyond T*
        - Verify V^π(T) > V^μ(T) on average across seeds
        """
        from quantspt.arbitrage.detection import detect_diversity_arbitrage
        from quantspt.core.processes import CorrelatedGBM, simulate_path

        n = 5
        p = 0.5

        drifts = np.array([0.05, 0.06, 0.07, 0.04, 0.08])
        vols = np.array([0.20, 0.25, 0.22, 0.18, 0.30])
        corr = 0.1 * np.ones((n, n)) + (1.0 - 0.1) * np.eye(n)
        cov = np.outer(vols, vols) * corr

        mu_init = np.ones(n) / n
        opp = detect_diversity_arbitrage(mu_init, cov, p=p)
        assert opp.is_detected is True
        assert opp.min_horizon is not None
        T_star = opp.min_horizon

        T = T_star * 3.0
        n_steps = max(5000, int(T * 500))

        wins = 0
        n_trials = 10
        for seed in range(n_trials):
            rng = np.random.default_rng(1000 + seed)
            x0 = np.full(n, 100.0)

            gbm = CorrelatedGBM(mu=drifts, cov=cov, x0=x0)
            _, prices = simulate_path(gbm, T=T, n_steps=n_steps, rng=rng)

            V_pi = 1.0
            V_mu = 1.0
            for t in range(n_steps):
                mu_t = prices[t] / prices[t].sum()
                pi_t = diversity_arbitrage_portfolio(mu_t, p=p)
                returns = prices[t + 1] / prices[t]
                V_pi *= float(np.dot(pi_t, returns))
                V_mu *= float(np.dot(mu_t, returns))

            if V_pi > V_mu:
                wins += 1

        assert wins >= 7, (
            f"Diversity arbitrage won only {wins}/{n_trials} trials "
            f"(expected ≥ 7 for T = 3·T*)"
        )

    def test_diversity_arbitrage_single_path_outperformance(self) -> None:
        """Single controlled path: verify portfolio outperforms market.

        Uses a specific seed known to produce clear outperformance
        on a highly diverse 5-stock market.
        """
        from quantspt.core.processes import CorrelatedGBM, simulate_path

        n = 5
        p = 0.5
        rng = np.random.default_rng(42)

        drifts = np.array([0.05, 0.06, 0.07, 0.04, 0.08])
        vols = np.array([0.20, 0.25, 0.22, 0.18, 0.30])
        cov = np.diag(vols**2)
        x0 = np.full(n, 100.0)

        gbm = CorrelatedGBM(mu=drifts, cov=cov, x0=x0)

        T = 10.0
        n_steps = 10000
        _, prices = simulate_path(gbm, T=T, n_steps=n_steps, rng=rng)

        V_pi = 1.0
        V_mu = 1.0
        for t in range(n_steps):
            mu_t = prices[t] / prices[t].sum()
            pi_t = diversity_arbitrage_portfolio(mu_t, p=p)
            returns = prices[t + 1] / prices[t]
            V_pi *= float(np.dot(pi_t, returns))
            V_mu *= float(np.dot(mu_t, returns))

        log_excess = np.log(V_pi / V_mu)
        assert (
            log_excess > 0
        ), f"Portfolio underperformed: log(V^π/V^μ) = {log_excess:.6f}"

    def test_constructed_portfolio_matches_direct(self) -> None:
        """construct_arbitrage_portfolio('diversity') = diversity_arbitrage_portfolio."""
        mu = np.array([0.4, 0.3, 0.2, 0.1])
        a = np.eye(4)
        pi_direct = diversity_arbitrage_portfolio(mu, p=0.5)
        pi_constructed = construct_arbitrage_portfolio(mu, a, method="diversity", p=0.5)
        assert_allclose(pi_constructed, pi_direct, atol=1e-14)

    def test_portfolio_value_path_positive(self) -> None:
        """Portfolio value stays positive throughout simulation."""
        from quantspt.core.processes import CorrelatedGBM, simulate_path

        n = 5
        rng = np.random.default_rng(123)
        cov = 0.04 * np.eye(n)
        x0 = np.full(n, 100.0)
        drifts = np.full(n, 0.05)

        gbm = CorrelatedGBM(mu=drifts, cov=cov, x0=x0)
        _, prices = simulate_path(gbm, T=2.0, n_steps=2000, rng=rng)

        V = 1.0
        for t in range(2000):
            mu_t = prices[t] / prices[t].sum()
            pi_t = diversity_arbitrage_portfolio(mu_t, p=0.5)
            returns = prices[t + 1] / prices[t]
            V *= float(np.dot(pi_t, returns))
            assert V > 0, f"Portfolio value went non-positive at t={t}"
