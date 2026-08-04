"""Tests for the Correlated GBM market model.

Validates:
- Construction and parameter validation
- Growth rates γ_i = μ_i − a_{ii}/2
- Covariance rate matrix retrieval
- Market weights sum to 1 at all simulation times
- Excess growth rate γ* matches core formula
- StochasticProcess protocol compliance
"""

from __future__ import annotations

import numpy as np
import pytest

from quantspt.core.growth_rates import excess_growth_rate
from quantspt.core.processes import simulate_path
from quantspt.errors import SPTInvariantError
from quantspt.models.gbm import CorrelatedGBMMarket


class TestCorrelatedGBMMarketConstruction:
    """Parameter validation and construction."""

    def test_basic_construction(self) -> None:
        mu = np.array([0.05, 0.08])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        model = CorrelatedGBMMarket(mu=mu, cov=cov)
        assert model.n_assets == 2

    def test_rejects_non_symmetric_cov(self) -> None:
        mu = np.array([0.05, 0.08])
        cov = np.array([[0.04, 0.01], [0.02, 0.09]])
        with pytest.raises(Exception, match="symmetric"):
            CorrelatedGBMMarket(mu=mu, cov=cov)

    def test_rejects_shape_mismatch(self) -> None:
        mu = np.array([0.05, 0.08, 0.10])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        with pytest.raises(Exception, match="shape"):
            CorrelatedGBMMarket(mu=mu, cov=cov)

    def test_rejects_non_psd(self) -> None:
        mu = np.array([0.05, 0.08])
        cov = np.array([[0.04, 0.10], [0.10, 0.04]])  # negative eigenvalue
        with pytest.raises(Exception, match="PSD"):
            CorrelatedGBMMarket(mu=mu, cov=cov)


class TestGrowthRates:
    """Growth rate γ_i = μ_i − a_{ii}/2 must hold exactly."""

    def test_growth_rates_formula(self) -> None:
        mu = np.array([0.10, 0.06, 0.12])
        cov = np.diag([0.04, 0.09, 0.16])
        model = CorrelatedGBMMarket(mu=mu, cov=cov)
        x = np.ones(3)  # state doesn't matter for constant model
        gamma = model.drift_rates(0.0, x)
        expected = mu - 0.5 * np.diag(cov)
        np.testing.assert_allclose(gamma, expected)

    def test_growth_rates_independent_of_state(self) -> None:
        mu = np.array([0.05, 0.08])
        cov = np.diag([0.04, 0.09])
        model = CorrelatedGBMMarket(mu=mu, cov=cov)
        g1 = model.drift_rates(0.0, np.array([100.0, 50.0]))
        g2 = model.drift_rates(5.0, np.array([200.0, 10.0]))
        np.testing.assert_allclose(g1, g2)

    def test_growth_rates_independent_of_time(self) -> None:
        mu = np.array([0.07])
        cov = np.diag([0.04])
        model = CorrelatedGBMMarket(mu=mu, cov=cov)
        g0 = model.drift_rates(0.0, np.ones(1))
        g100 = model.drift_rates(100.0, np.ones(1))
        np.testing.assert_allclose(g0, g100)


class TestCovarianceRate:
    """Covariance matrix retrieval."""

    def test_returns_correct_matrix(self) -> None:
        cov = np.array([[0.04, 0.02], [0.02, 0.09]])
        model = CorrelatedGBMMarket(mu=np.array([0.05, 0.08]), cov=cov)
        a = model.covariance_rate(0.0, np.ones(2))
        np.testing.assert_allclose(a, cov)

    def test_returns_copy(self) -> None:
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        model = CorrelatedGBMMarket(mu=np.array([0.05, 0.08]), cov=cov)
        a = model.covariance_rate(0.0, np.ones(2))
        a[0, 0] = 999.0
        a2 = model.covariance_rate(0.0, np.ones(2))
        assert a2[0, 0] != 999.0


class TestMarketWeightsInSimulation:
    """Market weights must sum to 1 at all times when computed from GBM paths."""

    def test_weights_sum_to_one(self) -> None:
        n = 5
        rng = np.random.default_rng(123)
        mu = rng.uniform(0.03, 0.12, n)
        L = rng.standard_normal((n, n)) * 0.1
        cov = L @ L.T + np.eye(n) * 0.02
        model = CorrelatedGBMMarket(mu=mu, cov=cov)

        x0 = rng.uniform(50, 200, n)
        proc = model.to_stochastic_process(x0)
        _, path = simulate_path(proc, T=1.0, n_steps=100, rng=rng)

        for step in range(path.shape[0]):
            weights = model.market_weights(path[step])
            np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-12)

    def test_weights_nonnegative(self) -> None:
        """GBM prices stay positive → weights are non-negative."""
        rng = np.random.default_rng(456)
        mu = np.array([0.05, 0.08, 0.03])
        cov = np.diag([0.04, 0.09, 0.01])
        model = CorrelatedGBMMarket(mu=mu, cov=cov)
        x0 = np.array([100.0, 100.0, 100.0])
        proc = model.to_stochastic_process(x0)
        _, path = simulate_path(proc, T=2.0, n_steps=200, rng=rng)

        for step in range(path.shape[0]):
            weights = model.market_weights(path[step])
            assert np.all(weights >= 0)


class TestExcessGrowthRateConsistency:
    """γ* from the model's covariance must match core.growth_rates."""

    def test_matches_core_function(self) -> None:
        rng = np.random.default_rng(789)
        n = 4
        mu = rng.uniform(0.03, 0.12, n)
        L = rng.standard_normal((n, n)) * 0.1
        cov = L @ L.T + np.eye(n) * 0.02
        model = CorrelatedGBMMarket(mu=mu, cov=cov)

        x = rng.uniform(50, 200, n)
        weights = model.market_weights(x)
        a = model.covariance_rate(0.0, x)

        gamma_star_model = excess_growth_rate(weights, a)

        expected = 0.5 * (np.dot(weights, np.diag(a)) - weights @ a @ weights)
        np.testing.assert_allclose(gamma_star_model, expected, atol=1e-14)

    def test_equal_weighted_excess_growth(self) -> None:
        """For uncorrelated stocks: γ*_η = (n-1)/(2n) · σ²."""
        n = 10
        sigma_sq = 0.04
        cov = np.eye(n) * sigma_sq
        mu = np.full(n, 0.05)
        model = CorrelatedGBMMarket(mu=mu, cov=cov)
        pi = np.full(n, 1.0 / n)
        a = model.covariance_rate(0.0, np.ones(n))
        gamma_star = excess_growth_rate(pi, a)
        expected = (n - 1) / (2.0 * n) * sigma_sq
        np.testing.assert_allclose(gamma_star, expected, atol=1e-14)


class TestStochasticProcessProtocol:
    """The returned process must implement all protocol methods."""

    def test_protocol_methods_exist(self) -> None:
        model = CorrelatedGBMMarket(
            mu=np.array([0.05, 0.08]),
            cov=np.diag([0.04, 0.09]),
        )
        proc = model.to_stochastic_process(np.array([100.0, 100.0]))
        assert proc.size() == 2
        assert proc.factors() == 2
        assert proc.initial_values().shape == (2,)

    def test_rejects_wrong_x0_length(self) -> None:
        model = CorrelatedGBMMarket(
            mu=np.array([0.05, 0.08]),
            cov=np.diag([0.04, 0.09]),
        )
        with pytest.raises(SPTInvariantError, match="mismatch"):
            model.to_stochastic_process(np.array([100.0]))

    def test_rejects_nonpositive_x0(self) -> None:
        model = CorrelatedGBMMarket(
            mu=np.array([0.05]),
            cov=np.diag([0.04]),
        )
        with pytest.raises(SPTInvariantError, match="positive"):
            model.to_stochastic_process(np.array([-1.0]))
