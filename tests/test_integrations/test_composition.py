"""Cross-module integration and composition tests.

Validates end-to-end workflows:
- data → causal structure → causal covariance → Neural FGP training → backtest
- simulate GBM → estimate covariance → optimize growth rate → backtest → attribution
- Custom model pipeline: user nn.Module → wrap → integrate with master formula
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from quantspt.backtesting.engine import BacktestConfig, BacktestEngine
from quantspt.core.covariance import relative_covariance
from quantspt.core.generating_functions import DiversityGenerator, GeneratingFunction
from quantspt.core.growth_rates import excess_growth_rate
from quantspt.core.master_formula import master_formula_decomposition
from quantspt.estimation.covariance.sample import sample_covariance
from quantspt.ml.neural_fgp import NeuralFGP, NeuralFGPConfig


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(2024)


@pytest.fixture
def synthetic_market(rng):
    """Synthetic 500-day, 5-asset market (GBM-like)."""
    T, n = 500, 5
    mu = np.array([0.05, 0.07, 0.03, 0.06, 0.04])
    sigma = np.array([0.2, 0.25, 0.15, 0.22, 0.18])
    dt = 1.0 / 252.0

    log_returns = np.zeros((T, n))
    for t in range(T):
        log_returns[t] = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.normal(
            size=n
        )

    prices = 100.0 * np.exp(np.cumsum(log_returns, axis=0))
    weights = prices / prices.sum(axis=1, keepdims=True)
    returns = prices[1:] / prices[:-1]
    return {
        "prices": prices,
        "weights": weights,
        "returns": returns,
        "log_returns": log_returns,
        "dt": dt,
        "n": n,
    }


# ---------------------------------------------------------------------------
# End-to-end: data → causal → covariance → Neural FGP → master formula
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCausalToNeuralFGP:
    """End-to-end pipeline through causal structure to Neural FGP."""

    def test_causal_structure_to_neural_fgp(self, rng) -> None:
        """Data → causal DAG → covariance → neural FGP → weights."""
        from quantspt.causal.covariance import CausalCovarianceEstimator
        from quantspt.causal.structure import CausalStructureLearner

        n_samples = 1000
        X = rng.normal(size=n_samples)
        Y = 0.8 * X + rng.normal(size=n_samples) * 0.3
        Z = 0.5 * Y + rng.normal(size=n_samples) * 0.25
        data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

        learner = CausalStructureLearner(
            method="pc", ci_test="pearsonr", significance_level=0.05
        )
        learner.fit(data)
        assert len(learner.edges) >= 2

        estimator = CausalCovarianceEstimator(edges=learner.edges)
        estimator.fit(data)
        obs_cov = estimator.observational_covariance()
        assert obs_cov.shape == (3, 3)
        eigvals = np.linalg.eigvalsh(obs_cov)
        assert np.all(eigvals >= -1e-10)

        n_assets = 3
        T = 200
        prices = (
            np.exp(
                np.cumsum(
                    rng.multivariate_normal(np.zeros(3), obs_cov * 0.01, size=T),
                    axis=0,
                )
            )
            * 100.0
        )
        mw = prices / prices.sum(axis=1, keepdims=True)
        rets = prices[1:] / prices[:-1]

        config = NeuralFGPConfig(
            hidden_dims=[16, 8],
            epochs=15,
            train_window=30,
            eval_window=5,
            seed=42,
        )
        model = NeuralFGP(n_assets=n_assets, config=config)
        model.fit(mw[:-1], returns=rets)

        G = model.to_generating_function()
        assert isinstance(G, GeneratingFunction)

        mu_test = mw[-1]
        w = G.weights(mu_test)
        assert abs(w.sum() - 1.0) < 1e-4
        assert np.all(np.isfinite(w))


# ---------------------------------------------------------------------------
# End-to-end: simulate → estimate → optimize → backtest → attribution
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSimulateToBacktest:
    """simulate GBM → estimate cov → diversity strategy → backtest."""

    def test_gbm_to_backtest(self, synthetic_market) -> None:
        """Full pipeline from synthetic market to backtest result."""
        mkt = synthetic_market
        weights = mkt["weights"]
        returns = mkt["returns"]
        n = mkt["n"]

        cov_result = sample_covariance(mkt["log_returns"], annualize=False)
        cov_est = cov_result["raw"]
        assert cov_est.shape == (n, n)
        eigvals = np.linalg.eigvalsh(cov_est)
        assert np.all(eigvals > 0)

        mu_mid = weights[len(weights) // 2]
        gamma = excess_growth_rate(mu_mid, cov_est)
        assert gamma >= 0

        G = DiversityGenerator(p=0.5)

        engine = BacktestEngine(
            weight_func=G.weights,
            returns=returns,
            initial_weights=weights[0],
            config=BacktestConfig(initial_value=1.0, dt=mkt["dt"]),
        )
        result_envelope = engine.run()
        result = result_envelope.data

        assert result.portfolio_values[-1] > 0
        assert len(result.portfolio_values) == len(returns) + 1
        assert np.all(np.isfinite(result.portfolio_values))

    def test_master_formula_on_backtest(self, synthetic_market) -> None:
        """Master formula decomposition is finite on simulated data."""
        mkt = synthetic_market
        weights = mkt["weights"]

        G = DiversityGenerator(p=0.5)

        T_steps = 50
        mu_path = weights[:T_steps]
        cov_result = sample_covariance(mkt["log_returns"][:T_steps], annualize=False)
        cov_est = cov_result["raw"]
        a_path = np.tile(cov_est, (T_steps, 1, 1))

        result = master_formula_decomposition(G, mu_path, a_path, mkt["dt"])
        assert np.isfinite(result["boundary"])
        assert np.isfinite(result["drift_integral"])
        assert np.isfinite(result["total"])
        assert (
            abs(result["total"] - result["boundary"] - result["drift_integral"]) < 1e-10
        )


# ---------------------------------------------------------------------------
# Custom model pipeline: nn.Module → wrap → master formula
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCustomModelPipeline:
    """User nn.Module → wrap_torch_model → integrate with master formula."""

    def test_custom_module_to_master_formula(self, synthetic_market, rng) -> None:
        import torch
        import torch.nn as nn

        from quantspt.ml.wrappers import wrap_torch_model

        mkt = synthetic_market
        n = mkt["n"]

        class UserNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(n, 1, bias=True)
                with torch.no_grad():
                    self.fc.weight.fill_(1.0)
                    self.fc.bias.fill_(0.0)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x**2).sum(dim=-1)

        model = UserNet()
        wrapper = wrap_torch_model(model, n_assets=n, positivity_offset=2.0)
        G = wrapper.to_generating_function()
        assert isinstance(G, GeneratingFunction)

        T_steps = 20
        mu_path = mkt["weights"][:T_steps]
        cov_est = np.cov(mkt["log_returns"][:T_steps], rowvar=False)
        a_path = np.tile(cov_est, (T_steps, 1, 1))

        result = master_formula_decomposition(G, mu_path, a_path, mkt["dt"])
        assert np.isfinite(result["total"])


# ---------------------------------------------------------------------------
# Covariance estimation → excess growth rate → diversity
# ---------------------------------------------------------------------------


class TestCovarianceToGrowthRate:
    """Covariance estimation feeds correctly into growth rate computation."""

    def test_sample_cov_to_excess_growth(self, synthetic_market) -> None:
        mkt = synthetic_market
        log_ret = mkt["log_returns"]

        cov_result = sample_covariance(log_ret, annualize=False)
        cov = cov_result["raw"]
        assert cov.shape == (mkt["n"], mkt["n"])

        mu = mkt["weights"][100]
        gamma = excess_growth_rate(mu, cov)
        assert gamma >= -1e-10
        assert np.isfinite(gamma)

    def test_relative_cov_null_space_property(self, synthetic_market) -> None:
        """τ^μ · μ = 0 (null space property of relative covariance)."""
        mkt = synthetic_market
        cov_result = sample_covariance(mkt["log_returns"], annualize=False)
        cov = cov_result["raw"]
        mu = mkt["weights"][50]

        tau = relative_covariance(cov, mu)
        assert_allclose(tau @ mu, 0.0, atol=1e-10)

    def test_excess_growth_rate_nonnegative(self, synthetic_market) -> None:
        """γ* ≥ 0 for market weights (long-only portfolios)."""
        mkt = synthetic_market
        cov_result = sample_covariance(mkt["log_returns"], annualize=False)
        cov = cov_result["raw"]

        for t in range(0, 100, 10):
            mu = mkt["weights"][t]
            gamma = excess_growth_rate(mu, cov)
            assert gamma >= -1e-10


# ---------------------------------------------------------------------------
# Neural FGP → Backtest integration
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestNeuralFGPBacktest:
    """NeuralFGP trained model feeds into backtest engine."""

    def test_neural_fgp_to_backtest(self, synthetic_market) -> None:
        mkt = synthetic_market
        weights = mkt["weights"]
        returns = mkt["returns"]
        n = mkt["n"]

        config = NeuralFGPConfig(
            hidden_dims=[32, 16],
            epochs=15,
            train_window=50,
            eval_window=10,
            seed=42,
        )
        model = NeuralFGP(n_assets=n, config=config)
        train_end = 200
        model.fit(weights[:train_end], returns=returns[:train_end])

        test_returns = returns[train_end:]
        initial_w = weights[train_end]

        engine = BacktestEngine(
            weight_func=model.weights,
            returns=test_returns,
            initial_weights=initial_w,
            config=BacktestConfig(initial_value=1.0, dt=mkt["dt"]),
        )
        result_envelope = engine.run()
        result = result_envelope.data

        assert result.portfolio_values[-1] > 0
        assert np.all(np.isfinite(result.portfolio_values))
        assert len(result.portfolio_values) == len(test_returns) + 1
