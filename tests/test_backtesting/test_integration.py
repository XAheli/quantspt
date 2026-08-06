"""End-to-end integration test for the backtesting pipeline.

Simulates a 5-stock GBM market → constructs DiversityGenerator(p=0.5) →
backtests with monthly rebalancing and 10bps proportional costs →
computes master formula attribution → verifies the identity holds →
reports performance metrics.

This single test validates the entire backtest → attribution → master
formula pipeline works correctly.

Mathematical References
-----------------------
- Master formula: F&K Survey Eq. 11.2
- Diversity generator: F&K Survey Remark 11.1 (Example 3)
- Drift process non-negativity for p ∈ (0,1): FKK
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from quantspt.backtesting.attribution import compute_attribution
from quantspt.backtesting.engine import BacktestConfig, BacktestEngine
from quantspt.backtesting.execution import (
    InstantExecution,
    ProportionalCostExecution,
)
from quantspt.backtesting.performance import (
    compute_performance,
    compute_turnover_stats,
    information_ratio,
    max_drawdown,
    sharpe_ratio,
    tracking_error,
)
from quantspt.backtesting.rebalancing import CalendarRebalancer, Frequency
from quantspt.backtesting.statistical_tests import (
    bootstrap_confidence_interval,
    permutation_test,
)
from quantspt.core.generating_functions import DiversityGenerator
from quantspt.core.processes import CorrelatedGBM, simulate_path


def _build_5stock_cov() -> np.ndarray:
    """Construct a realistic 5×5 covariance matrix."""
    rng = np.random.default_rng(0)
    L = rng.standard_normal((5, 5)) * 0.1
    cov = L @ L.T + np.eye(5) * 0.02
    return (cov + cov.T) / 2


@pytest.mark.integration
class TestFullPipelineIntegration:
    """End-to-end: simulate → backtest → attribute → verify master formula."""

    def test_full_pipeline_master_formula_holds(self) -> None:
        """THE critical integration test.

        Simulates 5-stock GBM → DiversityGenerator(p=0.5) → monthly
        rebalancing + 10bps costs → master formula attribution.

        Verifies:
        1. Backtest produces finite, reasonable values
        2. Master formula identity holds (boundary + drift ≈ actual)
        3. Performance metrics are computable and finite
        4. Turnover statistics are consistent
        """
        cov = _build_5stock_cov()
        drifts = np.array([0.06, 0.08, 0.05, 0.07, 0.04])
        x0 = np.array([100.0, 80.0, 120.0, 90.0, 110.0])

        rng = np.random.default_rng(2024)
        gbm = CorrelatedGBM(mu=drifts, cov=cov, x0=x0)
        n_steps = 2520  # ~10 years of daily data
        T = 10.0
        _, prices = simulate_path(gbm, T=T, n_steps=n_steps, rng=rng)

        mu_path = prices / prices.sum(axis=1, keepdims=True)
        returns = prices[1:] / prices[:-1]  # (T, n) gross returns
        dt = T / n_steps

        # -- 1) Run backtest with DiversityGenerator --
        G = DiversityGenerator(p=0.5)
        engine = BacktestEngine(
            weight_func=G.weights,
            returns=returns,
            initial_weights=mu_path[0],
            rebalancer=CalendarRebalancer(Frequency.MONTHLY),
            execution=ProportionalCostExecution(cost_bps=10.0),
            config=BacktestConfig(initial_value=1.0, dt=dt),
        )
        result = engine.run()
        bt = result.data

        # Basic sanity
        assert np.all(np.isfinite(bt.portfolio_values))
        assert np.all(bt.portfolio_values > 0)
        assert np.all(np.isfinite(bt.market_values))
        assert bt.n_rebalances > 0
        assert bt.total_turnover() > 0

        # -- 2) Master formula attribution (zero-cost version for theory) --
        engine_free = BacktestEngine(
            weight_func=G.weights,
            returns=returns,
            initial_weights=mu_path[0],
            rebalancer=CalendarRebalancer(Frequency.DAILY),
            execution=InstantExecution(),
            config=BacktestConfig(initial_value=1.0, dt=dt),
        )
        bt_free = engine_free.run().data
        log_rel_free = bt_free.log_relative_return()

        attribution = compute_attribution(G, mu_path, cov, log_rel_free, dt)

        assert np.isfinite(attribution.boundary)
        assert np.isfinite(attribution.drift_integral)
        assert (
            attribution.drift_integral > 0
        ), "Diversity drift should be positive for p=0.5"
        assert_allclose(
            attribution.actual_log_relative,
            attribution.predicted_log_relative,
            atol=0.15,
            err_msg=(
                f"Master formula VIOLATED in integration test: "
                f"actual={attribution.actual_log_relative:.6f}, "
                f"predicted={attribution.predicted_log_relative:.6f}, "
                f"boundary={attribution.boundary:.6f}, "
                f"drift={attribution.drift_integral:.6f}, "
                f"residual={attribution.residual:.6f}"
            ),
        )

        # -- 3) Performance metrics --
        perf = compute_performance(bt.portfolio_values, dt)
        assert np.isfinite(perf.annualized_return)
        assert np.isfinite(perf.annualized_volatility)
        assert perf.annualized_volatility > 0
        assert np.isfinite(perf.sharpe_ratio)
        assert np.isfinite(perf.max_drawdown)
        assert perf.max_drawdown <= 0

        mkt_perf = compute_performance(bt.market_values, dt)
        assert np.isfinite(mkt_perf.annualized_return)

        sr = sharpe_ratio(bt.portfolio_values, dt)
        assert np.isfinite(sr)

        dd = max_drawdown(bt.portfolio_values)
        assert dd <= 0

        track_err = tracking_error(bt.portfolio_values, bt.market_values, dt)
        assert track_err >= 0

        ir = information_ratio(bt.portfolio_values, bt.market_values, dt)
        assert np.isfinite(ir)

        # -- 4) Turnover statistics --
        turnover_stats = compute_turnover_stats(bt.turnover, bt.n_rebalances, dt)
        assert turnover_stats.total_turnover > 0
        assert turnover_stats.avg_turnover_per_rebalance > 0
        assert turnover_stats.annualized_turnover > 0

    def test_zero_cost_vs_costly_backtest(self) -> None:
        """Zero-cost backtest should outperform costly one."""
        cov = _build_5stock_cov()
        drifts = np.array([0.06, 0.08, 0.05, 0.07, 0.04])
        x0 = np.array([100.0, 80.0, 120.0, 90.0, 110.0])

        rng = np.random.default_rng(42)
        gbm = CorrelatedGBM(mu=drifts, cov=cov, x0=x0)
        n_steps = 504
        T = 2.0
        _, prices = simulate_path(gbm, T=T, n_steps=n_steps, rng=rng)

        mu_path = prices / prices.sum(axis=1, keepdims=True)
        returns = prices[1:] / prices[:-1]
        dt = T / n_steps
        G = DiversityGenerator(p=0.5)

        engine_free = BacktestEngine(
            weight_func=G.weights,
            returns=returns,
            initial_weights=mu_path[0],
            rebalancer=CalendarRebalancer(Frequency.MONTHLY),
            execution=InstantExecution(),
            config=BacktestConfig(dt=dt),
        )
        engine_costly = BacktestEngine(
            weight_func=G.weights,
            returns=returns,
            initial_weights=mu_path[0],
            rebalancer=CalendarRebalancer(Frequency.MONTHLY),
            execution=ProportionalCostExecution(cost_bps=50.0),
            config=BacktestConfig(dt=dt),
        )

        v_free = engine_free.run().data.portfolio_values[-1]
        v_costly = engine_costly.run().data.portfolio_values[-1]
        assert v_free > v_costly

    def test_statistical_tests_on_backtest_returns(self) -> None:
        """Bootstrap CI and permutation test on backtest log-returns."""
        cov = _build_5stock_cov()
        drifts = np.array([0.06, 0.08, 0.05, 0.07, 0.04])
        x0 = np.array([100.0, 80.0, 120.0, 90.0, 110.0])

        rng = np.random.default_rng(2024)
        gbm = CorrelatedGBM(mu=drifts, cov=cov, x0=x0)
        n_steps = 504
        T = 2.0
        _, prices = simulate_path(gbm, T=T, n_steps=n_steps, rng=rng)

        mu_path = prices / prices.sum(axis=1, keepdims=True)
        returns = prices[1:] / prices[:-1]
        dt = T / n_steps
        G = DiversityGenerator(p=0.5)

        engine = BacktestEngine(
            weight_func=G.weights,
            returns=returns,
            initial_weights=mu_path[0],
            rebalancer=CalendarRebalancer(Frequency.MONTHLY),
            execution=InstantExecution(),
            config=BacktestConfig(dt=dt),
        )
        bt = engine.run().data

        port_log_ret = np.diff(np.log(bt.portfolio_values))
        ci = bootstrap_confidence_interval(
            port_log_ret,
            np.mean,
            n_bootstrap=2000,
            rng=np.random.default_rng(1),
        )
        assert ci.ci_lower < ci.ci_upper
        assert np.isfinite(ci.estimate)

        mkt_log_ret = np.diff(np.log(bt.market_values))
        perm = permutation_test(
            port_log_ret,
            mkt_log_ret,
            n_permutations=1000,
            rng=np.random.default_rng(1),
        )
        assert 0.0 <= perm.p_value <= 1.0
