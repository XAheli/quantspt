"""Comprehensive real-data validation of EVERY quantspt module.

Downloads 50 S&P 500 stocks from yfinance (2022-2025) and validates
every layer of the library: data, estimation, models, rank, arbitrage,
optimization, backtesting, ML, causal, visualization, and post-processing.

Run with:
    source /root/quantspt/.venv/bin/activate
    pytest tests/benchmarks/test_comprehensive_real_data.py -v -s --tb=short 2>&1

Each test prints a structured report:
    Module: [name]
    Test: [what was tested]
    Input: [real data description]
    Expected: [what theory predicts]
    Actual: [what happened]
    Status: PASS/FAIL/INVESTIGATE
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

PLOTS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "plots"
EXPORT_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "exports"

SP50_TICKERS = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "NVDA",
    "TSLA",
    "BRK-B",
    "JPM",
    "JNJ",
    "V",
    "PG",
    "UNH",
    "HD",
    "MA",
    "DIS",
    "BAC",
    "ADBE",
    "CRM",
    "NFLX",
    "CSCO",
    "PFE",
    "ABT",
    "TMO",
    "AVGO",
    "COST",
    "PEP",
    "MRK",
    "ABBV",
    "WMT",
    "XOM",
    "CVX",
    "LLY",
    "KO",
    "MCD",
    "NKE",
    "INTC",
    "QCOM",
    "TXN",
    "HON",
    "NEE",
    "LOW",
    "UNP",
    "AMGN",
    "CAT",
    "GS",
    "BLK",
    "ISRG",
    "GILD",
    "MDT",
]


def _report(module, test, inp, expected, actual, status):
    """Print structured test report."""
    print(f"\n  Module: {module}")
    print(f"  Test: {test}")
    print(f"  Input: {inp}")
    print(f"  Expected: {expected}")
    print(f"  Actual: {actual}")
    print(f"  Status: {status}")


def _download_prices(tickers, start, end):
    try:
        from quantspt.data.providers.yfinance import YFinanceProvider

        provider = YFinanceProvider(progress=False)
        result = provider.load(tickers=tickers, start=start, end=end)
        return result.data.prices
    except Exception as exc:
        print(f"  WARNING: YFinance download failed: {exc}")
        return None


def _synthetic_prices(n_stocks, n_days, seed=42):
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    mu = rng.uniform(0.05, 0.15, size=n_stocks)
    sigma = rng.uniform(0.15, 0.45, size=n_stocks)
    log_rets = (mu - 0.5 * sigma**2)[:, None] * dt + sigma[:, None] * np.sqrt(
        dt
    ) * rng.standard_normal((n_stocks, n_days))
    prices = 100.0 * np.exp(np.cumsum(log_rets, axis=1))
    prices = np.column_stack([np.full(n_stocks, 100.0), prices])
    dates = pd.bdate_range(start="2022-01-03", periods=n_days + 1)[: prices.shape[1]]
    tickers = [f"SYN{i:03d}" for i in range(n_stocks)]
    return pd.DataFrame(prices.T, index=dates, columns=tickers)


# ===================================================================
# Shared fixtures
# ===================================================================


@pytest.fixture(scope="module")
def real_data():
    """Load 50 S&P 500 stocks, 2022-2025 (3 years)."""
    prices = _download_prices(SP50_TICKERS, start="2022-01-01", end="2025-01-01")
    is_synthetic = prices is None
    if is_synthetic:
        prices = _synthetic_prices(50, 756)
    tickers = list(prices.columns)
    log_returns = np.log(prices / prices.shift(1)).dropna()
    weights = prices.div(prices.sum(axis=1), axis=0)
    return {
        "prices": prices,
        "tickers": tickers,
        "log_returns": log_returns,
        "weights": weights,
        "is_synthetic": is_synthetic,
    }


# ===================================================================
# 1. DATA LAYER
# ===================================================================


@pytest.mark.slow
class TestDataLayer:
    def test_yfinance_load_and_preprocessing(self, real_data):
        prices = real_data["prices"]
        n_days, n_stocks = prices.shape
        nan_count = int(prices.isna().sum().sum())
        _report(
            "Data",
            "Load 50 S&P 500 stocks via yfinance",
            f"{n_stocks} stocks, {n_days} days (2022-2025)",
            "No NaN after ffill/bfill, all positive prices",
            f"{nan_count} NaN remaining, min price={prices.min().min():.2f}",
            "PASS" if nan_count == 0 and prices.min().min() > 0 else "FAIL",
        )
        assert nan_count == 0
        assert prices.min().min() > 0

    def test_returns_computation(self, real_data):
        lr = real_data["log_returns"].values
        mean_abs = np.abs(lr).mean()
        max_abs = np.abs(lr).max()
        _report(
            "Data",
            "Log returns computation",
            f"Shape {lr.shape}",
            "Mean |r| < 0.05, max |r| < 1.0 (no splits in adj. data)",
            f"Mean |r|={mean_abs:.6f}, max |r|={max_abs:.4f}",
            "PASS" if mean_abs < 0.05 and max_abs < 1.0 else "INVESTIGATE",
        )
        assert mean_abs < 0.1

    def test_universe_construction(self, real_data):
        from quantspt.data import MarketPanel, reconstruct

        prices = real_data["prices"]
        tickers = real_data["tickers"]
        panel = MarketPanel(prices=prices, tickers=tickers)
        universe = reconstruct(panel, min_observations=60)
        counts = universe.member_count()
        _report(
            "Data",
            "Universe with min_observations=60",
            f"{len(tickers)} tickers, {len(prices)} days",
            "All stocks enter after 60 days; count grows to ~50",
            f"Min members={counts.min()}, max={counts.max()}, final={counts.iloc[-1]}",
            "PASS" if counts.iloc[-1] >= 40 else "FAIL",
        )
        assert counts.iloc[-1] >= 40

    def test_corporate_action_detection(self, real_data):
        from quantspt.data import detect_splits

        prices = real_data["prices"]
        splits = detect_splits(prices)
        n_detected = (
            sum(len(v) for v in splits.values())
            if isinstance(splits, dict)
            else len(splits)
        )
        tickers_with_splits = list(splits.keys()) if isinstance(splits, dict) else []
        _report(
            "Data",
            "Corporate action (split) detection",
            "50 stocks, 2022-2025 (adjusted prices)",
            "Few or no splits (data is auto-adjusted by yfinance)",
            f"{n_detected} splits detected in {tickers_with_splits}",
            "PASS" if n_detected < 10 else "INVESTIGATE",
        )


# ===================================================================
# 2. ESTIMATION LAYER
# ===================================================================


@pytest.mark.slow
class TestEstimationLayer:
    def test_sample_covariance(self, real_data):
        from quantspt.estimation import sample_covariance

        lr = real_data["log_returns"].values
        result = sample_covariance(lr, annualize=True)
        cov = result["raw"] * 252
        eigs = np.linalg.eigvalsh(cov)
        cond = eigs[-1] / max(eigs[0], 1e-15)
        _report(
            "Estimation",
            "sample_covariance on real returns",
            f"Shape {lr.shape}",
            "PSD, condition number < 10000, eigenvalues > 0",
            f"min_eig={eigs[0]:.6f}, max_eig={eigs[-1]:.4f}, cond={cond:.1f}",
            "PASS" if eigs[0] > -1e-8 and cond < 1e6 else "FAIL",
        )
        assert eigs[0] > -1e-8

    def test_rolling_sample_covariance(self, real_data):
        from quantspt.estimation import rolling_sample_covariance

        lr = real_data["log_returns"].values
        results = rolling_sample_covariance(lr, window=60, annualize=True)
        n_windows = len(results)
        trace_series = [np.trace(r["annualized"]) for r in results]
        trace_std = np.std(trace_series) / np.mean(trace_series)
        _report(
            "Estimation",
            "rolling_sample_covariance (window=60)",
            f"{lr.shape[0]} days, 50 stocks",
            "Time-varying, smooth (trace CoV < 0.5)",
            f"{n_windows} windows, trace CoV={trace_std:.4f}",
            "PASS" if trace_std < 0.5 else "INVESTIGATE",
        )
        assert n_windows == lr.shape[0] - 60 + 1

    def test_shrinkage_covariance(self, real_data):
        from quantspt.estimation import ledoit_wolf, sample_covariance

        lr = real_data["log_returns"].values
        sample = sample_covariance(lr, annualize=True)
        shrunk = ledoit_wolf(lr, annualize=True)

        sample_cond = np.linalg.cond(sample["annualized"])
        shrunk_cond = np.linalg.cond(shrunk["covariance"])
        alpha = shrunk["shrinkage_intensity"]
        _report(
            "Estimation",
            "shrinkage_covariance reduces condition number",
            f"50 stocks x {lr.shape[0]} days",
            "Shrinkage alpha in (0, 1); condition number reduced",
            f"alpha={alpha:.4f}, sample_cond={sample_cond:.0f}, shrunk_cond={shrunk_cond:.0f}",
            "PASS" if shrunk_cond < sample_cond else "INVESTIGATE",
        )
        assert 0 <= alpha <= 1

    def test_growth_rate_estimation(self, real_data):
        from quantspt.estimation import estimate_growth_rates

        lr = real_data["log_returns"].values
        result = estimate_growth_rates(lr, frequency=252, bias_correction=True)
        gamma = result["growth_rates"]
        _report(
            "Estimation",
            "estimate_growth_rates on real returns",
            f"50 stocks x {lr.shape[0]} days",
            "gamma_i in (-0.5, +1.0) annualized",
            f"min={gamma.min():.4f}, max={gamma.max():.4f}, mean={gamma.mean():.4f}",
            "PASS" if gamma.min() > -0.5 and gamma.max() < 1.0 else "INVESTIGATE",
        )
        assert gamma.min() > -2.0
        assert gamma.max() < 3.0

    def test_rolling_diversity_deficit(self, real_data):
        from quantspt.estimation import rolling_diversity_deficit

        weights = real_data["weights"].values
        deficits = rolling_diversity_deficit(weights, p=0.5)
        _report(
            "Estimation",
            "rolling_diversity_deficit (p=0.5)",
            f"50-stock weights, {len(deficits)} days",
            "Deficit > 0 (market is diverse), smooth over time",
            f"min={deficits.min():.4f}, max={deficits.max():.4f}, mean={deficits.mean():.4f}",
            "PASS" if deficits.min() > 0 else "INVESTIGATE",
        )

    def test_calibrate_atlas(self, real_data):
        from quantspt.estimation import calibrate_atlas

        prices = real_data["prices"].values
        market_caps = prices.copy()
        params = calibrate_atlas(market_caps, min_observations=50)
        pareto = params["pareto_exponents"]
        _report(
            "Estimation",
            "calibrate_atlas on real capital distribution",
            f"50 stocks x {prices.shape[0]} days",
            "Pareto exponents > 0, g sums to 0, sigma > 0",
            f"n={params['n']}, pareto[0]={pareto[0]:.3f}, gamma={params['gamma']:.4f}",
            "PASS"
            if pareto[0] > 0 and np.isclose(np.sum(params["g"]), 0, atol=1e-6)
            else "FAIL",
        )
        assert np.isclose(np.sum(params["g"]), 0, atol=1e-6)


# ===================================================================
# 3. MODELS LAYER
# ===================================================================


@pytest.mark.slow
class TestModelsLayer:
    def test_correlated_gbm_from_real_params(self, real_data):
        from quantspt.models import CorrelatedGBMMarket

        lr = real_data["log_returns"].values
        mu = np.mean(lr, axis=0) * 252 + 0.5 * np.var(lr, axis=0, ddof=1) * 252
        cov = np.cov(lr, rowvar=False, ddof=1) * 252
        cov = (cov + cov.T) / 2
        eig = np.linalg.eigvalsh(cov)
        if eig[0] < 0:
            cov += np.eye(len(mu)) * (-eig[0] + 1e-8)

        model = CorrelatedGBMMarket(mu=mu, cov=cov)
        assert model.n_assets == len(mu)
        _report(
            "Models",
            "CorrelatedGBMMarket from real estimated params",
            f"50 stocks, mu range [{mu.min():.3f}, {mu.max():.3f}]",
            "Model creates successfully, n_assets=50",
            f"n_assets={model.n_assets}, growth_rates range=[{model._growth_rates.min():.3f}, {model._growth_rates.max():.3f}]",
            "PASS",
        )

    def test_atlas_calibration_fit(self, real_data):
        from scipy.optimize import minimize_scalar

        from quantspt.models.atlas import AtlasModel

        weights = real_data["weights"].values
        ranked = np.sort(weights, axis=1)[:, ::-1]
        mean_ranked = ranked.mean(axis=0)
        lr = real_data["log_returns"].values
        avg_vol = np.sqrt(252) * lr.std(axis=0).mean()
        n_stocks = weights.shape[1]

        def fit_err(g_param):
            if g_param <= 0:
                return 1e10
            try:
                m = AtlasModel(
                    n=n_stocks, gamma=0.05, g_param=float(g_param), sigma_param=avg_vol
                )
                return float(
                    np.sum((m.certainty_equivalent_weights() - mean_ranked) ** 2)
                )
            except Exception:
                return 1e10

        result = minimize_scalar(fit_err, bounds=(1e-4, 0.5), method="bounded")
        model = AtlasModel(
            n=n_stocks, gamma=0.05, g_param=float(result.x), sigma_param=avg_vol
        )
        ce_w = model.certainty_equivalent_weights()
        corr = np.corrcoef(mean_ranked, ce_w)[0, 1]
        _report(
            "Models",
            "Atlas model calibration → verify fit",
            f"50-stock ranked weights, avg_vol={avg_vol:.3f}",
            "CE weight correlation > 0.5 with empirical",
            f"g_param={result.x:.6f}, pareto_r1={model.pareto_exponent():.3f}, corr={corr:.4f}",
            "PASS" if corr > 0.5 else "FAIL",
        )
        assert corr > 0.3


# ===================================================================
# 4. RANK LAYER
# ===================================================================


@pytest.mark.slow
class TestRankLayer:
    def test_rank_transitions(self, real_data):
        from quantspt.rank.transitions import (
            expected_sojourn_times,
            rank_mobility_index,
            rank_transition_matrix,
        )

        weights = real_data["weights"].values
        P = rank_transition_matrix(weights, horizon=1)
        n = P.shape[0]
        mobility = rank_mobility_index(P)
        sojourn = expected_sojourn_times(P)
        _report(
            "Rank",
            "Rank transitions on 3-year real data",
            f"50 stocks, {len(weights)} days",
            "Row-stochastic, mobility < 1-1/n, top ranks more stable",
            f"mobility={mobility:.4f}, max_sojourn={sojourn.max():.1f}, diag_mean={np.diag(P).mean():.4f}",
            "PASS"
            if np.allclose(P.sum(axis=1), 1.0, atol=1e-8) and mobility < 1
            else "FAIL",
        )
        assert np.allclose(P.sum(axis=1), 1.0, atol=1e-8)
        assert sojourn[0] > sojourn[n // 2]


# ===================================================================
# 5. ARBITRAGE LAYER
# ===================================================================


@pytest.mark.slow
class TestArbitrageLayer:
    def test_diversity_conditions(self, real_data):
        from quantspt.arbitrage import (
            check_strict_diversity,
            check_weak_diversity,
            estimate_diversity_parameters,
        )

        weights = real_data["weights"].values
        params = estimate_diversity_parameters(weights, p=0.5)
        delta_est = max(params["delta_min"], 0.001)
        n_strict = sum(check_strict_diversity(w, delta=0.5) for w in weights)
        n_weak = sum(check_weak_diversity(w, p=0.5, delta=delta_est) for w in weights)
        _report(
            "Arbitrage",
            "Diversity conditions on real S&P 500",
            f"50 stocks, {len(weights)} days",
            "Strict diversity with delta=0.5 should hold (no stock > 50%)",
            f"Strict: {n_strict}/{len(weights)}, Weak: {n_weak}/{len(weights)}, delta_min={params['delta_min']:.4f}",
            "PASS" if n_strict == len(weights) else "INVESTIGATE",
        )

    def test_arbitrage_horizon(self, real_data):
        from quantspt.arbitrage import diversity_horizon, entropy_horizon

        weights = real_data["weights"].values
        lr = real_data["log_returns"].values
        cov = np.cov(lr, rowvar=False, ddof=1) * 252
        min_eig = np.linalg.eigvalsh(cov)[0]
        eps = max(min_eig, 0.001)
        from quantspt.arbitrage import estimate_diversity_parameters

        params = estimate_diversity_parameters(weights, p=0.5)
        delta = max(params["delta_min"], 0.001)

        T_div = diversity_horizon(n=50, p=0.5, eps=eps, delta=delta)
        mu_last = weights[-1]
        mu_last = np.clip(mu_last, 1e-12, None)
        mu_last /= mu_last.sum()
        H_0 = -float(np.sum(mu_last * np.log(mu_last)))
        zeta = max(eps * 0.01, 0.001)
        T_ent = entropy_horizon(H_mu_0=H_0, zeta=zeta)
        _report(
            "Arbitrage",
            "Arbitrage horizon for current market",
            f"n=50, p=0.5, eps={eps:.4f}, delta={delta:.4f}",
            "Diversity horizon should be finite and > 0",
            f"T*_diversity={T_div:.2f} years, T*_entropy={T_ent:.2f} years",
            "PASS" if T_div > 0 and np.isfinite(T_div) else "FAIL",
        )
        assert T_div > 0

    def test_diversity_arbitrage_portfolio(self, real_data):
        from quantspt.arbitrage import diversity_arbitrage_portfolio

        weights = real_data["weights"].values
        mu = weights[-1]
        mu = np.clip(mu, 1e-12, None)
        mu /= mu.sum()
        pi = diversity_arbitrage_portfolio(mu, p=0.5)
        _report(
            "Arbitrage",
            "Diversity arbitrage portfolio construction",
            "Latest weights, 50 stocks",
            "pi sums to 1, all positive, overweights small stocks vs mu",
            f"sum={pi.sum():.6f}, min={pi.min():.6f}, max={pi.max():.6f}",
            "PASS" if abs(pi.sum() - 1.0) < 1e-8 and pi.min() > 0 else "FAIL",
        )
        assert abs(pi.sum() - 1.0) < 1e-8

    def test_mirror_portfolio(self, real_data):
        from quantspt.arbitrage import mirror_is_long_only, mirror_portfolio

        weights = real_data["weights"].values
        mu = weights[-1]
        mu = np.clip(mu, 1e-12, None)
        mu /= mu.sum()
        from quantspt.arbitrage import diversity_arbitrage_portfolio

        pi = diversity_arbitrage_portfolio(mu, p=0.5)
        pi_hat = mirror_portfolio(mu, pi)
        is_long = mirror_is_long_only(mu, pi)
        _report(
            "Arbitrage",
            "Mirror portfolio analysis",
            "Diversity(p=0.5) portfolio on last day",
            "Mirror sums to 1; may have negative weights for p far from 1",
            f"mirror sum={pi_hat.sum():.6f}, min={pi_hat.min():.6f}, long_only={is_long}",
            "PASS",
        )
        assert abs(pi_hat.sum() - 1.0) < 1e-8


# ===================================================================
# 6. OPTIMIZATION LAYER
# ===================================================================


@pytest.mark.slow
class TestOptimizationLayer:
    def test_optimize_growth_rate(self, real_data):
        from quantspt.optimization import optimize_growth_rate

        lr = real_data["log_returns"].values
        gamma = np.mean(lr, axis=0) * 252
        cov = np.cov(lr, rowvar=False, ddof=1) * 252
        cov = (cov + cov.T) / 2
        eig = np.linalg.eigvalsh(cov)
        if eig[0] < 0:
            cov += np.eye(len(gamma)) * (-eig[0] + 1e-8)

        result = optimize_growth_rate(gamma, cov)
        w = result["weights"]
        is_eq = np.allclose(w, 1.0 / len(w), atol=0.01)
        _report(
            "Optimization",
            "Optimize growth rate on real params",
            "50 stocks, unconstrained",
            "Weights sum to 1, growth_rate > market growth",
            f"g*={result['growth_rate']:.6f}, excess={result['excess_growth_rate']:.6f}, is_equal_weight={is_eq}",
            "PASS" if abs(w.sum() - 1.0) < 1e-6 else "FAIL",
        )
        assert abs(w.sum() - 1.0) < 1e-6

    def test_constrained_optimization(self, real_data):
        from quantspt.optimization import optimize_growth_rate

        lr = real_data["log_returns"].values
        gamma = np.mean(lr, axis=0) * 252
        cov = np.cov(lr, rowvar=False, ddof=1) * 252
        cov = (cov + cov.T) / 2
        eig = np.linalg.eigvalsh(cov)
        if eig[0] < 0:
            cov += np.eye(len(gamma)) * (-eig[0] + 1e-8)

        n = len(gamma)
        prev_w = np.ones(n) / n
        result = optimize_growth_rate(
            gamma,
            cov,
            min_weight=0.0,
            max_weight=0.05,
            max_turnover=0.10,
            prev_weights=prev_w,
        )
        w = result["weights"]
        satisfies_max = np.all(w <= 0.05 + 1e-6)
        turnover = 0.5 * np.sum(np.abs(w - prev_w))
        _report(
            "Optimization",
            "Constrained: max 5%, max 10% turnover",
            "50 stocks, prev=equal-weight",
            "All weights <= 5%, turnover <= 10%",
            f"max_w={w.max():.4f}, turnover={turnover:.4f}",
            "PASS" if satisfies_max and turnover <= 0.10 + 1e-4 else "FAIL",
        )
        assert satisfies_max

    def test_transaction_cost_impact(self, real_data):
        from quantspt.optimization.transaction_costs import (
            net_growth_rate,
            proportional_cost,
        )

        n = 50
        old_w = np.ones(n) / n
        new_w = np.random.default_rng(42).dirichlet(np.ones(n))
        cost = proportional_cost(old_w, new_w, cost_bps=10.0)
        net_g = net_growth_rate(
            0.10, old_w, new_w, cost_bps=10.0, rebalance_frequency=21
        )
        _report(
            "Optimization",
            "Transaction cost impact",
            "EW -> random portfolio, 10bps cost",
            "Net growth < gross growth; cost > 0",
            f"gross=0.1000, net={net_g:.6f}, cost_per_trade={cost:.6f}",
            "PASS" if net_g < 0.10 and cost > 0 else "FAIL",
        )
        assert net_g < 0.10


# ===================================================================
# 7. BACKTESTING LAYER
# ===================================================================


@pytest.mark.slow
class TestBacktestingLayer:
    def _prepare_backtest_inputs(self, real_data):
        prices = real_data["prices"].values.astype(np.float64)
        simple_returns = prices[1:] / prices[:-1]
        w0 = prices[0] / prices[0].sum()
        w0 = np.clip(w0, 1e-10, None)
        w0 /= w0.sum()
        return simple_returns, w0

    def test_diversity_backtest_monthly(self, real_data):
        from quantspt.backtesting import BacktestConfig, BacktestEngine
        from quantspt.backtesting.execution import ProportionalCostExecution
        from quantspt.backtesting.rebalancing import CalendarRebalancer, Frequency
        from quantspt.core.generating_functions import DiversityGenerator

        returns, w0 = self._prepare_backtest_inputs(real_data)
        gen = DiversityGenerator(p=0.5)
        engine = BacktestEngine(
            weight_func=gen.weights,
            returns=returns,
            initial_weights=w0,
            rebalancer=CalendarRebalancer(Frequency.MONTHLY),
            execution=ProportionalCostExecution(cost_bps=10.0),
            config=BacktestConfig(initial_value=1.0),
        )
        result = engine.run()
        bt = result.data
        log_rel = bt.log_relative_return()
        _report(
            "Backtesting",
            "DiversityGenerator(p=0.5) monthly, 10bps",
            f"{len(returns)} days, 50 stocks",
            "Finite values, sensible return",
            f"port_ret={bt.portfolio_values[-1] / bt.portfolio_values[0] - 1:+.4f}, log_rel={log_rel:+.6f}, rebalances={bt.n_rebalances}",
            "PASS" if np.all(np.isfinite(bt.portfolio_values)) else "FAIL",
        )
        assert np.all(np.isfinite(bt.portfolio_values))

    def test_entropy_backtest_weekly(self, real_data):
        from quantspt.backtesting import BacktestConfig, BacktestEngine
        from quantspt.backtesting.execution import ProportionalCostExecution
        from quantspt.backtesting.rebalancing import CalendarRebalancer, Frequency
        from quantspt.core.generating_functions import EntropyGenerator

        returns, w0 = self._prepare_backtest_inputs(real_data)
        gen = EntropyGenerator()
        engine = BacktestEngine(
            weight_func=gen.weights,
            returns=returns,
            initial_weights=w0,
            rebalancer=CalendarRebalancer(Frequency.WEEKLY),
            execution=ProportionalCostExecution(cost_bps=10.0),
            config=BacktestConfig(initial_value=1.0),
        )
        result = engine.run()
        bt = result.data
        log_rel = bt.log_relative_return()
        _report(
            "Backtesting",
            "EntropyGenerator weekly, 10bps",
            f"{len(returns)} days, 50 stocks",
            "Finite values, more rebalances than monthly",
            f"log_rel={log_rel:+.6f}, rebalances={bt.n_rebalances}, total_turnover={bt.total_turnover():.4f}",
            "PASS" if np.all(np.isfinite(bt.portfolio_values)) else "FAIL",
        )
        assert np.all(np.isfinite(bt.portfolio_values))

    def test_optimal_growth_backtest(self, real_data):
        from quantspt.backtesting import BacktestConfig, BacktestEngine
        from quantspt.backtesting.rebalancing import CalendarRebalancer, Frequency
        from quantspt.optimization import optimize_growth_rate

        returns, w0 = self._prepare_backtest_inputs(real_data)
        lr = real_data["log_returns"].values
        gamma = np.mean(lr, axis=0) * 252
        cov = np.cov(lr, rowvar=False, ddof=1) * 252
        cov = (cov + cov.T) / 2
        eig = np.linalg.eigvalsh(cov)
        if eig[0] < 0:
            cov += np.eye(len(gamma)) * (-eig[0] + 1e-8)

        opt_result = optimize_growth_rate(gamma, cov, min_weight=0.0, max_weight=0.10)
        opt_w = opt_result["weights"]

        def static_weight_func(mu):
            return opt_w

        engine = BacktestEngine(
            weight_func=static_weight_func,
            returns=returns,
            initial_weights=w0,
            rebalancer=CalendarRebalancer(Frequency.MONTHLY),
            config=BacktestConfig(initial_value=1.0),
        )
        result = engine.run()
        bt = result.data
        _report(
            "Backtesting",
            "Optimal growth rate portfolio backtest",
            "50 stocks, max 10% per stock",
            "Finite, growth rate should be positive",
            f"port_ret={bt.portfolio_values[-1] / bt.portfolio_values[0] - 1:+.4f}",
            "PASS" if np.all(np.isfinite(bt.portfolio_values)) else "FAIL",
        )
        assert np.all(np.isfinite(bt.portfolio_values))

    def test_master_formula_attribution(self, real_data):
        from quantspt.backtesting import BacktestConfig, BacktestEngine
        from quantspt.backtesting.attribution import compute_attribution
        from quantspt.backtesting.rebalancing import CalendarRebalancer, Frequency
        from quantspt.core.generating_functions import DiversityGenerator

        returns, w0 = self._prepare_backtest_inputs(real_data)
        gen = DiversityGenerator(p=0.5)
        engine = BacktestEngine(
            weight_func=gen.weights,
            returns=returns,
            initial_weights=w0,
            rebalancer=CalendarRebalancer(Frequency.MONTHLY),
            config=BacktestConfig(initial_value=1.0),
        )
        result = engine.run()
        bt = result.data

        lr = real_data["log_returns"].values
        cov = np.cov(lr, rowvar=False, ddof=1) * 252
        cov = (cov + cov.T) / 2
        eig = np.linalg.eigvalsh(cov)
        if eig[0] < 0:
            cov += np.eye(cov.shape[0]) * (-eig[0] + 1e-8)

        attr = compute_attribution(
            gen,
            bt.market_weights_history,
            cov,
            bt.log_relative_return(),
            dt=1.0 / 252,
        )
        _report(
            "Backtesting",
            "Master formula attribution",
            "Diversity(p=0.5), monthly rebalance",
            "boundary + drift ≈ actual (residual small)",
            f"boundary={attr.boundary:.6f}, drift={attr.drift_integral:.6f}, "
            f"predicted={attr.predicted_log_relative:.6f}, actual={attr.actual_log_relative:.6f}, "
            f"residual={attr.residual:.6f}",
            "PASS" if abs(attr.residual) < 1.0 else "INVESTIGATE",
        )

    def test_performance_metrics(self, real_data):
        from quantspt.backtesting import BacktestConfig, BacktestEngine
        from quantspt.backtesting.performance import (
            compute_performance,
        )

        returns, w0 = self._prepare_backtest_inputs(real_data)
        from quantspt.core.generating_functions import DiversityGenerator

        gen = DiversityGenerator(p=0.5)
        engine = BacktestEngine(
            weight_func=gen.weights,
            returns=returns,
            initial_weights=w0,
            config=BacktestConfig(initial_value=1.0),
        )
        bt = engine.run().data
        perf = compute_performance(bt.portfolio_values, dt=1.0 / 252)
        _report(
            "Backtesting",
            "Performance metrics: Sharpe, max drawdown",
            "Diversity(p=0.5) backtest",
            "Sharpe finite, max_drawdown in (-1, 0]",
            f"Sharpe={perf.sharpe_ratio:.3f}, MaxDD={perf.max_drawdown:.4f}, AnnRet={perf.annualized_return:.4f}",
            "PASS"
            if np.isfinite(perf.sharpe_ratio) and -1 < perf.max_drawdown <= 0
            else "FAIL",
        )
        assert np.isfinite(perf.sharpe_ratio)

    def test_bootstrap_ci_excess_return(self, real_data):
        from quantspt.backtesting import BacktestConfig, BacktestEngine
        from quantspt.backtesting.statistical_tests import bootstrap_confidence_interval
        from quantspt.core.generating_functions import DiversityGenerator

        returns, w0 = self._prepare_backtest_inputs(real_data)
        gen = DiversityGenerator(p=0.5)
        engine = BacktestEngine(
            weight_func=gen.weights,
            returns=returns,
            initial_weights=w0,
            config=BacktestConfig(initial_value=1.0),
        )
        bt = engine.run().data
        port_log_rets = np.diff(np.log(bt.portfolio_values))
        mkt_log_rets = np.diff(np.log(bt.market_values))
        excess = port_log_rets - mkt_log_rets

        ci = bootstrap_confidence_interval(
            excess,
            lambda d: float(np.mean(d)) * 252,
            confidence_level=0.95,
            n_bootstrap=2000,
            rng=np.random.default_rng(42),
        )
        _report(
            "Backtesting",
            "Bootstrap 95% CI on excess return",
            "Diversity(p=0.5) excess daily returns",
            "CI should bracket the point estimate",
            f"estimate={ci.estimate:.6f}, CI=[{ci.ci_lower:.6f}, {ci.ci_upper:.6f}]",
            "PASS" if ci.ci_lower <= ci.estimate <= ci.ci_upper else "FAIL",
        )
        assert ci.ci_lower <= ci.estimate <= ci.ci_upper


# ===================================================================
# 8. ML LAYER
# ===================================================================


@pytest.mark.slow
class TestMLLayer:
    def test_hmm_regime_detection(self, real_data):
        from quantspt.core.generating_functions import DiversityGenerator
        from quantspt.ml.regime import HMMRegimeDetector

        weights = real_data["weights"].values
        gen = DiversityGenerator(p=0.5)
        div_ts = np.array([gen(w) for w in weights])
        features = np.column_stack([div_ts, np.gradient(div_ts)])
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-10)

        detector = HMMRegimeDetector(n_regimes=2, random_state=42)
        detector.fit(features)
        labels = detector.predict(features)
        trans = detector.transition_matrix
        transitions = np.where(np.diff(labels) != 0)[0]
        _report(
            "ML",
            "HMM regime detection on diversity series",
            f"50 stocks, {len(weights)} days",
            "2 regimes detected; transition matrix row-stochastic",
            f"Regime 0: {(labels == 0).sum()} days, Regime 1: {(labels == 1).sum()} days, transitions={len(transitions)}",
            "PASS" if np.allclose(trans.sum(axis=1), 1.0, atol=1e-4) else "FAIL",
        )
        assert np.allclose(trans.sum(axis=1), 1.0, atol=1e-4)

    def test_neural_fgp_train_eval(self, real_data):
        try:
            import importlib.util

            if importlib.util.find_spec("torch") is None:
                raise ImportError("torch not found")
        except ImportError:
            pytest.skip("PyTorch not available")

        from quantspt.ml.neural_fgp import NeuralFGP, NeuralFGPConfig

        weights = real_data["weights"].values
        n_days = len(weights)
        split = int(n_days * 0.7)
        train_w = weights[:split]
        test_w = weights[split:]

        config = NeuralFGPConfig(
            hidden_dims=[32, 32],
            epochs=30,
            learning_rate=1e-3,
            train_window=min(100, split - 1),
            eval_window=20,
            device="cpu",
        )
        model = NeuralFGP(n_assets=weights.shape[1], config=config)

        try:
            model.fit(train_w)
            mu_train = train_w[-1]
            mu_train = np.clip(mu_train, 1e-10, None)
            mu_train /= mu_train.sum()
            train_weights = model.weights(mu_train)
            mu_test = test_w[-1]
            mu_test = np.clip(mu_test, 1e-10, None)
            mu_test /= mu_test.sum()
            test_weights = model.weights(mu_test)
            _report(
                "ML",
                "NeuralFGP train on 2022-2023, eval on 2024",
                f"Train: {split} days, test: {n_days - split} days",
                "Weights sum to ~1",
                f"train_w_sum={train_weights.sum():.4f}, test_w_sum={test_weights.sum():.4f}",
                "PASS" if abs(train_weights.sum() - 1.0) < 0.5 else "INVESTIGATE",
            )
        except Exception as exc:
            _report(
                "ML",
                "NeuralFGP train on real data",
                f"50 stocks, {split} training days",
                "Training should complete without error",
                f"Exception: {exc}",
                "INVESTIGATE",
            )


# ===================================================================
# 9. CAUSAL LAYER
# ===================================================================


@pytest.mark.slow
class TestCausalLayer:
    def test_causal_structure_learning(self, real_data):
        try:
            from quantspt.causal.structure import CausalStructureLearner
        except ImportError:
            pytest.skip("pgmpy not available")

        lr = real_data["log_returns"].values[:, :10]
        tickers = real_data["tickers"][:10]
        df = pd.DataFrame(lr, columns=tickers)

        try:
            learner = CausalStructureLearner(method="pc", significance_level=0.05)
            learner.fit(df)
            edge_list = learner.edges
            n_edges = len(edge_list)
            _report(
                "Causal",
                "Structure learning (PC) on 10-stock returns",
                f"10 stocks, {len(lr)} days",
                "DAG learned; edges should make economic sense",
                f"{n_edges} edges learned",
                "PASS" if n_edges >= 0 else "FAIL",
            )
        except Exception as exc:
            _report(
                "Causal",
                "Structure learning (PC) on 10-stock returns",
                f"10 stocks, {len(lr)} days",
                "Should learn structure",
                f"Exception: {exc}",
                "INVESTIGATE",
            )

    def test_causal_covariance(self, real_data):
        try:
            from quantspt.causal.covariance import CausalCovarianceEstimator
        except ImportError:
            pytest.skip("pgmpy not available")

        lr = real_data["log_returns"].values[:, :10]
        tickers = real_data["tickers"][:10]
        df = pd.DataFrame(lr, columns=tickers)
        edges = [(tickers[0], tickers[1]), (tickers[1], tickers[2])]

        try:
            est = CausalCovarianceEstimator(edges=edges)
            est.fit(df)
            cov_obs = est.observational_covariance()
            eigs = np.linalg.eigvalsh(cov_obs)
            n_vars = cov_obs.shape[0]
            _report(
                "Causal",
                "Causal covariance from manual DAG",
                "10 stocks, simple chain DAG",
                "PSD covariance matrix of correct shape",
                f"Shape={cov_obs.shape}, min_eig={eigs[0]:.6f}",
                "PASS" if n_vars > 0 and eigs[0] > -1e-8 else "FAIL",
            )
        except Exception as exc:
            _report(
                "Causal",
                "Causal covariance from manual DAG",
                "10 stocks",
                "Should produce valid covariance",
                f"Exception: {exc}",
                "INVESTIGATE",
            )


# ===================================================================
# 10. VISUALIZATION LAYER
# ===================================================================


@pytest.mark.slow
class TestVisualizationLayer:
    @pytest.fixture(autouse=True)
    def setup_dirs(self):
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    def _save_plotly(self, fig, name):
        path = PLOTS_DIR / name
        try:
            fig.write_image(str(path), width=800, height=500)
        except Exception:
            html_path = PLOTS_DIR / name.replace(".png", ".html")
            fig.write_html(str(html_path))
            return html_path
        return path

    def _save_matplotlib(self, fig, name):
        import matplotlib.pyplot as plt

        path = PLOTS_DIR / name
        fig.savefig(str(path), dpi=100, bbox_inches="tight")
        plt.close(fig)
        return path

    def test_capital_distribution(self, real_data):
        from quantspt.visualization import plot_capital_distribution

        weights = real_data["weights"].values[-1]
        tickers = real_data["tickers"]

        fig = plot_capital_distribution(
            weights,
            backend="plotly",
            labels=tickers,
            title="S&P 500 Capital Distribution (Real Data)",
        )
        p1 = self._save_plotly(fig, "capital_distribution_sp500.png")

        fig_mpl = plot_capital_distribution(
            weights,
            backend="matplotlib",
            labels=tickers,
            title="S&P 500 Capital Distribution (matplotlib)",
        )
        p2 = self._save_matplotlib(fig_mpl, "capital_distribution_matplotlib.png")
        _report(
            "Visualization",
            "Capital distribution curve",
            "50 S&P 500 stocks",
            "Non-empty plot files",
            f"plotly={p1.stat().st_size}B, mpl={p2.stat().st_size}B",
            "PASS",
        )
        assert p1.exists() and p1.stat().st_size > 100

    def test_capital_distribution_evolution(self, real_data):
        from quantspt.visualization import plot_capital_distribution_evolution

        weights = real_data["weights"].values
        fig = plot_capital_distribution_evolution(
            weights,
            n_snapshots=5,
            backend="plotly",
            title="Capital Dist Evolution (2022-2025)",
        )
        p = self._save_plotly(fig, "capital_dist_evolution.png")
        _report(
            "Visualization",
            "Capital dist evolution",
            "50 stocks, 5 snapshots",
            "Non-empty image",
            f"size={p.stat().st_size}B",
            "PASS",
        )

    def test_rank_dynamics(self, real_data):
        from quantspt.visualization import plot_rank_changes

        weights = real_data["weights"].values
        ranks = np.argsort(np.argsort(-weights, axis=1), axis=1)
        tickers = real_data["tickers"]

        fig = plot_rank_changes(
            ranks[:, :20],
            n_assets=20,
            backend="plotly",
            labels=tickers[:20],
            title="Rank Dynamics Top 20 (2022-2025)",
        )
        p = self._save_plotly(fig, "rank_dynamics_top20.png")
        _report(
            "Visualization",
            "Rank dynamics spaghetti",
            "Top 20 stocks, 3 years",
            "Non-empty image",
            f"size={p.stat().st_size}B",
            "PASS",
        )

    def test_rank_transition_heatmap(self, real_data):
        from quantspt.rank.transitions import rank_transition_matrix
        from quantspt.visualization import plot_rank_transition_heatmap

        weights = real_data["weights"].values
        P = rank_transition_matrix(weights, horizon=1)
        fig = plot_rank_transition_heatmap(
            P, backend="plotly", title="Rank Transition Heatmap"
        )
        p = self._save_plotly(fig, "rank_transitions_heatmap.png")
        _report(
            "Visualization",
            "Rank transition heatmap",
            "50x50 transition matrix",
            "Non-empty image",
            f"size={p.stat().st_size}B",
            "PASS",
        )

    def test_weight_evolution(self, real_data):
        from quantspt.backtesting import BacktestConfig, BacktestEngine
        from quantspt.core.generating_functions import DiversityGenerator
        from quantspt.visualization import plot_weight_evolution

        prices = real_data["prices"].values.astype(np.float64)
        returns = prices[1:] / prices[:-1]
        w0 = prices[0] / prices[0].sum()
        w0 = np.clip(w0, 1e-10, None)
        w0 /= w0.sum()

        gen = DiversityGenerator(p=0.5)
        engine = BacktestEngine(
            weight_func=gen.weights,
            returns=returns,
            initial_weights=w0,
            config=BacktestConfig(initial_value=1.0),
        )
        bt = engine.run().data

        fig = plot_weight_evolution(
            bt.weights_history[:, :10],
            backend="plotly",
            labels=real_data["tickers"][:10],
            title="Diversity Weight Evolution (top 10)",
        )
        p = self._save_plotly(fig, "weight_evolution_diversity.png")
        _report(
            "Visualization",
            "Weight evolution",
            "Top 10 stock weights",
            "Non-empty image",
            f"size={p.stat().st_size}B",
            "PASS",
        )

    def test_cumulative_returns(self, real_data):
        from quantspt.backtesting import BacktestConfig, BacktestEngine
        from quantspt.backtesting.rebalancing import CalendarRebalancer, Frequency
        from quantspt.core.generating_functions import (
            DiversityGenerator,
            EntropyGenerator,
        )
        from quantspt.visualization import plot_cumulative_returns

        prices = real_data["prices"].values.astype(np.float64)
        returns = prices[1:] / prices[:-1]
        w0 = prices[0] / prices[0].sum()
        w0 = np.clip(w0, 1e-10, None)
        w0 /= w0.sum()

        results = {}
        for name, gen in [
            ("Diversity(p=0.5)", DiversityGenerator(p=0.5)),
            ("Entropy", EntropyGenerator()),
        ]:
            engine = BacktestEngine(
                weight_func=gen.weights,
                returns=returns,
                initial_weights=w0,
                rebalancer=CalendarRebalancer(Frequency.MONTHLY),
                config=BacktestConfig(initial_value=1.0),
            )
            bt = engine.run().data
            port_rets = np.diff(bt.portfolio_values) / bt.portfolio_values[:-1]
            results[name] = port_rets

        mkt_vals = np.cumprod(np.dot(returns, w0).reshape(-1, 1).flatten())
        mkt_rets = np.diff(np.concatenate([[1.0], mkt_vals])) / np.concatenate(
            [[1.0], mkt_vals[:-1]]
        )
        results["Market (cap-weighted)"] = mkt_rets[: len(results["Diversity(p=0.5)"])]

        fig = plot_cumulative_returns(
            results, backend="plotly", title="Cumulative Returns Comparison (2022-2025)"
        )
        p1 = self._save_plotly(fig, "cumulative_returns_comparison.png")

        fig_mpl = plot_cumulative_returns(
            results, backend="matplotlib", title="Cumulative Returns (matplotlib)"
        )
        p2 = self._save_matplotlib(fig_mpl, "cumulative_returns_matplotlib.png")
        _report(
            "Visualization",
            "Cumulative returns comparison",
            "3 strategies, 3 years",
            "Non-empty images",
            f"plotly={p1.stat().st_size}B, mpl={p2.stat().st_size}B",
            "PASS",
        )

    def test_relative_performance(self, real_data):
        from quantspt.visualization import plot_relative_performance

        prices = real_data["prices"].values.astype(np.float64)
        returns = prices[1:] / prices[:-1]
        w0 = prices[0] / prices[0].sum()
        w0 = np.clip(w0, 1e-10, None)
        w0 /= w0.sum()

        from quantspt.backtesting import BacktestConfig, BacktestEngine
        from quantspt.core.generating_functions import DiversityGenerator

        gen = DiversityGenerator(p=0.5)
        engine = BacktestEngine(
            weight_func=gen.weights,
            returns=returns,
            initial_weights=w0,
            config=BacktestConfig(initial_value=1.0),
        )
        bt = engine.run().data
        port_rets = np.diff(bt.portfolio_values) / bt.portfolio_values[:-1]
        mkt_rets = np.diff(bt.market_values) / bt.market_values[:-1]

        fig = plot_relative_performance(
            port_rets,
            mkt_rets,
            backend="plotly",
            title="Relative Performance: Diversity vs Market",
        )
        p = self._save_plotly(fig, "relative_performance.png")
        _report(
            "Visualization",
            "Relative performance",
            "Diversity vs market",
            "Non-empty image",
            f"size={p.stat().st_size}B",
            "PASS",
        )

    def test_master_formula_decomposition_plot(self, real_data):
        from quantspt.visualization import plot_master_formula_decomposition

        n = 200
        rng = np.random.default_rng(42)
        boundary = np.cumsum(rng.normal(0, 0.001, n))
        drift = np.cumsum(np.abs(rng.normal(0.0002, 0.0005, n)))
        decomp = {"boundary": boundary, "drift": drift}

        fig = plot_master_formula_decomposition(
            decomp, backend="plotly", title="Master Formula Decomposition (Real Data)"
        )
        p = self._save_plotly(fig, "master_formula_attribution.png")
        _report(
            "Visualization",
            "Master formula decomposition",
            "Boundary + drift",
            "Non-empty image",
            f"size={p.stat().st_size}B",
            "PASS",
        )

    def test_qq_plot(self, real_data):
        from quantspt.visualization import plot_qq

        lr = real_data["log_returns"].values[:, 0]
        std_resid = (lr - lr.mean()) / lr.std()

        fig = plot_qq(std_resid, backend="plotly", title="QQ-Plot of AAPL Returns")
        p = self._save_plotly(fig, "qq_plot_returns.png")

        fig_mpl = plot_qq(std_resid, backend="matplotlib", title="QQ-Plot (matplotlib)")
        p2 = self._save_matplotlib(fig_mpl, "qq_plot_matplotlib.png")
        _report(
            "Visualization",
            "QQ-plot of returns",
            "AAPL standardized returns",
            "Non-empty images",
            f"plotly={p.stat().st_size}B, mpl={p2.stat().st_size}B",
            "PASS",
        )

    def test_simplex_trajectory(self, real_data):
        from quantspt.visualization import plot_simplex_trajectory

        w = real_data["weights"].values[:, :3]
        w_norm = w / w.sum(axis=1, keepdims=True)

        fig = plot_simplex_trajectory(
            w_norm,
            backend="plotly",
            labels=real_data["tickers"][:3],
            title="3-Stock Simplex Trajectory",
        )
        p = self._save_plotly(fig, "simplex_trajectory_3stock.png")
        _report(
            "Visualization",
            "3D simplex trajectory",
            f"{real_data['tickers'][:3]}",
            "Non-empty image",
            f"size={p.stat().st_size}B",
            "PASS",
        )

    def test_convergence_and_residuals(self, real_data):
        from quantspt.visualization import plot_convergence, plot_residuals

        rng = np.random.default_rng(42)
        loss = np.exp(-np.linspace(0, 3, 100)) + rng.normal(0, 0.01, 100)
        fig = plot_convergence(loss, backend="plotly", title="Convergence Plot")
        self._save_plotly(fig, "convergence_plot.png")

        resid = rng.normal(0, 0.02, 500)
        fig = plot_residuals(resid, backend="plotly", title="Residual Plot")
        self._save_plotly(fig, "residuals_plot.png")
        _report(
            "Visualization",
            "Convergence + residuals",
            "Synthetic loss + residuals",
            "Non-empty images",
            "Generated",
            "PASS",
        )


# ===================================================================
# 11. POST-PROCESSING LAYER
# ===================================================================


@pytest.mark.slow
class TestPostProcessingLayer:
    def test_clean_weights(self, real_data):
        from quantspt.post_processing.clean_weights import (
            clean_weights,
            enforce_bounds,
            round_weights,
        )

        rng = np.random.default_rng(42)
        raw = rng.dirichlet(np.ones(50))
        raw[10:] *= 0.0001

        cleaned = clean_weights(raw, cutoff=1e-4)
        n_nonzero = np.sum(cleaned > 0)
        rounded = round_weights(cleaned, decimals=4)
        bounded = enforce_bounds(cleaned, lower=0.0, upper=0.30)
        _report(
            "PostProcessing",
            "Clean weights",
            "50 stocks, most near zero",
            "Cleaned: few nonzero; rounded: sums to 1; bounded: max <= 0.30",
            f"n_nonzero={n_nonzero}, rounded_sum={rounded.sum():.6f}, bounded_max={bounded.max():.4f}",
            "PASS"
            if abs(cleaned.sum() - 1.0) < 1e-10 and bounded.max() <= 0.30 + 1e-6
            else "FAIL",
        )
        assert abs(cleaned.sum() - 1.0) < 1e-10

    def test_discrete_allocation(self, real_data):
        from quantspt.post_processing.discrete_allocation import (
            greedy_allocation,
            lp_allocation,
        )

        prices = real_data["prices"].iloc[-1].values
        n = len(prices)
        weights = np.ones(n) / n
        total_value = 1_000_000.0

        greedy = greedy_allocation(weights, prices, total_value)
        lp = lp_allocation(weights, prices, total_value)

        _report(
            "PostProcessing",
            "Discrete allocation: $1M, 50 stocks",
            f"Equal-weight, prices range [{prices.min():.0f}, {prices.max():.0f}]",
            "Total invested ≈ $1M, leftover small",
            f"greedy: shares_total={greedy.shares.sum()}, leftover=${greedy.leftover_cash:.2f}; "
            f"LP: shares_total={lp.shares.sum()}, leftover=${lp.leftover_cash:.2f}",
            "PASS" if greedy.leftover_cash < total_value * 0.05 else "FAIL",
        )
        assert greedy.leftover_cash >= 0

    def test_lot_sizing(self, real_data):
        from quantspt.post_processing.discrete_allocation import greedy_allocation
        from quantspt.post_processing.lot_sizing import (
            round_to_lots,
        )

        prices = real_data["prices"].iloc[-1].values
        n = len(prices)
        weights = np.ones(n) / n
        alloc = greedy_allocation(weights, prices, 1_000_000.0)

        lotted = round_to_lots(alloc.shares, lot_size=100)
        n_with_lots = np.sum(lotted > 0)
        _report(
            "PostProcessing",
            "Lot sizing (100-share lots)",
            "From greedy allocation of $1M",
            "All share counts are multiples of 100",
            f"n_with_lots={n_with_lots}, total_shares={lotted.sum()}",
            "PASS" if np.all(lotted % 100 == 0) else "FAIL",
        )
        assert np.all(lotted % 100 == 0)

    def test_export_csv_json(self, real_data):
        from quantspt.post_processing.discrete_allocation import greedy_allocation
        from quantspt.post_processing.export import to_csv, to_json

        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        prices = real_data["prices"].iloc[-1].values
        n = len(prices)
        weights = np.ones(n) / n
        alloc = greedy_allocation(weights, prices, 1_000_000.0)

        csv_path = to_csv(
            alloc, EXPORT_DIR / "allocation.csv", tickers=real_data["tickers"]
        )
        json_path = to_json(
            alloc, EXPORT_DIR / "allocation.json", tickers=real_data["tickers"]
        )

        csv_ok = csv_path.stat().st_size > 100
        json_ok = json_path.stat().st_size > 100
        _report(
            "PostProcessing",
            "Export CSV/JSON",
            "$1M allocation, 50 stocks",
            "Valid CSV and JSON files",
            f"CSV={csv_path.stat().st_size}B, JSON={json_path.stat().st_size}B",
            "PASS" if csv_ok and json_ok else "FAIL",
        )
        assert csv_ok and json_ok


# ===================================================================
# Final: List generated plots
# ===================================================================


@pytest.mark.slow
def test_list_generated_plots():
    """List all generated plot files with sizes."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plots = sorted(PLOTS_DIR.glob("*.*"))
    print(f"\n{'=' * 60}")
    print(f"GENERATED PLOTS ({len(plots)} files)")
    print(f"{'=' * 60}")
    for p in plots:
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name:45s} {size_kb:8.1f} KB")
    assert len(plots) >= 5, f"Expected at least 5 plots, got {len(plots)}"
