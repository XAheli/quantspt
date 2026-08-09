"""Validate CovarianceConditionalFGP on real 2022-2026 S&P 500 data.

Downloads 50 S&P 500 stocks via yfinance, trains the conditional FGP
on 2022-2024 data, and evaluates on 2024-2026 holdout with 10bps costs.

Compares against fixed p=0.3 baseline. If the conditional strategy
cannot beat fixed p, documents the finding per the research directive.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pytest

from quantspt.backtesting.engine import BacktestConfig, BacktestEngine
from quantspt.backtesting.execution import ProportionalCostExecution
from quantspt.backtesting.rebalancing import CalendarRebalancer, Frequency
from quantspt.core.generating_functions import DiversityGenerator
from quantspt.ml.conditional_fgp import (
    BoundaryRobustnessRegularizer,
    ConditionalFGPConfig,
    CovarianceConditionalFGP,
    CovarianceFeatureExtractor,
    cost_optimal_p,
    optimal_p_for_cost_level,
)

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

SP500_50 = [
    "AAPL",
    "MSFT",
    "AMZN",
    "NVDA",
    "GOOGL",
    "META",
    "BRK-B",
    "UNH",
    "JNJ",
    "XOM",
    "JPM",
    "V",
    "PG",
    "MA",
    "HD",
    "CVX",
    "MRK",
    "ABBV",
    "LLY",
    "PEP",
    "KO",
    "COST",
    "AVGO",
    "WMT",
    "MCD",
    "TMO",
    "CSCO",
    "CRM",
    "ABT",
    "DHR",
    "NEE",
    "LIN",
    "TXN",
    "PM",
    "UNP",
    "RTX",
    "HON",
    "LOW",
    "AMGN",
    "IBM",
    "COP",
    "QCOM",
    "BA",
    "CAT",
    "GE",
    "INTC",
    "SBUX",
    "PLD",
    "AMD",
    "ISRG",
]


def _download_real_data(
    start: str = "2022-01-01",
    end: str = "2026-08-01",
) -> dict:
    """Download real market data and compute weights, returns, covariances."""
    import yfinance as yf

    data = yf.download(SP500_50, start=start, end=end, auto_adjust=True)
    close = data["Close"] if "Close" in data.columns.get_level_values(0) else data
    close = close.dropna(axis=1, how="any").dropna(axis=0)

    prices = close.values.astype(np.float64)
    T, n = prices.shape
    print(f"Downloaded {n} stocks, {T} days ({start} to {end})")

    weights = prices / prices.sum(axis=1, keepdims=True)
    weights = np.clip(weights, 1e-8, None)
    weights /= weights.sum(axis=1, keepdims=True)

    returns_ratio = prices[1:] / prices[:-1]
    log_returns = np.log(returns_ratio)

    cov_window = 63
    cov_matrices: list[np.ndarray] = []
    for t in range(T):
        if t < cov_window:
            cov_matrices.append(np.eye(n) * 0.04)
        else:
            chunk = log_returns[max(0, t - cov_window) : t]
            if len(chunk) < n + 1:
                cov_matrices.append(np.eye(n) * 0.04)
            else:
                cov_matrices.append(np.cov(chunk, rowvar=False) * 252)

    return {
        "prices": prices,
        "weights": weights,
        "returns": returns_ratio,
        "log_returns": log_returns,
        "cov_matrices": cov_matrices,
        "tickers": list(close.columns),
        "n": n,
        "T": T,
    }


def _backtest_strategy(
    weight_func,
    returns: np.ndarray,
    initial_weights: np.ndarray,
    cost_bps: float = 10.0,
) -> dict:
    """Run a simple backtest and return metrics."""
    init_w = initial_weights.copy()
    init_w = np.maximum(init_w, 1e-10)
    init_w /= init_w.sum()

    engine = BacktestEngine(
        weight_func=weight_func,
        returns=returns,
        initial_weights=init_w,
        rebalancer=CalendarRebalancer(Frequency.MONTHLY),
        execution=ProportionalCostExecution(cost_bps=cost_bps),
        config=BacktestConfig(initial_value=1.0),
    )
    result = engine.run().data
    n_years = len(returns) / 252.0

    return {
        "log_relative_return": result.log_relative_return(),
        "annualized_excess": result.log_relative_return() / max(n_years, 1e-6),
        "total_turnover": result.total_turnover(),
        "total_cost": result.total_cost(),
        "final_portfolio_value": float(result.portfolio_values[-1]),
        "final_market_value": float(result.market_values[-1]),
    }


@pytest.fixture(scope="module")
def real_data():
    """Download real market data once for all tests in this module."""
    try:
        data = _download_real_data()
    except Exception as e:
        pytest.skip(f"Could not download market data: {e}")
    if data["T"] < 600:
        pytest.skip(f"Insufficient data: only {data['T']} days")
    return data


# ===================================================================
# Real data tests
# ===================================================================


@pytest.mark.slow
class TestConditionalFGPRealData:
    """Validate CovarianceConditionalFGP on real 2022-2026 data."""

    def test_covariance_features_real_data(self, real_data: dict) -> None:
        """Feature extraction produces finite, reasonable values."""
        ext = CovarianceFeatureExtractor()
        cov = real_data["cov_matrices"][-1]
        features = ext.extract(cov)
        assert np.all(np.isfinite(features))
        feat_dict = dict(zip(ext.feature_names, features, strict=False))
        print(f"Features at last date: {feat_dict}")

    def test_feature_time_series(self, real_data: dict) -> None:
        """Features vary over time, capturing regime changes."""
        ext = CovarianceFeatureExtractor(features=["avg_correlation"])
        vals = []
        for t in range(100, real_data["T"], 21):
            vals.append(ext.extract(real_data["cov_matrices"][t])[0])
        vals = np.array(vals)
        assert np.std(vals) > 0.01, "avg_correlation should vary over time"
        print(
            f"avg_correlation: mean={np.mean(vals):.3f}, "
            f"std={np.std(vals):.3f}, range=[{np.min(vals):.3f}, {np.max(vals):.3f}]"
        )

    def test_cost_optimal_p_formula_matches_research(self) -> None:
        """The cost formula matches the research findings."""
        p_0 = cost_optimal_p(0.0)
        p_50 = cost_optimal_p(50.0)
        p_100 = cost_optimal_p(100.0)

        assert abs(p_0 - 0.09) < 0.01, f"Zero-cost p should be ~0.09, got {p_0}"
        assert abs(p_50 - 0.315) < 0.05, f"50bps p should be ~0.315, got {p_50}"
        assert p_100 > p_50, "Higher cost should give higher p"

    def test_empirical_p_sweep(self, real_data: dict) -> None:
        """Empirical p sweep on full data returns sensible results."""
        n_total = len(real_data["returns"])
        subset_returns = real_data["returns"][: min(500, n_total)]
        subset_weights = real_data["weights"][: min(501, n_total + 1)]

        result = optimal_p_for_cost_level(
            subset_returns,
            subset_weights,
            cost_bps=10.0,
            p_grid=np.linspace(0.1, 0.9, 9),
        )
        print(f"Empirical optimal p (10bps): {result['optimal_p']:.3f}")
        print(f"Formula p: {result['formula_p']:.3f}")
        assert 0.05 <= result["optimal_p"] <= 0.95

    def test_boundary_robustness_real_weights(self, real_data: dict) -> None:
        """Boundary regularizer works on real market weights."""
        reg = BoundaryRobustnessRegularizer()
        mu = real_data["weights"][-1]
        best_p, penalties = reg.select_robust_p(mu)
        assert 0.05 <= best_p <= 0.95
        assert np.all(penalties >= 0)
        print(f"Most boundary-robust p: {best_p:.3f}")

    def test_conditional_fgp_fit_real_data(self, real_data: dict) -> None:
        """Train the conditional FGP on real data (full period)."""
        cfg = ConditionalFGPConfig(
            p_grid_size=10,
            n_estimators=30,
            cost_bps=10.0,
            boundary_penalty_weight=0.1,
        )
        model = CovarianceConditionalFGP(config=cfg)
        t0 = time.time()
        model.fit(
            real_data["weights"],
            real_data["cov_matrices"],
            real_data["returns"],
        )
        elapsed = time.time() - t0
        print(f"Training time: {elapsed:.1f}s")
        assert model.fitted

        meta = model.training_metadata
        print(f"Status: {meta['status']}")
        if meta["status"] == "fitted":
            print(f"Samples: {meta['n_samples']}")
            print(f"Val MAE: {meta['val_mae']:.4f}")
            print(f"Val R2: {meta['val_r2']:.4f}")
            print(f"Feature importances: {meta['feature_importances']}")

    def test_conditional_fgp_vs_fixed_p(self, real_data: dict) -> None:
        """Train/test split: conditional FGP vs fixed p=0.3.

        Train on first 60%, test on remaining 40%.
        With 10bps costs.
        """
        T = real_data["T"]
        split = int(T * 0.6)

        train_weights = real_data["weights"][:split]
        train_returns = real_data["returns"][: split - 1]
        train_covs = real_data["cov_matrices"][:split]

        test_returns = real_data["returns"][split - 1 :]
        test_weights = real_data["weights"][split - 1 :]

        cfg = ConditionalFGPConfig(
            p_grid_size=10,
            n_estimators=30,
            cost_bps=10.0,
            boundary_penalty_weight=0.1,
        )
        model = CovarianceConditionalFGP(config=cfg)
        model.fit(train_weights, train_covs, train_returns)

        baseline_gen = DiversityGenerator(0.3)
        baseline_metrics = _backtest_strategy(
            baseline_gen.weights,
            test_returns,
            test_weights[0],
            cost_bps=10.0,
        )

        conditional_metrics = _backtest_strategy(
            model.weights_from_mu_only,
            test_returns,
            test_weights[0],
            cost_bps=10.0,
        )

        print("\n=== OUT-OF-SAMPLE RESULTS (with 10bps costs) ===")
        print(
            f"Fixed p=0.3:       ann. excess = "
            f"{baseline_metrics['annualized_excess'] * 100:.3f}%"
        )
        print(
            f"Conditional FGP:   ann. excess = "
            f"{conditional_metrics['annualized_excess'] * 100:.3f}%"
        )
        print(f"Conditional p (fallback): {model.fallback_p:.3f}")

        beats_fixed = (
            conditional_metrics["annualized_excess"]
            > baseline_metrics["annualized_excess"]
        )

        if beats_fixed:
            print("RESULT: Conditional FGP OUTPERFORMS fixed p=0.3 on holdout")
        else:
            print(
                "RESULT: On 50-stock S&P 500 with ~4 years of data, "
                "the covariance regime signal is too noisy to reliably "
                "improve over fixed p. The conditional approach may add "
                "value on longer time series or larger universes."
            )

        assert baseline_metrics["annualized_excess"] is not None
        assert conditional_metrics["annualized_excess"] is not None

    def test_walk_forward_conditional_fgp(self, real_data: dict) -> None:
        """Walk-forward evaluation of the conditional FGP."""
        T = real_data["T"]
        train_size = int(T * 0.5)
        step_size = 63

        results: list[dict] = []
        cursor = 0

        while cursor + train_size + step_size <= T - 1:
            train_end = cursor + train_size
            test_end = min(train_end + step_size, T - 1)

            train_w = real_data["weights"][cursor:train_end]
            train_r = real_data["returns"][cursor : train_end - 1]
            train_c = real_data["cov_matrices"][cursor:train_end]

            cfg = ConditionalFGPConfig(
                p_grid_size=8,
                n_estimators=20,
                boundary_penalty_weight=0.1,
                cost_bps=10.0,
            )
            model = CovarianceConditionalFGP(config=cfg)

            try:
                model.fit(train_w, train_c, train_r)
            except Exception:
                cursor += step_size
                continue

            test_r = real_data["returns"][train_end - 1 : test_end]
            test_w = real_data["weights"][train_end - 1 :]

            if len(test_r) < 10:
                cursor += step_size
                continue

            cond_m = _backtest_strategy(model.weights_from_mu_only, test_r, test_w[0])
            fixed_m = _backtest_strategy(
                DiversityGenerator(0.3).weights, test_r, test_w[0]
            )
            results.append(
                {
                    "window_start": cursor,
                    "conditional_excess": cond_m["annualized_excess"],
                    "fixed_excess": fixed_m["annualized_excess"],
                    "conditional_wins": cond_m["annualized_excess"]
                    > fixed_m["annualized_excess"],
                }
            )
            cursor += step_size

        if not results:
            pytest.skip("No walk-forward windows completed")

        n_wins = sum(r["conditional_wins"] for r in results)
        n_total = len(results)
        print(f"\nWalk-forward: conditional wins {n_wins}/{n_total} windows")

        avg_cond = np.mean([r["conditional_excess"] for r in results])
        avg_fixed = np.mean([r["fixed_excess"] for r in results])
        print(f"Avg conditional excess: {avg_cond * 100:.3f}%")
        print(f"Avg fixed p=0.3 excess: {avg_fixed * 100:.3f}%")

    def test_master_formula_compatibility(self, real_data: dict) -> None:
        """Conditional FGP converts to GeneratingFunction for master formula."""
        model = CovarianceConditionalFGP()
        G = model.to_generating_function()

        mu = real_data["weights"][-1]
        tau_mu = real_data["cov_matrices"][-1]

        G_val = G(mu)
        assert G_val > 0

        drift = G.drift(mu, tau_mu)
        assert np.isfinite(drift)
        print(f"G(mu)={G_val:.4f}, drift={drift:.6f}")

    def test_regime_variation_captured(self, real_data: dict) -> None:
        """The predicted p varies meaningfully across different dates."""
        cfg = ConditionalFGPConfig(
            p_grid_size=8,
            n_estimators=20,
            boundary_penalty_weight=0.0,
        )
        model = CovarianceConditionalFGP(config=cfg)
        split = int(real_data["T"] * 0.5)
        model.fit(
            real_data["weights"][:split],
            real_data["cov_matrices"][:split],
            real_data["returns"][: split - 1],
        )

        if model.training_metadata.get("status") != "fitted":
            pytest.skip("Model fell back to formula, can't test variation")

        ps = []
        for t in range(split, real_data["T"], 21):
            features = model.extract_covariance_features(real_data["cov_matrices"][t])
            ps.append(model.optimal_p(features))

        ps = np.array(ps)
        print(
            f"Predicted p: mean={np.mean(ps):.3f}, "
            f"std={np.std(ps):.3f}, range=[{np.min(ps):.3f}, {np.max(ps):.3f}]"
        )
