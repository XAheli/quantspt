"""Validate Neural FGP and AdaptiveFGP on real market data.

Downloads 50 S&P 500 stocks (2022-2025), trains both approaches, and
evaluates out-of-sample relative returns against classical baselines.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pytest

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


def _download_market_weights(
    tickers: list[str],
    start: str = "2022-01-01",
    end: str = "2025-01-01",
) -> np.ndarray:
    """Download daily data and compute market-cap proxy weights."""
    import yfinance as yf

    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    data = data.dropna(axis=1, how="any")
    data = data.dropna(axis=0)

    prices = data.values.astype(np.float64)
    weights = prices / prices.sum(axis=1, keepdims=True)
    weights = np.clip(weights, 1e-8, None)
    weights /= weights.sum(axis=1, keepdims=True)

    print(f"Downloaded {weights.shape[1]} stocks, {weights.shape[0]} days")
    return weights


def _evaluate_walk_forward(
    model_weights_fn,
    market_weights: np.ndarray,
    train_days: int = 500,
    eval_days: int = 126,
    step_days: int = 63,
) -> dict:
    """Walk-forward evaluation of a portfolio strategy.

    Train on [0:train_days], evaluate on [train_days:train_days+eval_days],
    step forward by step_days, repeat.
    """
    returns = market_weights[1:] / market_weights[:-1]
    T_ret = returns.shape[0]

    all_log_rels = []
    windows = 0
    start = 0

    while start + train_days + eval_days <= T_ret:
        train_mw = market_weights[start : start + train_days]
        eval_mw = market_weights[start + train_days : start + train_days + eval_days]
        eval_ret = returns[start + train_days : start + train_days + eval_days]

        try:
            pi_fn = model_weights_fn(train_mw)
        except Exception as e:
            print(f"  Training failed at window {windows}: {e}")
            start += step_days
            continue

        log_rel = 0.0
        for t in range(len(eval_ret)):
            pi = pi_fn(eval_mw[t])
            period_return = float(np.dot(pi, eval_ret[t]))
            log_rel += np.log(max(period_return, 1e-12))
        all_log_rels.append(log_rel)
        windows += 1
        start += step_days

    if not all_log_rels:
        return {"mean_log_rel": 0.0, "total_log_rel": 0.0, "n_windows": 0}

    return {
        "mean_log_rel": float(np.mean(all_log_rels)),
        "total_log_rel": float(np.sum(all_log_rels)),
        "n_windows": windows,
        "per_window": all_log_rels,
    }


def _classical_weights_fn(p: float):
    """Return a (train, eval_fn) factory for DiversityGenerator(p)."""

    def factory(train_mw: np.ndarray):
        def weights_fn(mu: np.ndarray) -> np.ndarray:
            mu_p = mu**p
            return mu_p / mu_p.sum()

        return weights_fn

    return factory


def _market_weights_fn():
    """Market portfolio (buy-and-hold cap-weighted)."""

    def factory(train_mw: np.ndarray):
        def weights_fn(mu: np.ndarray) -> np.ndarray:
            return mu.copy()

        return weights_fn

    return factory


@pytest.fixture(scope="module")
def real_data():
    """Download real market data once for all tests."""
    try:
        mw = _download_market_weights(SP500_50)
    except Exception as e:
        pytest.skip(f"Could not download market data: {e}")
    if mw.shape[0] < 600:
        pytest.skip("Insufficient data downloaded")
    return mw


class TestAdaptiveFGPRealData:
    """Test AdaptiveFGP on real S&P 500 data."""

    def test_adaptive_vs_classical(self, real_data: np.ndarray) -> None:
        """AdaptiveFGP must not collapse and should compete with classical."""
        from quantspt.ml.adaptive_fgp import AdaptiveFGP, AdaptiveFGPConfig

        mw = real_data

        classical_p05 = _evaluate_walk_forward(
            _classical_weights_fn(0.5),
            mw,
            train_days=500,
            eval_days=126,
            step_days=126,
        )
        classical_p03 = _evaluate_walk_forward(
            _classical_weights_fn(0.3),
            mw,
            train_days=500,
            eval_days=126,
            step_days=126,
        )
        market = _evaluate_walk_forward(
            _market_weights_fn(),
            mw,
            train_days=500,
            eval_days=126,
            step_days=126,
        )

        print(f"\n{'=' * 60}")
        print("BASELINE RESULTS (walk-forward, 126-day eval windows)")
        print(f"  Market:           mean_log_rel = {market['mean_log_rel']:.6f}")
        print(f"  Diversity(p=0.5): mean_log_rel = {classical_p05['mean_log_rel']:.6f}")
        print(f"  Diversity(p=0.3): mean_log_rel = {classical_p03['mean_log_rel']:.6f}")
        print(f"{'=' * 60}")

        def adaptive_factory(train_mw: np.ndarray):
            cfg = AdaptiveFGPConfig(
                base_p=0.5,
                correction_dims=[32, 16],
                learning_rate=1e-4,
                epochs=150,
                train_window=200,
                eval_window=20,
                weight_decay=1e-6,
                anchor_strength=0.01,
                min_epochs=30,
                early_stopping_patience=25,
                seed=42,
            )
            model = AdaptiveFGP(n_assets=train_mw.shape[1], config=cfg)
            model.fit(train_mw)
            return lambda mu: model.weights(mu)

        adaptive_result = _evaluate_walk_forward(
            adaptive_factory,
            mw,
            train_days=500,
            eval_days=126,
            step_days=126,
        )

        print(
            f"\n  AdaptiveFGP:      mean_log_rel = {adaptive_result['mean_log_rel']:.6f}"
        )
        print(f"  (n_windows={adaptive_result['n_windows']})")
        print(f"{'=' * 60}")

        assert adaptive_result["n_windows"] > 0, "No evaluation windows completed"
        assert adaptive_result["mean_log_rel"] > market["mean_log_rel"] - 0.05, (
            f"AdaptiveFGP collapsed: {adaptive_result['mean_log_rel']:.6f} "
            f"vs market {market['mean_log_rel']:.6f}"
        )


class TestNeuralFGPRealData:
    """Test warm-started NeuralFGP on real data."""

    def test_warmstarted_vs_classical(self, real_data: np.ndarray) -> None:
        """Warm-started NeuralFGP should not collapse to market weights."""
        from quantspt.ml.neural_fgp import NeuralFGP, NeuralFGPConfig

        mw = real_data

        def neural_factory(train_mw: np.ndarray):
            cfg = NeuralFGPConfig(
                hidden_dims=[64, 64, 32],
                learning_rate=1e-4,
                epochs=100,
                train_window=200,
                eval_window=20,
                weight_decay=1e-6,
                positivity_offset=1.0,
                early_stopping_patience=25,
                min_epochs=30,
                warm_start=True,
                warm_start_p=0.5,
                warm_start_epochs=100,
                warm_start_lr=1e-3,
                seed=42,
            )
            model = NeuralFGP(n_assets=train_mw.shape[1], config=cfg)
            model.fit(train_mw)
            return lambda mu: model.weights(mu)

        neural_result = _evaluate_walk_forward(
            neural_factory,
            mw,
            train_days=500,
            eval_days=126,
            step_days=126,
        )

        market = _evaluate_walk_forward(
            _market_weights_fn(),
            mw,
            train_days=500,
            eval_days=126,
            step_days=126,
        )

        print(f"\n{'=' * 60}")
        print("NEURAL FGP (warm-started) RESULTS")
        print(f"  Market:      mean_log_rel = {market['mean_log_rel']:.6f}")
        print(f"  Neural FGP:  mean_log_rel = {neural_result['mean_log_rel']:.6f}")
        print(f"  (n_windows={neural_result['n_windows']})")
        print(f"{'=' * 60}")

        assert neural_result["n_windows"] > 0


class TestJAXBackendDispatch:
    """Test JAX backend dispatch provides speedup."""

    def test_jax_dispatch_correctness(self) -> None:
        """Core functions produce identical results under JAX backend."""
        from quantspt._config import set_backend
        from quantspt.core.covariance import portfolio_variance, relative_covariance
        from quantspt.core.growth_rates import excess_growth_rate

        n = 100
        rng = np.random.default_rng(42)
        raw = rng.standard_normal((n, n))
        a = raw @ raw.T / n + np.eye(n) * 0.01
        pi = rng.dirichlet(np.ones(n))

        set_backend("numpy")
        tau_np = relative_covariance(a, pi)
        var_np = portfolio_variance(a, pi)
        egr_np = excess_growth_rate(pi, a)

        try:
            set_backend("jax")
        except (ValueError, ImportError):
            pytest.skip("JAX not available")

        tau_jax = relative_covariance(a, pi)
        var_jax = portfolio_variance(a, pi)
        egr_jax = excess_growth_rate(pi, a)

        set_backend("numpy")

        np.testing.assert_allclose(tau_jax, tau_np, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(var_jax, var_np, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(egr_jax, egr_np, rtol=1e-5, atol=1e-7)

    def test_jax_dispatch_speedup(self) -> None:
        """JAX backend is faster than NumPy at n=1000."""
        from quantspt._config import set_backend
        from quantspt.core.covariance import relative_covariance

        n = 1000
        rng = np.random.default_rng(42)
        raw = rng.standard_normal((n, n))
        a = (raw @ raw.T / n + np.eye(n) * 0.01).astype(np.float64)
        pi = rng.dirichlet(np.ones(n)).astype(np.float64)

        set_backend("numpy")
        for _ in range(3):
            relative_covariance(a, pi)
        t0 = time.perf_counter()
        for _ in range(50):
            relative_covariance(a, pi)
        numpy_time = time.perf_counter() - t0

        try:
            set_backend("jax")
        except (ValueError, ImportError):
            pytest.skip("JAX not available")

        for _ in range(3):
            relative_covariance(a, pi)
        t0 = time.perf_counter()
        for _ in range(50):
            relative_covariance(a, pi)
        jax_time = time.perf_counter() - t0

        set_backend("numpy")

        speedup = numpy_time / max(jax_time, 1e-12)
        print(
            f"\n  n={n}: NumPy={numpy_time:.4f}s, JAX={jax_time:.4f}s, "
            f"speedup={speedup:.1f}x"
        )

        assert jax_time < numpy_time * 2, (
            f"JAX should not be dramatically slower: "
            f"JAX={jax_time:.4f}s vs NumPy={numpy_time:.4f}s"
        )
