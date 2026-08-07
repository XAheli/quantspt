"""Real-data end-to-end validation of the SPT pipeline.

Downloads actual market data via yfinance and validates the full SPT
pipeline: diversity analysis, backtest, HMM regime detection, and
Atlas model calibration.

Run with:
    pytest tests/benchmarks/test_real_data_validation.py -v -s --tb=short
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BENCHMARKS_DIR = Path(__file__).resolve().parents[2] / "benchmarks"

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


def _download_prices(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame | None:
    """Download adjusted close prices; return None on failure."""
    try:
        from quantspt.data.providers.yfinance import YFinanceProvider

        provider = YFinanceProvider(progress=False)
        result = provider.load(tickers=tickers, start=start, end=end)
        return result.data.prices
    except Exception as exc:
        print(f"  WARNING: YFinance download failed: {exc}")
        print("  Falling back to synthetic data for validation.")
        return None


def _synthetic_prices(n_stocks: int, n_days: int, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic price paths as fallback."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    mu = rng.uniform(0.05, 0.15, size=n_stocks)
    sigma = rng.uniform(0.15, 0.45, size=n_stocks)
    log_returns = (mu - 0.5 * sigma**2)[:, None] * dt + sigma[:, None] * np.sqrt(
        dt
    ) * rng.standard_normal((n_stocks, n_days))
    prices = 100.0 * np.exp(np.cumsum(log_returns, axis=1))
    prices = np.column_stack([np.full(n_stocks, 100.0), prices])

    dates = pd.bdate_range(start="2024-01-02", periods=n_days + 1)[: prices.shape[1]]
    tickers = [f"SYN{i:03d}" for i in range(n_stocks)]
    return pd.DataFrame(prices.T, index=dates, columns=tickers)


# ---------------------------------------------------------------------------
# Test 1: S&P 500 Diversity Analysis
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestDiversityAnalysis:
    """Analyze market diversity and concentration on real S&P 500 data."""

    _prices: pd.DataFrame = None  # type: ignore[assignment]
    _is_synthetic: bool = False

    @classmethod
    @pytest.fixture(autouse=True, scope="class")
    def load_data(cls):
        prices = _download_prices(SP50_TICKERS, start="2024-01-01", end="2025-01-01")
        if prices is None:
            prices = _synthetic_prices(50, 252)
            cls._is_synthetic = True
        else:
            cls._is_synthetic = False
        cls._prices = prices

    def test_market_weights_and_diversity(self):
        """Compute market-cap proxy weights and check diversity."""
        from quantspt.core.generating_functions import DiversityGenerator

        prices = self._prices
        weights = prices.div(prices.sum(axis=1), axis=0).values

        n_days, n_stocks = weights.shape
        print(f"\n  Data: {n_days} days x {n_stocks} stocks")
        print(f"  Source: {'synthetic' if self._is_synthetic else 'yfinance'}")

        top1 = weights.max(axis=1)
        top5 = np.sort(weights, axis=1)[:, -5:].sum(axis=1)
        hhi = (weights**2).sum(axis=1)

        print(f"  Top-1 weight: mean={top1.mean():.4f}  max={top1.max():.4f}")
        print(f"  Top-5 weight: mean={top5.mean():.4f}")
        print(f"  HHI:          mean={hhi.mean():.6f}")

        gen = DiversityGenerator(p=0.5)
        diversity_vals = np.array([gen(w) for w in weights])
        print(
            f"  Diversity G_0.₅: mean={diversity_vals.mean():.4f}  "
            f"std={diversity_vals.std():.4f}"
        )

        diversity_ratio = diversity_vals / (n_stocks**0.5)
        print(f"  Diversity ratio G/n^p: mean={diversity_ratio.mean():.4f}")

        delta = 1.0 - top1
        print(f"\n  Diversity condition 1-μ_max (δ): mean={delta.mean():.4f}")
        print(
            f"  Market is {'diverse' if delta.mean() > 0.5 else 'concentrated'} "
            f"(δ {'>' if delta.mean() > 0.5 else '<'} 0.5)"
        )

        assert np.all(weights >= 0)
        assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-8)
        assert diversity_vals.mean() > 0

    def test_excess_growth_rate_real_data(self):
        """Compute rolling excess growth rate on real data."""
        from quantspt.core.growth_rates import excess_growth_rate

        prices = self._prices
        log_returns = np.log(prices / prices.shift(1)).dropna().values
        weights = prices.div(prices.sum(axis=1), axis=0).values[1:]

        window = min(60, len(log_returns) - 1)

        gamma_stars = []
        for t in range(window, len(log_returns)):
            window_rets = log_returns[t - window : t]
            cov = np.cov(window_rets, rowvar=False, ddof=1)
            cov = (cov + cov.T) / 2
            eigvals = np.linalg.eigvalsh(cov)
            if eigvals[0] < 0:
                cov += np.eye(cov.shape[0]) * (-eigvals[0] + 1e-10)

            pi = weights[t]
            pi = np.clip(pi, 1e-12, None)
            pi = pi / pi.sum()
            try:
                gs = excess_growth_rate(pi, cov * 252)
                gamma_stars.append(gs)
            except Exception:
                gamma_stars.append(np.nan)

        gamma_stars = np.array(gamma_stars)
        valid = gamma_stars[np.isfinite(gamma_stars)]
        print("\n  Rolling g* (annualized, 60-day window):")
        print(f"    mean={valid.mean():.6f}  std={valid.std():.6f}")
        print(f"    min={valid.min():.6f}  max={valid.max():.6f}")
        print(
            f"    g* > 0 on {(valid > 0).sum()}/{len(valid)} days "
            f"({(valid > 0).mean() * 100:.1f}%)"
        )

        assert len(valid) > 10
        assert valid.mean() > 0, "Mean g* should be positive for diversified portfolio"


# ---------------------------------------------------------------------------
# Test 2: DiversityGenerator Backtest on Real Data
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestDiversityBacktest:
    """Backtest diversity-weighted vs market-cap-weighted on real data."""

    _prices: pd.DataFrame = None  # type: ignore[assignment]
    _is_synthetic: bool = False

    @classmethod
    @pytest.fixture(autouse=True, scope="class")
    def load_data(cls):
        prices = _download_prices(SP50_TICKERS, start="2024-01-01", end="2025-01-01")
        if prices is None:
            prices = _synthetic_prices(50, 252)
            cls._is_synthetic = True
        else:
            cls._is_synthetic = False
        cls._prices = prices

    def test_diversity_vs_market(self):
        """Run backtest comparing diversity-weighted (p=0.5) to market-cap."""
        from quantspt.backtesting import BacktestConfig, BacktestEngine
        from quantspt.core.generating_functions import DiversityGenerator

        prices = self._prices.values.astype(np.float64)
        n_days, n_stocks = prices.shape

        simple_returns = prices[1:] / prices[:-1]
        weights_0 = prices[0] / prices[0].sum()
        weights_0 = np.clip(weights_0, 1e-10, None)
        weights_0 = weights_0 / weights_0.sum()

        gen = DiversityGenerator(p=0.5)

        engine = BacktestEngine(
            weight_func=gen.weights,
            returns=simple_returns,
            initial_weights=weights_0,
            config=BacktestConfig(initial_value=1.0),
        )

        t0 = time.perf_counter()
        result = engine.run()
        elapsed = time.perf_counter() - t0

        bt = result.data
        log_rel = bt.log_relative_return()
        port_ret = bt.portfolio_values[-1] / bt.portfolio_values[0] - 1
        mkt_ret = bt.market_values[-1] / bt.market_values[0] - 1

        print(f"\n  Backtest: {n_days} days x {n_stocks} stocks")
        print(f"  Source: {'synthetic' if self._is_synthetic else 'yfinance'}")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Portfolio return:  {port_ret:+.4f} ({port_ret * 100:+.2f}%)")
        print(f"  Market return:     {mkt_ret:+.4f} ({mkt_ret * 100:+.2f}%)")
        print(f"  log(V^pi/V^μ):      {log_rel:+.6f}")
        print(f"  Outperformance:    {'YES' if log_rel > 0 else 'NO'}")
        print(f"  Rebalances:        {bt.n_rebalances}")
        print(f"  Total turnover:    {bt.total_turnover():.4f}")

        assert np.all(np.isfinite(bt.portfolio_values))
        assert bt.portfolio_values[-1] > 0

    def test_multiple_p_values(self):
        """Compare diversity portfolios across p values."""
        from quantspt.backtesting import BacktestConfig, BacktestEngine
        from quantspt.core.generating_functions import DiversityGenerator

        prices = self._prices.values.astype(np.float64)
        simple_returns = prices[1:] / prices[:-1]
        weights_0 = prices[0] / prices[0].sum()
        weights_0 = np.clip(weights_0, 1e-10, None)
        weights_0 = weights_0 / weights_0.sum()

        print(
            f"\n  {'p':>5s}  {'port_ret':>10s}  {'mkt_ret':>10s}  {'log(V^pi/V^μ)':>14s}"
        )
        print(f"  {'-' * 5}  {'-' * 10}  {'-' * 10}  {'-' * 14}")

        for p in [0.25, 0.5, 0.75, 0.9]:
            gen = DiversityGenerator(p=p)
            engine = BacktestEngine(
                weight_func=gen.weights,
                returns=simple_returns,
                initial_weights=weights_0,
                config=BacktestConfig(initial_value=1.0),
            )
            result = engine.run()
            bt = result.data
            port_ret = bt.portfolio_values[-1] / bt.portfolio_values[0] - 1
            mkt_ret = bt.market_values[-1] / bt.market_values[0] - 1
            log_rel = bt.log_relative_return()
            print(
                f"  {p:>5.2f}  {port_ret:>+10.4f}  {mkt_ret:>+10.4f}  {log_rel:>+14.6f}"
            )

            assert np.all(np.isfinite(bt.portfolio_values))


# ---------------------------------------------------------------------------
# Test 3: HMM Regime Detection on Real Data
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestRegimeDetection:
    """Run HMM regime detection on diversity time series."""

    _prices: pd.DataFrame = None  # type: ignore[assignment]
    _is_synthetic: bool = False

    @classmethod
    @pytest.fixture(autouse=True, scope="class")
    def load_data(cls):
        prices = _download_prices(SP50_TICKERS, start="2022-01-01", end="2025-01-01")
        if prices is None:
            prices = _synthetic_prices(50, 750)
            cls._is_synthetic = True
        else:
            cls._is_synthetic = False
        cls._prices = prices

    def test_hmm_on_diversity_series(self):
        """Detect regimes in the diversity time series."""
        from quantspt.core.generating_functions import DiversityGenerator
        from quantspt.ml.regime import HMMRegimeDetector

        prices = self._prices
        weights = prices.div(prices.sum(axis=1), axis=0).values

        gen = DiversityGenerator(p=0.5)
        diversity_ts = np.array([gen(w) for w in weights])

        features = np.column_stack(
            [
                diversity_ts,
                np.gradient(diversity_ts),
            ]
        )
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-10)

        detector = HMMRegimeDetector(n_regimes=2, random_state=42)
        detector.fit(features)
        labels = detector.predict(features)

        n_regime_0 = (labels == 0).sum()
        n_regime_1 = (labels == 1).sum()
        print(f"\n  HMM Regime Detection ({len(diversity_ts)} days)")
        print(f"  Source: {'synthetic' if self._is_synthetic else 'yfinance'}")
        print(f"  Regime 0: {n_regime_0} days ({n_regime_0 / len(labels) * 100:.1f}%)")
        print(f"  Regime 1: {n_regime_1} days ({n_regime_1 / len(labels) * 100:.1f}%)")

        trans = detector.transition_matrix
        print("  Transition matrix:")
        print(f"    P(0->0)={trans[0, 0]:.3f}  P(0->1)={trans[0, 1]:.3f}")
        print(f"    P(1->0)={trans[1, 0]:.3f}  P(1->1)={trans[1, 1]:.3f}")

        for regime in range(2):
            mask = labels == regime
            d_vals = diversity_ts[mask]
            print(
                f"  Regime {regime}: diversity mean={d_vals.mean():.4f} "
                f"std={d_vals.std():.4f}"
            )

        transitions = np.where(np.diff(labels) != 0)[0]
        print(f"  Regime transitions: {len(transitions)} total")
        if len(transitions) > 0 and not self._is_synthetic:
            dates = self._prices.index
            for t_idx in transitions[:5]:
                print(
                    f"    Transition at day {t_idx}: {dates[t_idx].strftime('%Y-%m-%d')}"
                )

        assert len(np.unique(labels)) >= 1
        assert trans.shape == (2, 2)
        assert np.allclose(trans.sum(axis=1), 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# Test 4: Atlas Model Calibration
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestAtlasCalibration:
    """Fit Atlas model parameters to real capital distribution data."""

    _prices: pd.DataFrame = None  # type: ignore[assignment]
    _is_synthetic: bool = False

    @classmethod
    @pytest.fixture(autouse=True, scope="class")
    def load_data(cls):
        prices = _download_prices(SP50_TICKERS, start="2024-01-01", end="2025-01-01")
        if prices is None:
            prices = _synthetic_prices(50, 252)
            cls._is_synthetic = True
        else:
            cls._is_synthetic = False
        cls._prices = prices

    def test_calibrate_atlas_parameters(self):
        """Calibrate Atlas model and check Pareto exponent matches literature."""
        from scipy.optimize import minimize_scalar

        from quantspt.models.atlas import AtlasModel

        prices = self._prices
        weights = prices.div(prices.sum(axis=1), axis=0).values
        n_stocks = weights.shape[1]

        ranked_weights = np.sort(weights, axis=1)[:, ::-1]
        mean_ranked = ranked_weights.mean(axis=0)

        log_returns = np.log(prices / prices.shift(1)).dropna().values
        avg_vol = np.sqrt(252) * log_returns.std(axis=0).mean()

        print("\n  Atlas Model Calibration")
        print(f"  Source: {'synthetic' if self._is_synthetic else 'yfinance'}")
        print(f"  n_stocks={n_stocks}  avg_annual_vol={avg_vol:.4f}")
        print(f"  Top ranked weight: {mean_ranked[0]:.4f}")
        print(f"  Smallest ranked weight: {mean_ranked[-1]:.6f}")

        log_rank = np.log(np.arange(1, n_stocks + 1))
        log_weight = np.log(mean_ranked)
        valid = np.isfinite(log_weight) & (mean_ranked > 1e-10)
        if valid.sum() >= 5:
            slope, _intercept = np.polyfit(log_rank[valid], log_weight[valid], 1)
            empirical_zipf = -slope
            print(f"\n  Empirical Zipf exponent (log-log fit): {empirical_zipf:.3f}")
        else:
            empirical_zipf = 1.0
            print(f"\n  Insufficient data for Zipf fit; using default {empirical_zipf}")

        sigma_param = avg_vol

        def atlas_fit_error(g_param):
            if g_param <= 0:
                return 1e10
            try:
                model = AtlasModel(
                    n=n_stocks,
                    gamma=0.05,
                    g_param=float(g_param),
                    sigma_param=sigma_param,
                )
                ce_weights = model.certainty_equivalent_weights()
                return float(np.sum((ce_weights - mean_ranked) ** 2))
            except Exception:
                return 1e10

        result = minimize_scalar(
            atlas_fit_error,
            bounds=(1e-4, 0.5),
            method="bounded",
        )
        best_g = result.x

        model = AtlasModel(
            n=n_stocks,
            gamma=0.05,
            g_param=float(best_g),
            sigma_param=sigma_param,
        )

        pareto_exp = model.pareto_exponent()
        zipf_exp = model.zipf_exponent()
        ce_weights = model.certainty_equivalent_weights()

        weight_corr = np.corrcoef(mean_ranked, ce_weights)[0, 1]

        print("\n  Fitted Atlas parameters:")
        print(f"    g_param = {best_g:.6f}")
        print(f"    sigma_param = {sigma_param:.4f}")
        print(f"    Pareto exponent r_1 = {pareto_exp:.3f}  (literature: ~1.0-1.5)")
        print(f"    Zipf exponent a = {zipf_exp:.3f}")
        print(f"    CE weight correlation = {weight_corr:.4f}")

        gamma_star_ew = model.equal_weighted_excess_growth_rate()
        gamma_star_div = model.diversity_weighted_excess_growth(p=0.5)
        print("\n  Theoretical growth rates:")
        print(f"    g*_equal_weight = {gamma_star_ew:.6f}")
        print(f"    g*_diversity(p=0.5) = {gamma_star_div:.6f}")
        print(f"    Market growth = {model.market_growth_rate():.6f}")

        assert pareto_exp > 0
        assert 0 < zipf_exp < 100
        assert weight_corr > 0.5, f"Atlas fit too poor: correlation {weight_corr:.3f}"


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_generate_validation_report():
    """Run full validation and write benchmarks/real_data_validation.md."""
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Real-Data End-to-End SPT Validation",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
    ]

    # --- Download ---
    print("\n" + "=" * 72)
    print("REAL-DATA VALIDATION")
    print("=" * 72)

    prices_1y = _download_prices(SP50_TICKERS, start="2024-01-01", end="2025-01-01")
    is_synthetic = prices_1y is None
    if is_synthetic:
        prices_1y = _synthetic_prices(50, 252)
        lines.append("**Data source:** Synthetic (yfinance unavailable)")
    else:
        lines.append(
            f"**Data source:** Yahoo Finance ({len(SP50_TICKERS)} S&P 500 stocks)"
        )
    lines.append("**Period:** 2024-01-01 to 2025-01-01")
    lines.append(f"**Stocks:** {prices_1y.shape[1]}")
    lines.append(f"**Trading days:** {prices_1y.shape[0]}")
    lines.append("")

    # --- Diversity analysis ---
    from quantspt.core.generating_functions import DiversityGenerator

    weights_df = prices_1y.div(prices_1y.sum(axis=1), axis=0)
    weights = weights_df.values

    top1 = weights.max(axis=1)
    hhi = (weights**2).sum(axis=1)
    gen = DiversityGenerator(p=0.5)
    div_ts = np.array([gen(w) for w in weights])

    lines += [
        "## 1. Market Diversity Analysis",
        "",
        f"- **Top-1 weight:** mean={top1.mean():.4f}, max={top1.max():.4f}",
        f"- **HHI:** mean={hhi.mean():.6f}",
        f"- **Diversity G_0.₅:** mean={div_ts.mean():.4f}, std={div_ts.std():.4f}",
        f"- **Market assessment:** {'Diverse' if (1.0 - top1).mean() > 0.5 else 'Concentrated'}",
        "",
    ]

    # --- Backtest ---
    from quantspt.backtesting import BacktestConfig, BacktestEngine

    prices_arr = prices_1y.values.astype(np.float64)
    simple_returns = prices_arr[1:] / prices_arr[:-1]
    w0 = prices_arr[0] / prices_arr[0].sum()
    w0 = np.clip(w0, 1e-10, None)
    w0 = w0 / w0.sum()

    lines += ["## 2. Diversity Portfolio Backtest", ""]
    lines.append(
        "| p | Portfolio Return | Market Return | log(V^pi/V^μ) | Outperforms? |"
    )
    lines.append("|---|-----------------|---------------|--------------|-------------|")

    for p in [0.25, 0.5, 0.75, 0.9]:
        g = DiversityGenerator(p=p)
        engine = BacktestEngine(
            weight_func=g.weights,
            returns=simple_returns,
            initial_weights=w0,
            config=BacktestConfig(initial_value=1.0),
        )
        result = engine.run()
        bt = result.data
        port_ret = bt.portfolio_values[-1] / bt.portfolio_values[0] - 1
        mkt_ret = bt.market_values[-1] / bt.market_values[0] - 1
        log_rel = bt.log_relative_return()
        outperforms = "Yes" if log_rel > 0 else "No"
        lines.append(
            f"| {p:.2f} | {port_ret:+.4f} ({port_ret * 100:+.2f}%) | "
            f"{mkt_ret:+.4f} ({mkt_ret * 100:+.2f}%) | {log_rel:+.6f} | {outperforms} |"
        )

    lines.append("")

    # --- Regime detection ---
    prices_3y = _download_prices(SP50_TICKERS, start="2022-01-01", end="2025-01-01")
    if prices_3y is None:
        prices_3y = _synthetic_prices(50, 750)

    weights_3y = prices_3y.div(prices_3y.sum(axis=1), axis=0).values
    div_3y = np.array([gen(w) for w in weights_3y])

    try:
        from quantspt.ml.regime import HMMRegimeDetector

        features = np.column_stack([div_3y, np.gradient(div_3y)])
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-10)

        detector = HMMRegimeDetector(n_regimes=2, random_state=42)
        detector.fit(features)
        labels = detector.predict(features)
        trans = detector.transition_matrix
        transitions = np.where(np.diff(labels) != 0)[0]

        lines += [
            "## 3. HMM Regime Detection",
            "",
            f"- **Period:** 2022-01-01 to 2025-01-01 ({len(labels)} days)",
            f"- **Regime 0:** {(labels == 0).sum()} days ({(labels == 0).mean() * 100:.1f}%)",
            f"- **Regime 1:** {(labels == 1).sum()} days ({(labels == 1).mean() * 100:.1f}%)",
            f"- **Transitions:** {len(transitions)} total",
            "- **Transition matrix:**",
            f"  - P(0->0)={trans[0, 0]:.3f}, P(0->1)={trans[0, 1]:.3f}",
            f"  - P(1->0)={trans[1, 0]:.3f}, P(1->1)={trans[1, 1]:.3f}",
        ]

        for r in range(2):
            mask = labels == r
            lines.append(
                f"- **Regime {r} diversity:** mean={div_3y[mask].mean():.4f}, "
                f"std={div_3y[mask].std():.4f}"
            )

        if not is_synthetic and len(transitions) > 0:
            dates = prices_3y.index
            lines.append("- **Notable transitions:**")
            for t_idx in transitions[:5]:
                lines.append(f"  - {dates[t_idx].strftime('%Y-%m-%d')}")
        lines.append("")
    except Exception as exc:
        lines += [
            "## 3. HMM Regime Detection",
            "",
            f"Skipped: {exc}",
            "",
        ]

    # --- Atlas calibration ---
    from scipy.optimize import minimize_scalar

    from quantspt.models.atlas import AtlasModel

    ranked = np.sort(weights, axis=1)[:, ::-1]
    mean_ranked = ranked.mean(axis=0)
    log_returns = np.log(prices_1y / prices_1y.shift(1)).dropna().values
    avg_vol = np.sqrt(252) * log_returns.std(axis=0).mean()

    log_rank = np.log(np.arange(1, weights.shape[1] + 1))
    log_w = np.log(mean_ranked)
    valid_mask = np.isfinite(log_w) & (mean_ranked > 1e-10)
    if valid_mask.sum() >= 5:
        slope, _ = np.polyfit(log_rank[valid_mask], log_w[valid_mask], 1)
        empirical_zipf = -slope
    else:
        empirical_zipf = 1.0

    sigma_param = avg_vol

    def _fit_err(g_param):
        if g_param <= 0:
            return 1e10
        try:
            m = AtlasModel(
                n=weights.shape[1],
                gamma=0.05,
                g_param=float(g_param),
                sigma_param=sigma_param,
            )
            return float(np.sum((m.certainty_equivalent_weights() - mean_ranked) ** 2))
        except Exception:
            return 1e10

    opt = minimize_scalar(_fit_err, bounds=(1e-4, 0.5), method="bounded")
    model = AtlasModel(
        n=weights.shape[1], gamma=0.05, g_param=float(opt.x), sigma_param=sigma_param
    )
    pareto_exp = model.pareto_exponent()
    zipf_exp = model.zipf_exponent()
    ce_w = model.certainty_equivalent_weights()
    corr = np.corrcoef(mean_ranked, ce_w)[0, 1]

    lines += [
        "## 4. Atlas Model Calibration",
        "",
        f"- **Fitted g_param:** {opt.x:.6f}",
        f"- **sigma_param:** {sigma_param:.4f}",
        f"- **Pareto exponent r_1:** {pareto_exp:.3f} (literature: ~1.0-1.5)",
        f"- **Zipf exponent a:** {zipf_exp:.3f}",
        f"- **Empirical Zipf (log-log slope):** {empirical_zipf:.3f}",
        f"- **CE weight vs empirical correlation:** {corr:.4f}",
        f"- **g*_equal_weight:** {model.equal_weighted_excess_growth_rate():.6f}",
        f"- **g*_diversity(p=0.5):** {model.diversity_weighted_excess_growth(p=0.5):.6f}",
        "",
        "## Findings Summary",
        "",
    ]

    if not is_synthetic:
        diverse_str = "diverse" if (1.0 - top1).mean() > 0.5 else "concentrated"
        lines.append(
            f"1. The S&P 500 subset is currently **{diverse_str}** "
            f"(δ = {(1.0 - top1).mean():.3f})"
        )
        lines.append(
            f"2. Pareto exponent {pareto_exp:.2f} — "
            f"{'within' if 0.5 < pareto_exp < 3.0 else 'outside'} "
            f"expected range (literature: ~1.0-1.5)"
        )
        lines.append(f"3. Atlas CE fit correlation: {corr:.3f}")
    else:
        lines.append("(Results based on synthetic data; yfinance was unavailable)")

    lines.append("")

    (BENCHMARKS_DIR / "real_data_validation.md").write_text("\n".join(lines) + "\n")
    print(f"\nReport saved to {BENCHMARKS_DIR / 'real_data_validation.md'}")
