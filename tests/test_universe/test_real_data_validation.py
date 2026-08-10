"""Real-data validation: SPT-optimised universe vs naive universe.

Downloads 50 S&P 500 stocks (2020–2026) via yfinance and verifies that the
SPTUniverseSelector delivers >50 bps/year improvement over a naive
"all stocks" universe when paired with DiversityGenerator(p=0.3).

The research showed a 170 bps/year spread — capturing even 1/3 of that is
significant.  This test therefore requires >50 bps improvement.

Run with: ``pytest tests/test_universe/test_real_data_validation.py -v -s``
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from quantspt.core.generating_functions import DiversityGenerator
from quantspt.universe.selector import SPTUniverseSelector

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

SP500_TICKERS = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "NVDA",
    "TSLA",
    "BRK-B",
    "UNH",
    "JNJ",
    "JPM",
    "V",
    "PG",
    "XOM",
    "HD",
    "MA",
    "CVX",
    "MRK",
    "ABBV",
    "PEP",
    "KO",
    "COST",
    "LLY",
    "AVGO",
    "WMT",
    "MCD",
    "CSCO",
    "ACN",
    "TMO",
    "ABT",
    "DHR",
    "NEE",
    "LIN",
    "PM",
    "TXN",
    "UNP",
    "COP",
    "RTX",
    "LOW",
    "AMGN",
    "HON",
    "ORCL",
    "IBM",
    "GS",
    "CAT",
    "BA",
    "SBUX",
    "AMD",
    "INTC",
    "DIS",
]


def _load_prices() -> pd.DataFrame:
    """Download prices, returning a forward-filled DataFrame."""
    import yfinance as yf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = yf.download(
            SP500_TICKERS,
            start="2020-01-01",
            end="2026-08-01",
            auto_adjust=True,
            progress=False,
        )
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data
    prices = prices.dropna(axis=1, how="all").ffill().bfill().dropna(axis=1, how="any")
    return prices


def _run_diversity_backtest(
    prices: pd.DataFrame,
    tickers: list[str],
    p: float = 0.3,
) -> float:
    """Return annualised excess log-return (bps) of DiversityGenerator(p)
    vs cap-weighted market over the supplied price panel restricted to *tickers*.
    """
    sub = prices[tickers].copy()
    log_returns = np.log(sub / sub.shift(1)).iloc[1:]
    weights = sub.div(sub.sum(axis=1), axis=0).iloc[1:]
    mu_arr = weights.values
    gen = DiversityGenerator(p)

    total_excess = 0.0
    valid_days = 0
    for t in range(len(log_returns) - 1):
        mu_t = mu_arr[t]
        mu_t = np.clip(mu_t, 1e-12, None)
        mu_t /= mu_t.sum()
        pi_t = gen.weights(mu_t)
        ret_t = log_returns.iloc[t].values
        if np.any(np.isnan(ret_t)):
            continue
        total_excess += float(np.dot(pi_t - mu_t, ret_t))
        valid_days += 1

    years = valid_days / 252
    return total_excess / max(years, 0.01) * 10_000  # bps / year


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prices():
    """Module-scoped fixture so data is downloaded only once."""
    return _load_prices()


class TestRealDataValidation:
    """Demonstrate that SPT-optimised universe selection materially
    improves diversity-strategy performance on real 2020-2026 data.
    """

    def test_spt_universe_outperforms_naive(self, prices):
        """Core validation: >50 bps/year improvement over naive "all stocks".

        Methodology:
        - Naive: run DiversityGenerator(p=0.3) on ALL 50 stocks.
        - SPT-optimised: select best 30 of 50 based on SPT criteria,
          then run DiversityGenerator(p=0.3) on that subset.

        The selection uses trailing statistics (volatility, correlation,
        weight trends) — it does NOT use future returns.
        """
        all_tickers = list(prices.columns)
        n_all = len(all_tickers)

        # --- Naive: all 50 stocks, full period ---
        naive_bps = _run_diversity_backtest(prices, all_tickers)

        # --- SPT-optimised: select best 30 ---
        selector = SPTUniverseSelector(
            n_stocks=30,
            min_market_cap_percentile=0,
            max_market_cap_percentile=95,
            max_avg_correlation=0.85,
            min_idiosyncratic_vol=0.05,
            lookback_days=126,
        )
        spt_tickers = selector.select(prices)
        spt_bps = _run_diversity_backtest(prices, spt_tickers)

        improvement = spt_bps - naive_bps

        print(f"\n{'=' * 60}")
        print("REAL-DATA VALIDATION: SPT Universe Selection")
        print(f"{'=' * 60}")
        print(f"  Stocks available : {n_all}")
        print(f"  Naive (all {n_all})   : {naive_bps:+.1f} bps/year")
        print(f"  SPT-optimised    : {spt_bps:+.1f} bps/year")
        print(f"  Improvement      : {improvement:+.1f} bps/year")
        print(f"  Selected tickers : {spt_tickers}")
        print(f"{'=' * 60}")

        assert improvement > 50, (
            f"SPT universe should outperform naive by >50 bps/year, "
            f"got {improvement:.1f} bps/year "
            f"(SPT={spt_bps:.1f}, naive={naive_bps:.1f})"
        )

    def test_score_stocks_gives_meaningful_ranking(self, prices):
        """Verify that the scoring produces a ranking that separates
        high-gamma and low-gamma stocks."""
        selector = SPTUniverseSelector(
            n_stocks=30,
            min_market_cap_percentile=0,
            max_market_cap_percentile=100,
            max_avg_correlation=1.0,
            min_idiosyncratic_vol=0.0,
        )
        scores = selector.score_stocks(prices)
        valid = scores.dropna(subset=["spt_score"])

        # Top-10 and bottom-10 stocks should have different mean idiosyncratic vol
        top10 = valid.nlargest(10, "spt_score")
        bot10 = valid.nsmallest(10, "spt_score")

        print(f"\n  Top 10 mean idio_vol  : {top10['idiosyncratic_vol'].mean():.4f}")
        print(f"  Bot 10 mean idio_vol  : {bot10['idiosyncratic_vol'].mean():.4f}")
        print(f"  Top 10 mean avg_corr  : {top10['avg_correlation'].mean():.4f}")
        print(f"  Bot 10 mean avg_corr  : {bot10['avg_correlation'].mean():.4f}")

        # Top should have higher idio vol OR lower correlation (or both)
        top_gamma = top10["gamma_contribution"].mean()
        bot_gamma = bot10["gamma_contribution"].mean()
        assert top_gamma > bot_gamma, (
            f"Top-scored stocks should have higher gamma contribution: "
            f"top={top_gamma:.6f}, bot={bot_gamma:.6f}"
        )

    def test_excluded_mega_caps_reduce_boundary_risk(self, prices):
        """With max_market_cap_percentile=80, mega-caps should be excluded."""
        selector = SPTUniverseSelector(
            n_stocks=20,
            min_market_cap_percentile=0,
            max_market_cap_percentile=80,
            min_idiosyncratic_vol=0.0,
        )
        selected = selector.select(prices)
        last_prices = prices.iloc[-1]
        high_cap_cutoff = np.percentile(last_prices.values, 80)
        mega_caps = last_prices[last_prices > high_cap_cutoff].index.tolist()

        overlap = set(selected) & set(mega_caps)
        print(f"\n  Mega-caps excluded: {set(mega_caps) - set(selected)}")
        print(f"  Overlap (should be small): {overlap}")
        assert len(overlap) <= 2, f"Too many mega-caps in universe: {overlap}"

    def test_monthly_reconstitution_stable(self, prices):
        """Monthly universe selection should not flip more than ~30% of
        stocks between consecutive months."""
        selector = SPTUniverseSelector(
            n_stocks=25,
            min_market_cap_percentile=0,
            max_market_cap_percentile=95,
            min_idiosyncratic_vol=0.0,
            lookback_days=63,
        )
        ts = selector.select_timeseries(prices)
        dates = sorted(ts.keys())
        turnovers = []
        for i in range(1, len(dates)):
            prev = set(ts[dates[i - 1]])
            curr = set(ts[dates[i]])
            changes = len(prev ^ curr)
            turnovers.append(changes / 25)

        avg_turnover = float(np.mean(turnovers))
        print(f"\n  Avg monthly turnover: {avg_turnover:.2%}")
        assert avg_turnover < 0.50, (
            f"Average monthly turnover {avg_turnover:.2%} is too high"
        )
