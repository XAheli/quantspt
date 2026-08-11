"""SPT-optimised universe selection.

The diversity strategy earns excess growth proportional to

    γ*_π ≈ (1/2) Σ π_i(1 − π_i) σ²_i     (for uncorrelated stocks)

To maximise this the universe should contain stocks with large
idiosyncratic variance and low pairwise correlation, while avoiding
mega-caps that drive the boundary term adversely.

This module turns that insight into a production-ready selector.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .._preconditions import require
from .criteria import (
    boundary_risk_score,
    gamma_star_contribution,
    idiosyncratic_volatility,
    pairwise_correlation_score,
)

__all__ = [
    "SPTUniverseSelector",
]


@dataclass
class SPTUniverseSelector:
    """Select stocks that maximise the expected excess growth rate γ*.

    Parameters
    ----------
    n_stocks : int
        Target number of stocks in the output universe.
    min_market_cap_percentile : float
        Exclude micro-caps below this percentile of the input universe.
    max_market_cap_percentile : float
        Exclude mega-caps above this percentile.
    max_avg_correlation : float
        Drop any stock whose average |ρ| with the universe exceeds this.
    min_idiosyncratic_vol : float
        Minimum annualised idiosyncratic vol to be considered.
    rebalance_frequency : str
        ``"monthly"`` or ``"quarterly"`` — cadence for ``select_timeseries``.
    lookback_days : int
        Number of trailing trading days used to estimate criteria.
    """

    n_stocks: int = 50
    min_market_cap_percentile: float = 20.0
    max_market_cap_percentile: float = 90.0
    max_avg_correlation: float = 0.7
    min_idiosyncratic_vol: float = 0.10
    rebalance_frequency: str = "monthly"
    lookback_days: int = 126
    _score_weights: dict[str, float] = field(
        default_factory=lambda: {
            "idio_vol": 0.30,
            "low_corr": 0.30,
            "gamma_contrib": 0.25,
            "low_boundary": 0.15,
        }
    )

    def __post_init__(self) -> None:
        require(self.n_stocks >= 5, "n_stocks must be >= 5")
        require(
            0 <= self.min_market_cap_percentile < self.max_market_cap_percentile <= 100,
            "market-cap percentile bounds must satisfy 0 <= min < max <= 100",
        )
        require(
            self.rebalance_frequency in ("monthly", "quarterly"),
            f"rebalance_frequency must be 'monthly' or 'quarterly', got {self.rebalance_frequency!r}",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        prices: pd.DataFrame,
        market_caps: pd.DataFrame | None = None,
    ) -> list[str]:
        """Select the optimal universe from the trailing window of *prices*.

        Parameters
        ----------
        prices : DataFrame, shape (T, n)
            Adjusted close prices, columns = tickers.
        market_caps : DataFrame, optional
            Market capitalisation.  If ``None``, prices are used as a proxy.

        Returns
        -------
        list of str
            Tickers in the selected universe (length <= ``n_stocks``).
        """
        scores = self.score_stocks(prices, market_caps)
        scores = scores.dropna(subset=["spt_score"])
        ranked = scores.sort_values("spt_score", ascending=False)
        return ranked.index[: self.n_stocks].tolist()

    def score_stocks(
        self,
        prices: pd.DataFrame,
        market_caps: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Score every stock on SPT-benefit criteria.

        Returns
        -------
        DataFrame indexed by ticker with columns:

        - ``idiosyncratic_vol`` — σ after removing the market factor
        - ``avg_correlation`` — mean |ρ| with the rest of the universe
        - ``gamma_contribution`` — marginal γ* contribution
        - ``boundary_risk`` — weight-trend proxy for boundary risk
        - ``spt_score`` — composite (higher = better)
        """
        require(prices.shape[0] >= 30, "need >= 30 price observations")
        require(prices.shape[1] >= 5, "need >= 5 stocks")

        tail = (
            prices.iloc[-self.lookback_days :]
            if len(prices) > self.lookback_days
            else prices
        )
        returns = tail.pct_change().iloc[1:]
        returns = returns.dropna(axis=1, how="all")
        returns = returns.fillna(0.0)
        tickers = list(returns.columns)

        # ----- Market-cap filter ----- #
        caps = market_caps if market_caps is not None else prices
        last_caps = caps.iloc[-1].reindex(tickers)
        lo = np.nanpercentile(last_caps, self.min_market_cap_percentile)
        hi = np.nanpercentile(last_caps, self.max_market_cap_percentile)
        cap_mask = (last_caps >= lo) & (last_caps <= hi)

        # ----- Individual criteria ----- #
        mkt_ret = returns.mean(axis=1)
        idio_vol = idiosyncratic_volatility(returns, mkt_ret)
        avg_corr = pairwise_correlation_score(returns)

        # Equal-weighted proxy for gamma contribution
        n = len(tickers)
        ew = np.ones(n) / n
        cov = np.asarray(returns.cov().values * 252, dtype=np.float64)
        gamma_c = gamma_star_contribution(ew, cov)
        gamma_c_s = pd.Series(gamma_c, index=tickers, name="gamma_contribution")

        # Boundary risk from weight trend — use full available history
        full_weight_history = prices[tickers].div(prices[tickers].sum(axis=1), axis=0)
        current_weights = np.asarray(
            full_weight_history.iloc[-1].values, dtype=np.float64
        )
        bnd_risk = boundary_risk_score(current_weights, full_weight_history)

        # ----- Build score DataFrame ----- #
        df = pd.DataFrame(
            {
                "idiosyncratic_vol": idio_vol.reindex(tickers),
                "avg_correlation": avg_corr.reindex(tickers),
                "gamma_contribution": gamma_c_s,
                "boundary_risk": bnd_risk.reindex(tickers),
            },
            index=tickers,
        )

        # ----- Hard filters ----- #
        df.loc[~cap_mask, :] = np.nan
        df.loc[df["avg_correlation"] > self.max_avg_correlation, :] = np.nan
        vol_mask = df["idiosyncratic_vol"] >= self.min_idiosyncratic_vol
        df.loc[~vol_mask, :] = np.nan

        # ----- Rank-normalise each criterion to [0, 1] ----- #
        valid = df.dropna()
        if valid.empty:
            df["spt_score"] = np.nan
            return df

        def _rank_norm(s: pd.Series) -> pd.Series:
            r = s.rank(method="average")
            return (r - r.min()) / max(r.max() - r.min(), 1e-12)

        w = self._score_weights
        norm_idio = _rank_norm(valid["idiosyncratic_vol"])
        norm_corr = _rank_norm(-valid["avg_correlation"])  # lower corr → higher score
        norm_gamma = _rank_norm(valid["gamma_contribution"])
        norm_bnd = _rank_norm(-valid["boundary_risk"])  # lower risk → higher score

        composite = (
            w["idio_vol"] * norm_idio
            + w["low_corr"] * norm_corr
            + w["gamma_contrib"] * norm_gamma
            + w["low_boundary"] * norm_bnd
        )
        df["spt_score"] = np.nan
        df.loc[composite.index, "spt_score"] = composite
        return df

    def select_timeseries(
        self,
        prices: pd.DataFrame,
        market_caps: pd.DataFrame | None = None,
    ) -> dict[pd.Timestamp, list[str]]:
        """Run selection at each rebalance date over the full history.

        Returns
        -------
        dict mapping rebalance date → list of selected tickers.
        """
        freq_days = 21 if self.rebalance_frequency == "monthly" else 63
        start = max(self.lookback_days, 60)
        dates = prices.index
        selections: dict[pd.Timestamp, list[str]] = {}

        for i in range(start, len(dates), freq_days):
            window = prices.iloc[: i + 1]
            caps_window = market_caps.iloc[: i + 1] if market_caps is not None else None
            selected = self.select(window, caps_window)
            selections[dates[i]] = selected

        return selections
