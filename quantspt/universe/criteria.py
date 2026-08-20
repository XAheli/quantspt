"""Individual stock selection criteria for SPT-optimized universes.

Each criterion scores stocks on a dimension that affects the excess growth
rate γ* of a diversity-weighted portfolio.  The key identity for uncorrelated
stocks is:

    γ*_π = (1/2) Σ π_i (1 − π_i) σ²_i

To maximize γ* the universe should contain stocks with:
- high idiosyncratic variance (large σ²_i after removing market factor),
- low average pairwise correlation (so the uncorrelated approximation holds),
- weights that are not too concentrated (moderate market cap).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .._preconditions import require

__all__ = [
    "boundary_risk_score",
    "gamma_star_contribution",
    "idiosyncratic_volatility",
    "liquidity_filter",
    "pairwise_correlation_score",
]


def idiosyncratic_volatility(
    returns: pd.DataFrame,
    market_returns: pd.Series,
) -> pd.Series:
    """Residual annualised volatility after removing the market factor.

    For each stock, regresses daily returns on the equal-weighted market
    return and reports ``std(residual) * sqrt(252)``.

    Parameters
    ----------
    returns : DataFrame, shape (T, n)
        Daily simple or log returns for each stock.
    market_returns : Series, shape (T,)
        Daily market (or equal-weighted) return.

    Returns
    -------
    Series indexed by ticker
        Annualised idiosyncratic volatility.
    """
    require(len(returns) == len(market_returns), "returns / market length mismatch")
    require(len(returns) >= 20, "need >= 20 observations for a meaningful estimate")

    mkt = np.asarray(market_returns.values, dtype=np.float64)

    result: dict[str, float] = {}
    for ticker in returns.columns:
        y = np.asarray(returns[ticker].values, dtype=np.float64)
        valid = np.isfinite(y) & np.isfinite(mkt)
        if valid.sum() < 20:
            result[ticker] = np.nan
            continue
        y_v, m_v = y[valid], mkt[valid]
        cov_matrix = np.cov(y_v, m_v, ddof=1)
        beta = cov_matrix[0, 1] / max(cov_matrix[1, 1], 1e-15)
        resid = y_v - beta * m_v
        result[ticker] = float(np.std(resid, ddof=1) * np.sqrt(252))

    return pd.Series(result, name="idiosyncratic_vol")


def pairwise_correlation_score(returns: pd.DataFrame) -> pd.Series:
    """Average absolute pairwise correlation for each stock.

    Lower values indicate that a stock is more independent of the rest of
    the universe — desirable for maximising γ*.

    Parameters
    ----------
    returns : DataFrame, shape (T, n)
        Daily returns.

    Returns
    -------
    Series indexed by ticker
        Mean absolute pairwise correlation (0 = perfectly independent).
    """
    require(returns.shape[1] >= 2, "need >= 2 stocks")
    corr = returns.corr()
    np.fill_diagonal(corr.values, 0.0)
    avg = corr.abs().mean(axis=1)
    avg.name = "avg_correlation"
    return avg


def gamma_star_contribution(
    weights: NDArray[np.float64],
    covariance: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Marginal contribution of each stock to the portfolio excess growth rate.

    From γ* = (1/2)[Σ π_i a_{ii} − π^T a π], the marginal contribution
    of stock *i* (holding other weights fixed) is:

        ∂γ*/∂π_i = (1/2)[a_{ii} − 2(a π)_i]

    which measures how much adding a bit more of stock *i* changes γ*.

    Parameters
    ----------
    weights : ndarray, shape (n,)
        Current portfolio weights.
    covariance : ndarray, shape (n, n)
        Annualised covariance matrix.

    Returns
    -------
    ndarray, shape (n,)
        Marginal gamma contribution per stock.
    """
    n = len(weights)
    require(covariance.shape == (n, n), "covariance shape mismatch")
    diag = np.diag(covariance)
    port_cov = covariance @ weights
    return 0.5 * (diag - 2.0 * port_cov)


def boundary_risk_score(
    weights: NDArray[np.float64],
    concentration_history: pd.DataFrame,
) -> pd.Series:
    """Measure how much each stock's inclusion increases boundary risk.

    Stocks whose weight has been rising (concentrating) tend to make the
    boundary term log(G(μ_T)/G(μ_0)) more negative, which hurts the
    diversity strategy.  We proxy this with the slope of the stock's
    weight share over the look-back window.

    Parameters
    ----------
    weights : ndarray, shape (n,)
        Current portfolio weights.
    concentration_history : DataFrame, shape (T, n)
        Historical market-weight time series.

    Returns
    -------
    Series indexed by ticker
        Score in [0, 1] — higher means more boundary risk.
    """
    require(
        concentration_history.shape[1] == len(weights),
        "concentration_history / weights dimension mismatch",
    )
    T = len(concentration_history)
    t_idx = np.arange(T, dtype=np.float64)
    t_idx -= t_idx.mean()

    scores: dict[str, float] = {}
    for j, ticker in enumerate(concentration_history.columns):
        col = np.asarray(concentration_history.iloc[:, j].values, dtype=np.float64)
        valid = np.isfinite(col)
        if valid.sum() < 5:
            scores[ticker] = 0.5
            continue
        slope = float(np.polyfit(t_idx[valid], col[valid], 1)[0])
        # Positive slope → concentrating → higher risk.
        # Normalise to [0, 1] with a sigmoid-like transform.
        scores[ticker] = float(1.0 / (1.0 + np.exp(-slope * T * 50)))

    return pd.Series(scores, name="boundary_risk")


def liquidity_filter(
    volumes: pd.DataFrame,
    min_daily_volume: float = 1e6,
) -> pd.Index:
    """Return tickers whose median daily dollar volume exceeds *min_daily_volume*.

    Parameters
    ----------
    volumes : DataFrame, shape (T, n)
        Daily dollar trading volume (price × shares).
    min_daily_volume : float
        Minimum median daily dollar volume.

    Returns
    -------
    Index
        Tickers that pass the filter.
    """
    median_vol = volumes.median()
    return median_vol[median_vol >= min_daily_volume].index
