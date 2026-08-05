"""SPT performance metrics for backtested strategies.

Computes standard portfolio performance metrics plus SPT-specific
measures derived from the master formula.

Mathematical References
-----------------------
- Sharpe ratio: (annualized return - risk-free) / annualized volatility
- Max drawdown: max peak-to-trough decline
- Information ratio: excess return / tracking error
- Excess growth rate: F&K Survey Eq. 1.13
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require

__all__ = [
    "PerformanceMetrics",
    "TurnoverStats",
    "compute_performance",
    "compute_turnover_stats",
    "information_ratio",
    "max_drawdown",
    "sharpe_ratio",
    "tracking_error",
]


@dataclass(frozen=True)
class PerformanceMetrics:
    """Standard portfolio performance metrics.

    Attributes
    ----------
    annualized_return : float
        Compound annual growth rate.
    annualized_volatility : float
        Annualized standard deviation of returns.
    sharpe_ratio : float
        Risk-adjusted return (excess over risk-free / volatility).
    max_drawdown : float
        Maximum peak-to-trough decline (negative).
    total_return : float
        Cumulative return over the backtest period.
    """

    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float


@dataclass(frozen=True)
class TurnoverStats:
    """Turnover statistics for a backtested strategy.

    Attributes
    ----------
    total_turnover : float
        Sum of all turnover across the backtest.
    avg_turnover_per_rebalance : float
        Average turnover when rebalancing occurs.
    n_rebalances : int
        Number of rebalancing events.
    annualized_turnover : float
        Total turnover scaled to annual rate.
    """

    total_turnover: float
    avg_turnover_per_rebalance: float
    n_rebalances: int
    annualized_turnover: float


def sharpe_ratio(
    portfolio_values: NDArray[np.float64],
    dt: float,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute annualized Sharpe ratio from portfolio value series.

    Parameters
    ----------
    portfolio_values : ndarray of shape (T+1,)
        Portfolio value time series.
    dt : float
        Time step in years (e.g. 1/252).
    risk_free_rate : float
        Annualized risk-free rate.

    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
    require(len(portfolio_values) >= 2, "Need at least 2 values for Sharpe")
    log_returns = np.diff(np.log(portfolio_values))
    periods_per_year = 1.0 / dt
    mean_ret = float(np.mean(log_returns)) * periods_per_year
    std_ret = float(np.std(log_returns, ddof=1)) * np.sqrt(periods_per_year)
    if std_ret < 1e-15:
        return 0.0
    return (mean_ret - risk_free_rate) / std_ret


def max_drawdown(portfolio_values: NDArray[np.float64]) -> float:
    """Compute maximum drawdown from portfolio value series.

    Parameters
    ----------
    portfolio_values : ndarray of shape (T+1,)
        Portfolio value time series.

    Returns
    -------
    float
        Maximum drawdown (negative value, e.g. -0.15 for 15% drawdown).
        Returns 0.0 if values are monotonically increasing.
    """
    require(len(portfolio_values) >= 2, "Need at least 2 values")
    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - running_max) / running_max
    return float(np.min(drawdowns))


def tracking_error(
    portfolio_values: NDArray[np.float64],
    benchmark_values: NDArray[np.float64],
    dt: float,
) -> float:
    """Compute annualized tracking error vs a benchmark.

    Parameters
    ----------
    portfolio_values : ndarray of shape (T+1,)
        Portfolio value time series.
    benchmark_values : ndarray of shape (T+1,)
        Benchmark value time series.
    dt : float
        Time step in years.

    Returns
    -------
    float
        Annualized tracking error (standard deviation of excess returns).
    """
    require(
        len(portfolio_values) == len(benchmark_values),
        "portfolio and benchmark must have same length",
    )
    port_returns = np.diff(np.log(portfolio_values))
    bench_returns = np.diff(np.log(benchmark_values))
    excess = port_returns - bench_returns
    periods_per_year = 1.0 / dt
    return float(np.std(excess, ddof=1) * np.sqrt(periods_per_year))


def information_ratio(
    portfolio_values: NDArray[np.float64],
    benchmark_values: NDArray[np.float64],
    dt: float,
) -> float:
    """Compute annualized information ratio vs a benchmark.

    Parameters
    ----------
    portfolio_values : ndarray of shape (T+1,)
        Portfolio value time series.
    benchmark_values : ndarray of shape (T+1,)
        Benchmark value time series.
    dt : float
        Time step in years.

    Returns
    -------
    float
        Annualized information ratio.
    """
    track_err = tracking_error(portfolio_values, benchmark_values, dt)
    if track_err < 1e-15:
        return 0.0
    port_returns = np.diff(np.log(portfolio_values))
    bench_returns = np.diff(np.log(benchmark_values))
    periods_per_year = 1.0 / dt
    mean_excess = float(np.mean(port_returns - bench_returns)) * periods_per_year
    return mean_excess / track_err


def compute_performance(
    portfolio_values: NDArray[np.float64],
    dt: float,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """Compute standard performance metrics from portfolio value series.

    Parameters
    ----------
    portfolio_values : ndarray of shape (T+1,)
        Portfolio value time series.
    dt : float
        Time step in years.
    risk_free_rate : float
        Annualized risk-free rate.

    Returns
    -------
    PerformanceMetrics
        Full suite of standard metrics.
    """
    require(len(portfolio_values) >= 2, "Need at least 2 values")
    total_time = (len(portfolio_values) - 1) * dt
    total_ret = float(portfolio_values[-1] / portfolio_values[0]) - 1.0

    if total_time > 0:
        ann_ret = float(
            (portfolio_values[-1] / portfolio_values[0]) ** (1.0 / total_time) - 1.0
        )
    else:
        ann_ret = 0.0

    log_returns = np.diff(np.log(portfolio_values))
    periods_per_year = 1.0 / dt
    ann_vol = float(np.std(log_returns, ddof=1) * np.sqrt(periods_per_year))

    return PerformanceMetrics(
        annualized_return=ann_ret,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe_ratio(portfolio_values, dt, risk_free_rate),
        max_drawdown=max_drawdown(portfolio_values),
        total_return=total_ret,
    )


def compute_turnover_stats(
    turnover: NDArray[np.float64],
    n_rebalances: int,
    dt: float,
) -> TurnoverStats:
    """Compute turnover statistics from turnover series.

    Parameters
    ----------
    turnover : ndarray of shape (T+1,)
        Turnover at each time step.
    n_rebalances : int
        Number of rebalancing events.
    dt : float
        Time step in years.

    Returns
    -------
    TurnoverStats
        Turnover statistics.
    """
    total = float(np.sum(turnover))
    avg_per_rebal = total / n_rebalances if n_rebalances > 0 else 0.0
    total_time = (len(turnover) - 1) * dt
    ann_turnover = total / total_time if total_time > 0 else 0.0

    return TurnoverStats(
        total_turnover=total,
        avg_turnover_per_rebalance=avg_per_rebal,
        n_rebalances=n_rebalances,
        annualized_turnover=ann_turnover,
    )
