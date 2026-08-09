"""Transaction cost models for net-of-cost growth rate computation.

Provides proportional and market-impact cost models for evaluating
the true net performance of portfolio strategies after trading costs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "net_growth_rate",
    "optimal_rebalancing_frequency",
    "proportional_cost",
    "sqrt_market_impact",
]


def proportional_cost(
    old_weights: NDArray[np.float64],
    new_weights: NDArray[np.float64],
    *,
    cost_bps: float = 10.0,
) -> float:
    r"""Compute proportional transaction cost.

    .. math::
        C = \frac{c}{10000} \sum_i |w^{\text{new}}_i - w^{\text{old}}_i|

    Parameters
    ----------
    old_weights : ndarray of shape (n,)
        Current portfolio weights.
    new_weights : ndarray of shape (n,)
        Target portfolio weights.
    cost_bps : float
        Cost in basis points (default 10 bps = 0.1%).

    Returns
    -------
    float
        Total proportional cost as a fraction of portfolio value.
    """
    old_weights = np.asarray(old_weights, dtype=np.float64)
    new_weights = np.asarray(new_weights, dtype=np.float64)
    require(
        len(old_weights) == len(new_weights),
        f"Weight vectors must match: {len(old_weights)} vs {len(new_weights)}",
    )
    turnover = float(np.sum(np.abs(new_weights - old_weights)))
    return turnover * cost_bps / 10000.0


def sqrt_market_impact(
    old_weights: NDArray[np.float64],
    new_weights: NDArray[np.float64],
    daily_volumes: NDArray[np.float64],
    portfolio_value: float,
    *,
    impact_coeff: float = 0.1,
) -> float:
    r"""Square-root market impact cost model.

    The cost of trading asset i is proportional to the square root of
    the fraction of daily volume traded:

    .. math::
        C_i = \eta \cdot \sigma_i \cdot \sqrt{\frac{|\Delta w_i| \cdot V}
              {\text{ADV}_i}}

    This simplified version uses a constant coefficient:

    .. math::
        C = \eta \sum_i \sqrt{\frac{|\Delta w_i| \cdot V}{\text{ADV}_i}}
            \cdot |\Delta w_i|

    Parameters
    ----------
    old_weights : ndarray of shape (n,)
        Current portfolio weights.
    new_weights : ndarray of shape (n,)
        Target portfolio weights.
    daily_volumes : ndarray of shape (n,)
        Average daily trading volume in currency units for each asset.
    portfolio_value : float
        Total portfolio value in currency units.
    impact_coeff : float
        Market impact coefficient (default 0.1).

    Returns
    -------
    float
        Total market impact cost as a fraction of portfolio value.
    """
    old_weights = np.asarray(old_weights, dtype=np.float64)
    new_weights = np.asarray(new_weights, dtype=np.float64)
    daily_volumes = np.asarray(daily_volumes, dtype=np.float64)

    n = len(old_weights)
    require(len(new_weights) == n, "Weight vector length mismatch")
    require(len(daily_volumes) == n, "Volume vector length mismatch")
    require(portfolio_value > 0, "Portfolio value must be positive")
    require(bool(np.all(daily_volumes > 0)), "All daily volumes must be positive")

    delta_w = np.abs(new_weights - old_weights)
    trade_fraction = delta_w * portfolio_value / daily_volumes
    cost_per_asset = impact_coeff * np.sqrt(trade_fraction) * delta_w
    return float(np.sum(cost_per_asset))


def optimal_rebalancing_frequency(
    weight_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    returns: NDArray[np.float64],
    initial_weights: NDArray[np.float64],
    *,
    cost_bps: float = 10.0,
    candidate_days: Sequence[int] | None = None,
) -> dict[str, int | float | dict[int, float]]:
    """Find the rebalancing frequency that maximizes net-of-cost return.

    Runs a backtest at each candidate frequency and returns the one
    with the highest annualised net return, solving the "break-even
    frequency" problem automatically.

    Parameters
    ----------
    weight_func : callable
        ``mu → weights`` (e.g. ``DiversityGenerator(p=0.5).weights``).
    returns : ndarray of shape (T, n)
        Asset return series (1+r per period).
    initial_weights : ndarray of shape (n,)
        Starting market weights.
    cost_bps : float
        Proportional trading cost in basis points.
    candidate_days : sequence of int, optional
        Rebalancing periods in trading days to evaluate.
        Defaults to ``[1, 5, 21, 63]`` (daily → quarterly).

    Returns
    -------
    dict
        ``{"optimal_days": int, "optimal_net_return": float,
           "all_results": {days: net_return, ...}}``.
    """

    from ..backtesting.engine import BacktestConfig, BacktestEngine
    from ..backtesting.execution import ProportionalCostExecution
    from ..backtesting.rebalancing import CalendarRebalancer, Frequency

    if candidate_days is None:
        candidate_days = [1, 5, 21, 63]

    freq_map = {
        1: Frequency.DAILY,
        5: Frequency.WEEKLY,
        21: Frequency.MONTHLY,
        63: Frequency.QUARTERLY,
    }

    n_years = len(returns) / 252.0
    best_days = candidate_days[0]
    best_net = -np.inf
    all_results: dict[int, float] = {}

    for days in candidate_days:
        freq = freq_map.get(days)
        if freq is None:
            msg = f"Unsupported frequency {days}; use 1, 5, 21, or 63"
            raise ValueError(msg)
        engine = BacktestEngine(
            weight_func=weight_func,
            returns=returns,
            initial_weights=initial_weights,
            rebalancer=CalendarRebalancer(freq),
            execution=ProportionalCostExecution(cost_bps=cost_bps),
            config=BacktestConfig(initial_value=1.0),
        )
        result = engine.run().data
        ann_net = (result.log_relative_return() - result.total_cost()) / n_years
        all_results[days] = ann_net
        if ann_net > best_net:
            best_net = ann_net
            best_days = days

    return {
        "optimal_days": best_days,
        "optimal_net_return": best_net,
        "all_results": all_results,
    }


def net_growth_rate(
    gross_growth_rate: float,
    old_weights: NDArray[np.float64],
    new_weights: NDArray[np.float64],
    *,
    cost_bps: float = 10.0,
    rebalance_frequency: int = 21,
) -> float:
    """Compute net-of-cost annualised growth rate.

    Subtracts annualised proportional trading costs from the gross
    growth rate, assuming periodic rebalancing.

    Parameters
    ----------
    gross_growth_rate : float
        Annualised gross growth rate.
    old_weights : ndarray of shape (n,)
        Current weights.
    new_weights : ndarray of shape (n,)
        Target weights.
    cost_bps : float
        Proportional cost in basis points.
    rebalance_frequency : int
        Days between rebalances (default 21 = monthly).

    Returns
    -------
    float
        Net annualised growth rate.
    """
    per_trade_cost = proportional_cost(old_weights, new_weights, cost_bps=cost_bps)
    annual_cost = per_trade_cost * (252.0 / rebalance_frequency)
    return gross_growth_rate - annual_cost
