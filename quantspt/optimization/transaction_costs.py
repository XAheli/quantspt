"""Transaction cost models for net-of-cost growth rate computation.

Provides proportional and market-impact cost models for evaluating
the true net performance of portfolio strategies after trading costs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require

__all__ = [
    "net_growth_rate",
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
