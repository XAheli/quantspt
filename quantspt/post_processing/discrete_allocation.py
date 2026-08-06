"""Discrete share allocation from continuous portfolio weights.

Converts continuous optimisation output into executable integer share
counts, accounting for total portfolio value and share prices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .._preconditions import ensure, require

__all__ = [
    "AllocationResult",
    "greedy_allocation",
    "lp_allocation",
]


@dataclass(frozen=True)
class AllocationResult:
    """Result of converting continuous weights to integer shares.

    Attributes
    ----------
    shares : ndarray of shape (n,)
        Integer number of shares for each asset.
    leftover_cash : float
        Remaining cash after allocation.
    actual_weights : ndarray of shape (n,)
        Realised portfolio weights (shares * prices / total_value).
    """

    shares: NDArray[np.intp]
    leftover_cash: float
    actual_weights: NDArray[np.float64]


def greedy_allocation(
    weights: NDArray[np.float64],
    prices: NDArray[np.float64],
    total_value: float,
) -> AllocationResult:
    """Greedy integer share allocation (largest weight first).

    Allocates shares one at a time to the asset with the greatest
    remaining deficit (target allocation minus current allocation),
    continuing until no more whole shares can be purchased.

    Parameters
    ----------
    weights : ndarray of shape (n,)
        Target portfolio weights (non-negative, sum to 1).
    prices : ndarray of shape (n,)
        Current share prices (positive).
    total_value : float
        Total portfolio value to allocate.

    Returns
    -------
    AllocationResult
        Integer share counts, leftover cash, and actual weights.
    """
    require(weights.ndim == 1, f"weights must be 1-D, got ndim={weights.ndim}")
    require(prices.ndim == 1, f"prices must be 1-D, got ndim={prices.ndim}")
    require(
        len(weights) == len(prices),
        f"weights ({len(weights)}) and prices ({len(prices)}) must have same length",
    )
    require(bool(np.all(weights >= 0)), "weights must be non-negative")
    require(bool(np.all(prices > 0)), "prices must be positive")
    require(total_value > 0, f"total_value must be positive, got {total_value}")

    n = len(weights)
    target_allocations = weights * total_value
    shares = np.zeros(n, dtype=np.intp)

    remaining_cash = total_value
    for _ in range(int(total_value / prices.min()) + 1):
        current_allocations = shares.astype(np.float64) * prices
        deficits = target_allocations - current_allocations

        affordable = prices <= remaining_cash
        if not np.any(affordable):
            break

        candidates = np.where(affordable, deficits, -np.inf)
        best = int(np.argmax(candidates))

        if candidates[best] <= 0 and not np.any((deficits > 0) & affordable):
            break

        shares[best] += 1
        remaining_cash -= prices[best]

    actual_values = shares.astype(np.float64) * prices
    invested = actual_values.sum()
    if invested > 0:
        actual_weights = actual_values / total_value
    else:
        actual_weights = np.zeros(n, dtype=np.float64)

    ensure(remaining_cash >= -1e-10, "leftover cash must be non-negative")

    return AllocationResult(
        shares=shares,
        leftover_cash=float(max(0.0, remaining_cash)),
        actual_weights=actual_weights,
    )


def lp_allocation(
    weights: NDArray[np.float64],
    prices: NDArray[np.float64],
    total_value: float,
) -> AllocationResult:
    """LP-based optimal integer allocation (minimise tracking error).

    Solves a linear program relaxation to find integer share counts
    that minimise the sum of absolute deviations between actual and
    target allocations. Falls back to greedy if scipy is unavailable.

    Parameters
    ----------
    weights : ndarray of shape (n,)
        Target portfolio weights (non-negative, sum to 1).
    prices : ndarray of shape (n,)
        Current share prices (positive).
    total_value : float
        Total portfolio value to allocate.

    Returns
    -------
    AllocationResult
        Integer share counts, leftover cash, and actual weights.
    """
    require(weights.ndim == 1, f"weights must be 1-D, got ndim={weights.ndim}")
    require(prices.ndim == 1, f"prices must be 1-D, got ndim={prices.ndim}")
    require(
        len(weights) == len(prices),
        f"weights ({len(weights)}) and prices ({len(prices)}) must have same length",
    )
    require(bool(np.all(weights >= 0)), "weights must be non-negative")
    require(bool(np.all(prices > 0)), "prices must be positive")
    require(total_value > 0, f"total_value must be positive, got {total_value}")

    try:
        from scipy.optimize import linprog
    except ImportError:
        return greedy_allocation(weights, prices, total_value)

    n = len(weights)
    target_values = weights * total_value
    max_shares = np.floor(total_value / prices).astype(np.intp)

    # Minimise sum of absolute deviations:
    # |shares_i * price_i - target_i| for each i
    # Introduce slack variables d_i >= |shares_i * price_i - target_i|
    # min sum(d_i) subject to:
    #   shares_i * price_i - target_i <= d_i  (positive deviation)
    #   target_i - shares_i * price_i <= d_i  (negative deviation)
    #   sum(shares_i * price_i) <= total_value (budget)
    #   0 <= shares_i <= max_shares_i (integrality relaxed to continuous)

    # Variables: [shares_0..shares_{n-1}, d_0..d_{n-1}]
    c = np.zeros(2 * n)
    c[n:] = 1.0  # minimise sum of d_i

    # Inequality constraints: A_ub @ x <= b_ub
    # For each i:
    #   price_i * shares_i - d_i <= target_i
    #   -price_i * shares_i - d_i <= -target_i
    A_ub = np.zeros((2 * n + 1, 2 * n))
    b_ub = np.zeros(2 * n + 1)

    for i in range(n):
        # shares_i * price_i - d_i <= target_i
        A_ub[i, i] = prices[i]
        A_ub[i, n + i] = -1.0
        b_ub[i] = target_values[i]

        # -shares_i * price_i - d_i <= -target_i
        A_ub[n + i, i] = -prices[i]
        A_ub[n + i, n + i] = -1.0
        b_ub[n + i] = -target_values[i]

    # Budget: sum(shares_i * price_i) <= total_value
    A_ub[2 * n, :n] = prices
    b_ub[2 * n] = total_value

    bounds = [(0, int(ms)) for ms in max_shares] + [(0, None)] * n

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not result.success:
        return greedy_allocation(weights, prices, total_value)

    shares = np.round(result.x[:n]).astype(np.intp)
    # Ensure budget not exceeded after rounding
    while (shares.astype(np.float64) * prices).sum() > total_value:
        excess = shares.astype(np.float64) * prices - target_values
        worst = int(np.argmax(excess))
        if shares[worst] > 0:
            shares[worst] -= 1
        else:
            break

    actual_values = shares.astype(np.float64) * prices
    leftover = total_value - actual_values.sum()
    if actual_values.sum() > 0:
        actual_weights = actual_values / total_value
    else:
        actual_weights = np.zeros(n, dtype=np.float64)

    ensure(leftover >= -1e-10, "leftover cash must be non-negative")

    return AllocationResult(
        shares=shares,
        leftover_cash=float(max(0.0, leftover)),
        actual_weights=actual_weights,
    )
