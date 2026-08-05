"""Rank-based portfolio construction.

Portfolios whose weights depend only on the *rank* of each stock,
not its identity.  These are the natural portfolios in SPT's
rank-based framework.

Mathematical References
-----------------------
- Top/bottom portfolios: F&K Survey §11
- Rank-weighted FGP: F&K Survey §11
- Leaking portfolios: F&K Survey Eq. 11.17–11.19
- Diversity-weighted rank portfolios: FKK Eq. 4.4
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "bottom_m_portfolio",
    "leaking_portfolio",
    "rank_weighted_portfolio",
    "top_m_portfolio",
]


def top_m_portfolio(
    mu: NDArray[np.float64],
    m: int,
) -> NDArray[np.float64]:
    r"""Equal-weighted portfolio of the m largest stocks.

    Assigns weight 1/m to each of the m stocks with the highest market
    weights, and zero to the rest.

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Market weights (positive, sum to 1).
    m : int
        Number of top stocks to include (1 ≤ m ≤ n).

    Returns
    -------
    ndarray of shape (n,)
        Portfolio weights (sum to 1).

    References
    ----------
    F&K Survey §11
    """
    n = len(mu)
    require(mu.ndim == 1, "mu must be 1-D")
    require(1 <= m <= n, f"m must be in [1, {n}], got {m}")

    order = np.argsort(-mu)
    pi = np.zeros(n, dtype=np.float64)
    pi[order[:m]] = 1.0 / m
    return pi


def bottom_m_portfolio(
    mu: NDArray[np.float64],
    m: int,
) -> NDArray[np.float64]:
    r"""Equal-weighted portfolio of the m smallest stocks.

    Assigns weight 1/m to each of the m stocks with the lowest market
    weights, and zero to the rest.

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Market weights (positive, sum to 1).
    m : int
        Number of bottom stocks to include (1 ≤ m ≤ n).

    Returns
    -------
    ndarray of shape (n,)
        Portfolio weights (sum to 1).

    References
    ----------
    F&K Survey §11
    """
    n = len(mu)
    require(mu.ndim == 1, "mu must be 1-D")
    require(1 <= m <= n, f"m must be in [1, {n}], got {m}")

    order = np.argsort(mu)
    pi = np.zeros(n, dtype=np.float64)
    pi[order[:m]] = 1.0 / m
    return pi


def rank_weighted_portfolio(
    mu: NDArray[np.float64],
    weight_func: Callable[[int, int], float],
) -> NDArray[np.float64]:
    r"""Portfolio with rank-determined weights.

    Each stock receives weight proportional to w(rank, n) where
    rank is the 0-indexed position (0 = largest).  Weights are
    normalised to sum to 1.

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Market weights (used only for determining ranks).
    weight_func : callable (rank: int, n: int) -> float
        Function mapping (rank, total_stocks) to a non-negative weight.
        Must return positive values for at least one rank.

    Returns
    -------
    ndarray of shape (n,)
        Portfolio weights (sum to 1).

    References
    ----------
    F&K Survey §11
    """
    n = len(mu)
    require(mu.ndim == 1, "mu must be 1-D")
    require(n >= 2, f"Need ≥ 2 stocks, got {n}")

    order = np.argsort(-mu)
    ranks = np.empty(n, dtype=np.intp)
    ranks[order] = np.arange(n)

    raw_weights = np.array([weight_func(int(ranks[i]), n) for i in range(n)])
    require(
        bool(np.all(raw_weights >= 0)),
        "weight_func must return non-negative values",
    )
    total = float(np.sum(raw_weights))
    require(total > 0, "weight_func must be positive for at least one rank")
    return raw_weights / total


def leaking_portfolio(
    mu: NDArray[np.float64],
    m: int,
    p: float,
) -> NDArray[np.float64]:
    r"""Diversity-weighted portfolio restricted to top m stocks.

    Applies p-diversity weighting to only the m largest stocks:

    .. math::
        \pi_i = \frac{\mu_i^p}{\sum_{j \in \text{top-}m} \mu_j^p}
        \quad \text{for } i \in \text{top-}m,
        \qquad \pi_i = 0 \text{ otherwise}

    This portfolio "leaks" value at rank boundaries (positions m and m+1)
    because stocks crossing the boundary trigger portfolio rebalancing.

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Market weights (positive, sum to 1).
    m : int
        Number of top stocks to include (1 ≤ m ≤ n).
    p : float
        Diversity parameter p ∈ (0, 1).  Smaller p tilts more toward
        equal weighting among the top m.

    Returns
    -------
    ndarray of shape (n,)
        Portfolio weights (sum to 1).

    References
    ----------
    F&K Survey Eq. 11.17–11.19
    """
    n = len(mu)
    require(mu.ndim == 1, "mu must be 1-D")
    require(1 <= m <= n, f"m must be in [1, {n}], got {m}")
    require(0 < p < 1, f"p must be in (0, 1), got {p}")
    require(bool(np.all(mu > 0)), "All weights must be positive")

    order = np.argsort(-mu)
    top_indices = order[:m]

    pi = np.zeros(n, dtype=np.float64)
    mu_top_p = mu[top_indices] ** p
    pi[top_indices] = mu_top_p / float(np.sum(mu_top_p))
    return pi
