"""Rank transition analysis for equity markets.

Rank transitions — how stocks move between rank positions over time —
reveal the dynamic structure of capital distribution and are central
to understanding the local time accumulation that drives rank-based
portfolio performance.

Mathematical References
-----------------------
- Rank dynamics and local times: BFK §3
- Ergodic property (uniform occupation): BFK Prop. 2.3
- Rank stability and capital distribution: BFK §4
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require

__all__ = [
    "expected_sojourn_times",
    "rank_mobility_index",
    "rank_transition_matrix",
]


def rank_transition_matrix(
    weight_paths: NDArray[np.float64],
    horizon: int = 1,
) -> NDArray[np.float64]:
    r"""Compute the rank transition matrix over a given horizon.

    Entry P_{ij} is the empirical probability that a stock at rank i
    moves to rank j after ``horizon`` time steps.  Ranks are 0-indexed
    (0 = largest).

    Parameters
    ----------
    weight_paths : ndarray of shape (T, n)
        Market weight paths over T time steps for n stocks.
    horizon : int
        Number of time steps over which to measure transitions.

    Returns
    -------
    ndarray of shape (n, n)
        Row-stochastic transition matrix: P[i, j] = Prob(rank j at t+h | rank i at t).

    References
    ----------
    BFK §3, Prop. 2.3
    """
    require(weight_paths.ndim == 2, "weight_paths must be 2-D (T, n)")
    T, n = weight_paths.shape
    require(horizon < T, f"Need > {horizon} time steps, got {T}")
    require(horizon >= 1, f"horizon must be ≥ 1, got {horizon}")
    require(n >= 2, f"Need ≥ 2 stocks, got {n}")

    counts = np.zeros((n, n), dtype=np.float64)

    for t in range(T - horizon):
        ranks_t = np.argsort(np.argsort(-weight_paths[t]))
        ranks_th = np.argsort(np.argsort(-weight_paths[t + horizon]))
        for stock in range(n):
            counts[ranks_t[stock], ranks_th[stock]] += 1.0

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return counts / row_sums


def expected_sojourn_times(
    transition_matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Expected sojourn time at each rank position.

    The expected number of periods a stock stays at rank k before
    leaving is 1 / (1 − P_{kk}).

    Parameters
    ----------
    transition_matrix : ndarray of shape (n, n)
        Row-stochastic transition matrix from :func:`rank_transition_matrix`.

    Returns
    -------
    ndarray of shape (n,)
        Expected sojourn time at each rank.  ``inf`` if P_{kk} = 1.

    References
    ----------
    BFK §3
    """
    require(transition_matrix.ndim == 2, "transition_matrix must be 2-D")
    n = transition_matrix.shape[0]
    require(
        transition_matrix.shape == (n, n),
        f"transition_matrix must be square, got {transition_matrix.shape}",
    )

    diag = np.diag(transition_matrix)
    leaving_prob = 1.0 - diag
    return np.where(leaving_prob > 0, 1.0 / leaving_prob, np.inf)


def rank_mobility_index(
    transition_matrix: NDArray[np.float64],
) -> float:
    r"""Compute the rank mobility index.

    Measures the average probability of a rank change per period:

    .. math::
        M = 1 - \frac{1}{n} \sum_{k=1}^n P_{kk}

    The index is 0 when no stock ever changes rank (perfectly static)
    and approaches 1 − 1/n for maximal mixing.

    Parameters
    ----------
    transition_matrix : ndarray of shape (n, n)
        Row-stochastic transition matrix.

    Returns
    -------
    float
        Mobility index in [0, 1 − 1/n].

    References
    ----------
    BFK Prop. 2.3 (ergodic property implies M → 1 − 1/n)
    """
    require(transition_matrix.ndim == 2, "transition_matrix must be 2-D")
    return 1.0 - float(np.mean(np.diag(transition_matrix)))
