"""Ranked capitalisation dynamics for rank-based market models.

The ranked capitalisation processes Z_{(k)}(t) = log X_{(k)}(t) track the
log-cap of the stock currently at rank k.  In first-order models the
dynamics depend only on rank, producing:

    Z_k(t) = Z_k(0) + (g_k + γ)t + σ_k B_k(t)
             + ½[Λ_{k,k+1}(t) − Λ_{k−1,k}(t)]

where Λ_{k,k+1}(t) is the local time at the collision boundary Z_k = Z_{k+1}.

Mathematical References
-----------------------
- Ranked capitalisation definition: BFK Eq. 3.1
- Ranked dynamics with local times: BFK Eq. 3.3
- Rank assignment from market weights: F&K Survey Eq. 1.18
- Ranked weight dynamics: FKK Eq. 5.7
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require

__all__ = [
    "rank_assignment",
    "ranked_capitalizations",
    "ranked_capitalizations_path",
    "ranked_drift_coefficients",
    "ranked_weights_from_path",
]


def ranked_capitalizations(
    log_caps: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Sort log-capitalisations in descending order.

    Returns Z_{(1)} ≥ Z_{(2)} ≥ … ≥ Z_{(n)}, the ranked log-caps
    at a single time instant.

    Parameters
    ----------
    log_caps : ndarray of shape (n,)
        Log-capitalisations of each stock.

    Returns
    -------
    ndarray of shape (n,)
        Ranked (sorted descending) log-capitalisations.

    References
    ----------
    BFK Eq. 3.1
    """
    require(log_caps.ndim == 1, "log_caps must be 1-D")
    require(len(log_caps) >= 2, f"Need ≥ 2 stocks, got {len(log_caps)}")
    return np.sort(log_caps)[::-1].copy()


def ranked_capitalizations_path(
    log_caps_path: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Sort log-capitalisations in descending order at each time step.

    Parameters
    ----------
    log_caps_path : ndarray of shape (T, n)
        Log-capitalisations over T time steps for n stocks.

    Returns
    -------
    ndarray of shape (T, n)
        Ranked log-capitalisations per time step.

    References
    ----------
    BFK Eq. 3.1
    """
    require(log_caps_path.ndim == 2, "log_caps_path must be 2-D (T, n)")
    require(
        log_caps_path.shape[1] >= 2,
        f"Need ≥ 2 stocks, got {log_caps_path.shape[1]}",
    )
    return np.sort(log_caps_path, axis=1)[:, ::-1].copy()


def rank_assignment(
    values: NDArray[np.float64],
) -> NDArray[np.intp]:
    r"""Assign 0-indexed ranks to stocks (0 = largest).

    Given values v_1, …, v_n, stock i receives rank r_i such that
    v_{p(0)} ≥ v_{p(1)} ≥ … ≥ v_{p(n−1)} where p is the permutation
    induced by sorting.

    Parameters
    ----------
    values : ndarray of shape (n,)
        Stock values (capitalisations, weights, or log-caps).

    Returns
    -------
    ndarray of shape (n,) dtype intp
        Rank of each stock. ``ranks[i] = 0`` means stock i is the largest.

    References
    ----------
    F&K Survey Eq. 1.18
    """
    require(values.ndim == 1, "values must be 1-D")
    require(len(values) >= 2, f"Need ≥ 2 stocks, got {len(values)}")
    order = np.argsort(-values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(values))
    return ranks


def ranked_drift_coefficients(
    g: NDArray[np.float64],
    gamma: float,
) -> NDArray[np.float64]:
    r"""Drift coefficients for ranked capitalisation dynamics.

    In the first-order model (BFK Eq. 3.3), the drift of the k-th
    ranked log-capitalisation is:

    .. math::
        \text{drift}_k = \gamma + g_k

    where γ is the common drift and g_k is the rank-dependent increment.

    Parameters
    ----------
    g : ndarray of shape (n,)
        Rank-dependent growth-rate increments, indexed by rank
        (g[0] for rank 0 = largest, g[n−1] for smallest).
    gamma : float
        Common drift parameter.

    Returns
    -------
    ndarray of shape (n,)
        Drift coefficient for each rank position.

    References
    ----------
    BFK Eq. 3.3
    """
    return gamma + g


def ranked_weights_from_path(
    log_caps_path: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Compute ranked market weights from log-capitalisation paths.

    At each time step, converts log-caps to weights on the simplex and
    sorts them in descending order.

    Parameters
    ----------
    log_caps_path : ndarray of shape (T, n)
        Log-capitalisations over T time steps.

    Returns
    -------
    ndarray of shape (T, n)
        Ranked market weights μ_{(1)}(t) ≥ … ≥ μ_{(n)}(t) for each t.

    References
    ----------
    F&K Survey Eq. 1.18, BFK §4
    """
    require(log_caps_path.ndim == 2, "log_caps_path must be 2-D (T, n)")
    shifted = log_caps_path - np.max(log_caps_path, axis=1, keepdims=True)
    caps = np.exp(shifted)
    weights = caps / np.sum(caps, axis=1, keepdims=True)
    return np.sort(weights, axis=1)[:, ::-1].copy()
