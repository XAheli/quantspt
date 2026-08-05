"""Local time estimation for ranked capitalisation processes.

The local time Λ_{k,k+1}(t) measures the accumulated time that adjacent
ranked capitalisations Z_{(k)} and Z_{(k+1)} spend at the collision
boundary.  It drives the rank-switching dynamics in first-order models.

Mathematical References
-----------------------
- Local time definition: BFK Eq. 3.5
- Asymptotic rate formula: BFK Eq. 3.7
- Relation to Pareto exponents: BFK Eq. 4.3
- Empirical estimation via Tanaka formula: BFK §3
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require

__all__ = [
    "empirical_local_time_rates",
    "empirical_local_times",
    "local_time_rates_analytical",
    "local_time_rates_atlas",
]


def local_time_rates_analytical(
    g: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Asymptotic local time accumulation rates from model parameters.

    For a first-order model with rank-dependent increments g_1, …, g_n
    satisfying the stability condition, the local time rate between
    ranks k and k+1 is:

    .. math::
        \lambda_{k,k+1} = -2\,(g_1 + \cdots + g_k), \quad k = 1, \dots, n-1

    (using 1-indexed ranks).  The stability condition g_1+…+g_k < 0 for
    k < n guarantees all rates are positive.

    Parameters
    ----------
    g : ndarray of shape (n,)
        Rank-dependent growth-rate increments (0-indexed: g[0] is the
        largest-stock increment).  Must satisfy the stability condition.

    Returns
    -------
    ndarray of shape (n−1,)
        Local time rates λ_{k,k+1} for adjacent rank pairs.

    References
    ----------
    BFK Eq. 3.7
    """
    require(len(g) >= 2, f"Need ≥ 2 stocks, got {len(g)}")
    cumsum = np.cumsum(g)
    require(
        bool(np.all(cumsum[:-1] < 0)),
        "Stability condition violated: partial sums g[0]+...+g[k] must be < 0 "
        "for k < n-1",
    )
    return -2.0 * cumsum[:-1]


def local_time_rates_atlas(
    n: int,
    g_param: float,
) -> NDArray[np.float64]:
    r"""Local time rates for the basic Atlas model.

    In the basic Atlas model with g_k = −g for k < n and g_n = (n−1)g:

    .. math::
        \lambda_{k,k+1} = 2\,k\,g, \quad k = 1, \dots, n-1

    Parameters
    ----------
    n : int
        Number of stocks (≥ 2).
    g_param : float
        Atlas growth parameter g > 0.

    Returns
    -------
    ndarray of shape (n−1,)
        Local time rates.

    References
    ----------
    BFK Eq. 3.7 applied to the basic Atlas (BFK Eq. 1.7)
    """
    require(n >= 2, f"Need ≥ 2 stocks, got {n}")
    require(g_param > 0, f"g_param must be positive, got {g_param}")
    return 2.0 * g_param * np.arange(1, n, dtype=np.float64)


def empirical_local_times(
    log_caps_path: NDArray[np.float64],
    dt: float = 1.0,
    epsilon: float | None = None,
) -> NDArray[np.float64]:
    r"""Estimate cumulative local times from discrete observations.

    Uses a Tanaka-formula discretisation: the local time Λ_{k,k+1}
    is approximated by counting time steps where the gap between adjacent
    ranked log-caps falls below a threshold ε:

    .. math::
        \Lambda_{k,k+1}(T) \approx
        \sum_t \mathbf{1}\{Z_{(k)}(t) - Z_{(k+1)}(t) < \varepsilon\}
        \cdot \frac{\Delta t}{\varepsilon}

    Parameters
    ----------
    log_caps_path : ndarray of shape (T, n)
        Log-capitalisations over T time steps.
    dt : float
        Time increment between observations.
    epsilon : float, optional
        Proximity threshold.  If ``None``, set to 10% of the standard
        deviation of adjacent ranked gaps.

    Returns
    -------
    ndarray of shape (n−1,)
        Estimated cumulative local times Λ_{k,k+1}(T).

    References
    ----------
    BFK §3 (Tanaka formula discretisation)
    """
    require(log_caps_path.ndim == 2, "log_caps_path must be 2-D (T, n)")
    T, n = log_caps_path.shape
    require(n >= 2, f"Need ≥ 2 stocks, got {n}")
    require(T >= 2, f"Need ≥ 2 time steps, got {T}")
    require(dt > 0, f"dt must be positive, got {dt}")

    sorted_caps = np.sort(log_caps_path, axis=1)[:, ::-1]
    gaps = np.diff(sorted_caps, axis=1)  # (T, n-1), non-positive
    gaps = -gaps  # make non-negative

    if epsilon is None:
        gap_std = float(np.std(gaps))
        epsilon = max(gap_std * 0.1, 1e-10)

    require(epsilon > 0, f"epsilon must be positive, got {epsilon}")

    local_times = np.zeros(n - 1)
    for k in range(n - 1):
        local_times[k] = float(np.sum(gaps[:, k] < epsilon)) * dt / epsilon

    return local_times


def empirical_local_time_rates(
    log_caps_path: NDArray[np.float64],
    dt: float = 1.0,
    epsilon: float | None = None,
) -> NDArray[np.float64]:
    r"""Estimate local time accumulation rates (per unit time).

    Returns Λ_{k,k+1}(T) / T, the rate at which local time accumulates
    between adjacent ranks.

    Parameters
    ----------
    log_caps_path : ndarray of shape (T, n)
        Log-capitalisations over T time steps.
    dt : float
        Time increment between observations.
    epsilon : float, optional
        Proximity threshold (see :func:`empirical_local_times`).

    Returns
    -------
    ndarray of shape (n−1,)
        Estimated local time rates λ_{k,k+1}.

    References
    ----------
    BFK Eq. 3.7
    """
    T = log_caps_path.shape[0]
    total_time = (T - 1) * dt
    require(total_time > 0, "Total time must be positive")
    lt = empirical_local_times(log_caps_path, dt=dt, epsilon=epsilon)
    return lt / total_time
