"""Capital distribution curve analysis and Pareto fitting.

The capital distribution curve — ranked market weights plotted against
rank — reveals the structural properties of equity markets.  In the
Atlas model the curve follows a Pareto/Zipf power law in steady state.

Mathematical References
-----------------------
- Capital distribution curve: BFK §4, F&K Survey §2
- Pareto exponents: BFK Eq. 4.3–4.4
- Log-log structure: BFK Figures 4.1, 4.2
- Stability of the distribution: BFK §4, §6.4
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require

__all__ = [
    "capital_distribution_curve",
    "capital_distribution_stability",
    "log_log_fit",
    "pareto_exponents_empirical",
]


def capital_distribution_curve(
    mu: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Compute the capital distribution curve (ranked market weights).

    Returns μ_{(1)} ≥ μ_{(2)} ≥ … ≥ μ_{(n)}.

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Market weights (positive, sum to 1).

    Returns
    -------
    ndarray of shape (n,)
        Ranked weights in descending order.

    References
    ----------
    BFK §4, F&K Survey §2
    """
    require(mu.ndim == 1, "mu must be 1-D")
    require(len(mu) >= 2, f"Need ≥ 2 stocks, got {len(mu)}")
    return np.sort(mu)[::-1].copy()


def log_log_fit(
    mu: NDArray[np.float64],
) -> tuple[float, float, float]:
    r"""Fit a linear model to the log-log capital distribution.

    Fits log(μ_{(k)}) = α + β · log(k) via ordinary least squares
    on ranks k = 1, …, n.  A Zipf/Pareto law corresponds to β ≈ −α_Z
    where α_Z is the Zipf exponent.

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Market weights (positive, sum to 1).

    Returns
    -------
    tuple of (slope, intercept, r_squared)
        slope : float
            Log-log slope β (negative indicates power-law decay).
        intercept : float
            Log-log intercept α.
        r_squared : float
            Coefficient of determination R².

    References
    ----------
    BFK §4 (Figures 4.1, 4.2)
    """
    require(mu.ndim == 1, "mu must be 1-D")
    n = len(mu)
    require(n >= 3, "Need ≥ 3 stocks for meaningful log-log fit")
    require(bool(np.all(mu > 0)), "All weights must be positive")

    ranked = np.sort(mu)[::-1]
    log_rank = np.log(np.arange(1, n + 1, dtype=np.float64))
    log_weight = np.log(ranked)

    x_mean = float(np.mean(log_rank))
    y_mean = float(np.mean(log_weight))
    ss_xx = float(np.sum((log_rank - x_mean) ** 2))
    ss_xy = float(np.sum((log_rank - x_mean) * (log_weight - y_mean)))
    ss_yy = float(np.sum((log_weight - y_mean) ** 2))

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    r_squared = ss_xy**2 / (ss_xx * ss_yy) if ss_yy > 0 else 0.0

    return slope, intercept, r_squared


def pareto_exponents_empirical(
    mu: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Estimate Pareto exponents from successive ranked weight ratios.

    In the Atlas model at stationarity (BFK Eq. 4.4):

    .. math::
        P[\mu_{(k)} / \mu_{(k+1)} > y] \to y^{-r_k}

    This function estimates r_k from the log-ratio of successive ranked
    weights via the moment estimator:

    .. math::
        \hat{r}_k = 1 / \log(\mu_{(k)} / \mu_{(k+1)})

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Market weights (positive, sum to 1).

    Returns
    -------
    ndarray of shape (n−1,)
        Estimated Pareto exponents for each adjacent rank pair.
        Entries are ``inf`` when successive weights are equal.

    References
    ----------
    BFK Eq. 4.3–4.4
    """
    require(mu.ndim == 1, "mu must be 1-D")
    require(bool(np.all(mu > 0)), "All weights must be positive")
    ranked = np.sort(mu)[::-1]
    log_ratios = np.log(ranked[:-1] / ranked[1:])
    nonzero = log_ratios > 0
    exponents = np.full_like(log_ratios, np.inf)
    exponents[nonzero] = 1.0 / log_ratios[nonzero]
    return exponents


def capital_distribution_stability(
    weight_paths: NDArray[np.float64],
) -> float:
    r"""Measure stability of the capital distribution over time.

    Computes the mean L² distance between the ranked weight vector at
    each time step and the time-averaged ranked weight vector.  Low
    values indicate a stable capital distribution.

    Parameters
    ----------
    weight_paths : ndarray of shape (T, n)
        Market weight paths over T time steps.

    Returns
    -------
    float
        Mean L² deviation of the ranked capital distribution from its
        time average.  Lower is more stable.

    References
    ----------
    BFK §4, §6.4
    """
    require(weight_paths.ndim == 2, "weight_paths must be 2-D (T, n)")
    T, n = weight_paths.shape
    require(T >= 2, f"Need ≥ 2 time steps, got {T}")
    require(n >= 2, f"Need ≥ 2 stocks, got {n}")

    ranked = np.sort(weight_paths, axis=1)[:, ::-1]
    mean_ranked = np.mean(ranked, axis=0)
    deviations = ranked - mean_ranked[np.newaxis, :]
    l2_norms = np.sqrt(np.sum(deviations**2, axis=1))
    return float(np.mean(l2_norms))
