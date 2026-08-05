"""Master formula performance attribution for backtested strategies.

Decomposes backtest log-relative return into the boundary term and drift
integral from the master formula, enabling verification that theoretical
SPT predictions hold in discrete simulation.

Mathematical References
-----------------------
- Master formula: F&K Survey Eq. 11.2
    log(V^π(T) / V^μ(T)) = log(G(μ(T))/G(μ(0))) + ∫₀ᵀ g(t) dt
- Drift process: F&K Survey Eq. 11.3
- Boundary term: log(G(μ_T)/G(μ_0))
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require
from ..core.covariance import relative_covariance
from ..core.generating_functions import GeneratingFunction

__all__ = [
    "AttributionResult",
    "compute_attribution",
]


@dataclass(frozen=True)
class AttributionResult:
    """Master formula attribution over a backtest period.

    Attributes
    ----------
    boundary : float
        Boundary term: log(G(μ(T)) / G(μ(0))).
    drift_integral : float
        Drift integral: ∫₀ᵀ g(t) dt.
    predicted_log_relative : float
        Predicted log-relative return: boundary + drift_integral.
    actual_log_relative : float
        Observed log(V^π(T) / V^μ(T)) from the backtest.
    residual : float
        Actual - predicted. Captures discretization error and
        transaction costs.
    boundary_series : NDArray[np.float64]
        Cumulative boundary term at each time step.
    drift_series : NDArray[np.float64]
        Cumulative drift integral at each time step.

    References
    ----------
    F&K Survey Eq. 11.2
    """

    boundary: float
    drift_integral: float
    predicted_log_relative: float
    actual_log_relative: float
    residual: float
    boundary_series: NDArray[np.float64]
    drift_series: NDArray[np.float64]


def compute_attribution(
    G: GeneratingFunction,
    market_weights: NDArray[np.float64],
    covariance_path: NDArray[np.float64],
    actual_log_relative: float,
    dt: float,
) -> AttributionResult:
    r"""Decompose backtest performance using the master formula.

    Computes the boundary term and drift integral from the master formula:

    .. math::
        \log\frac{V^{\pi}(T)}{V^{\mu}(T)}
        = \underbrace{\log\frac{G(\mu(T))}{G(\mu(0))}}_{\text{boundary}}
        + \underbrace{\int_0^T g(t)\,dt}_{\text{drift}}

    The residual (actual - predicted) captures effects not in the
    continuous-time formula: discretization error, transaction costs,
    and rebalancing frequency effects.

    Parameters
    ----------
    G : GeneratingFunction
        The generating function used for the strategy.
    market_weights : ndarray of shape (T+1, n)
        Market weight time series.
    covariance_path : ndarray of shape (T+1, n, n) or (n, n)
        Covariance rate matrices. If 2-D, treated as constant.
    actual_log_relative : float
        Observed log(V^π(T) / V^μ(T)).
    dt : float
        Time step in years.

    Returns
    -------
    AttributionResult
        Full attribution decomposition.

    References
    ----------
    F&K Survey Eq. 11.2, 11.3
    """
    T_plus_1 = len(market_weights)
    require(T_plus_1 >= 2, "Need at least 2 time steps for attribution")
    n = market_weights.shape[1]

    if covariance_path.ndim == 2:
        require(
            covariance_path.shape == (n, n),
            "Constant covariance must be (n, n)",
        )
        a_path = np.tile(covariance_path, (T_plus_1, 1, 1))
    else:
        require(
            covariance_path.shape[0] == T_plus_1,
            "covariance_path length must match market_weights",
        )
        a_path = covariance_path

    G_0 = G(market_weights[0])
    require(G_0 > 0, "G(μ(0)) must be positive")

    boundary_series = np.zeros(T_plus_1)
    drift_series = np.zeros(T_plus_1)

    cumulative_drift = 0.0
    for t in range(T_plus_1):
        mu_t = market_weights[t]
        G_t = G(mu_t)
        boundary_series[t] = float(np.log(G_t / G_0))

        if t > 0:
            tau_mu = relative_covariance(a_path[t], mu_t)
            g_t = G.drift(mu_t, tau_mu)
            cumulative_drift += g_t * dt
        drift_series[t] = cumulative_drift

    boundary = boundary_series[-1]
    drift_int = drift_series[-1]
    predicted = boundary + drift_int
    residual = actual_log_relative - predicted

    return AttributionResult(
        boundary=boundary,
        drift_integral=drift_int,
        predicted_log_relative=predicted,
        actual_log_relative=actual_log_relative,
        residual=residual,
        boundary_series=boundary_series,
        drift_series=drift_series,
    )
