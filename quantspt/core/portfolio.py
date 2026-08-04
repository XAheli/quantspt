"""Portfolio value process and relative returns.

This module implements the portfolio value process V^π, its log-return
decomposition into growth rate and martingale components, relative
return dynamics V^π / V^μ, and turnover computation.

Mathematical References
-----------------------
- Portfolio value SDE: F&K Survey Eq. 1.9
- Log-return decomposition: F&K Survey Eq. 1.12
- Relative return dynamics: F&K Survey Eq. 3.4
- Rebalancing / turnover: F&K Survey §1.3
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require

__all__ = [
    "portfolio_log_return",
    "portfolio_value_weights",
    "relative_return",
    "log_relative_return",
    "drift_of_relative_return",
    "rebalancing_turnover",
    "cumulative_turnover",
    "holding_drift",
]


# ---------------------------------------------------------------------------
# Portfolio value process  (F&K Survey Eq. 1.9)
# ---------------------------------------------------------------------------


def portfolio_log_return(
    pi: NDArray[np.float64],
    log_returns: NDArray[np.float64],
) -> float:
    r"""Single-period log-return of a portfolio (first-order approximation).

    For small Δt the discrete analogue of the value process is:

    .. math::
        \log\frac{V^{\pi}(t+\Delta t)}{V^{\pi}(t)}
        \approx \sum_i \pi_i\,r_i

    where r_i = log(S_i(t+Δt) / S_i(t)) are individual log-returns.

    Parameters
    ----------
    pi : ndarray of shape (n,)
        Portfolio weights at start of period.
    log_returns : ndarray of shape (n,)
        Per-stock log-returns over the period.

    Returns
    -------
    float
        Portfolio log-return for this period.

    Notes
    -----
    This is the *first-order* approximation (Itô correction is
    second-order in dt).  For the continuous-time version, see
    :func:`~quantspt.core.growth_rates.portfolio_growth_rate`.

    References
    ----------
    F&K Survey Eq. 1.9 (discrete analogue)
    """
    return float(np.dot(pi, log_returns))


def portfolio_value_weights(
    pi: NDArray[np.float64],
    simple_returns: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Compute portfolio weights after one period of buy-and-hold.

    If the portfolio starts at weights π and stocks realise simple
    returns r_i = S_i(t+Δt)/S_i(t) − 1, the weights drift to:

    .. math::
        \tilde{\pi}_i = \frac{\pi_i (1 + r_i)}{\sum_j \pi_j (1 + r_j)}

    The difference π − π̃ drives rebalancing trades.

    Parameters
    ----------
    pi : ndarray of shape (n,)
        Portfolio weights at the start of the period.
    simple_returns : ndarray of shape (n,)
        Per-stock simple returns over the period.

    Returns
    -------
    ndarray of shape (n,)
        Drifted weights before rebalancing.

    References
    ----------
    F&K Survey §1.3
    """
    values = pi * (1.0 + simple_returns)
    total = float(np.sum(values))
    require(total > 0, "Portfolio value became non-positive")
    return values / total


# ---------------------------------------------------------------------------
# Relative returns  (F&K Survey Eq. 3.4)
# ---------------------------------------------------------------------------


def relative_return(
    V_pi: NDArray[np.float64],
    V_mu: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Compute the relative return path V^π / V^μ.

    Parameters
    ----------
    V_pi : ndarray of shape (T,)
        Portfolio value path.
    V_mu : ndarray of shape (T,)
        Market (benchmark) value path.

    Returns
    -------
    ndarray of shape (T,)
        Relative value process. Starts at V_pi[0] / V_mu[0].

    References
    ----------
    F&K Survey §3
    """
    require(len(V_pi) == len(V_mu), "Value paths must have equal length")
    require(bool(np.all(V_mu > 0)), "Benchmark values must be positive")
    return V_pi / V_mu


def log_relative_return(
    V_pi: NDArray[np.float64],
    V_mu: NDArray[np.float64],
) -> float:
    r"""Compute log(V^π(T) / V^μ(T)) from value paths.

    Parameters
    ----------
    V_pi : ndarray of shape (T,)
        Portfolio value path.
    V_mu : ndarray of shape (T,)
        Market value path.

    Returns
    -------
    float
        Terminal log-relative return.

    References
    ----------
    F&K Survey Eq. 3.4 (integrated form)
    """
    require(V_pi[-1] > 0, "Terminal portfolio value must be positive")
    require(V_mu[-1] > 0, "Terminal benchmark value must be positive")
    return np.log(V_pi[-1] / V_mu[-1])


def drift_of_relative_return(
    pi: NDArray[np.float64],
    mu: NDArray[np.float64],
    gamma: NDArray[np.float64],
    a: NDArray[np.float64],
) -> float:
    r"""Instantaneous drift of log(V^π / V^μ).

    .. math::
        d\log\frac{V^{\pi}}{V^{\mu}}
        = \bigl[\gamma^*_\pi - \gamma^*_\mu
          + \sum_i (\pi_i - \mu_i)(\gamma_i - \bar\gamma_\mu)\bigr]\,dt
          + \text{martingale}

    This function returns the drift coefficient.

    Parameters
    ----------
    pi : ndarray of shape (n,)
        Portfolio weights.
    mu : ndarray of shape (n,)
        Market weights.
    gamma : ndarray of shape (n,)
        Individual stock growth rates γ_i.
    a : ndarray of shape (n, n)
        Covariance rate matrix.

    Returns
    -------
    float
        Instantaneous drift of the log-relative process.

    References
    ----------
    F&K Survey Eq. 3.4
    """
    from .growth_rates import relative_performance_rate

    return relative_performance_rate(pi, mu, gamma, a)


# ---------------------------------------------------------------------------
# Holding / buy-and-hold drift
# ---------------------------------------------------------------------------


def holding_drift(
    pi: NDArray[np.float64],
    mu: NDArray[np.float64],
    a: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Natural drift of portfolio weights under buy-and-hold.

    Without rebalancing, portfolio weights evolve like market weights:

    .. math::
        d\tilde\pi_i \approx \tilde\pi_i \bigl[
            r_i - \sum_j \tilde\pi_j r_j
        \bigr]

    This is the infinitesimal analogue of :func:`portfolio_value_weights`
    drift.  We approximate it via the covariance structure.

    The drift of π_i under buy-and-hold is:

    .. math::
        \tilde\pi_i (a^{\pi}_i - a_{\pi\pi})

    where a^π_i = Σ_j π_j a_{ij} and a_{ππ} = π' a π.  (Only the
    volatility part matters for the diffusion-driven drift of
    weights on the simplex.)

    Parameters
    ----------
    pi : ndarray of shape (n,)
        Current portfolio weights.
    mu : ndarray of shape (n,)
        Market weights (not used in this pure diffusion calculation,
        but kept for API consistency).
    a : ndarray of shape (n, n)
        Covariance rate matrix.

    Returns
    -------
    ndarray of shape (n,)
        Drift vector on the simplex due to buy-and-hold.

    References
    ----------
    F&K Survey §1.3
    """
    a_pi = a @ pi
    a_pipi = float(pi @ a_pi)
    return pi * (a_pi - a_pipi)


# ---------------------------------------------------------------------------
# Rebalancing & turnover  (F&K Survey §1.3)
# ---------------------------------------------------------------------------


def rebalancing_turnover(
    pi_target: NDArray[np.float64],
    pi_current: NDArray[np.float64],
) -> float:
    r"""Compute single-period turnover.

    .. math::
        \text{turnover} = \frac{1}{2} \sum_i |\pi^{\text{target}}_i
                          - \pi^{\text{current}}_i|

    This is the fraction of portfolio value that must be traded to
    rebalance from current to target weights.

    Parameters
    ----------
    pi_target : ndarray of shape (n,)
        Desired portfolio weights after rebalancing.
    pi_current : ndarray of shape (n,)
        Actual portfolio weights before rebalancing.

    Returns
    -------
    float
        One-way turnover in [0, 1].

    References
    ----------
    F&K Survey §1.3
    """
    return 0.5 * float(np.sum(np.abs(pi_target - pi_current)))


def cumulative_turnover(
    weight_path: NDArray[np.float64],
    simple_returns: NDArray[np.float64] | None = None,
) -> float:
    r"""Compute cumulative turnover along a rebalancing path.

    If ``simple_returns`` is provided, weights are first drifted via
    buy-and-hold before comparing to the next period's target.  Otherwise,
    consecutive rows of ``weight_path`` are compared directly.

    Parameters
    ----------
    weight_path : ndarray of shape (T, n)
        Portfolio weight targets at each rebalancing date.
    simple_returns : ndarray of shape (T-1, n), optional
        Per-stock simple returns between rebalancing dates.

    Returns
    -------
    float
        Total one-way turnover over all periods.
    """
    T = weight_path.shape[0]
    require(T >= 2, "Need at least 2 periods for turnover")

    total = 0.0
    for t in range(T - 1):
        if simple_returns is not None:
            drifted = portfolio_value_weights(weight_path[t], simple_returns[t])
        else:
            drifted = weight_path[t]
        total += rebalancing_turnover(weight_path[t + 1], drifted)

    return total
