"""Excess growth rates and portfolio growth rate decomposition.

The excess growth rate γ*_π is THE fundamental quantity of Stochastic
Portfolio Theory. It measures the "diversification return" that portfolios
earn purely from rebalancing and volatility, independent of any return
forecasts.

Mathematical References
-----------------------
- Excess growth rate γ*_π: F&K Survey Eq. 1.13, FKK Eq. 2.8
- Portfolio growth rate γ_π: F&K Survey Eq. 1.12, BFK Eq. 5.2-5.4
- Relative performance drift: F&K Survey Eq. 3.4
- Atlas model special cases: BFK Eq. 5.3, 5.9, 5.14
- Bounds on γ*_π: FKK Eq. 5.12
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require
from .covariance import portfolio_variance

__all__ = [
    "atlas_excess_growth_rate_equal_weighted",
    "atlas_excess_growth_rate_uncorrelated",
    "atlas_market_growth_rate",
    "excess_growth_rate",
    "excess_growth_rate_bounds",
    "excess_growth_rate_from_tau",
    "portfolio_growth_rate",
    "relative_performance_rate",
]


def excess_growth_rate(
    pi: NDArray[np.float64],
    a: NDArray[np.float64],
) -> float:
    r"""Compute excess growth rate γ*_π(t).

    .. math::
        \gamma^*_{\pi}(t) = \frac{1}{2}\left[
            \sum_i \pi_i a_{ii} - \pi^T a \pi
        \right]

    This is the "diversification return" — the amount by which a portfolio's
    growth rate exceeds the weighted average of individual stock growth rates.
    It is always non-negative for long-only portfolios and equals zero only
    for single-stock portfolios.

    Parameters
    ----------
    pi : ndarray of shape (n,)
        Portfolio weights summing to 1. Must be non-negative for the
        non-negativity guarantee to hold.
    a : ndarray of shape (n, n)
        Instantaneous covariance rate matrix.

    Returns
    -------
    float
        Excess growth rate. Non-negative for long-only portfolios.

    References
    ----------
    F&K Survey Eq. 1.13, FKK Eq. 2.8

    Examples
    --------
    Equal-weighted 2-stock portfolio with uncorrelated σ² = 0.04:

    >>> excess_growth_rate(np.array([0.5, 0.5]), np.diag([0.04, 0.04]))
    0.01

    Concentrated portfolio always has γ* = 0:

    >>> excess_growth_rate(np.array([1.0, 0.0]), np.diag([0.04, 0.04]))
    0.0
    """
    weighted_var = float(np.dot(pi, np.diag(a)))
    port_var = portfolio_variance(a, pi)
    return 0.5 * (weighted_var - port_var)


def excess_growth_rate_from_tau(
    pi: NDArray[np.float64],
    tau_pi: NDArray[np.float64],
) -> float:
    r"""Compute excess growth rate from the relative covariance matrix.

    .. math::
        \gamma^*_{\pi}(t) = \frac{1}{2} \sum_i \pi_i \tau^{\pi}_{ii}

    This is the numéraire-invariant form of the excess growth rate.

    Parameters
    ----------
    pi : ndarray of shape (n,)
        Portfolio weights.
    tau_pi : ndarray of shape (n, n) or (n,)
        Relative covariance matrix τ^π, or just its diagonal.

    Returns
    -------
    float
        Excess growth rate.

    References
    ----------
    F&K Survey Eq. 3.6, FKK Eq. 5.4
    """
    if tau_pi.ndim == 2:
        diag = np.diag(tau_pi)
    else:
        diag = tau_pi  # already just the diagonal
    return 0.5 * float(np.dot(pi, diag))


def portfolio_growth_rate(
    pi: NDArray[np.float64],
    gamma: NDArray[np.float64],
    a: NDArray[np.float64],
) -> float:
    r"""Compute portfolio growth rate γ_π(t).

    .. math::
        \gamma_{\pi}(t) = \sum_i \pi_i \gamma_i + \gamma^*_{\pi}

    where γ_i = b_i - a_{ii}/2 is the individual stock growth rate.

    The long-term growth of portfolio value is determined by this quantity:

    .. math::
        \lim_{T \to \infty} \frac{1}{T} \log V^{\pi}(T)
        = \lim_{T \to \infty} \frac{1}{T} \int_0^T \gamma_{\pi}(t)\,dt
        \quad \text{a.s.}

    Parameters
    ----------
    pi : ndarray of shape (n,)
        Portfolio weights.
    gamma : ndarray of shape (n,)
        Individual stock growth rates γ_i = b_i - a_{ii}/2.
    a : ndarray of shape (n, n)
        Instantaneous covariance rate matrix.

    Returns
    -------
    float
        Portfolio growth rate.

    References
    ----------
    F&K Survey Eq. 1.12, BFK Eq. 5.2-5.4
    """
    weighted_growth = float(np.dot(pi, gamma))
    return weighted_growth + excess_growth_rate(pi, a)


def relative_performance_rate(
    pi: NDArray[np.float64],
    mu: NDArray[np.float64],
    gamma: NDArray[np.float64],
    a: NDArray[np.float64],
) -> float:
    r"""Drift rate of relative performance log(V^π / V^μ).

    .. math::
        \text{rate} = \gamma^*_{\pi} - \gamma^*_{\mu}
            + \sum_i (\pi_i - \mu_i)(\gamma_i - \bar{\gamma}_{\mu})

    where :math:`\bar{\gamma}_{\mu} = \sum_i \mu_i \gamma_i`.

    The first two terms capture the excess-growth differential (pure
    diversification effect). The third captures the growth-rate tilt
    toward stocks with higher/lower individual growth rates.

    Parameters
    ----------
    pi : ndarray of shape (n,)
        Portfolio weights.
    mu : ndarray of shape (n,)
        Market weights.
    gamma : ndarray of shape (n,)
        Individual stock growth rates γ_i = b_i - a_{ii}/2.
    a : ndarray of shape (n, n)
        Instantaneous covariance rate matrix.

    Returns
    -------
    float
        Instantaneous drift rate of log-relative value.

    References
    ----------
    F&K Survey Eq. 3.4
    """
    gamma_star_pi = excess_growth_rate(pi, a)
    gamma_star_mu = excess_growth_rate(mu, a)
    gamma_bar_mu = float(np.dot(mu, gamma))
    growth_tilt = float(np.dot(pi - mu, gamma - gamma_bar_mu))
    return gamma_star_pi - gamma_star_mu + growth_tilt


def excess_growth_rate_bounds(
    pi: NDArray[np.float64],
    eps: float,
    M: float,
) -> tuple[float, float]:
    r"""Theoretical bounds on γ*_π under the non-degeneracy condition.

    .. math::
        \frac{\varepsilon}{2}(1 - \pi_{(1)})
        \leq \gamma^*_{\pi}
        \leq M(1 - \pi_{(1)})

    where π_{(1)} = max_i π_i is the largest portfolio weight.

    Parameters
    ----------
    pi : ndarray of shape (n,)
        Portfolio weights.
    eps : float
        Lower non-degeneracy constant.
    M : float
        Upper non-degeneracy constant.

    Returns
    -------
    tuple of (float, float)
        (lower_bound, upper_bound) on γ*_π.

    References
    ----------
    FKK Eq. 5.12
    """
    pi_max = float(np.max(pi))
    lower = 0.5 * eps * (1.0 - pi_max)
    upper = M * (1.0 - pi_max)
    return lower, upper


# ---------------------------------------------------------------------------
# Atlas model special cases (BFK 2005)
# ---------------------------------------------------------------------------


def atlas_excess_growth_rate_uncorrelated(
    pi: NDArray[np.float64],
    sigma_sq: NDArray[np.float64],
) -> float:
    r"""Excess growth rate for uncorrelated stocks (Atlas model).

    .. math::
        \gamma^*_{\pi} = \frac{1}{2} \sum_i \pi_i (1 - \pi_i) \sigma^2_i

    This is the special case of the general formula when stocks are
    driven by independent Brownian motions (a_{ij} = 0 for i ≠ j).

    Parameters
    ----------
    pi : ndarray of shape (n,)
        Portfolio weights.
    sigma_sq : ndarray of shape (n,)
        Individual stock variance rates σ²_i.

    Returns
    -------
    float
        Excess growth rate in the uncorrelated case.

    References
    ----------
    BFK Eq. 5.3
    """
    return 0.5 * float(np.sum(pi * (1.0 - pi) * sigma_sq))


def atlas_excess_growth_rate_equal_weighted(
    n: int,
    sigma_sq: NDArray[np.float64],
) -> float:
    r"""Excess growth rate of equally-weighted portfolio in Atlas model.

    .. math::
        \gamma^*_{\eta} = \frac{n-1}{2n^2} \sum_k \sigma^2_k

    For constant volatility σ²:

    .. math::
        \gamma^*_{\eta} = \frac{(n-1)\sigma^2}{2n} \to \frac{\sigma^2}{2}
        \quad \text{as } n \to \infty

    Parameters
    ----------
    n : int
        Number of stocks.
    sigma_sq : ndarray of shape (n,)
        Variance rates for each rank position.

    Returns
    -------
    float
        Excess growth rate of the equal-weighted portfolio.

    References
    ----------
    BFK Eq. 5.14
    """
    require(n >= 2, f"Need at least 2 stocks, got {n}")
    return (n - 1) / (2 * n**2) * float(np.sum(sigma_sq))


def atlas_market_growth_rate(gamma: float) -> float:
    r"""Long-term growth rate of the market portfolio in the Atlas model.

    In the Atlas model, the market portfolio has growth rate equal to
    the common drift parameter γ:

    .. math::
        G^{\mu}(n) = \gamma

    Parameters
    ----------
    gamma : float
        Common drift parameter of the Atlas model.

    Returns
    -------
    float
        Market portfolio growth rate.

    References
    ----------
    BFK Eq. 5.10
    """
    return gamma
