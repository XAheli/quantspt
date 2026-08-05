"""Mirror portfolios for short-horizon arbitrage construction.

The mirror portfolio π̂ = 2μ − π reflects a portfolio through the market.
The key identity relating π and π̂ enables arbitrage construction on
any time horizon when the relative covariance is bounded away from zero.

Mathematical References
-----------------------
- Mirror definition: FKK Eq. 8.1
- Mirror covariance: FKK Eq. 8.3–8.4
- Performance identity: FKK Eq. 8.7
- Short-horizon arbitrage: FKK §8, Lemma 8.1
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require

__all__ = [
    "mirror_covariance_rate",
    "mirror_is_long_only",
    "mirror_performance_residual",
    "mirror_portfolio",
]


def mirror_portfolio(
    mu: NDArray[np.float64],
    pi: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Compute the mirror portfolio.

    .. math::
        \hat{\pi}_i = 2\mu_i - \pi_i

    The mirror reflects π through the market weights μ.  If π
    overweights stock i (π_i > μ_i), then π̂ underweights it by the
    same amount, and vice versa.

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Market weights (positive, sum to 1).
    pi : ndarray of shape (n,)
        Portfolio weights (sum to 1).

    Returns
    -------
    ndarray of shape (n,)
        Mirror portfolio weights (sum to 1).

    References
    ----------
    FKK Eq. 8.1
    """
    require(len(mu) == len(pi), "mu and pi must have the same length")
    return 2.0 * mu - pi


def mirror_is_long_only(
    mu: NDArray[np.float64],
    pi: NDArray[np.float64],
) -> bool:
    r"""Check whether the mirror portfolio has non-negative weights.

    The mirror π̂_i = 2μ_i − π_i ≥ 0 requires π_i ≤ 2μ_i for all i.

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Market weights.
    pi : ndarray of shape (n,)
        Portfolio weights.

    Returns
    -------
    bool
        True if all mirror weights are non-negative.

    References
    ----------
    FKK §8
    """
    pi_hat = mirror_portfolio(mu, pi)
    return bool(np.all(pi_hat >= -1e-12))


def mirror_covariance_rate(
    pi: NDArray[np.float64],
    mu: NDArray[np.float64],
    a: NDArray[np.float64],
) -> float:
    r"""Relative covariance rate τ^μ_{ππ} appearing in the mirror identity.

    .. math::
        \tau^{\mu}_{\pi\pi} = (\pi - \mu)' a (\pi - \mu)

    This quantity drives the mirror performance identity (FKK Eq. 8.7).

    Parameters
    ----------
    pi : ndarray of shape (n,)
        Portfolio weights.
    mu : ndarray of shape (n,)
        Market weights.
    a : ndarray of shape (n, n)
        Covariance rate matrix.

    Returns
    -------
    float
        Relative covariance rate (non-negative).

    References
    ----------
    FKK Eq. 8.3–8.4
    """
    n = len(pi)
    require(len(mu) == n, "pi and mu must have the same length")
    require(a.shape == (n, n), f"Covariance shape {a.shape} vs {n} assets")
    diff = pi - mu
    return float(diff @ a @ diff)


def mirror_performance_residual(
    V_pi: NDArray[np.float64],
    V_pi_hat: NDArray[np.float64],
    V_mu: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Compute the residual of the mirror performance identity.

    The mirror identity (FKK Eq. 8.7) states:

    .. math::
        \log V^{\pi}(t) + \log V^{\hat{\pi}}(t)
        = 2\log V^{\mu}(t) + \int_0^t \tau^{\mu}_{\pi\pi}(s)\,ds

    Since the integral is non-negative, a weaker inequality always holds:

    .. math::
        \log V^{\pi}(t) + \log V^{\hat{\pi}}(t) \geq 2\log V^{\mu}(t)

    This function computes the residual:
        log V^π(t) + log V^{π̂}(t) − 2 log V^μ(t)
    which should be non-negative.

    Parameters
    ----------
    V_pi : ndarray of shape (T,)
        Portfolio value path.
    V_pi_hat : ndarray of shape (T,)
        Mirror portfolio value path.
    V_mu : ndarray of shape (T,)
        Market portfolio value path.

    Returns
    -------
    ndarray of shape (T,)
        Residual at each time step (should be ≥ 0).

    References
    ----------
    FKK Eq. 8.7
    """
    T = len(V_pi)
    require(len(V_pi_hat) == T, "Value paths must have equal length")
    require(len(V_mu) == T, "Value paths must have equal length")
    require(bool(np.all(V_pi > 0)), "Portfolio values must be positive")
    require(bool(np.all(V_pi_hat > 0)), "Mirror values must be positive")
    require(bool(np.all(V_mu > 0)), "Market values must be positive")

    return np.log(V_pi) + np.log(V_pi_hat) - 2.0 * np.log(V_mu)
