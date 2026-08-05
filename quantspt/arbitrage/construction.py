"""Explicit arbitrage portfolio construction.

Given that diversity conditions are satisfied, this module constructs
the explicit portfolio that achieves relative arbitrage over a
sufficient time horizon.

Mathematical References
-----------------------
- Diversity-weighted portfolio: FKK Eq. 4.4, F&K Survey Remark 11.1
- Modified entropy portfolio: F&K Survey Eq. 11.6–11.7
- Performance bound: FKK Eq. 4.5
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require

__all__ = [
    "construct_arbitrage_portfolio",
    "diversity_arbitrage_portfolio",
    "modified_entropy_arbitrage_portfolio",
]


def diversity_arbitrage_portfolio(
    mu: NDArray[np.float64],
    p: float = 0.5,
) -> NDArray[np.float64]:
    r"""Construct the p-diversity arbitrage portfolio.

    The diversity-weighted portfolio π^{(p)} with weights:

    .. math::
        \pi_i^{(p)} = \frac{\mu_i^p}{\sum_j \mu_j^p}

    Under the weak diversity condition (FKK Eq. 4.2), this portfolio
    outperforms the market almost surely for T ≥ T*.

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Market weights (positive, sum to 1).
    p : float
        Diversity exponent p ∈ (0, 1).

    Returns
    -------
    ndarray of shape (n,)
        Arbitrage portfolio weights (sum to 1).

    References
    ----------
    FKK Eq. 4.4, F&K Survey Remark 11.1 (Example 3)
    """
    require(mu.ndim == 1, "mu must be 1-D")
    require(0 < p < 1, f"p must be in (0, 1), got {p}")
    require(bool(np.all(mu > 0)), "All weights must be positive")

    mu_p = mu**p
    return mu_p / float(np.sum(mu_p))


def modified_entropy_arbitrage_portfolio(
    mu: NDArray[np.float64],
    c: float | None = None,
) -> NDArray[np.float64]:
    r"""Construct the modified entropy arbitrage portfolio.

    The generating function H_c(μ) = c − Σ μ_i log μ_i produces
    portfolio weights via the Fernholz formula:

    .. math::
        \pi_i = \mu_i \left[
            \frac{-(1 + \log\mu_i)}{H_c(\mu)}
            + 1 + \frac{\sum_k \mu_k (1 + \log\mu_k)}{H_c(\mu)}
        \right]

    Under the sufficient intrinsic volatility condition (γ*_μ ≥ ζ > 0),
    this portfolio achieves relative arbitrage for T > H_c(μ(0))/ζ.

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Market weights (positive, sum to 1).
    c : float, optional
        Shift parameter.  If ``None``, set to 1.0 (ensures H_c > 0
        since H ∈ [0, log n]).

    Returns
    -------
    ndarray of shape (n,)
        Portfolio weights (sum to 1).

    References
    ----------
    F&K Survey Eq. 11.6–11.7
    """
    require(mu.ndim == 1, "mu must be 1-D")
    require(bool(np.all(mu > 0)), "All weights must be positive")

    if c is None:
        c = 1.0
    require(c > 0, f"c must be positive, got {c}")

    H = -float(np.sum(mu * np.log(mu)))
    Hc = c + H
    require(Hc > 0, f"H_c must be positive, got {Hc}")

    log_grad = -(1.0 + np.log(mu)) / Hc
    S = float(np.dot(mu, log_grad))
    pi = (log_grad + 1.0 - S) * mu

    total = float(np.sum(pi))
    if abs(total - 1.0) > 1e-8:
        pi = pi / total

    return pi


def construct_arbitrage_portfolio(
    mu: NDArray[np.float64],
    a: NDArray[np.float64],
    method: str = "diversity",
    **kwargs: Any,
) -> NDArray[np.float64]:
    r"""Construct an explicit arbitrage portfolio.

    Dispatcher that selects and constructs the appropriate arbitrage
    portfolio based on the specified method.

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Market weights (positive, sum to 1).
    a : ndarray of shape (n, n)
        Covariance rate matrix (used for detection, not directly
        in portfolio construction).
    method : str
        Construction method: ``'diversity'`` or ``'entropy'``.
    **kwargs
        Passed to the underlying constructor (e.g., ``p=0.5``
        for diversity, ``c=1.0`` for entropy).

    Returns
    -------
    ndarray of shape (n,)
        Arbitrage portfolio weights (sum to 1).

    Raises
    ------
    ValueError
        If the method is not recognised.

    References
    ----------
    FKK Eq. 4.4 (diversity), F&K Survey Eq. 11.6–11.7 (entropy)
    """
    if method == "diversity":
        p = kwargs.get("p", 0.5)
        return diversity_arbitrage_portfolio(mu, p=p)
    elif method == "entropy":
        c = kwargs.get("c", None)
        return modified_entropy_arbitrage_portfolio(mu, c=c)
    else:
        msg = f"Unknown method '{method}'. Use 'diversity' or 'entropy'."
        raise ValueError(msg)
