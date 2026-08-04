"""Market weight dynamics and capital distribution.

The market weight μ_i(t) = X_i(t) / (X_1(t) + ... + X_n(t)) is the
fraction of total capitalisation held by stock i.  This module provides
the market weight SDE, ranked weights, and the coherence condition that
links individual growth rates to market-level quantities.

Mathematical References
-----------------------
- Market weight SDE: F&K Survey Eq. 2.4
- Ranked market weights: F&K Survey Eq. 1.18
- Growth rate of market portfolio: F&K Survey Eq. 2.7
- Coherence condition: F&K Survey Eq. 2.9
- Capital distribution: BFK §4, F&K Survey §2
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require

__all__ = [
    "capital_distribution_curve",
    "coherence_residual",
    "log_log_capital_curve",
    "market_excess_growth_rate",
    "market_weight_diffusion",
    "market_weight_drift",
    "rank_permutation",
    "ranked_weights",
    "validate_weights",
    "verify_coherence",
]


# ---------------------------------------------------------------------------
# Weight validation
# ---------------------------------------------------------------------------


def validate_weights(
    mu: NDArray[np.float64],
    *,
    tol: float = 1e-8,
    label: str = "Market weights",
) -> None:
    r"""Validate that a weight vector lies on the open simplex.

    Checks that all entries are strictly positive and sum to 1 within
    tolerance.  Raises :class:`~quantspt.errors.SPTInvariantError` on
    failure.

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Weight vector to validate.
    tol : float
        Tolerance for the sum-to-one check.
    label : str
        Name used in error messages.
    """
    require(mu.ndim == 1, f"{label}: expected 1-D array, got ndim={mu.ndim}")
    require(len(mu) >= 2, f"{label}: need ≥ 2 assets, got {len(mu)}")
    require(
        bool(np.all(mu > 0)),
        f"{label}: all entries must be > 0, min={float(np.min(mu)):.2e}",
    )
    require(
        abs(float(np.sum(mu)) - 1.0) < tol,
        f"{label}: must sum to 1, got {float(np.sum(mu)):.8f}",
    )


# ---------------------------------------------------------------------------
# Market weight dynamics  (F&K Survey Eq. 2.4)
# ---------------------------------------------------------------------------


def market_weight_drift(
    mu: NDArray[np.float64],
    gamma: NDArray[np.float64],
    a: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Drift vector of the market weight SDE.

    The market weight process satisfies (F&K Survey Eq. 2.4):

    .. math::
        d\mu_i = \mu_i \bigl[(b_i - b_\mu)\,dt
                 + \sum_\nu (\sigma_{i\nu} - \sigma_{\mu\nu})\,dW_\nu \bigr]

    The drift component for stock *i* (without the dW term) is:

    .. math::
        \mu_i \bigl[\gamma_i - \gamma_\mu
               + \tfrac{1}{2}(a_{\mu\mu} - a_{ii})
               + a^\mu_i - a_{\mu\mu}\bigr]
        = \mu_i \bigl[\gamma_i - \gamma_\mu
               + a^\mu_i - \tfrac{1}{2}(a_{ii} + a_{\mu\mu})\bigr]

    where we use b_i = γ_i + a_{ii}/2 and b_μ = γ_μ + a_{μμ}/2.

    A simpler equivalent form uses the relative covariance diagonal:

    .. math::
        \text{drift}_i = \mu_i \bigl[(\gamma_i - \gamma_\mu)
                          + \tfrac{1}{2}\tau^\mu_{ii}\bigr]
                        \quad \text{(not exact — see note)}

    We implement the exact formula derived from b_i − b_μ expanded.

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Current market weights (positive, sum to 1).
    gamma : ndarray of shape (n,)
        Individual stock growth rates γ_i = b_i − a_{ii}/2.
    a : ndarray of shape (n, n)
        Instantaneous covariance rate matrix.

    Returns
    -------
    ndarray of shape (n,)
        Drift vector of dμ (before the dW term).

    References
    ----------
    F&K Survey Eq. 2.4
    """
    n = len(mu)
    require(a.shape == (n, n), f"Covariance shape {a.shape} vs {n} assets")

    a_mu = a @ mu  # a^μ_i = Σ_j μ_j a_{ij}
    a_mumu = float(mu @ a_mu)  # a_{μμ} = μ' a μ
    gamma_mu = float(np.dot(mu, gamma)) + 0.5 * (float(np.dot(mu, np.diag(a))) - a_mumu)

    b = gamma + 0.5 * np.diag(a)  # rates of return b_i
    b_mu = gamma_mu + 0.5 * a_mumu  # b_μ = γ_μ + a_{μμ}/2

    return mu * (b - b_mu)


def market_weight_diffusion(
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Diffusion matrix of the market weight SDE.

    From F&K Survey Eq. 2.4, the volatility loading of dμ_i is:

    .. math::
        \mu_i (\sigma_{i\nu} - \sigma_{\mu\nu})

    where σ_{μν} = Σ_j μ_j σ_{jν}.

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Current market weights.
    sigma : ndarray of shape (n, m)
        Volatility matrix (n assets × m factors).

    Returns
    -------
    ndarray of shape (n, m)
        Diffusion matrix of dμ.

    References
    ----------
    F&K Survey Eq. 2.4
    """
    sigma_mu = mu @ sigma  # (m,) market volatility loading
    return mu[:, np.newaxis] * (sigma - sigma_mu[np.newaxis, :])


# ---------------------------------------------------------------------------
# Ranked weights and permutations  (F&K Survey Eq. 1.18)
# ---------------------------------------------------------------------------


def ranked_weights(
    mu: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Sort market weights in descending order.

    Returns μ_{(1)} ≥ μ_{(2)} ≥ … ≥ μ_{(n)}, the ranked market weights.

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Market weights.

    Returns
    -------
    ndarray of shape (n,)
        Sorted weights in descending order.

    References
    ----------
    F&K Survey Eq. 1.18
    """
    return np.sort(mu)[::-1].copy()


def rank_permutation(
    mu: NDArray[np.float64],
) -> NDArray[np.intp]:
    r"""Return the permutation that sorts weights into descending order.

    If ``p = rank_permutation(mu)`` then ``mu[p]`` equals
    :func:`ranked_weights` (up to tie-breaking).

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Market weights.

    Returns
    -------
    ndarray of shape (n,) dtype intp
        Index permutation such that ``mu[p[0]] ≥ mu[p[1]] ≥ …``.
    """
    return np.argsort(mu)[::-1]


# ---------------------------------------------------------------------------
# Market portfolio growth rate  (F&K Survey Eq. 2.7)
# ---------------------------------------------------------------------------


def market_excess_growth_rate(
    mu: NDArray[np.float64],
    a: NDArray[np.float64],
) -> float:
    r"""Excess growth rate of the market portfolio γ*_μ.

    This is a convenience alias that calls the general excess growth rate
    formula with the market weight vector as the portfolio.

    .. math::
        \gamma^*_\mu = \frac{1}{2}\bigl[
            \sum_i \mu_i a_{ii} - \mu^T a \mu
        \bigr]

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Market weights.
    a : ndarray of shape (n, n)
        Covariance rate matrix.

    Returns
    -------
    float
        Excess growth rate of the market portfolio.

    References
    ----------
    F&K Survey Eq. 1.13 applied with π = μ
    """
    from .growth_rates import excess_growth_rate

    return excess_growth_rate(mu, a)


# ---------------------------------------------------------------------------
# Coherence condition  (F&K Survey Eq. 2.9)
# ---------------------------------------------------------------------------


def coherence_residual(
    gamma: NDArray[np.float64],
    mu: NDArray[np.float64],
    a: NDArray[np.float64],
    gamma_mu: float | None = None,
) -> float:
    r"""Compute the residual of the coherence condition.

    The coherence condition (F&K Survey Eq. 2.9) states:

    .. math::
        \sum_i \gamma_i \mu_i + \gamma^*_\mu = \gamma_\mu

    This function returns:

    .. math::
        \sum_i \gamma_i \mu_i + \gamma^*_\mu - \gamma_\mu

    which should be zero when the parameters are self-consistent.

    Parameters
    ----------
    gamma : ndarray of shape (n,)
        Individual stock growth rates γ_i.
    mu : ndarray of shape (n,)
        Market weights.
    a : ndarray of shape (n, n)
        Covariance rate matrix.
    gamma_mu : float, optional
        Market portfolio growth rate.  If ``None``, it is computed from
        the individual rates and excess growth rate.

    Returns
    -------
    float
        Residual; should be ≈ 0 for consistent parameters.

    References
    ----------
    F&K Survey Eq. 2.9
    """
    from .growth_rates import excess_growth_rate, portfolio_growth_rate

    weighted_gamma = float(np.dot(gamma, mu))
    gamma_star_mu = excess_growth_rate(mu, a)

    if gamma_mu is None:
        gamma_mu = portfolio_growth_rate(mu, gamma, a)

    return weighted_gamma + gamma_star_mu - gamma_mu


def verify_coherence(
    gamma: NDArray[np.float64],
    mu: NDArray[np.float64],
    a: NDArray[np.float64],
    gamma_mu: float | None = None,
    tol: float = 1e-10,
) -> bool:
    r"""Check whether the coherence condition holds.

    Parameters
    ----------
    gamma : ndarray of shape (n,)
        Individual stock growth rates.
    mu : ndarray of shape (n,)
        Market weights.
    a : ndarray of shape (n, n)
        Covariance rate matrix.
    gamma_mu : float, optional
        Market growth rate.  Computed if not supplied.
    tol : float
        Tolerance for the residual.

    Returns
    -------
    bool
        ``True`` if |residual| < tol.

    References
    ----------
    F&K Survey Eq. 2.9
    """
    return abs(coherence_residual(gamma, mu, a, gamma_mu)) < tol


# ---------------------------------------------------------------------------
# Capital distribution curve  (BFK §4, F&K Survey §2)
# ---------------------------------------------------------------------------


def capital_distribution_curve(
    mu: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Compute the capital distribution curve.

    Returns the ranked market weights — the same as :func:`ranked_weights`,
    but named for its role in analysing the shape of the capital distribution.

    In an Atlas model the steady-state curve follows a Pareto law
    (BFK §4, Eq. 4.2–4.4).

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Market weights.

    Returns
    -------
    ndarray of shape (n,)
        Ranked weights μ_{(1)} ≥ … ≥ μ_{(n)}.

    References
    ----------
    BFK §4, F&K Survey §2
    """
    return ranked_weights(mu)


def log_log_capital_curve(
    mu: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Compute the log-log capital distribution curve.

    Returns (log(rank), log(weight)) for the classic log-log plot that
    reveals Pareto-like structure.  Rank indices are 1-based.

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Market weights.

    Returns
    -------
    tuple of (ndarray, ndarray)
        (log_rank, log_weight) each of shape (n,).

    References
    ----------
    BFK §4 (Figures 4.1, 4.2)
    """
    ranked = ranked_weights(mu)
    log_rank = np.log(np.arange(1, len(mu) + 1, dtype=np.float64))
    log_weight = np.log(ranked)
    return log_rank, log_weight
