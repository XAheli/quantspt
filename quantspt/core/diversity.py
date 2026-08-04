"""Market diversity measures and conditions.

Diversity is the central structural property in SPT that enables relative
arbitrage. A market is "diverse" if no single stock ever dominates the
entire market in terms of relative capitalization.

Mathematical References
-----------------------
- Strict diversity: FKK Eq. 4.1
- Weak diversity: FKK Eq. 4.2
- Asymptotic weak diversity: FKK Eq. 4.3
- p-Diversity measure: Fernholz (2002), F&K Survey Remark 11.1
- Entropy: F&K Survey Eq. 11.5
- Arbitrage horizon bound: FKK Eq. 4.5
- Sufficient conditions for diversity: FKK Theorem 6.1
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require

__all__ = [
    "p_diversity",
    "entropy",
    "herfindahl_hirschman_index",
    "concentration_ratio",
    "is_diverse",
    "is_weakly_diverse",
    "diversity_deficit",
    "arbitrage_horizon_bound",
    "intrinsic_volatility_condition",
]


def p_diversity(
    mu: NDArray[np.float64],
    p: float,
) -> float:
    r"""Compute the p-diversity measure D_p(μ).

    .. math::
        D_p(\mu) = \left(\sum_{i=1}^n \mu_i^p\right)^{1/(1-p)}

    For p ∈ (0, 1), this is a concave function on the simplex that
    measures how "spread out" the market weights are. It equals 1
    when all weight is concentrated in one stock, and equals n^{1/(1-p)}
    when weights are uniform.

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Market weights on the simplex (positive, sum to 1).
    p : float
        Diversity parameter, must be in (0, 1).

    Returns
    -------
    float
        p-diversity measure. Higher values indicate more diverse markets.

    References
    ----------
    Fernholz (2002), F&K Survey Remark 11.1 (Example 3)
    """
    require(0 < p < 1, f"Diversity parameter p must be in (0, 1), got {p}")
    require(bool(np.all(mu > 0)), "All market weights must be strictly positive")
    return float(np.sum(mu**p)) ** (1.0 / (1.0 - p))


def entropy(mu: NDArray[np.float64]) -> float:
    r"""Compute Shannon entropy of market weights.

    .. math::
        H(\mu) = -\sum_{i=1}^n \mu_i \log \mu_i

    Entropy is maximized (= log n) at uniform weights and minimized (= 0)
    at full concentration. It serves as the generating function G = exp(H)
    for the entropy-weighted portfolio.

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Market weights (positive, sum to 1).

    Returns
    -------
    float
        Shannon entropy. In [0, log n].

    References
    ----------
    F&K Survey Eq. 11.5, Lukacs Lectures §11
    """
    require(
        bool(np.all(mu > 0)), "All market weights must be strictly positive for entropy"
    )
    return -float(np.sum(mu * np.log(mu)))


def herfindahl_hirschman_index(mu: NDArray[np.float64]) -> float:
    r"""Compute the Herfindahl-Hirschman Index (HHI).

    .. math::
        \text{HHI} = \sum_{i=1}^n \mu_i^2

    HHI is 1/n for uniform weights and 1 for full concentration.
    It equals 1 - 2γ*_μ/trace(a) for the market portfolio when
    all stocks have equal variance.

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Market weights.

    Returns
    -------
    float
        HHI in [1/n, 1].
    """
    return float(np.sum(mu**2))


def concentration_ratio(
    mu: NDArray[np.float64],
    k: int = 5,
) -> float:
    r"""Compute top-k concentration ratio.

    .. math::
        CR_k = \sum_{i=1}^k \mu_{(i)}

    where μ_{(i)} denotes the i-th largest market weight.

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Market weights.
    k : int
        Number of top stocks to include.

    Returns
    -------
    float
        Sum of k largest weights. In [k/n, 1].
    """
    require(1 <= k <= len(mu), f"k must be in [1, {len(mu)}], got {k}")
    sorted_desc = np.sort(mu)[::-1]
    return float(np.sum(sorted_desc[:k]))


def is_diverse(
    mu: NDArray[np.float64],
    delta: float,
) -> bool:
    r"""Check strict diversity condition (FKK Eq. 4.1).

    A market is strictly diverse with parameter δ if:

    .. math::
        \max_i \mu_i(t) \leq 1 - \delta \quad \text{a.s., for all } t

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Market weights at a given time.
    delta : float
        Diversity parameter δ ∈ (0, 1).

    Returns
    -------
    bool
        True if the market satisfies strict diversity at this instant.

    References
    ----------
    FKK Eq. 4.1
    """
    require(0 < delta < 1, f"δ must be in (0, 1), got {delta}")
    return float(np.max(mu)) <= 1.0 - delta


def is_weakly_diverse(
    mu: NDArray[np.float64],
    delta: float,
    p: float,
) -> bool:
    r"""Check weak diversity condition (FKK Eq. 4.2).

    A market is weakly diverse with parameters (δ, p) if:

    .. math::
        \sum_{i=1}^n \mu_i^p(t) \geq 1 + \delta \quad \text{a.s., for all } t

    This is weaker than strict diversity: it allows individual stocks to
    temporarily dominate, as long as the overall distribution remains
    sufficiently spread.

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Market weights.
    delta : float
        Diversity parameter δ > 0.
    p : float
        Exponent parameter p ∈ (0, 1).

    Returns
    -------
    bool
        True if the market satisfies weak diversity at this instant.

    References
    ----------
    FKK Eq. 4.2
    """
    require(delta > 0, f"δ must be positive, got {delta}")
    require(0 < p < 1, f"p must be in (0, 1), got {p}")
    return float(np.sum(mu**p)) >= 1.0 + delta


def diversity_deficit(
    mu: NDArray[np.float64],
    p: float,
) -> float:
    r"""Measure how far a market is from the weak diversity condition.

    Returns Σ μ_i^p - 1. Weak diversity (FKK Eq. 4.2) requires this to
    exceed δ > 0 at all times.

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Market weights.
    p : float
        Exponent parameter p ∈ (0, 1).

    Returns
    -------
    float
        Diversity deficit. Positive means weakly diverse; negative means
        the diversity condition fails.
    """
    require(0 < p < 1, f"p must be in (0, 1), got {p}")
    return float(np.sum(mu**p)) - 1.0


def arbitrage_horizon_bound(
    n: int,
    p: float,
    eps: float,
    delta: float,
) -> float:
    r"""Minimum time horizon for relative arbitrage under weak diversity.

    Under the weak diversity condition with parameters (δ, p) and
    non-degeneracy constant ε, relative arbitrage exists for:

    .. math::
        T \geq \frac{2 \log n}{p \varepsilon \delta}

    Parameters
    ----------
    n : int
        Number of stocks in the market.
    p : float
        Diversity exponent, p ∈ (0, 1).
    eps : float
        Non-degeneracy constant (smallest eigenvalue of a(t)).
    delta : float
        Diversity parameter δ > 0.

    Returns
    -------
    float
        Minimum horizon T* for relative arbitrage.

    References
    ----------
    FKK Eq. 4.5
    """
    require(n >= 2, f"Need at least 2 stocks, got {n}")
    require(0 < p < 1, f"p must be in (0, 1), got {p}")
    require(eps > 0, f"Non-degeneracy constant must be positive, got {eps}")
    require(delta > 0, f"Diversity parameter must be positive, got {delta}")
    return 2.0 * np.log(n) / (p * eps * delta)


def intrinsic_volatility_condition(
    gamma_star_mu: float,
    zeta: float,
) -> bool:
    r"""Check the sufficient intrinsic volatility condition for arbitrage.

    If the market's excess growth rate satisfies γ*_μ ≥ ζ > 0 at all times,
    then relative arbitrage exists for:

    .. math::
        T > \frac{H(\mu(0))}{\zeta}

    where H is the entropy of market weights.

    Parameters
    ----------
    gamma_star_mu : float
        Current excess growth rate of the market portfolio.
    zeta : float
        Required lower bound ζ > 0.

    Returns
    -------
    bool
        True if the sufficient condition holds.

    References
    ----------
    F&K Survey Eq. 11.8-11.12
    """
    return gamma_star_mu >= zeta
