"""Diversity conditions for relative arbitrage.

Diversity conditions are the structural properties of a market that
guarantee the existence of relative arbitrage opportunities.  Strict
diversity bounds the maximum weight of any single stock; weak diversity
requires sufficient spread in the p-norm of market weights.

Mathematical References
-----------------------
- Strict diversity: FKK Eq. 4.1
- Weak diversity: FKK Eq. 4.2
- Asymptotic weak diversity: FKK Eq. 4.3
- Diversity parameter estimation: FKK §4
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require

__all__ = [
    "check_asymptotic_weak_diversity",
    "check_strict_diversity",
    "check_weak_diversity",
    "estimate_diversity_parameters",
]


def check_strict_diversity(
    mu: NDArray[np.float64],
    delta: float,
) -> bool:
    r"""Check strict diversity condition (FKK Eq. 4.1).

    A market is strictly diverse with parameter δ if:

    .. math::
        \max_i \mu_i(t) \leq 1 - \delta

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Market weights at a given time.
    delta : float
        Diversity parameter δ ∈ (0, 1).

    Returns
    -------
    bool
        True if strict diversity holds at this instant.

    References
    ----------
    FKK Eq. 4.1
    """
    require(0 < delta < 1, f"δ must be in (0, 1), got {delta}")
    return float(np.max(mu)) <= 1.0 - delta


def check_weak_diversity(
    mu: NDArray[np.float64],
    delta: float,
    p: float,
) -> bool:
    r"""Check weak diversity condition (FKK Eq. 4.2).

    A market is weakly diverse with parameters (δ, p) if:

    .. math::
        \sum_{i=1}^n \mu_i^p(t) \geq 1 + \delta

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Market weights (positive, sum to 1).
    delta : float
        Diversity parameter δ > 0.
    p : float
        Exponent parameter p ∈ (0, 1).

    Returns
    -------
    bool
        True if weak diversity holds at this instant.

    References
    ----------
    FKK Eq. 4.2
    """
    require(delta > 0, f"δ must be positive, got {delta}")
    require(0 < p < 1, f"p must be in (0, 1), got {p}")
    require(bool(np.all(mu > 0)), "All weights must be positive")
    return float(np.sum(mu**p)) >= 1.0 + delta


def check_asymptotic_weak_diversity(
    mu_path: NDArray[np.float64],
    p: float,
    delta: float,
) -> bool:
    r"""Check asymptotic weak diversity condition (FKK Eq. 4.3).

    Asymptotic weak diversity holds if the time-averaged p-norm
    exceeds the threshold:

    .. math::
        \liminf_{T \to \infty} \frac{1}{T} \int_0^T
        \sum_{i=1}^n \mu_i^p(t)\,dt \geq 1 + \delta

    This is approximated by the sample mean of Σ μ_i^p over the path.

    Parameters
    ----------
    mu_path : ndarray of shape (T, n)
        Market weight path over T time steps.
    p : float
        Exponent parameter p ∈ (0, 1).
    delta : float
        Diversity parameter δ > 0.

    Returns
    -------
    bool
        True if the empirical time average satisfies the condition.

    References
    ----------
    FKK Eq. 4.3
    """
    require(mu_path.ndim == 2, "mu_path must be 2-D (T, n)")
    require(delta > 0, f"δ must be positive, got {delta}")
    require(0 < p < 1, f"p must be in (0, 1), got {p}")
    require(bool(np.all(mu_path > 0)), "All weights must be positive")

    p_norms = np.sum(mu_path**p, axis=1)
    return float(np.mean(p_norms)) >= 1.0 + delta


def estimate_diversity_parameters(
    mu_path: NDArray[np.float64],
    p: float,
) -> dict[str, float]:
    r"""Estimate diversity parameters from observed market weight paths.

    Computes statistics of the diversity deficit Σ μ_i^p − 1 over
    the observation period to determine whether diversity conditions
    are satisfied and with what confidence.

    Parameters
    ----------
    mu_path : ndarray of shape (T, n)
        Market weight path over T time steps.
    p : float
        Exponent parameter p ∈ (0, 1).

    Returns
    -------
    dict with keys:
        ``delta_mean`` : float
            Time-averaged diversity deficit (Σ μ_i^p − 1).
        ``delta_min`` : float
            Minimum observed diversity deficit.
        ``delta_std`` : float
            Standard deviation of the diversity deficit.
        ``fraction_diverse`` : float
            Fraction of time steps satisfying Σ μ_i^p > 1.
        ``max_weight`` : float
            Maximum individual weight observed (for strict diversity).

    References
    ----------
    FKK §4
    """
    require(mu_path.ndim == 2, "mu_path must be 2-D (T, n)")
    require(0 < p < 1, f"p must be in (0, 1), got {p}")
    require(bool(np.all(mu_path > 0)), "All weights must be positive")

    deficits = np.sum(mu_path**p, axis=1) - 1.0
    return {
        "delta_mean": float(np.mean(deficits)),
        "delta_min": float(np.min(deficits)),
        "delta_std": float(np.std(deficits)),
        "fraction_diverse": float(np.mean(deficits > 0)),
        "max_weight": float(np.max(mu_path)),
    }
