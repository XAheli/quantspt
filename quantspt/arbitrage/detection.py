"""Arbitrage opportunity detection.

Combines diversity conditions, non-degeneracy estimation, and horizon
computation to determine whether relative arbitrage opportunities exist
for a given market configuration.

Mathematical References
-----------------------
- Sufficient intrinsic volatility: F&K Survey Eq. 11.8–11.12
- Non-degeneracy condition: FKK Eq. 2.3
- Diversity-based detection: FKK §4, Eq. 4.5
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require

__all__ = [
    "ArbitrageOpportunity",
    "check_sufficient_intrinsic_volatility",
    "detect_diversity_arbitrage",
    "estimate_nondegeneracy",
]


@dataclass(frozen=True)
class ArbitrageOpportunity:
    """Result of an arbitrage detection scan.

    Attributes
    ----------
    is_detected : bool
        Whether the conditions for relative arbitrage are satisfied.
    method : str
        Detection method used (``'diversity'``, ``'intrinsic_volatility'``).
    min_horizon : float | None
        Minimum horizon T* for guaranteed outperformance, if detected.
    delta : float
        Estimated diversity parameter.
    epsilon : float
        Estimated non-degeneracy constant (smallest eigenvalue of a).
    expected_rate : float
        Expected outperformance rate for large T.
    basis : str
        Theoretical reference for the detection result.
    """

    is_detected: bool
    method: str
    min_horizon: float | None
    delta: float
    epsilon: float
    expected_rate: float
    basis: str


def estimate_nondegeneracy(
    a: NDArray[np.float64],
) -> float:
    r"""Estimate the non-degeneracy constant ε.

    Returns the smallest eigenvalue of the covariance rate matrix a,
    which serves as the non-degeneracy constant in FKK Eq. 2.3:

    .. math::
        \varepsilon \|\xi\|^2 \leq \xi' a \xi
        \quad \forall\, \xi \in \mathbb{R}^n

    Parameters
    ----------
    a : ndarray of shape (n, n)
        Symmetric covariance rate matrix.

    Returns
    -------
    float
        Smallest eigenvalue of a.  Positive for non-degenerate markets.

    References
    ----------
    FKK Eq. 2.3
    """
    require(a.ndim == 2, "a must be 2-D")
    require(a.shape[0] == a.shape[1], "a must be square")
    eigenvalues = np.linalg.eigvalsh(a)
    return float(eigenvalues[0])


def check_sufficient_intrinsic_volatility(
    gamma_star_mu: float,
    zeta: float,
) -> bool:
    r"""Check the sufficient intrinsic volatility condition.

    If the market's excess growth rate satisfies γ*_μ ≥ ζ > 0 at all
    times, then relative arbitrage exists for sufficiently long horizons.

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
    F&K Survey Eq. 11.8–11.12
    """
    require(zeta > 0, f"ζ must be positive, got {zeta}")
    return gamma_star_mu >= zeta


def detect_diversity_arbitrage(
    mu: NDArray[np.float64],
    a: NDArray[np.float64],
    p: float = 0.5,
) -> ArbitrageOpportunity:
    r"""Detect whether diversity-based relative arbitrage is available.

    Checks the weak diversity condition and computes the minimum
    arbitrage horizon from FKK Eq. 4.5:

    .. math::
        T^* = \frac{2 \log n}{p\,\varepsilon\,\delta}

    where ε is the non-degeneracy constant and δ is the diversity
    deficit.

    Parameters
    ----------
    mu : ndarray of shape (n,)
        Market weights (positive, sum to 1).
    a : ndarray of shape (n, n)
        Covariance rate matrix (symmetric PSD).
    p : float
        Diversity exponent p ∈ (0, 1).

    Returns
    -------
    ArbitrageOpportunity
        Detection result with horizon estimate and parameters.

    References
    ----------
    FKK Eq. 4.5
    """
    require(mu.ndim == 1, "mu must be 1-D")
    n = len(mu)
    require(n >= 2, f"Need ≥ 2 stocks, got {n}")
    require(0 < p < 1, f"p must be in (0, 1), got {p}")
    require(bool(np.all(mu > 0)), "All weights must be positive")
    require(a.shape == (n, n), f"Covariance shape {a.shape} vs {n} assets")

    delta = float(np.sum(mu**p)) - 1.0
    epsilon = estimate_nondegeneracy(a)

    if delta <= 0 or epsilon <= 0:
        return ArbitrageOpportunity(
            is_detected=False,
            method="diversity",
            min_horizon=None,
            delta=max(delta, 0.0),
            epsilon=max(epsilon, 0.0),
            expected_rate=0.0,
            basis="FKK Eq. 4.5 (conditions not met)",
        )

    T_star = 2.0 * np.log(n) / (p * epsilon * delta)
    expected_rate = (1.0 - p) * epsilon * delta / 2.0

    return ArbitrageOpportunity(
        is_detected=True,
        method="diversity",
        min_horizon=float(T_star),
        delta=delta,
        epsilon=epsilon,
        expected_rate=expected_rate,
        basis="FKK Eq. 4.5",
    )
