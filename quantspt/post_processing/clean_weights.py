"""Weight cleaning utilities for post-optimisation processing.

Provides functions to remove noise from continuous optimisation solutions:
zero out negligible weights, round for display, and clip to bounds while
maintaining the simplex constraint (weights sum to 1).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._preconditions import ensure, require

__all__ = [
    "clean_weights",
    "enforce_bounds",
    "round_weights",
]


def clean_weights(
    weights: NDArray[np.float64],
    cutoff: float = 1e-4,
) -> NDArray[np.float64]:
    """Zero out negligible weights and renormalise to the simplex.

    Optimisers often produce very small but non-zero allocations to assets
    that should effectively be excluded. This function sets weights below
    the cutoff to zero and rescales the remainder so the portfolio still
    sums to 1.

    Parameters
    ----------
    weights : ndarray of shape (n,)
        Portfolio weights (must be non-negative, sum to 1).
    cutoff : float
        Absolute threshold below which weights are zeroed (default 1e-4).

    Returns
    -------
    ndarray of shape (n,)
        Cleaned weights summing to 1.

    Raises
    ------
    SPTInvariantError
        If *weights* is not 1-D, contains negatives, or if all weights
        fall below the cutoff.
    """
    require(weights.ndim == 1, f"weights must be 1-D, got ndim={weights.ndim}")
    require(
        bool(np.all(weights >= -1e-10)),
        "weights must be non-negative",
    )
    require(cutoff >= 0, f"cutoff must be non-negative, got {cutoff}")

    cleaned = np.where(weights >= cutoff, weights, 0.0)
    total = cleaned.sum()
    require(total > 0, "all weights are below the cutoff; cannot renormalise")

    result = cleaned / total
    ensure(
        abs(result.sum() - 1.0) < 1e-10,
        "cleaned weights must sum to 1",
    )
    return result


def round_weights(
    weights: NDArray[np.float64],
    decimals: int = 4,
) -> NDArray[np.float64]:
    """Round weights to a fixed number of decimal places and renormalise.

    Useful for display or reporting where exact floating-point precision
    is not needed, while preserving the simplex constraint.

    Parameters
    ----------
    weights : ndarray of shape (n,)
        Portfolio weights (non-negative, sum to 1).
    decimals : int
        Number of decimal places to round to (default 4).

    Returns
    -------
    ndarray of shape (n,)
        Rounded weights summing to 1.
    """
    require(weights.ndim == 1, f"weights must be 1-D, got ndim={weights.ndim}")
    require(decimals >= 0, f"decimals must be non-negative, got {decimals}")

    rounded = np.round(weights, decimals)
    total = rounded.sum()
    if total == 0.0:
        return rounded

    result = rounded / total
    ensure(
        abs(result.sum() - 1.0) < 1e-10,
        "rounded weights must sum to 1",
    )
    return result


def enforce_bounds(
    weights: NDArray[np.float64],
    lower: float = 0.0,
    upper: float = 1.0,
) -> NDArray[np.float64]:
    """Clip weights to [lower, upper] and renormalise to the simplex.

    Enforces box constraints that the optimiser may have slightly violated
    due to numerical tolerance, then rescales to maintain unit sum.

    Parameters
    ----------
    weights : ndarray of shape (n,)
        Portfolio weights.
    lower : float
        Lower bound for each weight (default 0.0).
    upper : float
        Upper bound for each weight (default 1.0).

    Returns
    -------
    ndarray of shape (n,)
        Clipped weights summing to 1.

    Raises
    ------
    SPTInvariantError
        If bounds are invalid or if clipping produces an all-zero vector.
    """
    require(weights.ndim == 1, f"weights must be 1-D, got ndim={weights.ndim}")
    require(lower <= upper, f"lower ({lower}) must be <= upper ({upper})")
    require(lower >= 0.0, f"lower bound must be non-negative, got {lower}")
    require(upper <= 1.0, f"upper bound must be <= 1.0, got {upper}")

    clipped = np.clip(weights, lower, upper)
    total = clipped.sum()
    require(total > 0, "all clipped weights are zero; cannot renormalise")

    result = clipped / total
    ensure(
        abs(result.sum() - 1.0) < 1e-10,
        "bounded weights must sum to 1",
    )
    return result
