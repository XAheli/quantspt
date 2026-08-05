r"""Diversity parameter estimation from market weight time series.

Estimates the diversity parameter delta from observed market weights,
including rolling diversity deficit computation and bootstrap
confidence intervals.

Mathematical References
-----------------------
- Diversity deficit: FKK Eq. 4.2
- Weak diversity condition: FKK Eq. 4.2
- p-Diversity measure: Fernholz (2002), F&K Survey Remark 11.1
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._preconditions import require

__all__ = [
    "bootstrap_diversity_ci",
    "estimate_diversity_parameter",
    "rolling_diversity_deficit",
]


def rolling_diversity_deficit(
    weights: NDArray[np.float64],
    p: float,
) -> NDArray[np.float64]:
    r"""Compute the diversity deficit at each time step.

    The diversity deficit is:

    .. math::
        \Delta(t) = \sum_{i=1}^n \mu_i(t)^p - 1

    The weak diversity condition (FKK Eq. 4.2) requires Delta(t) >= delta > 0
    at all times.

    Parameters
    ----------
    weights : ndarray of shape (T, n)
        Time series of market weights.
    p : float
        Diversity exponent, p in (0, 1).

    Returns
    -------
    ndarray of shape (T,)
        Diversity deficit at each time step.

    References
    ----------
    FKK Eq. 4.2
    """
    weights = np.asarray(weights, dtype=np.float64)
    require(weights.ndim == 2, f"weights must be 2-D, got shape {weights.shape}")
    require(0 < p < 1, f"p must be in (0, 1), got {p}")

    deficits: NDArray[np.float64] = np.sum(weights**p, axis=1) - 1.0
    return deficits


def estimate_diversity_parameter(
    weights: NDArray[np.float64],
    p: float,
    *,
    quantile: float = 0.05,
) -> dict[str, float]:
    r"""Estimate the diversity parameter delta from market weight data.

    The diversity parameter delta is estimated as a quantile of the
    observed diversity deficits. Under the weak diversity condition
    (FKK Eq. 4.2), we need delta > 0 at all times, so the estimate
    uses a conservative quantile (default 5th percentile).

    Parameters
    ----------
    weights : ndarray of shape (T, n)
        Time series of market weights.
    p : float
        Diversity exponent, p in (0, 1).
    quantile : float
        Quantile level for conservative delta estimate (default 0.05).

    Returns
    -------
    dict with keys:
        ``'delta'`` : float
            Estimated diversity parameter (quantile of deficits).
        ``'mean_deficit'`` : float
            Mean diversity deficit across the sample.
        ``'min_deficit'`` : float
            Minimum observed deficit.
        ``'is_weakly_diverse'`` : float
            1.0 if all deficits > 0, else 0.0.

    References
    ----------
    FKK Eq. 4.2
    """
    deficits = rolling_diversity_deficit(weights, p)

    return {
        "delta": float(np.quantile(deficits, quantile)),
        "mean_deficit": float(np.mean(deficits)),
        "min_deficit": float(np.min(deficits)),
        "is_weakly_diverse": 1.0 if float(np.min(deficits)) > 0 else 0.0,
    }


def bootstrap_diversity_ci(
    weights: NDArray[np.float64],
    p: float,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> dict[str, float]:
    r"""Bootstrap confidence interval for the diversity parameter.

    Resamples blocks of the time series to construct a confidence
    interval for delta, accounting for temporal dependence.

    Parameters
    ----------
    weights : ndarray of shape (T, n)
        Time series of market weights.
    p : float
        Diversity exponent, p in (0, 1).
    n_bootstrap : int
        Number of bootstrap resamples.
    confidence : float
        Confidence level (default 0.95).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        ``'delta_mean'`` : float
            Mean delta across bootstrap resamples.
        ``'ci_lower'`` : float
            Lower bound of confidence interval.
        ``'ci_upper'`` : float
            Upper bound of confidence interval.

    References
    ----------
    FKK Eq. 4.2
    """
    weights = np.asarray(weights, dtype=np.float64)
    require(weights.ndim == 2, f"weights must be 2-D, got shape {weights.shape}")
    require(0 < p < 1, f"p must be in (0, 1), got {p}")
    require(0 < confidence < 1, f"confidence must be in (0, 1), got {confidence}")

    T = weights.shape[0]
    rng = np.random.default_rng(seed)

    block_size = max(1, int(np.sqrt(T)))

    delta_samples = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        n_blocks = (T + block_size - 1) // block_size
        block_starts = rng.integers(0, T - block_size + 1, size=n_blocks)
        indices = np.concatenate(
            [np.arange(s, min(s + block_size, T)) for s in block_starts]
        )[:T]
        boot_weights = weights[indices]
        deficits = np.sum(boot_weights**p, axis=1) - 1.0
        delta_samples[b] = float(np.quantile(deficits, 0.05))

    alpha = 1.0 - confidence
    return {
        "delta_mean": float(np.mean(delta_samples)),
        "ci_lower": float(np.quantile(delta_samples, alpha / 2)),
        "ci_upper": float(np.quantile(delta_samples, 1.0 - alpha / 2)),
    }
