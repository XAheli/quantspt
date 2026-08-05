r"""Shrinkage covariance estimators.

Implements the Ledoit-Wolf (2004) linear shrinkage estimator and Oracle
Approximating Shrinkage (OAS) for the covariance rate matrix.  These
estimators reduce estimation error by shrinking the sample covariance
toward a structured target (scaled identity), with an analytically
optimal shrinkage intensity.

The shrinkage estimator takes the form:

.. math::
    \hat{\Sigma}_{\text{shrunk}} = \alpha\,F + (1 - \alpha)\,S

where S is the sample covariance, F is the shrinkage target, and
α ∈ [0, 1] is the shrinkage intensity.

Mathematical References
-----------------------
- Covariance rate estimation: F&K Survey Eq. 1.3
- Non-degeneracy condition: FKK Eq. 2.3
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..._preconditions import require

__all__ = [
    "ledoit_wolf",
    "oracle_approximating_shrinkage",
]


def _shrinkage_target_identity(
    S: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Scaled identity target: F = trace(S)/n · I_n."""
    n = S.shape[0]
    mu = np.trace(S) / n
    return mu * np.eye(n)


def ledoit_wolf(
    returns: NDArray[np.float64],
    *,
    annualize: bool = True,
    frequency: int = 252,
    min_observations: int | None = None,
) -> dict[str, NDArray[np.float64] | float | int]:
    r"""Ledoit-Wolf linear shrinkage covariance estimator.

    Shrinks the sample covariance matrix toward a scaled identity target
    to minimise expected Frobenius loss.  The optimal shrinkage intensity
    is computed analytically following the Ledoit-Wolf (2004) formula.

    The estimator is:

    .. math::
        \hat{\Sigma} = \hat{\alpha}\,\frac{\mathrm{tr}(S)}{n}\,I_n
                       + (1 - \hat{\alpha})\,S

    where the optimal intensity :math:`\hat{\alpha}` balances bias
    (from shrinking) against variance (from the noisy sample estimate).
    This is particularly important for SPT applications where n/T is
    not negligible, ensuring the covariance rate matrix satisfies the
    non-degeneracy condition (FKK Eq. 2.3).

    Parameters
    ----------
    returns : ndarray of shape (T, n)
        Matrix of returns (log-returns recommended).
    annualize : bool
        If ``True``, scale the result by *frequency*.
    frequency : int
        Trading days per year, default 252.
    min_observations : int or None
        Minimum required observations.  Defaults to ``n + 1``.

    Returns
    -------
    dict with keys:
        ``'covariance'`` : ndarray of shape (n, n)
            Shrinkage-estimated covariance (annualised if requested).
        ``'raw'`` : ndarray of shape (n, n)
            Un-annualised shrinkage estimate.
        ``'sample_covariance'`` : ndarray of shape (n, n)
            Un-annualised sample covariance S.
        ``'shrinkage_target'`` : ndarray of shape (n, n)
            Scaled identity target F.
        ``'shrinkage_intensity'`` : float
            Optimal α ∈ [0, 1].
        ``'n_observations'`` : int
            Number of observations used.

    Raises
    ------
    SPTInvariantError
        If the number of observations is insufficient.

    References
    ----------
    F&K Survey Eq. 1.3 (covariance rate definition)
    """
    returns = np.asarray(returns, dtype=np.float64)
    require(returns.ndim == 2, f"returns must be 2-D, got shape {returns.shape}")

    T, n = returns.shape
    if min_observations is None:
        min_observations = n + 1
    require(
        min_observations <= T,
        f"Need at least {min_observations} observations, got {T}",
    )

    X = returns - returns.mean(axis=0)
    S = (X.T @ X) / (T - 1)

    target = _shrinkage_target_identity(S)
    mu_val = np.trace(S) / n

    alpha = _ledoit_wolf_intensity(X, S, mu_val, T, n)

    shrunk = alpha * target + (1.0 - alpha) * S

    result: dict[str, NDArray[np.float64] | float | int] = {
        "covariance": shrunk * frequency if annualize else shrunk,
        "raw": shrunk,
        "sample_covariance": S,
        "shrinkage_target": target,
        "shrinkage_intensity": alpha,
        "n_observations": T,
    }
    return result


def _ledoit_wolf_intensity(
    X: NDArray[np.float64],
    S: NDArray[np.float64],
    mu: float,
    T: int,
    n: int,
) -> float:
    r"""Compute the optimal shrinkage intensity (Ledoit-Wolf 2004).

    The three quantities needed are:

    - δ² = ‖S − μI‖²_F / n²
    - β̄² = (1/(n²T²)) Σ_t ‖x_t x_t' − S‖²_F
    - β² = min(β̄², δ²)

    Then α = β² / δ².
    """
    delta_sq = np.sum((S - mu * np.eye(n)) ** 2) / n**2

    if delta_sq < 1e-30:
        return 1.0

    beta_bar_sq = 0.0
    for t in range(T):
        x_t = X[t : t + 1].T  # (n, 1)
        outer = x_t @ x_t.T
        beta_bar_sq += np.sum((outer - S) ** 2)
    beta_bar_sq /= n**2 * T**2

    beta_sq = min(beta_bar_sq, delta_sq)
    alpha = float(np.clip(beta_sq / delta_sq, 0.0, 1.0))
    return alpha


def oracle_approximating_shrinkage(
    returns: NDArray[np.float64],
    *,
    annualize: bool = True,
    frequency: int = 252,
    min_observations: int | None = None,
) -> dict[str, NDArray[np.float64] | float | int]:
    r"""Oracle Approximating Shrinkage (OAS) covariance estimator.

    An improved shrinkage estimator that better approximates the oracle
    shrinkage (which requires knowledge of the true covariance).  The
    OAS formula yields a closed-form intensity that converges faster
    to the oracle than the Ledoit-Wolf estimator.

    Uses the same shrinkage structure as :func:`ledoit_wolf` but with
    the Chen-Wiesel-Eldar-Hero (2010) intensity formula.

    Parameters
    ----------
    returns : ndarray of shape (T, n)
        Matrix of returns.
    annualize : bool
        If ``True``, scale by *frequency*.
    frequency : int
        Trading days per year.
    min_observations : int or None
        Minimum required observations.

    Returns
    -------
    dict
        Same structure as :func:`ledoit_wolf`.

    References
    ----------
    F&K Survey Eq. 1.3 (covariance rate definition)
    """
    returns = np.asarray(returns, dtype=np.float64)
    require(returns.ndim == 2, f"returns must be 2-D, got shape {returns.shape}")

    T, n = returns.shape
    if min_observations is None:
        min_observations = n + 1
    require(
        min_observations <= T,
        f"Need at least {min_observations} observations, got {T}",
    )

    X = returns - returns.mean(axis=0)
    S = (X.T @ X) / (T - 1)

    target = _shrinkage_target_identity(S)
    mu_val = np.trace(S) / n

    alpha = _oas_intensity(S, mu_val, T, n)

    shrunk = alpha * target + (1.0 - alpha) * S

    result: dict[str, NDArray[np.float64] | float | int] = {
        "covariance": shrunk * frequency if annualize else shrunk,
        "raw": shrunk,
        "sample_covariance": S,
        "shrinkage_target": target,
        "shrinkage_intensity": alpha,
        "n_observations": T,
    }
    return result


def _oas_intensity(
    S: NDArray[np.float64],
    mu: float,
    T: int,
    n: int,
) -> float:
    """OAS shrinkage intensity (Chen et al. 2010).

    Closed-form approximation to the oracle intensity.
    """
    trace_S = np.trace(S)
    trace_S2 = float(np.sum(S**2))

    rho_num = (1.0 - 2.0 / n) * trace_S2 + trace_S**2
    rho_den = (T + 1.0 - 2.0 / n) * (trace_S2 - trace_S**2 / n)

    if abs(rho_den) < 1e-30:
        return 1.0

    alpha = float(np.clip(rho_num / rho_den, 0.0, 1.0))
    return alpha
